"""HTTP/WebSocket surface: local-only security, frame encodings, validation errors, run lifecycle through the API."""
from __future__ import annotations

from datetime import datetime
import json
import os
import re
import stat
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import wait_for

BASE = "http://127.0.0.1:8799"


@pytest.fixture
def app(config):
    from sopa_companion.server import create_app
    application = create_app(config, token="t" * 40)
    with TestClient(application, base_url=BASE) as client:
        yield application, client


def auth(extra=None):
    return {"Authorization": "Bearer " + "t" * 40, "Origin": BASE, **(extra or {})}


def page(application, extra=None):
    """A page's request: this engine's page key (which a launch code is traded for) in its header, and the page's own
    Origin."""
    return {"Origin": BASE, "X-SoPA-Page-Key": application.state.page_key, **(extra or {})}


def sign_in(client, code):
    """The page's trade of a launch code for its key (frontend/src/page-key.mjs): one POST, the code in a header."""
    return client.post("/api/v1/session/sign-in", headers={"Origin": BASE, "X-SoPA-Launch-Code": code})


def test_security(app):
    application, client = app
    assert client.get("/api/v1/health").status_code == 401                       # no credentials
    assert client.get("/api/v1/health", headers=auth()).status_code == 200
    assert client.get("/api/v1/health", headers={**auth(), "Host": "evil.example:8799"}).status_code == 403
    assert client.post("/api/v1/runs", json={}, headers={"Authorization": "Bearer " + "t" * 40}).status_code == 403  # no Origin
    assert client.post("/api/v1/runs", json={}, headers=auth({"Origin": "http://evil.example"})).status_code == 403
    refused = client.get("/?token=" + "t" * 40, follow_redirects=False)         # the bearer never signs a browser in
    assert refused.status_code == 401 and "set-cookie" not in refused.headers
    assert sign_in(client, "nonsense").status_code == 401
    code = client.post("/api/v1/session/launch-code", headers=auth()).json()
    assert code["expires_in"] == 45                                             # the window the launcher opens now
    # The app's address with the code serves the page (public, no cookie), spending nothing; the page trades the code.
    response = client.get("/?code=" + code["code"], follow_redirects=False)
    assert response.status_code == 200 and "set-cookie" not in response.headers and "location" not in response.headers
    assert '<script type="module" src="./src/page-key.mjs"></script>' in response.text
    assert "connect-src 'self' ws://127.0.0.1:8799;" in response.headers["content-security-policy"]
    traded = sign_in(client, code["code"])
    assert traded.json() == {"key": application.state.page_key} and traded.headers["cache-control"] == "no-store"
    assert sign_in(client, code["code"]).status_code == 401                                   # single use
    assert client.get("/api/v1/health", headers=page(application)).status_code == 200
    # A link printed for a person (launch --print-url, serve in a terminal): one use too, 10 minutes for an SSH tunnel.
    assert client.post("/api/v1/session/launch-code?link=1", headers=auth()).json()["expires_in"] == 600


def test_outside_the_api_the_engine_answers_with_the_apps_own_files_alone(app):
    """The app's files need no credential, so nothing but them may live outside /api/ (no schema route either), and
    the page itself loads with nothing but its address."""
    from fastapi.routing import APIRoute, APIWebSocketRoute
    from starlette.routing import Mount
    application, client = app
    outside = [route for route in application.routes if not getattr(route, "path", "").startswith("/api/")]
    assert [(type(route), route.path) for route in outside] == [(Mount, "")], outside
    assert all(isinstance(route, (APIRoute, APIWebSocketRoute)) for route in application.routes if route not in outside)
    assert client.get("/openapi.json").status_code == 404
    for path in ("/", "/index.html", "/src/page-key.mjs", "/src/main.js", "/styles/app.css"):
        assert client.get(path).status_code == 200, path
    assert "browser" not in client.get("/api/v1/health", headers=auth()).json(), "the engine never reads [desktop] browser"


def test_serve_in_a_terminal_prints_a_one_use_link_never_the_bearer(app):
    """`serve` in a terminal printed the bearer token in its URL (?token=), which a browser keeps
    in its history, and which issues launch codes and stops the engine for as long as it runs. It prints a one-use link
    now (10 minutes); into a log (the launcher's), the address alone, as before."""
    from sopa_companion.cli import opening_lines
    application, client = app
    text = opening_lines(BASE + "/", application, True)
    assert "t" * 40 not in text and "token" not in text and "within 10 minutes" in text
    link = re.search(r"^SoPA companion: http://127\.0\.0\.1:8799/\?code=([\w-]+)$", text, re.M).group(1)
    assert client.get("/?code=" + link, follow_redirects=False).status_code == 200            # the page, nothing spent
    assert sign_in(client, link).status_code == 200
    assert sign_in(client, link).status_code == 401                                          # once
    assert opening_lines(BASE + "/", application, False) == (f"SoPA companion: {BASE}/\nFor a private URL: python -m "
                                                             "sopa_companion launch --print-url")


def test_the_page_key_is_never_the_bearer(app):
    """The page's key is its own, new at each start and never on disk, and it travels in a header, never a cookie
    (a cookie goes to every port of 127.0.0.1): never taken as a bearer, the bearer never as the page's key, and
    neither a launch code nor /shutdown takes it."""
    application, client = app
    bearer, key = "t" * 40, application.state.page_key
    code = client.post("/api/v1/session/launch-code", headers=auth()).json()["code"]
    signed_in = sign_in(client, code)
    assert signed_in.json() == {"key": key} and bearer not in signed_in.text and key != bearer
    assert "set-cookie" not in signed_in.headers
    stolen = {"Origin": BASE, "Authorization": "Bearer " + key}
    assert client.get("/api/v1/health", headers=stolen).status_code == 401                 # the page's key as a bearer
    assert client.post("/api/v1/session/launch-code", headers=stolen).status_code == 401
    assert client.post("/api/v1/shutdown", headers=stolen).status_code == 401
    for old in ({"Origin": BASE, "X-SoPA-Page-Key": bearer},                                 # the bearer as the key
                {"Origin": BASE, "Cookie": "sopa_companion_session=" + key}):               # the key as a cookie
        assert client.get("/api/v1/health", headers=old).status_code == 401
    assert client.get("/api/v1/health", headers=page(application)).status_code == 200      # the page's own key
    assert client.post("/api/v1/session/launch-code", headers=page(application)).status_code == 401
    assert client.post("/api/v1/shutdown", headers=page(application)).status_code == 403
    assert application.state.stop_forced is False
    # A page reloaded after its engine started again holds an old key: refused, in JSON (the page says it in words).
    gone = client.get("/api/v1/health", headers={"Origin": BASE, "X-SoPA-Page-Key": "o" * 43, "Accept": "text/html"})
    assert gone.status_code == 401 and gone.json()["code"] == "E_AUTH"
    # Nothing of the page's key is on disk; the bearer is new at each start, and an older version's session.key goes.
    state = __import__("pathlib").Path(application.state.runs.root).parents[1] / "state"
    assert not any(key in path.read_text(errors="ignore") for path in state.rglob("*") if path.is_file())


def test_the_bearer_is_new_at_every_start_and_kept_nowhere_but_the_runtime_file(config, tmp_path):
    from sopa_companion.server.app import create_app, fresh_token
    config["server"]["token"] = "auto"
    old = tmp_path / "state" / "run" / "session.key"
    old.parent.mkdir(parents=True)
    old.write_text("s" * 43)
    first, second = create_app(config), create_app(config)
    assert len({first.state.token, second.state.token, "s" * 43}) == 3 and not old.exists()
    assert first.state.page_key not in (first.state.token, second.state.page_key)
    assert fresh_token(config) != fresh_token(config)


def test_a_pages_socket_is_answered_with_its_subprotocol_and_named_by_its_address(app):
    """A page's socket signs in in its handshake (server/security.py): the engine answers the page's subprotocol, which
    a browser that offered some requires, never the key; the page's name comes from the socket's address, and a page
    whose address names none (or none usable) is named by the engine. A tool's socket offers none and gets none. A page
    whose key is not this engine's is accepted, then closed at once with 4401, so it knows it is signed out."""
    from starlette.websockets import WebSocketDisconnect
    application, client = app
    with client.websocket_connect("/api/v1/ws?page=page-1", headers=page_ws(application)) as ws:
        assert ws.accepted_subprotocol == "sopa-companion" and ws.receive_json()["ev"] == "runs"
    with client.websocket_connect("/api/v1/ws", headers=TOOL_WS) as ws:
        assert ws.accepted_subprotocol is None and ws.receive_json()["ev"] == "runs"
    before = next(application.state.socket_ids)
    for url in ("/api/v1/ws", "/api/v1/ws?page=a%20b", "/api/v1/ws?page=page-2"):
        with client.websocket_connect(url, headers=page_ws(application)) as ws:
            ws.receive_json()
    assert next(application.state.socket_ids) == before + 3, "two named by the engine, one by its address"
    with client.websocket_connect("/api/v1/ws?page=page-3", headers=page_ws(application, key="x" * 43)) as ws:
        assert ws.accepted_subprotocol == "sopa-companion"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 4401 and application.state.presence.open() == 0


def test_no_idle_or_refused_connection_keeps_a_page_out_of_a_real_engine(tmp_path, xenium_root, fake_binary):
    """Nothing waits on a client (a socket's credential is checked in its handshake): 50 local connections that never
    say anything, 50 handshakes without a credential (refused) and 50 with a key not this engine's (closed with 4401)
    hold no slot a page needs. A page then signs in by POST, and its socket gets its first push at once, as a page's (it
    takes an answer to closing back, as a reload does, and /health's presence counts it alone). A real engine on a free
    port, as `serve` runs one."""
    import socket
    import subprocess
    import httpx
    from websockets.exceptions import ConnectionClosed, InvalidStatus
    from websockets.sync.client import connect
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    script = tmp_path / "serve.py"
    script.write_text(f"""
import socket, uvicorn
from sopa_companion.server import create_app
config = {{"server": {{"port": {port}}}, "desktop": {{"exit_when_closed": False}},
          "paths": {{"data_dir": {str(tmp_path / 'var')!r}, "state_dir": {str(tmp_path / 'state')!r},
                    "sample_roots": [{str(xenium_root)!r}]}},
          "proseg": {{"binary": {str(fake_binary)!r}, "threads": 2}}, "analysis": {{"python": ""}}}}
app = create_app(config, token="t" * 40)
listener = socket.socket()
listener.bind(("127.0.0.1", {port}))
uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False)).run(sockets=[listener])
""")
    env = {**os.environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[2]),
           "SOPA_COMPANION_ZARR_LOCK_DIR": str(tmp_path / "locks"), "SOPA_COMPANION_QC_FIT_LOCK": str(tmp_path / "fit.lock")}
    env.pop("DISPLAY", None)
    log = tmp_path / "engine.log"
    base, idle = f"http://127.0.0.1:{port}", []
    bearer = {"Authorization": "Bearer " + "t" * 40, "Origin": base}
    with open(log, "wb") as out:
        child = subprocess.Popen([sys.executable, str(script)], env=env, stdout=out, stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url=base, timeout=30) as http:
            def up():
                try:
                    return http.get("/api/v1/health", headers=bearer).status_code == 200
                except httpx.HTTPError:
                    return False
            wait_for(lambda: child.poll() is not None or up(), timeout=60, interval=0.1)
            assert child.poll() is None, log.read_text()[-2000:]
            code = http.post("/api/v1/session/launch-code", headers=bearer).json()["code"]
            key = http.post("/api/v1/session/sign-in", headers={"Origin": base, "X-SoPA-Launch-Code": code}).json()["key"]
            page = {"Origin": base, "X-SoPA-Page-Key": key}
            assert http.post("/api/v1/closing", json={"answer": "discard"}, headers=page).json()["answer"] == "discard"
            idle = [socket.create_connection(("127.0.0.1", port), timeout=10) for _ in range(50)]
            address = f"ws://127.0.0.1:{port}/api/v1/ws"
            for _ in range(50):
                with pytest.raises(InvalidStatus) as refused:
                    connect(address, origin=base, proxy=None, open_timeout=10).close()
                assert refused.value.response.status_code == 403
            for _ in range(50):
                with connect(address + "?page=stranger", origin=base, subprotocols=["sopa-companion", "key." + "x" * 43],
                             proxy=None, open_timeout=10) as stranger:
                    with pytest.raises(ConnectionClosed) as closed:
                        stranger.recv(timeout=10)
                    assert closed.value.rcvd.code == 4401
            started = time.monotonic()
            with connect(address + "?page=page-1", origin=base, subprotocols=["sopa-companion", "key." + key], proxy=None,
                         open_timeout=10) as live:
                first = json.loads(live.recv(timeout=2))
                assert live.subprotocol == "sopa-companion" and first["ev"] == "runs" and time.monotonic() - started < 2
                assert http.get("/api/v1/closing", headers=bearer).json()["answer"] is None, "taken back: a page's socket"
                # The one page here, under the name its address gives: the strangers never counted.
                here = http.get("/api/v1/health", headers=bearer).json()["presence"]
                assert here["pages"] == 1 and here["state"] == "here" and here["connects"] == 1
            assert child.poll() is None
    finally:
        for connection in idle:
            connection.close()
        child.kill()
        child.wait()


def test_run_through_api(app):
    application, client = app
    assert client.get("/api/v1/samples", headers=auth()).json() == []           # fresh: nothing is preset
    sample = application.state.samples.list()[0]["id"]                           # a Xenium bundle (legacy runs)
    bad = client.post("/api/v1/runs", json={"sample": sample, "box": [0, 0, 5, 5]}, headers=auth())
    assert bad.status_code == 422 and "at least 20" in bad.json()["detail"]
    # An older page's "schedule" is ignored: every run follows proseg's own schedule and starts paused on its prior.
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150], "schedule": "open"}, headers=auth()).json()
    assert run["paused"] is True and "schedule" not in run
    assert "schedule_defaults" not in client.get("/api/v1/health", headers=auth()).json()
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "speed", "speed": "max"}, headers=auth())
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "resume"}, headers=auth())
    runs = application.state.runs
    wait_for(lambda: len(runs.get(run["id"]).frames) >= 3)

    frame = client.get(f"/api/v1/runs/{run['id']}/frames/1", headers=auth({"Accept-Encoding": "gzip"}))
    assert frame.status_code == 200
    header = runs.get(run["id"]).frames[1]
    data = np.frombuffer(frame.content, "<u4")                                    # the client inflates gzip
    assert data.size == (2 + header["nz"]) * header["nx"] * header["ny"]
    plain = client.get(f"/api/v1/runs/{run['id']}/frames/1", headers=auth({"Accept-Encoding": "identity"}))
    assert plain.headers.get("content-encoding") is None and len(plain.content) == data.size * 4
    assert client.get(f"/api/v1/runs/{run['id']}/frames/99999", headers=auth()).status_code == 404
    assert client.get(f"/api/v1/runs/{run['id']}/image/..%2Frun.json", headers=auth()).status_code == 404
    transcripts = client.get(f"/api/v1/runs/{run['id']}/transcripts", headers=auth())
    assert transcripts.status_code == 200 and len(transcripts.content) % 16 == 0
    assert client.get(f"/api/v1/runs/{run['id']}/cells", headers=auth()).json() == ["aaa-1", "bbb-1"]

    ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
    with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
        ws.send_json({"op": "watch", "run": run["id"], "since": 0})
        kinds = set()
        for _ in range(6):
            kinds.add(ws.receive_json()["ev"])
        assert {"runs", "frames", "run"} <= kinds

    assert client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "nonsense"}, headers=auth()).status_code == 422
    assert client.delete(f"/api/v1/runs/{run['id']}", headers=auth()).status_code == 409   # still active
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "finish"}, headers=auth())
    wait_for(lambda: runs.get(run["id"]).meta["state"] == "done")
    assert client.delete(f"/api/v1/runs/{run['id']}", headers=auth()).status_code == 200
    assert client.get(f"/api/v1/runs/{run['id']}", headers=auth()).status_code == 404


def test_proseg_s_count_of_each_cell_is_served_for_a_counted_frame(app):
    """GET /runs/{id}/counted/{seq}: a counted frame's sidecar as proseg-live writes it (f32 per cell, each cell's
    averaged count summed over its genes), zstd as it is for a client that takes it, kept by the browser for good;
    404 for a frame proseg did not count (or not there yet), 500 for a record that names another file."""
    from sopa_companion.analysis import counts_file
    application, client = app
    sample = application.state.samples.list()[0]["id"]
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    record = application.state.runs.get(run["id"])
    wait_for(lambda: len(record.frames) == 1 and "counted_file" in record.frames[0])
    url = f"/api/v1/runs/{run['id']}/counted/0"
    expected = counts_file.counted_of(counts_file.read(record.live / record.frames[0]["counts_file"]))
    assert expected.size == 2 and expected.sum() > 0
    plain = client.get(url, headers=auth({"Accept-Encoding": "identity"}))
    assert plain.status_code == 200 and plain.headers.get("content-encoding") is None
    assert np.array_equal(np.frombuffer(plain.content, "<f4"), expected)
    assert plain.headers["cache-control"] == "private, max-age=31536000, immutable"
    packed = client.get(url, headers=auth({"Accept-Encoding": "zstd"}))
    assert packed.headers["content-encoding"] == "zstd"
    assert np.array_equal(np.frombuffer(packed.content, "<f4"), expected)
    assert client.get(f"/api/v1/runs/{run['id']}/counted/99", headers=auth()).status_code == 404
    assert client.get("/api/v1/runs/nobody/counted/0", headers=auth()).status_code == 404
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "speed", "speed": "max"}, headers=auth())
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "resume"}, headers=auth())
    wait_for(lambda: any("counted_file" not in f for f in list(record.frames)))
    seq = next(f["seq"] for f in list(record.frames) if "counted_file" not in f)     # sampled, never counted
    missing = client.get(f"/api/v1/runs/{run['id']}/counted/{seq}", headers=auth())
    assert missing.status_code == 404 and missing.json()["detail"] == "No proseg count for this frame"
    with record.lock:
        record.frames[0] = {**record.frames[0], "counted_file": "../run.json"}
    assert client.get(url, headers=auth()).status_code == 500
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "abort"}, headers=auth())
    wait_for(lambda: record.meta["state"] == "stopped")


def test_a_watch_from_the_frames_a_page_holds_names_the_counted_ones_among_them(app):
    """A page that watches from frame `since` on holds the frames before it (it reconnected, or read them as it opened
    the run): their counts lines may have merged after their records went out, so the first push names every counted
    frame among them ({ev: "counted", seqs}) and those of them read on the count proseg saves (`sampled`: a pause's,
    the hold's, a finish's or the end's line), beyond the 64 newest proseg_counts.recent and .sampled name; a watch
    from 0 gets them with the records themselves, and no such event."""
    application, client = app
    sample = application.state.samples.list()[0]["id"]
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    record = application.state.runs.get(run["id"])
    wait_for(lambda: len(record.frames) >= 1 and record.counted)
    with record.lock:
        since, counted, sampled = len(record.frames), list(record.counted), list(record.sampled)
        record.counted = list(range(0, 200, 2))               # more counted frames than the 64 recent names
        record.sampled = list(range(0, 200, 6))
    try:
        ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
        for watch_from, expected in ((max(since, 150), [s for s in range(0, 200, 2) if s < max(since, 150)]), (0, None)):
            with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
                ws.send_json({"op": "watch", "run": run["id"], "since": watch_from})
                seen = []
                while "run" not in seen:                      # the run's own push comes after the counted list
                    message = ws.receive_json()
                    seen.append(message["ev"])
                    if message["ev"] == "counted":
                        assert (message["run"], message["seqs"]) == (run["id"], expected)
                        assert message["sampled"] == [s for s in range(0, 200, 6) if s < watch_from]
                assert ("counted" in seen) == (expected is not None), seen
                if expected is not None:
                    assert seen.index("counted") < seen.index("run") and len(expected) > 64
    finally:
        with record.lock:
            record.counted = sorted(set(record.counted) | set(counted))
            record.sampled = sampled
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "abort"}, headers=auth())
    wait_for(lambda: record.meta["state"] == "stopped")


def test_a_frame_analysed_again_on_another_count_is_announced_again(app, monkeypatch):
    """The socket announces each Clusters & UMAP result once for the count it was read on (AnalysisManager
    result_counts): a frame analysed on a token's count (averaged), then again once the hold counted it with f (the
    count proseg saves), is pushed again with its new result."""
    application, client = app
    sample = application.state.samples.list()[0]["id"]
    run_id = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()["id"]
    analysis, counts = application.state.analysis, {1: "averaged"}
    monkeypatch.setattr(analysis, "describe", lambda run: {"enabled": True, "status": {"state": "ready", "seq": 1},
                                                           "generation": 0, "seqs": [1]})
    monkeypatch.setattr(analysis, "result_counts", lambda run: dict(counts))
    monkeypatch.setattr(analysis, "summary", lambda run, seq, generation=None: {"seq": seq, "iter": seq,
                                                                                "count": counts[seq]})
    ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
    with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
        ws.send_json({"op": "watch", "run": run_id, "since": 0})
        announced = []
        for count in ("averaged", "sampled"):
            counts[1] = count
            analysis.version += 1
            deadline = 400
            while deadline and len(announced) < (1 if count == "averaged" else 2):
                message = ws.receive_json()
                deadline -= 1
                if message["ev"] == "analysis" and message.get("state") == "ready":
                    announced.append((message["seq"], message["summary"]["count"]))
    assert announced == [(1, "averaged"), (1, "sampled")]
    client.post(f"/api/v1/runs/{run_id}/control", json={"action": "abort"}, headers=auth())
    wait_for(lambda: application.state.runs.get(run_id).meta["state"] == "stopped")


def test_frame_records_reach_pages_without_the_counts_keys_only_the_engine_reads(app):
    """A counted frame's record holds its whole counts line ("counts") and its count file ("counts_file": for a frame
    counted when asked, an absolute path in the spool), which the engine alone reads: /frames and the socket's frames
    event send each record without them, counted_file kept (all the page reads: main.js frameCounted), with the count
    every number of the frame reads (`count`: "averaged" for frame 0's start and a count asked for); the run keeps its
    records whole."""
    from sopa_companion.analysis.counts_file import count_of
    application, client = app
    sample = application.state.samples.list()[0]["id"]
    run_id = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()["id"]
    runs = application.state.runs
    record = runs.get(run_id)
    client.post(f"/api/v1/runs/{run_id}/control", json={"action": "resume"}, headers=auth())
    wait_for(lambda: len(record.frames) >= 2)
    assert runs.request_counts(run_id)                        # the next frame counted, into the spool
    wait_for(lambda: any(os.path.isabs(frame.get("counts_file") or "") for frame in list(record.frames)))
    client.post(f"/api/v1/runs/{run_id}/control", json={"action": "abort"}, headers=auth())
    wait_for(lambda: record.meta["state"] == "stopped")
    with record.lock:
        whole = list(record.frames)
    spooled = [frame for frame in whole if os.path.isabs(frame.get("counts_file") or "")]
    assert spooled and whole[0].get("counts_file") and all("counts" in frame for frame in spooled), "the engine's whole"
    expected = [{**{key: value for key, value in frame.items() if key not in ("counts", "counts_file")},
                 **({"count": count_of(frame["counts"])} if "counts" in frame else {})} for frame in whole]
    assert [frame["seq"] for frame in expected if frame.get("counted_file")] == sorted(record.counted)
    asked = [frame["seq"] for frame in spooled]
    assert {frame["seq"]: frame["count"] for frame in expected if frame["seq"] in (0, *asked)} \
        == dict.fromkeys((0, *asked), "averaged")
    sent = client.get(f"/api/v1/runs/{run_id}/frames", headers=auth())
    assert sent.json() == {"from": 0, "frames": expected}
    assert os.path.dirname(spooled[0]["counts_file"]) not in sent.text, "no spool path"
    assert client.get(f"/api/v1/runs/{run_id}/frames?since=1", headers=auth()).json()["frames"] == expected[1:]
    ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
    with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
        ws.send_json({"op": "watch", "run": run_id, "since": 0})
        while (message := ws.receive_json())["ev"] != "frames":
            pass
    assert (message["from"], message["frames"]) == (0, expected)
    with record.lock:
        assert record.frames == whole, "the run keeps its records whole"


def test_the_run_record_reaches_pages_without_what_only_the_engine_reads(app):
    """A run's record (Run.public: GET /runs/{id}, GET /runs, the socket's run push, a run action's answer) names its
    newest frame (`last`) as /frames sends frames, without its counts line and count file, and gives proseg's info.json
    (`info`) without its counts block's spool folder (a path on the RAM disk, which the engine alone reads); the run
    keeps both whole. A new run is paused on its prior, its frame 0 counted ("init": `count` "averaged")."""
    application, client = app
    sample = application.state.samples.list()[0]["id"]
    run_id = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()["id"]
    record = application.state.runs.get(run_id)
    wait_for(lambda: record.counted == [0] and record.info is not None)
    with record.lock:
        newest, info = record.frames[-1], record.info
    spool = info["counts"]["spool"]
    assert {"counts", "counts_file", "counted_file"} <= set(newest) and os.path.isabs(spool), "the engine's whole"
    last = {**{key: value for key, value in newest.items() if key not in ("counts", "counts_file")},
            "count": "averaged"}
    counts = {key: value for key, value in info["counts"].items() if key != "spool"}
    got = client.get(f"/api/v1/runs/{run_id}", headers=auth())
    listed = next(run for run in client.get("/api/v1/runs", headers=auth()).json() if run["id"] == run_id)
    answered = client.post(f"/api/v1/runs/{run_id}/control", json={"action": "speed", "speed": "max"}, headers=auth())
    ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
    with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
        ws.send_json({"op": "watch", "run": run_id, "since": 1})
        while (message := ws.receive_json())["ev"] != "run":
            pass
    for name, sent in (("GET /runs/{id}", got.json()), ("GET /runs", listed), ("an action's answer", answered.json()),
                       ("the run push", message["run"])):
        assert sent["last"] == last and sent["info"] == {**info, "counts": counts}, name
    assert spool not in got.text and spool not in answered.text, "no spool path"
    with record.lock:
        assert record.frames[-1] == newest and record.info["counts"]["spool"] == spool, "the run keeps them"
    client.post(f"/api/v1/runs/{run_id}/control", json={"action": "abort"}, headers=auth())
    wait_for(lambda: record.meta["state"] == "stopped")


def test_the_test_view_answers_under_qc_where_the_page_keeps_refusals_quiet(app):
    """GET /runs/{id}/qc/test/view and GET /runs/{id}/qc/test/pool live under /qc: the page's fetch wrapper
    (frontend/src/api.js) raises no "Request failed" notice for /qc paths, so the corner's fetch after its test was
    dismissed (404) and a pool asked of a run that went stay quiet; the refusal is the engine's usual problem JSON,
    never a server error."""
    import re
    from pathlib import Path
    application, client = app
    sample = application.state.samples.list()[0]["id"]
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    for run_id, detail in ((run["id"], "No test for this run"), ("nope", "Unknown run")):
        refused = client.get(f"/api/v1/runs/{run_id}/qc/test/view", headers=auth())
        assert refused.status_code == 404 and refused.headers["content-type"] == "application/problem+json"
        assert refused.json() == {"detail": detail, "status": 404}
    refused = client.get("/api/v1/runs/nope/qc/test/pool?cell=0&wait=5", headers=auth())
    assert refused.status_code == 404 and refused.headers["content-type"] == "application/problem+json"
    assert refused.json() == {"detail": "Unknown run", "status": 404}
    source = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "api.js").read_text()
    quiet = re.search(r"const quiet = (.+?);\n", source, re.S).group(1)       # its regular expressions, as Python's
    patterns = [part[part.index("/") + 1:part.rindex("/.test(path)")] for part in quiet.split("||")]
    for path in (f"/api/v1/runs/{run['id']}/qc/test/view", f"/api/v1/runs/{run['id']}/qc/test/pool?cell=0&wait=5",
                 f"/api/v1/runs/{run['id']}/qc/test/pool?cell=0"):
        assert any(re.search(p, path) for p in patterns), (path, patterns)
    assert not any(re.search(p, f"/api/v1/runs/{run['id']}/frames/3/view?box=0,0,1,1") for p in patterns)


def test_a_test_body_is_checked_before_the_run_is(app):
    """POST /runs/{id}/qc/test {cell, flag, value}: a body that is not one is refused (422, problem JSON) before the
    run's state is looked at, so a refused body pauses nothing and makes no run; `changes` (two settings at once) is
    no longer taken; a body that is not an object gets FastAPI's own 422 first. A good one passes to the pool's
    reasons: the mixing check can't run on this sample (a Xenium bundle), so it is refused (409, that reason alone);
    that the run has not reached the end of its schedule (a wait), its region is under the measured 300 µm and its
    suspects come from proseg's own count, which the constants were not measured with (caveats), would not refuse it,
    and the pool route lists them by kind."""
    application, client = app
    sample = application.state.samples.list()[0]["id"]
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    url = f"/api/v1/runs/{run['id']}/qc/test"
    good = {"cell": 0, "flag": "cell-compactness", "value": 0.02}
    hold = {"flag": "prior-weight", "value": 0.5}
    for body, words in (({"cell": 0, "changes": [hold]}, "A test changes one setting: send flag and value"),
                        ({**good, "changes": []}, "A test changes one setting: send flag and value"),
                        ({**good, "cell": -1}, "cell must be a cell number (label - 1)"),
                        ({**good, "cell": True}, "cell must be a cell number (label - 1)"),
                        ({**good, "flag": "voxel-size"},
                         "A test changes one of the settings the test form offers"),
                        ({**good, "value": "0.02"}, "value must be a number"),
                        ({**good, "value": 50}, "Choose a value from 0.001 to 10"),
                        ({**good, "flag": "diffusion-probability", "value": 0.5}, "Choose a value from 0.1 to 0.3")):
        refused = client.post(url, json=body, headers=auth())
        assert refused.status_code == 422 and refused.headers["content-type"] == "application/problem+json", body
        assert refused.json() == {"detail": words, "status": 422}, body
    assert client.post(url, json=[good], headers=auth()).status_code == 422
    answer = client.post(url, json={**good, "suspects": [{"x": 1, "y": 1, "gene": "G1", "p": 0.9}]}, headers=auth())
    assert answer.status_code == 409, answer.text
    refused = answer.json()
    assert refused == {"detail": "The mixing check can't run on this sample", "status": 409,
                       "reasons": [{"code": "check_unavailable", "words": "The mixing check can't run on this sample"}]}
    pool = client.get(f"{url}/pool", params={"cell": "0"}, headers=auth()).json()
    assert [[r["code"] for r in pool[key]] for key in ("reasons", "waits", "caveats")] == \
        [["check_unavailable"], ["not_at_end"], ["region_small", "count_basis"]]
    assert pool["caveats"][0] == {"code": "region_small", "words": "Not measured yet for regions under 300 µm",
                                  "min_um": 300.0}
    assert [r["id"] for r in client.get("/api/v1/runs", headers=auth()).json()] == [run["id"]]
    assert client.get(url, headers=auth()).status_code == 404


@pytest.fixture
def app_with_fake_analysis(config, monkeypatch):
    import sys
    from sopa_companion.server import create_app
    config["analysis"] = {"python": sys.executable, "worker_module": "tests.fake_analysis_worker", "gpu": 0}
    monkeypatch.setenv("PYTHONPATH", str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    application = create_app(config, token="t" * 40)
    with TestClient(application, base_url=BASE) as client:
        yield application, client


def test_analysis_follows_the_run_and_answers_requests(app_with_fake_analysis, monkeypatch):
    application, client = app_with_fake_analysis
    manager = application.state.analysis
    manager.module = "fake_analysis_worker"           # resolved from tests/ via PYTHONPATH below
    import os
    tests_dir = str(__import__("pathlib").Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", tests_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))
    assert client.get("/api/v1/samples", headers=auth()).json() == []           # fresh: nothing is preset
    sample = application.state.samples.list()[0]["id"]                           # a Xenium bundle (legacy runs)
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "speed", "speed": "max"}, headers=auth())
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "resume"}, headers=auth())
    runs = application.state.runs
    wait_for(lambda: runs.get(run["id"]).info is not None and len(runs.get(run["id"]).frames) >= 3)
    client.post(f"/api/v1/runs/{run['id']}/analysis", json={"enabled": True}, headers=auth())
    wait_for(lambda: len(client.get(f"/api/v1/runs/{run['id']}/analysis", headers=auth()).json()["seqs"]) >= 2, timeout=60)
    state = client.get(f"/api/v1/runs/{run['id']}/analysis", headers=auth()).json()
    summary = client.get(f"/api/v1/runs/{run['id']}/analysis/{state['latest_seq']}", headers=auth()).json()
    cells = client.get(f"/api/v1/runs/{run['id']}/analysis/{state['latest_seq']}/cells", headers=auth()).content
    assert summary["nclusters"] == 2 and len(cells) == summary["ncells"] * 14
    # A frame asked for goes as the counted frame it snaps to (proseg's count: the reply's "requested" says which).
    url = f"/api/v1/runs/{run['id']}/analysis"
    asked = client.post(url, json={"enabled": False, "seq": 1}, headers=auth()).json()["requested"]
    assert asked in (0, 1), asked
    wait_for(lambda: asked in client.get(url, headers=auth()).json()["seqs"])
    assert client.post(f"/api/v1/runs/{run['id']}/analysis", json={"seq": 99999}, headers=auth()).status_code == 404
    for bad in ({"seq": "last"}, {"enabled": True, "params": {"resolution": "high"}}):   # refused, not a server error
        assert client.post(f"/api/v1/runs/{run['id']}/analysis", json=bad, headers=auth()).status_code == 422
    assert client.get(f"/api/v1/runs/{run['id']}/analysis", headers=auth()).json()["enabled"] is False
    # Graceful when the bundle has no Xenium masks / boundaries / prior.
    assert client.get(f"/api/v1/runs/{run['id']}/xenium/masks", headers=auth()).status_code == 404
    from sopa_companion.engine import vectors
    assert client.get(f"/api/v1/runs/{run['id']}/xenium", headers=auth()).status_code == 404   # only the binary form
    empty = client.get(f"/api/v1/runs/{run['id']}/xenium", params={"format": "bin", "v": 1}, headers=auth())
    assert empty.status_code == 200 and vectors.to_json(vectors.decode(empty.content)) == {"cells": [], "nuclei": []}
    assert client.get(f"/api/v1/runs/{run['id']}/prior", headers=auth()).status_code == 404
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "finish"}, headers=auth())
    wait_for(lambda: runs.get(run["id"]).meta["state"] == "done")


def test_analysis_worker_crash_recovers(app_with_fake_analysis, monkeypatch):
    application, client = app_with_fake_analysis
    import os
    tests_dir = str(__import__("pathlib").Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", tests_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("FAKE_ANALYSIS_CRASH_AT", "0")
    manager = application.state.analysis
    manager.module = "fake_analysis_worker"
    assert client.get("/api/v1/samples", headers=auth()).json() == []           # fresh: nothing is preset
    sample = application.state.samples.list()[0]["id"]                           # a Xenium bundle (legacy runs)
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "resume"}, headers=auth())
    runs = application.state.runs
    wait_for(lambda: runs.get(run["id"]).info is not None and len(runs.get(run["id"]).frames) >= 2)
    client.post(f"/api/v1/runs/{run['id']}/analysis", json={"seq": 0}, headers=auth())
    wait_for(lambda: client.get(f"/api/v1/runs/{run['id']}/analysis", headers=auth()).json()["status"].get("state") == "error", timeout=30)
    monkeypatch.delenv("FAKE_ANALYSIS_CRASH_AT")
    manager.failed_at = 0
    url = f"/api/v1/runs/{run['id']}/analysis"
    asked = client.post(url, json={"seq": 1}, headers=auth()).json()["requested"]
    assert asked in (0, 1)                           # the counted frame seq 1 snaps to
    wait_for(lambda: asked in client.get(url, headers=auth()).json()["seqs"], timeout=30)
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "abort"}, headers=auth())


def test_analysis_cadence_and_view_bounds(app_with_fake_analysis, monkeypatch):
    """A sampling run has its newest frame analysed at most once per interval (frames arrive every few ms here), while
    a frame asked for explicitly (scrubbing) is analysed at once; summaries carry the view box and centroids."""
    import os
    import time
    application, client = app_with_fake_analysis
    tests_dir = str(__import__("pathlib").Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", tests_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))
    manager = application.state.analysis
    manager.module = "fake_analysis_worker"
    manager.min_interval = 1.0                       # [analysis] min_interval_s = 1.0
    started, analyse = [], manager._analyse
    manager._analyse = lambda run_id, seq: (started.append((time.monotonic(), seq)), analyse(run_id, seq))[1]
    sample = application.state.samples.list()[0]["id"]
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "speed", "speed": "max"}, headers=auth())
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "resume"}, headers=auth())
    runs = application.state.runs
    wait_for(lambda: runs.get(run["id"]).info is not None and len(runs.get(run["id"]).frames) >= 3)
    state = client.post(f"/api/v1/runs/{run['id']}/analysis", json={"enabled": True}, headers=auth()).json()
    assert state["interval_s"] == 1.0
    wait_for(lambda: len(started) >= 3, timeout=30)
    asked = time.monotonic()
    # The frame asked for goes as the counted frame it snaps to (the reply's "requested"), at once.
    want = client.post(f"/api/v1/runs/{run['id']}/analysis", json={"seq": 1}, headers=auth()).json()["requested"]
    assert want in (0, 1)
    wait_for(lambda: any(seq == want and t >= asked for t, seq in started), timeout=10)
    first = next((t, seq) for t, seq in started if seq == want and t >= asked)
    assert first[0] - asked < 0.5
    automatic = [t for t, seq in started if (t, seq) != first]
    assert min(np.diff(automatic)) >= 0.95 and len(runs.get(run["id"]).frames) > 10 * len(automatic)
    latest = client.get(f"/api/v1/runs/{run['id']}/analysis", headers=auth()).json()["latest_seq"]
    summary = client.get(f"/api/v1/runs/{run['id']}/analysis/{latest}", headers=auth()).json()
    (x0, x1), (y0, y1) = summary["bounds"]["x"], summary["bounds"]["y"]
    assert x0 <= x1 and y0 <= y1 and all(len(c["centroid"]) == 3 for c in summary["clusters"])
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "abort"}, headers=auth())
    wait_for(lambda: runs.get(run["id"]).meta["state"] == "stopped")


def test_clusters_and_umap_are_always_on_and_a_tool_switch_reaches_the_page(app_with_fake_analysis, monkeypatch):
    """No switch on the page: a new run is analysed from its first frames on, and when a tool (the harness) switches its
    analysis off or on again, the page hears of it over the WebSocket (it follows the engine's switch)."""
    import os
    application, client = app_with_fake_analysis
    tests_dir = str(__import__("pathlib").Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", tests_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))
    application.state.analysis.module = "fake_analysis_worker"
    sample = application.state.samples.list()[0]["id"]
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    runs = application.state.runs
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "resume"}, headers=auth())
    wait_for(lambda: client.get(f"/api/v1/runs/{run['id']}/analysis", headers=auth()).json()["seqs"], timeout=60)
    assert client.get(f"/api/v1/runs/{run['id']}/analysis", headers=auth()).json()["enabled"] is True
    ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
    with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
        ws.send_json({"op": "watch", "run": run["id"], "since": 0})
        for wanted in (False, True):
            client.post(f"/api/v1/runs/{run['id']}/analysis", json={"enabled": wanted}, headers=auth())
            for _ in range(400):
                event = ws.receive_json()
                if event["ev"] == "analysis" and event.get("enabled") is wanted:
                    break
            else:
                raise AssertionError(f"the page never heard that the run's analysis is {'on' if wanted else 'off'}")
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "abort"}, headers=auth())
    wait_for(lambda: runs.get(run["id"]).meta["state"] == "stopped")


def test_health_names_what_the_mixing_check_is_missing(config, monkeypatch):
    """GET /health's qc.reason names the missing piece: "analysis" where Clusters & UMAP cannot run ([analysis] python:
    nothing is scored without them, whatever [qc] says), "software" without the check's own ([qc] python), and nothing
    when both are there; qc.state is what the check's probe says (its fake here, through the shared config). The page
    has words for each reason (qc-copy.mjs)."""
    from sopa_companion.server import create_app

    def health_of() -> dict:
        application = create_app(config, token="t" * 40)
        with TestClient(application, base_url=BASE) as client:
            wait_for(lambda: client.get("/api/v1/health", headers=auth()).json()["qc"]["state"] != "checking")
            return client.get("/api/v1/health", headers=auth()).json()
    for analysis_python, qc_python, reason, state in ((False, False, "analysis", "unavailable"),
                                                      (False, True, "analysis", "ready"),
                                                      (True, False, "software", "unavailable"),
                                                      (True, True, None, "ready")):
        # "off", not "auto" (""): auto takes this Python where it has the modules, as CI's one environment has them all
        config["analysis"]["python"] = sys.executable if analysis_python else "off"
        config["qc"]["python"] = sys.executable if qc_python else "off"
        health = health_of()
        assert health["qc"] == {"available": reason is None, "state": state, "reason": reason}, (analysis_python,
                                                                                                 qc_python)
        assert health["analysis"] is analysis_python
    # What the check's probe found, where Clusters & UMAP can run: each reason the page has words for; no answer
    # (its probe crashed) keeps the check offered.
    config["analysis"]["python"] = config["qc"]["python"] = sys.executable
    for mode, state, reason in (("no_nvidia", "unavailable", "no_gpu"), ("hidden", "unavailable", "no_gpu"),
                                ("torch_cpu", "unavailable", "torch_cpu"), ("driver", "unavailable", "driver"),
                                ("missing:scvi-tools,scviva-tools", "unavailable", "software"),
                                ("crash", "unknown", None)):
        monkeypatch.setenv("FAKE_QC_PROBE", mode)
        assert health_of()["qc"] == {"available": reason is None, "state": state, "reason": reason}, mode


def test_a_python_that_cannot_run_clusters_turns_them_off_and_the_page_hears(app_with_fake_analysis, monkeypatch):
    """The worker's hello says this Python lacks what Clusters & UMAP need (lasting): the watched run's analysis is
    pushed as "unavailable" (the page then reads describe, which names the packages), /health says they cannot run,
    and the mixing check's reason becomes "analysis"."""
    import os
    application, client = app_with_fake_analysis
    tests_dir = str(__import__("pathlib").Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", tests_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("FAKE_ANALYSIS_LASTING", "igraph,umap-learn")
    application.state.analysis.module = "fake_analysis_worker"
    assert client.get("/api/v1/health", headers=auth()).json()["analysis"] is True
    sample = application.state.samples.list()[0]["id"]
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
    with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
        ws.send_json({"op": "watch", "run": run["id"], "since": 0})
        for _ in range(400):
            event = ws.receive_json()
            if event["ev"] == "analysis" and event.get("state") == "unavailable":
                break
        else:
            raise AssertionError("the page never heard that Clusters & UMAP cannot run")
    described = client.get(f"/api/v1/runs/{run['id']}/analysis", headers=auth()).json()
    assert described["available"] is False and described["missing"] == ["igraph", "umap-learn"]
    health = client.get("/api/v1/health", headers=auth()).json()
    assert health["analysis"] is False and health["qc"]["available"] is False and health["qc"]["reason"] == "analysis"
    refused = client.post(f"/api/v1/runs/{run['id']}/analysis", json={"seq": 0}, headers=auth())
    assert refused.status_code == 503
    assert refused.json()["detail"] == "Clusters and the UMAP can't run here: this Python environment lacks igraph and " \
                                       "umap-learn"
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "abort"}, headers=auth())
    wait_for(lambda: application.state.runs.get(run["id"]).meta["state"] == "stopped")

def test_analysis_settings_are_checked_and_the_page_hears_of_a_change(app_with_fake_analysis, monkeypatch):
    """The UMAP pane's Advanced settings through the API: refusals in plain words, a change that starts generation 1, the
    WebSocket's settings push before any result of it (the same frame announced again, of the new settings), results of
    other generations never served, and a run created afterwards starting with the settings last applied."""
    import os
    application, client = app_with_fake_analysis
    tests_dir = str(__import__("pathlib").Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", tests_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))
    application.state.analysis.module = "fake_analysis_worker"
    sample = application.state.samples.list()[0]["id"]
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    url, runs = f"/api/v1/runs/{run['id']}/analysis", application.state.runs
    wait_for(lambda: 0 in client.get(url, headers=auth()).json()["seqs"], timeout=60)   # paused on frame 0, analysed
    first = client.get(url, headers=auth()).json()
    assert first["generation"] == 0 and first["changed"] is False and first["settings"] == first["defaults"]
    names = "resolution, n_neighbors, min_dist, min_counts and n_top_genes"
    for body, detail in (({"params": "high"}, f"params must be an object of settings ({names.replace(' and', ',')})"),
                         ({"params": {"n_neighbours": 20}}, f"unknown setting 'n_neighbours' (the settings are {names})"),
                         ({"params": {"resolution": "high"}}, "resolution must be a number"),
                         ({"params": {"n_neighbors": 15.5}}, "n_neighbors must be a whole number"),
                         ({"params": {"resolution": 9}}, "resolution must be from 0.1 to 3")):
        refused = client.post(url, json=body, headers=auth())
        assert refused.status_code == 422 and refused.json()["detail"] == detail
    assert client.get(url, headers=auth()).json()["generation"] == 0             # a refusal changed nothing
    ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
    with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
        ws.send_json({"op": "watch", "run": run["id"], "since": 0})
        for _ in range(400):                                 # the frame of the defaults, announced first
            event = ws.receive_json()
            if event["ev"] == "analysis" and event.get("state") == "ready" and event.get("seq") == 0:
                break
        assert event["generation"] == 0 and event["summary"]["params"]["resolution"] == 1.0
        changed = client.post(url, json={"params": {"resolution": 2.0}, "seq": 0}, headers=auth()).json()
        assert changed["generation"] == 1 and changed["changed"] is True and changed["settings"]["resolution"] == 2.0
        heard = []
        for _ in range(400):
            event = ws.receive_json()
            if event["ev"] == "analysis":
                heard.append(event)
                if event.get("state") == "ready":
                    break
        else:
            raise AssertionError("the page never heard of a result of the new settings")
        assert heard[0]["state"] == "settings" and heard[0]["generation"] == 1 and heard[0]["changed"] is True
        assert heard[0]["settings"]["resolution"] == 2.0 and heard[0]["defaults"]["resolution"] == 1.0
        assert heard[0]["limits"]["resolution"]["max"] == 3.0
        ready = heard[-1]                                    # the same frame again, of the new settings
        assert ready["seq"] == 0 and ready["generation"] == 1 and ready["summary"]["params"]["resolution"] == 2.0
        assert all(e.get("generation") == 1 for e in heard)
    assert client.get(f"{url}/0", params={"generation": 1}, headers=auth()).json()["params"]["resolution"] == 2.0
    for path in (f"{url}/0", f"{url}/0/cells"):
        assert client.get(path, params={"generation": 0}, headers=auth()).status_code == 404   # earlier settings
        assert client.get(path, params={"generation": -1}, headers=auth()).status_code == 422
    assert client.get(f"{url}/0/cells", params={"generation": 1}, headers=auth()).status_code == 200
    # A run made after the change starts with the settings last applied: its own generation 1.
    assert client.post(f"/api/v1/runs/{run['id']}/leave", json={"answer": "discard"}, headers=auth()).status_code == 200
    wait_for(lambda: run["id"] not in {r["id"] for r in runs.list()}, timeout=60)
    second = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    later = client.get(f"/api/v1/runs/{second['id']}/analysis", headers=auth()).json()
    assert later["generation"] == 1 and later["changed"] is True and later["settings"]["resolution"] == 2.0
    client.post(f"/api/v1/runs/{second['id']}/control", json={"action": "abort"}, headers=auth())
    wait_for(lambda: runs.get(second["id"]).meta["state"] == "stopped")

# A report as the page sends it every 10 s while a run is open (frontend/src/telemetry.mjs, Telemetry.report()).
PAGE_REPORT = {
    "v": 1, "period_s": 10.0, "hidden": False,
    "run": {"mode": "tiled", "state": "running", "cells": 40000, "live": True, "umap": True, "view": "2d",
            "level": "overview", "stage": [3840, 2160]},
    "rates": {"stage": 2.2, "umap": 0.9, "timeline": 2.0, "batches": 2.1, "snapshots": 2.0, "raf_frames": 2.3,
              "raf_callbacks": 4.6},
    "raf_lag_ms": {"p95": 41.3, "max": 12345.6},
    "long_tasks": {"count": 2, "max_ms": 230.0},
    "input": {"count": 20, "delay_p95_ms": 19.0, "delay_max_ms": 1500.5, "duration_p95_ms": 190.0, "duration_max_ms": 2000.0},
    "heap_mb": 312.4,
    "display": {"dpr": 2, "window": [1920, 1080], "screen": [2560, 1440],
                "renderer": "ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA Example GPU), NVIDIA)"},
    "fetch": {"view": {"n": 20, "bytes": 38000000, "wire": 2000000}, "image": {"n": 1, "bytes": 4000000, "wire": 2000000}},
}


def test_telemetry_reports_are_kept_for_the_next_bug_report(app):
    application, client = app
    path = application.state.telemetry.path
    report = {"t": 1234.5, **PAGE_REPORT}
    assert client.post("/api/v1/telemetry", json=report, headers={"Origin": BASE}).status_code == 401      # no session
    assert client.post("/api/v1/telemetry", json=report, headers={"Authorization": "Bearer " + "t" * 40}).status_code == 403
    # The browser's same-origin keepalive fetch: the page's key in its header, the page's Origin, a JSON body.
    keepalive = page(application, {"Content-Type": "application/json"})
    assert client.post("/api/v1/telemetry", content=json.dumps(report), headers=keepalive).status_code == 204
    assert client.post("/api/v1/telemetry", content=json.dumps(report), headers={**keepalive, "Origin": "null"}).status_code == 403
    line = json.loads(path.read_text())
    assert list(line)[0] == "t" and datetime.fromisoformat(line["t"]).utcoffset().total_seconds() == 0
    assert line["page_t"] == 1234.5 and {k: v for k, v in line.items() if k not in ("t", "page_t")} == PAGE_REPORT
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    json_headers = auth({"Content-Type": "application/json"})
    for body, status in ((b"[1, 2]", 422), (b"{not json", 422), (b'{"fps": NaN}', 422), (b"", 422),
                         (b'{"rates": {"stage": 1e999}}', 422), (b'{"rates": {"stage": -1e400}}', 422),   # inf
                         (b'{"s": "\\ud800"}', 422),                    # a lone surrogate: no UTF-8 for it
                         (b'{"pad": "' + b"x" * 17000 + b'"}', 413)):
        assert client.post("/api/v1/telemetry", content=body, headers=json_headers).status_code == status, body[:20]

    def chunked():                                           # no Content-Length: still cut off at 16 KB
        yield b'{"pad": "'
        for _ in range(40):
            yield b"x" * 1000
        yield b'"}'
    assert client.post("/api/v1/telemetry", content=chunked(), headers=json_headers).status_code == 413
    assert len(path.read_text().splitlines()) == 1          # refused reports are not written


def test_telemetry_rotates_and_summarises_at_most_every_30_s(tmp_path):
    from sopa_companion.server.app import TelemetryLog
    now, logs = [0.0], []
    log = TelemetryLog(tmp_path / "logs" / "telemetry.jsonl", keep_bytes=400, clock=lambda: now[0], log=logs.append)
    for i in range(12):
        log.write({"fps": i + 0.5, "note": "x" * 40})
        now[0] += 5
    folder = tmp_path / "logs"
    assert sorted(p.name for p in folder.iterdir()) == ["telemetry.jsonl", "telemetry.jsonl.1"]
    lines = [json.loads(l) for name in ("telemetry.jsonl.1", "telemetry.jsonl") for l in (folder / name).read_text().splitlines()]
    assert [l["fps"] for l in lines] == [i + 0.5 for i in range(12 - len(lines), 12)]   # the newest, in order
    assert all((folder / name).stat().st_size <= 400 for name in ("telemetry.jsonl", "telemetry.jsonl.1"))
    assert len(logs) == 2 and all("\n" not in line for line in logs)    # at 0 s and 30 s, one line each
    assert "(1 report since" in logs[0] and "(6 reports since" in logs[1] and 'fps=6.5 note="xxxx' in logs[1]


def test_the_log_line_holds_the_pages_whole_report():
    from sopa_companion.server.app import TELEMETRY_LINE_CHARS, telemetry_summary
    line = telemetry_summary({"t": "2026-10-16T14:00:00.000+00:00", **PAGE_REPORT})
    assert not line.endswith("…") and len(line) <= TELEMETRY_LINE_CHARS and "\n" not in line
    for part in ('run.mode="tiled"', "run.stage=[3840,2160]", "rates.stage=2.2", "raf_lag_ms.max=12345.6",
                 "long_tasks.count=2", "input.delay_max_ms=1500.5", "heap_mb=312.4", "display.screen=[2560,1440]",
                 'display.renderer="ANGLE (NVIDIA, Vulkan', "fetch.view.bytes=38000000", "fetch.image.n=1"):
        assert part in line, part
    assert "page_t" not in line and "2026-10-16" not in line                       # the engine's time is in the prefix
    # Lists other than a few numbers, deep objects and long strings stay compact; an odd report is cut at the limit.
    odd = telemetry_summary({"names": ["a", "b"], "many": list(range(9)), "deep": {"a": {"b": {"c": 1}}}, "s": "y" * 99,
                             "big": 1.5e9, "small": 0.0001234})
    assert odd == 'names=[2] many=[9] deep.a.b={1} s="' + "y" * 60 + '" big=1.5e+09 small=0'
    long = telemetry_summary({f"k{i}": i for i in range(1000)})
    assert len(long) == TELEMETRY_LINE_CHARS and long.endswith("…")
    # Keys at every level are cleaned and strings escaped: nothing in a report can forge a line of the engine log.
    forged = telemetry_summary({"a\nb": 1, "deep": {"x\nSoPA companion: the last window was closed; shutting down": 1,
                                                   "y\x1b[2J z": {"w\r\n": "v \n"}}})
    assert forged == ('a_b=1 deep.x_SoPA_companion:_the_last_window_was_closed;_shutting_down=1 '
                      'deep.y_[2J_z.w_="v\\u2028\\n"')
    assert not any(c in forged for c in "\n\r\x1b ")


def test_a_page_gone_without_a_goodbye_is_away_and_the_engine_stays(config, capsys):
    """A crashed page (a sad tab, a killed browser) says no goodbye: it is away, whatever browser showed it. The engine
    keeps going (its runs paused: tests/test_presence.py); a tool's socket (the bearer) never counted as a page."""
    from sopa_companion.server import create_app
    config["desktop"] = {"exit_when_closed": True, "close_grace_s": 0.2, "close_grace_pending_s": 0.2}
    shut = []
    application = create_app(config, token="t" * 40, shutdown=lambda: shut.append(1))
    with TestClient(application, base_url=BASE) as client:
        with client.websocket_connect("/api/v1/ws", headers=TOOL_WS) as ws:
            assert ws.receive_json()["ev"] == "runs"
        with client.websocket_connect("/api/v1/ws?page=page-1", headers=page_ws(application)) as ws:
            assert ws.receive_json()["ev"] == "runs"
            assert application.state.presence.open() == 1
        wait_for(lambda: application.state.presence.public()["state"] == "away", timeout=10)    # gone without a word
        time.sleep(1.5)
        assert shut == [] and application.state.page_closed is False                   # ... the engine stays
    assert "the last window went without closing (a crash, a sleeping computer" in capsys.readouterr().out


def test_stop_force_finishes_a_test_engines_runs_into_their_folders(config, monkeypatch):
    config["proseg"]["keep_unsaved"] = True
    from sopa_companion.server import app as server
    asked, shut = [], []
    monkeypatch.setattr(server, "ask_to_save", lambda unsaved, **kw: (asked.append(unsaved), False)[1])
    application = server.create_app(config, token="t" * 40, shutdown=lambda: shut.append(1))
    with TestClient(application, base_url=BASE) as client:
        runs = application.state.runs
        sample = runs.samples.list()[0]["id"]
        rid = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()["id"]
        client.post(f"/api/v1/runs/{rid}/control", json={"action": "resume"}, headers=auth())
        wait_for(lambda: len(runs.get(rid).frames) >= 2)
        assert client.post("/api/v1/shutdown", headers=page(application)).status_code == 403    # never the page's key
        assert client.post("/api/v1/shutdown", headers={"Authorization": "Bearer " + "x" * 40, "Origin": BASE}).status_code == 401
        assert shut == [] and application.state.stop_forced is False
        response = client.post("/api/v1/shutdown", headers=auth())                  # sopa_companion stop --force
        assert response.status_code == 200 and response.json() == {"discarding": [], "saving": [rid]} and shut == [1]
        run = runs.get(rid)
    assert asked == []                                                              # no dialog: finished
    assert run.meta["state"] == "done" and run.meta["outputs"] and run.dir.exists()


def closing_engine(config, monkeypatch, rows, *, asked=None, says=True):
    """A user engine whose pending runs are `rows` (a list the test changes), with Runs.leave and Runs.close recorded
    instead of done (Runs.close takes {run id: answer} when the runs were answered apart); the dialog (ask_to_save)
    records whom it asked and answers `says`."""
    from sopa_companion.server import app as server
    if asked is not None:
        monkeypatch.setattr(server, "ask_to_save",
                            lambda unsaved, **kw: (asked.append([r["id"] for r in unsaved if not r.get("decided")]), says)[1])
    application = server.create_app(config, token="t" * 40, shutdown=lambda: None)
    runs, done = application.state.runs, []
    monkeypatch.setattr(runs, "pending", lambda: [{"sample": "sample-1", "sample_name": "Sample", "iter": 812,
                                                   "state": "paused", "decided": False, **row} for row in rows])
    monkeypatch.setattr(runs, "leave", lambda rid, answer: done.append(("leave", rid, answer)))
    monkeypatch.setattr(runs, "close", lambda policy=None, **kw: done.append(("close", policy)))
    return application, done


def test_stop_force_discards_what_nobody_answered_and_saves_what_was_decided(config, monkeypatch, capsys):
    """`stop --force` no longer saves every run. A run being saved (a leave or a
    save under way) is finished, a page's answer from the last minute is carried out for its runs, and the runs nobody
    answered for are discarded without the dialog; the launcher says which (discarding, saving)."""
    asked, rows = [], [{"id": "answered"}, {"id": "nobody"}, {"id": "saving", "decided": True}]
    application, done = closing_engine(config, monkeypatch, rows, asked=asked)
    with TestClient(application, base_url=BASE) as client:
        reply = client.post("/api/v1/closing", json={"answer": "add", "runs": ["answered"], "revision": 1},
                            headers=page(application)).json()
        assert reply["taken"] is True and reply["covers"] == ["answered"] and reply["applies"] is True
        stopping = client.post("/api/v1/shutdown", headers=auth())
        assert stopping.json() == {"discarding": ["nobody"], "saving": ["answered", "saving"]}
    assert asked == [] and done == [("close", {"answered": "add", "nobody": "discard"})]     # "saving": completed
    log = capsys.readouterr().out
    assert ("closing with 1 unsaved run(s): saving (dialog answer: not asked, the page answered add 0 s before); "
            "waiting for 1 run(s) already being saved or closed") in log
    assert ("closing with 1 unsaved run(s): discarding (dialog answer: not asked, stop --force); waiting for 1 run(s) "
            "already being saved or closed") in log


def test_a_discard_never_covers_a_run_started_after_it(config, monkeypatch, capsys):
    """Window A answered Discard for its run; window B then started another; a shutdown within the
    minute discarded B's run too, which nobody was asked about. An answer is for the runs the page showed (those it
    names, else every run pending when it came), kept as far as they are pending: B's run takes the dialog (here it
    answers Save) and A's is discarded."""
    asked, rows = [], [{"id": "r1"}]
    application, done = closing_engine(config, monkeypatch, rows, asked=asked)
    with TestClient(application, base_url=BASE) as client:
        old_page = client.post("/api/v1/closing", content='{"answer":"discard"}', headers=beacon(application)).json()
        assert old_page["covers"] == ["r1"] and old_page["applies"] is True        # a page that names no runs
        named = client.post("/api/v1/closing", json={"answer": "discard", "runs": ["r1", "never-pending"]},
                            headers=page(application)).json()
        assert named["covers"] == ["r1"]                                           # named and pending
        rows.append({"id": "r2"})                                                  # window B starts a run
        state = client.get("/api/v1/closing", headers=auth()).json()
        assert state["covers"] == ["r1"] and [r["id"] for r in state["runs"]] == ["r1", "r2"]
    assert asked == [["r2"]] and done == [("close", {"r1": "discard", "r2": "add"})]
    log = capsys.readouterr().out
    assert "closing with 1 unsaved run(s): discarding (dialog answer: not asked, the page answered discard 0 s before)" in log
    assert "SoPA companion: the page answered how to close: discard (1 run(s) still sampling, 1 answered for)" in log


def test_stop_force_carries_out_the_answer_it_said_it_would(config, monkeypatch):
    """/shutdown said which runs are saved from the page's answer as it stood then, and the
    shutdown read the answer again: one 59.5 s old was said to be "being saved first", was too old to count by the
    time the shutdown ran, and its run was discarded. The shutdown carries out the answer as /shutdown took it."""
    from sopa_companion.server import app as server
    asked = []
    application, done = closing_engine(config, monkeypatch, [{"id": "r1"}], asked=asked)
    closing, real = application.state.closing, application.state.closing.clock
    with TestClient(application, base_url=BASE) as client:
        client.post("/api/v1/closing", json={"answer": "add", "runs": ["r1"]}, headers=page(application))
        closing.clock = lambda: real() + server.CLOSING_ANSWER_S - 0.5         # 59.5 s old at /shutdown
        assert client.post("/api/v1/shutdown", headers=auth()).json() == {"discarding": [], "saving": ["r1"]}
        closing.clock = lambda: real() + server.CLOSING_ANSWER_S + 5           # 65 s old as the shutdown runs
    assert asked == [] and done == [("close", {"r1": "add"})]


@pytest.mark.parametrize("answer, closed", [("add", {"r1": "add"}), ("discard", "discard")])
def test_an_add_names_its_runs_to_runs_close(config, monkeypatch, answer, closed):
    """Runs.close gets {run id: answer} once a run is added: a run that became pending after the answers were taken
    (nobody answered for it) is then discarded by Runs.close's own rule, never added under another run's answer. A
    close that adds nothing says "discard"."""
    application, done = closing_engine(config, monkeypatch, [{"id": "r1"}])
    with TestClient(application, base_url=BASE) as client:
        client.post("/api/v1/closing", json={"answer": answer, "runs": ["r1"]}, headers=page(application))
    assert done == [("close", closed)]


def test_an_answer_overtaken_on_its_way_changes_nothing(config, monkeypatch):
    """Discard and close, then Keep working, whose answer arrived first: the late discard was kept,
    and a stop within the minute deleted the run the user went on working with. Each choice of a window carries a
    revision that grows (`page` names the window): one lower than the window's last taken changes nothing."""
    application, _ = closing_engine(config, monkeypatch, [{"id": "r1"}])
    with TestClient(application, base_url=BASE) as client:
        def say(**body):
            return client.post("/api/v1/closing", json=body, headers=page(application)).json()
        assert say(answer="withdraw", revision=4, page="A")["taken"] is True       # Keep working, arrived first
        late = say(answer="discard", runs=["r1"], revision=3, page="A")            # the Discard sent before it
        assert late["taken"] is False and late["answer"] is None and late["applies"] is False
        assert say(answer="discard", runs=["r1"], revision=4, page="A")["answer"] == "discard"   # its ✕ beacon: taken
        assert say(answer="add", revision=1, page="B")["answer"] == "add"          # another window counts on its own
        assert say(answer="discard", revision=0, page="B")["taken"] is False
        assert say(answer="discard", revision=3, page="A")["taken"] is False       # A's choices before its last one
        assert say(answer="withdraw", revision=5, page="A")["answer"] is None      # any window can take it back
        # A page that names no window: its revisions are compared with each other, until a page connects again.
        assert say(answer="add", revision=7)["taken"] is True and say(answer="discard", revision=6)["taken"] is False
        with client.websocket_connect("/api/v1/ws", headers=page_ws(application)) as ws:   # a reload
            ws.receive_json()
        assert say(answer="discard", revision=1)["answer"] == "discard"
        for body in ({"answer": "add", "runs": "r1"}, {"answer": "add", "runs": [1]}, {"answer": "add", "runs": [""]},
                     {"answer": "add", "runs": ["r"] * 65}, {"answer": "add", "revision": "3"},
                     {"answer": "add", "revision": 1.5},
                     {"answer": "add", "revision": -1}, {"answer": "add", "revision": True},
                     {"answer": "add", "page": "p" * 101}, {"answer": "add", "page": 3}, {"answer": "add", "end": "yes"},
                     {"answer": "stay"}):
            refused = client.post("/api/v1/closing", json=body, headers=page(application))
            assert refused.status_code == 422, body
        assert client.get("/api/v1/closing", headers=auth()).json()["answer"] == "discard"


# ---- closing the app with runs still sampling -----------------------------------------------------------------------
def beacon(application):
    """The page's goodbye as it goes (frontend/src/page-key.mjs send): a keepalive POST of JSON with the page's key in
    its header and the page's own Origin."""
    return page(application, {"Content-Type": "application/json"})


def page_ws(application, key=None):
    """The page's WebSocket handshake: the page's own Origin, and its key offered beside the page's subprotocol
    (frontend/src/api.js LiveSocket), which the test client reads from this header as a browser sends it."""
    return {"Host": "127.0.0.1:8799", "Origin": BASE,
            "Sec-WebSocket-Protocol": "sopa-companion, key." + (key or application.state.page_key)}


TOOL_WS = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}   # e.g. the harness's watch


def test_the_page_answers_how_to_close_until_a_page_connects_again(app, monkeypatch, capsys):
    """POST /closing keeps the page's answer with its time and its runs, sent as the popup's JSON or as the page's
    keepalive goodbye (a text/plain body too); refused answers change nothing; a page that connects again (a reload)
    withdraws it."""
    application, client = app
    row = {"id": "r1", "sample": "sample-1", "iter": 812, "state": "paused", "decided": False}
    monkeypatch.setattr(application.state.runs, "pending", lambda: [dict(row)])
    nothing = {"answer": None, "age_s": None, "applies": False, "covers": [], "runs": [row]}
    assert client.get("/api/v1/closing", headers=auth()).json() == nothing
    reply = client.post("/api/v1/closing", json={"answer": "discard"}, headers=page(application))
    assert reply.status_code == 200 and reply.json()["answer"] == "discard" and reply.json()["applies"] is True
    assert reply.json()["taken"] is True and reply.json()["covers"] == ["r1"]
    assert client.post("/api/v1/closing", content='{"answer":"save"}', headers=beacon(application)).status_code == 200
    text_body = page(application, {"Content-Type": "text/plain;charset=UTF-8"})              # an earlier page's beacon
    assert client.post("/api/v1/closing", content='{"answer":"save"}', headers=text_body).status_code == 200
    state = client.get("/api/v1/closing", headers=auth()).json()
    assert state["answer"] == "add" and 0 <= state["age_s"] < 5 and state["applies"] is True and state["runs"] == [row]
    # Refused: no key; another origin (another port of this host); a body that is no answer; more than 4 KB, declared or
    # not.
    unsigned = {k: v for k, v in beacon(application).items() if k != "X-SoPA-Page-Key"}
    assert client.post("/api/v1/closing", content='{"answer":"discard"}', headers=unsigned).status_code == 401
    for origin in ("http://127.0.0.1:8888", "null"):
        refused = client.post("/api/v1/closing", content='{"answer":"discard"}', headers={**beacon(application), "Origin": origin})
        assert refused.status_code == 403, origin
    for body, status in ((b"", 422), (b"discard", 422), (b'["discard"]', 422), (b'{"answer": "Discard"}', 422),
                         (b'{"answer": true}', 422), (b'{"answer": ["discard"]}', 422), (b'{"answer": NaN}', 422),
                         (b'{"answer": "discard"', 422), (b'{"answer": "\xff"}', 422), (b'{"other": "discard"}', 422),
                         (b'{"answer": "discard", "revision": 1e999}', 422),
                         (b'{"answer": "discard", "pad": "' + b"x" * 5000 + b'"}', 413)):
        assert client.post("/api/v1/closing", content=body, headers=beacon(application)).status_code == status, body[:30]

    def chunked():                                           # no Content-Length: still cut off at 4 KB
        yield b'{"answer": "discard", "pad": "'
        for _ in range(8):
            yield b"x" * 1000
        yield b'"}'
    assert client.post("/api/v1/closing", content=chunked(), headers=beacon(application)).status_code == 413
    assert client.get("/api/v1/closing", headers=auth()).json()["answer"] == "add"      # refused ones change nothing
    with client.websocket_connect("/api/v1/ws", headers=TOOL_WS) as ws:                 # a tool connecting is no page
        assert ws.receive_json()["ev"] == "runs"
        assert client.get("/api/v1/closing", headers=auth()).json()["answer"] == "add"
    with client.websocket_connect("/api/v1/ws", headers=page_ws(application)) as ws:    # a reload: the page connects
        assert ws.receive_json()["ev"] == "runs"
        assert client.get("/api/v1/closing", headers=auth()).json() == nothing
    assert client.post("/api/v1/closing", json={"answer": "withdraw"}, headers=page(application)).json()["taken"] is True
    log = capsys.readouterr().out
    assert "SoPA companion: the page answered how to close: discard (1 run(s) still sampling, 1 answered for)" in log
    assert "SoPA companion: a page connected again: the answer to closing (add) no longer counts" in log
    assert "SoPA companion: the page took its answer to closing back" in log


@pytest.mark.parametrize("answer", ["save", "discard"])
def test_closing_carries_out_the_pages_answer_without_the_dialog(config, monkeypatch, capsys, answer):
    """The app's popup (or the page's beacon) answered: the engine saves, as the dialog's Save results, or discards, as
    its Discard, without the dialog; the log says so in the dialog's line, which log readers parse."""
    config["proseg"]["keep_unsaved"] = True
    from sopa_companion.server import app as server
    asked = []
    monkeypatch.setattr(server, "ask_to_save", lambda unsaved, **kw: (asked.append(unsaved), True)[1])
    application = server.create_app(config, token="t" * 40)
    with TestClient(application, base_url=BASE) as client:
        runs = application.state.runs
        sample = runs.samples.list()[0]["id"]
        rid = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()["id"]
        client.post(f"/api/v1/runs/{rid}/control", json={"action": "resume"}, headers=auth())
        wait_for(lambda: len(runs.get(rid).frames) >= 2)
        reply = client.post("/api/v1/closing", content=json.dumps({"answer": answer}), headers=beacon(application)).json()
        assert [(r["id"], r["decided"]) for r in reply["runs"]] == [(rid, False)] and reply["covers"] == [rid]
        run = runs.get(rid)
    # (the lifespan's shutdown ran when the client closed)
    assert asked == []                                                              # no dialog
    if answer == "save":
        assert run.meta["state"] == "done" and run.meta["outputs"] and run.dir.exists()
    else:
        assert run.dir.exists() and run.meta["state"] == "stopped"
    # The line as a log reader parses it (its "unsaved" pattern, repeated here: that reader never ships, and
    # the deploy runs these tests in production).
    line = re.search(r"^SoPA companion: closing with (\d+) unsaved run\(s\): (\w+) \(dialog answer:? ?([^)]*)\)",
                     capsys.readouterr().out, re.M)
    assert line and int(line.group(1)) == 1 and line.group(2) == ("saving" if answer == "save" else "discarding")
    assert line.group(3).startswith(f"not asked, the page answered {'add' if answer == 'save' else answer} ")


def test_an_old_or_withdrawn_answer_leaves_the_question_to_the_dialog(config, monkeypatch, capsys):
    """Only an answer from the last 60 s before the shutdown counts, and only until a page connects again; otherwise
    the dialog asks, as for a close nobody answered (a crash)."""
    from sopa_companion.server import app as server
    asked, closed = [], []
    monkeypatch.setattr(server, "ask_to_save", lambda unsaved, **kw: (asked.append([r["id"] for r in unsaved]), True)[1])
    for case in ("fresh", "old", "reload"):
        application = server.create_app(config, token="t" * 40)
        application.state.runs.pending = lambda: [{"id": case, "sample": "sample-1", "iter": 812, "decided": False}]
        application.state.runs.close = lambda policy=None, **kw: closed.append(policy)
        with TestClient(application, base_url=BASE) as client:
            assert client.post("/api/v1/closing", json={"answer": "discard"}, headers=page(application)).status_code == 200
            if case == "old":
                closing, real = application.state.closing, application.state.closing.clock
                closing.clock = lambda: real() + server.CLOSING_ANSWER_S + 1
                assert client.get("/api/v1/closing", headers=auth()).json()["applies"] is False
            elif case == "reload":
                with client.websocket_connect("/api/v1/ws", headers=page_ws(application)) as ws:
                    ws.receive_json()
    assert asked == [["old"], ["reload"]]
    assert closed == ["discard", {"old": "add"}, {"reload": "add"}]       # the dialog's Save, for the runs it asked about
    log = capsys.readouterr().out
    assert "closing with 1 unsaved run(s): discarding (dialog answer: not asked, the page answered discard 0 s before)" in log
    assert "SoPA companion: the answer to closing (discard, 61 s before the shutdown) is too old to count" in log
    assert "SoPA companion: a page connected again: the answer to closing (discard) no longer counts" in log


def test_an_answer_counts_for_a_minute_and_runs_being_saved_are_kept_whatever_it_says(capsys):
    from sopa_companion.server.app import CLOSING_ANSWER_S, ClosingAnswer, carry_out_answer, decide_closing
    now, logs = [100.0], []
    closing = ClosingAnswer(clock=lambda: now[0], log=logs.append)
    assert closing.at_shutdown() is None and closing.public() == {"answer": None, "age_s": None, "applies": False,
                                                                  "covers": []}
    taken = closing.give("discard")
    closing.cover(taken, ["r1", "r2"])
    now[0] += CLOSING_ANSWER_S
    assert closing.at_shutdown() == ("discard", CLOSING_ANSWER_S, frozenset({"r1", "r2"})) and logs == []
    now[0] += 0.5
    assert closing.public() == {"answer": "discard", "age_s": 60.5, "applies": False, "covers": ["r1", "r2"]}
    assert closing.at_shutdown() is None
    assert logs == ["SoPA companion: the answer to closing (discard, 60.5 s before the shutdown) is too old to count"]
    closing.give("save", ["r1"])                               # a newer answer replaces it
    assert closing.at_shutdown() == ("add", 0.0, frozenset({"r1"}))
    closing.cover(taken, [])                                   # an older answer's runs never narrow a newer one
    assert closing.public()["covers"] == ["r1"]
    closing.withdraw()
    closing.withdraw()                                         # nothing left to withdraw: nothing logged
    assert closing.at_shutdown() is None and closing.public()["answer"] is None
    assert logs[1:] == ["SoPA companion: a page connected again: the answer to closing (add) no longer counts"]
    # As the dialog: only "discard" deletes, and only runs still sampling; with only runs being saved there is nothing
    # to decide (nothing logged, as the dialog is not shown).
    sampling = {"id": "r1", "sample": "sample-1", "iter": 812, "decided": False}
    finishing = {"id": "r2", "sample": "sample-2", "iter": 400, "decided": True}
    assert carry_out_answer([finishing], "discard", 3) is True
    assert carry_out_answer([sampling, finishing], "discard", 3.4) is False
    assert carry_out_answer([sampling], "save", 0) is True
    assert capsys.readouterr().out.splitlines() == [
        "SoPA companion: closing with 1 unsaved run(s): discarding (dialog answer: not asked, the page answered discard 3 s "
        "before); waiting for 1 run(s) already being saved or closed",
        "SoPA companion: closing with 1 unsaved run(s): saving (dialog answer: not asked, the page answered save 0 s before)"]
    # decide_closing: the answer for its runs, nothing for decided ones, and no dialog on stop --force.
    later = {"id": "r3", "sample": "sample-3", "iter": 9, "decided": False}
    assert decide_closing([sampling, finishing, later], ("add", 1.0, frozenset({"r1"})), forced=True) == {
        "r1": "add", "r3": "discard"}
    assert decide_closing([sampling, later], ("discard", 1.0, None), forced=True) == {"r1": "discard", "r3": "discard"}
    assert decide_closing([finishing], None) == {}


def test_the_name_users_see_is_sopa_companion_and_the_log_says_it_too(app, tmp_path, monkeypatch, capsys):
    """The name users see (the app's title, the command line's description, the closing dialog) is SoPA companion, and
    the engine log's lines begin with it ("SoPA companion: "; its readers also take the previous version's "Proseg
    Live: ", engine/crashlog.py and the harness's field probe)."""
    from sopa_companion.cli import parser
    from sopa_companion.server import app as server
    application, client = app
    assert application.title == "SoPA companion" and parser().description.startswith("SoPA companion: ")
    folder = tmp_path / "bin"
    folder.mkdir()
    (folder / "zenity").write_text(f"#!{sys.executable}\nimport json, sys\n"
                                   f"open({str(tmp_path / 'args.json')!r}, 'w').write(json.dumps(sys.argv[1:]))\n")
    (folder / "zenity").chmod(0o755)                         # it answers Save results (exit 0)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    capsys.readouterr()
    assert server.ask_to_save([{"id": "r1", "sample": "sample-1", "iter": 812}], has_display=lambda: True) is True
    args = json.loads((tmp_path / "args.json").read_text())
    assert "--title=SoPA companion" in args and args[-1].startswith("--text=SoPA companion is closing.\n\n")
    assert not any("Proseg Live" in a for a in args)
    assert capsys.readouterr().out == "SoPA companion: closing with 1 unsaved run(s): saving (dialog answer: Save to the sample)\n"


# ---- what went wrong reaches the engine log and the Log ------------------------------------------------------------------
def test_failed_requests_reach_the_engine_log_and_the_log_counts_them(config, monkeypatch, capsys):
    """A sample that the helper could not open (its own lines, its traceback, are printed by the helper call) and an
    error the engine did not expect: each request says in one engine log line what failed, and the Log's summary counts
    them, so it never says nothing went wrong after them (three 500s, and "nothing went wrong"). Refusals that
    are answers (an unknown run) are not counted."""
    from sopa_companion.engine.zarr_samples import HelperFailed
    from sopa_companion.server import create_app
    application = create_app(config, token="t" * 40)
    with TestClient(application, base_url=BASE, raise_server_exceptions=False) as client:
        assert "Engine noted    nothing went wrong" in client.get("/api/v1/log", headers=auth()).text

        def unreadable(path):
            raise HelperFailed("KeyError: 'ome'", 422)
        monkeypatch.setattr(application.state.zarr, "open", unreadable)
        capsys.readouterr()
        refused = client.post("/api/v1/samples/open", json={"path": "/data/a.zarr"}, headers=auth())
        assert refused.status_code == 422 and refused.json() == {"detail": "KeyError: 'ome'", "status": 422}
        assert capsys.readouterr().out == "SoPA companion: POST /api/v1/samples/open failed (422): KeyError: 'ome'\n"
        assert client.get("/api/v1/runs/nope", headers=auth()).status_code == 404     # an answer: not noted

        def taken(path):           # a helper's refusal with more to say than its words: in the answer beside them
            raise HelperFailed("a already exists in the sample", 409, error="a already exists in the sample",
                               clash=["a"], existing=["a", "b"])
        monkeypatch.setattr(application.state.zarr, "open", taken)
        assert client.post("/api/v1/samples/open", json={"path": "/data/a.zarr"}, headers=auth()).json() == {
            "detail": "a already exists in the sample", "status": 409, "error": "a already exists in the sample",
            "clash": ["a"], "existing": ["a", "b"]}
        capsys.readouterr()
        monkeypatch.setattr(application.state.zarr, "list", lambda: {}["recent"])
        broken = client.get("/api/v1/samples", headers=auth())
        assert broken.status_code == 500 and broken.json() == {
            "detail": "The engine could not do this (KeyError). The Log has the details.", "status": 500}
        assert "SoPA companion: GET /api/v1/samples failed (500): KeyError: 'recent'" in capsys.readouterr().out
        for _ in range(3):                                               # a poll that keeps failing: kept once
            client.get("/api/v1/samples", headers=auth())
        # ... and printed once: a line per poll pushed the earlier lines out of the 150 the Log shows.
        assert "failed (500)" not in capsys.readouterr().out
        log = client.get("/api/v1/log", headers=auth()).text
    assert "Engine noted    3 things: GET /api/v1/samples failed (500): KeyError: 'recent' (4 times, the last at" in log
    assert "Engine events (what failed: requests, helpers, runs and saves)" in log
    assert "POST /api/v1/samples/open failed (422): KeyError: 'ome'" in log.split("Engine events (what failed")[1]


def test_a_run_that_fails_is_noted_in_its_log_and_the_sessions(app, monkeypatch, capsys):
    """A run that fails says so in the engine log (a line naming it, which its Log lists) and in the Log's summary, its
    own and the session's."""
    application, client = app
    sample = application.state.samples.list()[0]["id"]

    def offline(*args, **kwargs):
        raise OSError("the sample share did not answer")
    monkeypatch.setattr(application.state.samples, "crop_transcripts", offline)
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    wait_for(lambda: application.state.runs.get(run["id"]).meta["state"] == "failed")
    noted = f"run {run['id']} failed: the sample share did not answer"
    assert f"SoPA companion: {noted}\n" in capsys.readouterr().out
    assert f"Engine noted    1 thing: {noted}" in client.get(f"/api/v1/runs/{run['id']}/log", headers=auth()).text
    assert f"Engine noted    1 thing: {noted}" in client.get("/api/v1/log", headers=auth()).text


# ---- a run's sample answers once it left the open dialog's list ---------------------------------------------------------
def test_a_runs_sample_answers_after_it_left_the_open_dialogs_list(app, monkeypatch):
    """The detail of a sample no longer known (forgotten, or past the list) answered 404 while the layer routes opened
    it again, on top of the open dialog's list: now every sample route opens a run's sample again where the run found
    it, left off the list (recent=False); a sample no run uses is still "open it again"."""
    from sopa_companion.engine.runs import Run
    application, client = app
    runs, zarr = application.state.runs, application.state.zarr
    folder = runs.root / "r-zarr"
    folder.mkdir()
    runs.runs["r-zarr"] = Run(folder, {"id": "r-zarr", "kind": "proseg", "sample": "zsample", "sample_kind": "zarr",
                                       "sample_path": "/data/sample.zarr", "state": "done", "created": "2026-01-01"})
    known, opened = set(), []
    monkeypatch.setattr(zarr, "known", lambda sid: sid in known)

    def reopen(path, recent=True):
        opened.append((path, recent))
        known.add("zsample")
        return {"id": "zsample"}
    monkeypatch.setattr(zarr, "open", reopen)
    monkeypatch.setattr(zarr, "detail", lambda sid: {"id": sid, "kind": "zarr", "name": "sample"})
    detail = client.get("/api/v1/samples/zsample", headers=auth())
    assert detail.status_code == 200 and detail.json()["id"] == "zsample"
    assert opened == [("/data/sample.zarr", False)]
    known.clear()
    monkeypatch.setattr(zarr, "segmentations", lambda sid: [])
    assert client.get("/api/v1/samples/zsample/segmentations", headers=auth()).json() == {"segmentations": []}
    assert opened[-1] == ("/data/sample.zarr", False)                    # the layer routes too: left off the list
    unknown = client.get("/api/v1/samples/znobody", headers=auth())
    assert unknown.status_code == 404 and len(opened) == 2


# ---- routes no page calls --------------------------------------------------------------------------------------------
def test_routes_no_page_calls_are_gone(app):
    application, client = app
    sample = application.state.samples.list()[0]["id"]
    run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
    assert client.get(f"/api/v1/runs/{run['id']}/genes", headers=auth()).status_code == 404
    assert client.get(f"/api/v1/samples/{sample}/xenium", params={"box": "0,0,10,10"}, headers=auth()).status_code == 404
    assert client.get(f"/api/v1/runs/{run['id']}/cells", headers=auth()).status_code == 200     # still used
    import sopa_companion.server.app as server
    assert not hasattr(server, "_reachable")                 # nor a helper nothing called (a folder's listing check)


# ---- an engine reached from another computer -----------------------------------------------------------------------------
def test_a_page_lost_over_a_tunnel_neither_stops_the_engine_nor_deletes_its_runs(config, capsys):
    """An engine whose page came from another computer (launch --print-url through an SSH tunnel): the page goes without
    a goodbye (the tunnel dropped), so it is away: the engine keeps its run, paused (paused_away), until the app is
    opened again; when it is stopped later without an answer, the run is discarded."""
    from sopa_companion.server import create_app
    config["desktop"] = {"exit_when_closed": True, "close_grace_s": 0.2, "close_grace_pending_s": 0.2}
    shut = []
    application = create_app(config, token="t" * 40, shutdown=lambda: shut.append(1))
    with TestClient(application, base_url=BASE) as client:
        runs = application.state.runs
        sample = runs.samples.list()[0]["id"]
        rid = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()["id"]
        client.post(f"/api/v1/runs/{rid}/control", json={"action": "resume"}, headers=auth())
        wait_for(lambda: len(runs.get(rid).frames) >= 2)
        with client.websocket_connect("/api/v1/ws?page=tunnel", headers=page_ws(application)) as ws:
            assert ws.receive_json()["ev"] == "runs"
        wait_for(lambda: "1 run(s) paused; the engine waits for the app to be opened again" in capsys.readouterr().out,
                 timeout=10)
        assert shut == [] and runs.get(rid).meta["paused"] is True and runs.get(rid).meta["paused_away"]
        run = runs.get(rid)
    assert run.meta["state"] == "stopped" and not run.dir.exists()


def serve_with(tmp_path, xenium_root, fake_binary, hang_up: str, then=None):
    """A `serve` in its own process, SIGHUP set to `hang_up` ("SIG_DFL", or "SIG_IGN" as nohup leaves it) before the
    engine starts; once it has started, SIGHUP, then `then` (a signal) after 2 s if it is still running. -> (its output,
    its exit status, whether it still ran 2 s after the hang-up)."""
    import signal
    import subprocess
    script = tmp_path / "serve.py"
    script.write_text(f"""
import signal, socket, sys
signal.signal(signal.SIGHUP, signal.{hang_up})
import uvicorn
from sopa_companion.server import create_app
config = {{"server": {{"port": 8799}}, "desktop": {{"exit_when_closed": False}},
          "paths": {{"data_dir": {str(tmp_path / 'var')!r}, "state_dir": {str(tmp_path / 'state')!r},
                    "sample_roots": [{str(xenium_root)!r}]}},
          "proseg": {{"binary": {str(fake_binary)!r}, "threads": 2}}, "analysis": {{"python": "off"}}}}
app = create_app(config, token="t" * 40)
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
uvicorn.Server(uvicorn.Config(app, log_level="info", access_log=False)).run(sockets=[listener])
""")
    env = {**os.environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[2]),
           "SOPA_COMPANION_ZARR_LOCK_DIR": str(tmp_path / "locks"), "SOPA_COMPANION_QC_FIT_LOCK": str(tmp_path / "fit.lock")}
    env.pop("DISPLAY", None)
    child = subprocess.Popen([sys.executable, str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
    lines, running = [], False
    try:
        for line in child.stdout:                                       # until the engine has started
            lines.append(line)
            if "Application startup complete" in line:
                break
        child.send_signal(signal.SIGHUP)
        if then is not None:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                running = True
                child.send_signal(then)
        lines += child.stdout.readlines()
        child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    return "".join(lines), child.returncode, running


def test_a_hang_up_closes_the_engine_as_a_term_does(tmp_path, xenium_root, fake_binary):
    """SIGHUP (a `serve` whose SSH session dropped) shuts the engine down through its lifespan, as SIGTERM does: uvicorn
    handles only SIGINT and SIGTERM, and a hang-up killed the engine with no teardown at all."""
    import signal
    text, code, _ = serve_with(tmp_path, xenium_root, fake_binary, "SIG_DFL")
    assert "Application startup complete" in text and "Application shutdown complete" in text, text[-2000:]
    assert code in (0, -signal.SIGTERM)                                 # uvicorn raises the SIGTERM again once done


def test_an_engine_started_under_nohup_outlives_its_terminal(tmp_path, xenium_root, fake_binary):
    """`nohup python -m sopa_companion serve` ignores SIGHUP so the engine outlives the session (a logout): the handler
    that closes the engine on a hang-up is set only over SIGHUP's default, never over nohup's ignore."""
    import signal
    text, code, running = serve_with(tmp_path, xenium_root, fake_binary, "SIG_IGN", then=signal.SIGTERM)
    assert running, text[-2000:]                                        # still serving 2 s after the hang-up
    assert "Application shutdown complete" in text and code in (0, -signal.SIGTERM), text[-2000:]


def test_a_hang_up_sends_what_went_to_the_terminal_to_dev_null(monkeypatch):
    """After a real hang-up the terminal is no terminal to isatty any more, and a write to it fails (EIO): the shutdown
    then failed while saving the runs. The outputs that were the terminal when the handler was set go to /dev/null
    whatever isatty says by then; a file or a pipe keeps the engine's lines; nohup's ignore is left alone."""
    import pty
    import signal
    from sopa_companion.server import app as A
    master, slave = pty.openpty()
    held = os.dup(slave)                                                # the engine's output, a terminal
    os.close(slave)
    os.close(master)                                                    # the session hangs up
    assert not os.isatty(held)
    killed = []
    monkeypatch.setattr(A.os, "kill", lambda pid, sig: killed.append(sig))
    try:
        A.hang_up_like_term(terminals=(held,))
        assert os.fstat(held).st_rdev == os.stat(os.devnull).st_rdev and killed == [signal.SIGTERM]
        os.write(held, b"the shutdown's lines\n")                      # no longer fails
    finally:
        os.close(held)
    assert A.hang_up_handler(lambda signum: signal.SIG_IGN) is None      # nohup's: left as it is
    assert A.hang_up_handler(lambda signum: signal.SIG_DFL) is not None



def this_sessions_run(application, config, run_id, **meta):
    """A run of this engine session as the engine made it (its one-run marker), put in once the engine has started: at
    a start a marked folder is a crash's leftover and goes, and an unmarked one is a run from an earlier version,
    neither of them in the registry."""
    from test_runs_save import ended_run
    from sopa_companion.engine.runs import Run
    runs = application.state.runs
    ended_run(config, run_id, session_model=2, **meta)
    folder = runs.root / run_id
    run = runs.runs[run_id] = Run(folder, json.loads((folder / "run.json").read_text()))
    runs.changed()
    return run


def test_leave_route_pushes_gone_and_forgets_run_state(config, tmp_path, monkeypatch):
    from test_runs_save import Sample
    from sopa_companion.server import create_app
    from sopa_companion.engine.qc import RunQc
    application = create_app(config, token="t" * 40)
    runs = application.state.runs
    runs.zarr = Sample(tmp_path / "sample.zarr")
    this_sessions_run(application, config, "left")
    forgotten = []
    monkeypatch.setattr(application.state.qc_tests, "cancel", lambda rid: forgotten.append(("test", rid)))
    monkeypatch.setattr(application.state.deeper, "forget", lambda rid: forgotten.append(("deeper", rid)))
    monkeypatch.setattr(application.state.compare, "forget", lambda rid: forgotten.append(("compare", rid)))
    application.state.qc.states["left"] = RunQc("left")
    application.state.analysis.enabled["left"] = True
    with TestClient(application, base_url=BASE) as client:
        health = client.get("/api/v1/health", headers=auth()).json()
        assert health["one_run"] is True and health["older"]["runs"] == [] and health["session"] == {"mode": "desktop"}
        pending = client.get("/api/v1/closing", headers=auth()).json()["runs"]
        assert pending[0]["id"] == "left" and pending[0]["state"] == "done" and pending[0]["can_add"] is True
        for kind in ("proseg", "stain", "patches"):
            refused = client.post("/api/v1/runs", json={"kind": kind}, headers=auth())
            assert refused.status_code == 409 and refused.json()["detail"].startswith("Another run is open:")
        with client.websocket_connect("/api/v1/ws", headers=TOOL_WS) as socket:
            socket.send_json({"op": "watch", "run": "left"})
            response = client.post("/api/v1/runs/left/leave", json={"answer": "add"}, headers=auth())
            assert response.status_code == 200 and response.json()["leaving"]["answer"] == "add"
            last_run = None
            for _ in range(30):
                event = socket.receive_json()
                if event["ev"] == "run":
                    last_run = event["run"]
                if event["ev"] == "gone":
                    break
            assert event == {"ev": "gone", "run": "left", "reason": "left"}
            assert last_run["zarr_save"]["state"] == "saved"
            assert last_run["zarr_save"]["shapes_key"] == "proseg_boundaries"
        assert client.get("/api/v1/closing", headers=auth()).json()["runs"] == []
        assert "left" not in application.state.qc.states and "left" not in application.state.analysis.enabled
    assert forgotten == [("test", "left"), ("deeper", "left"), ("compare", "left")]


def test_shutdown_of_test_engine_leaves_sample_byte_identical(config, tmp_path, monkeypatch):
    import hashlib
    from test_runs_save import ended_run, Sample
    from sopa_companion.server import create_app
    config["proseg"]["keep_unsaved"] = True
    folder = tmp_path / "sample.zarr"
    folder.mkdir()
    (folder / "zarr.json").write_text('{"zarr_format":3,"node_type":"group","attributes":{}}')
    before = hashlib.sha256((folder / "zarr.json").read_bytes()).hexdigest()
    ended_run(config, "test-close", sample_path=str(folder))
    stopped = []
    application = create_app(config, token="t" * 40, shutdown=lambda: stopped.append(True))
    sample = Sample(folder)
    application.state.runs.zarr = sample
    monkeypatch.setattr(sample, "save", lambda *a, **kw: pytest.fail("Shutdown wrote into a sample"))
    with TestClient(application, base_url=BASE) as client:
        assert client.get("/api/v1/health", headers=auth()).json()["one_run"] is False
        assert client.post("/api/v1/shutdown", headers=auth()).json() == {"discarding": [], "saving": ["test-close"]}
    assert stopped == [True] and application.state.runs.get("test-close").dir.exists()
    assert hashlib.sha256((folder / "zarr.json").read_bytes()).hexdigest() == before


def test_stain_range_commit_routes_and_cell_values_delegate_once(config, monkeypatch):
    from sopa_companion.server import create_app
    application = create_app(config, token="t" * 40)
    this_sessions_run(application, config, "filters", kind="stain")
    calls = []
    exclusions = application.state.runs.stain.exclusions
    def commit(run, body, **kwargs):
        calls.append((body, kwargs))
        return {"version": body["version"] + 1, "digest": "new"}
    monkeypatch.setattr(exclusions, "change", commit)
    monkeypatch.setattr(exclusions, "cell_bytes", lambda run, set, channel: b"PLCV" + set.encode(), raising=False)
    with TestClient(application, base_url=BASE) as client:
        for kind, body in (("size", {"version": 0, "min": 5, "max": 40}),
                            ("brightness", {"version": 1, "channel": 0, "min": None, "max": 100})):
            response = client.post(f"/api/v1/runs/filters/stain/exclusions/{kind}", json=body, headers=auth())
            assert response.status_code == 200 and response.json()["version"] == body["version"] + 1
            assert calls[-1] == (body, {kind: True})
        assert len(calls) == 2
        response = client.get("/api/v1/runs/filters/stain/cells?set=resolved-6&channel=0", headers=auth())
        assert response.status_code == 200 and response.content == b"PLCVresolved-6"
        run = application.state.runs.get("filters")
        run.meta["leaving"] = {"answer": "add"}
        assert client.post("/api/v1/runs/filters/stain/exclusions/size", json={"version": 2, "min": 6, "max": 40},
                           headers=auth()).status_code == 409
        assert len(calls) == 2
        run.meta["leaving"] = None


def test_leave_rejects_bad_answers_and_shutdown_requests(app):
    application, client = app
    for answer in (None, "save", "later", True, ["add"]):
        assert client.post("/api/v1/runs/nope/leave", json={"answer": answer}, headers=auth()).status_code == 422
    application.state.runs.closing = True
    assert client.post("/api/v1/runs/nope/leave", json={"answer": "add"}, headers=auth()).status_code == 503


def pushed(socket, wanted, tries=40):
    """The first event `wanted(event)` accepts, asking with a ping after each quiet spell (a pong ends a spell) so a
    missing push fails the test instead of hanging it."""
    for _ in range(tries):
        socket.send_json({"op": "ping"})
        while (event := socket.receive_json())["ev"] != "pong":
            if wanted(event):
                return event
        time.sleep(0.1)
    raise AssertionError("no such push")


def test_older_runs_are_listed_answered_through_runs_and_pushed(app, monkeypatch):
    """GET /runs/older lists the runs from earlier versions never saved to their
    samples (health.older too); POST /runs/older/{id} {answer} answers one (Save to the sample, never as its
    segmentation, or Discard) and POST /runs/older {answer} all of them (Save all, Discard all, or Later, which sets
    them aside until the next start and stops a Save all), through Runs.answer_older and Runs.answer_all_older. Their
    rows go out as {ev: "older", runs} on a change, [] once none is left; new runs never wait for them (one run counts
    only this version's runs)."""
    from sopa_companion.engine.runs import RunError
    application, client = app
    runs = application.state.runs
    rows = [{"id": "old-1", "sample_name": "Sample", "kind": "proseg", "state": "done"},
            {"id": "old-2", "sample_name": "Sample", "kind": "stain", "state": "stopped"}]
    calls = []
    monkeypatch.setattr(runs, "older_version", 0, raising=False)      # grows with every change of the rows (Runs)

    def answer_older(run_id, answer):
        calls.append((run_id, answer))
        if run_id not in [row["id"] for row in rows]:
            raise RunError("Unknown run", 404)
        rows[:] = [row for row in rows if row["id"] != run_id]
        runs.older_version += 1

    def answer_all_older(answer):
        calls.append(("all", answer))
        if answer == "later":                                         # set aside: listed again at the next start
            rows[:] = [{**row, "later": True} for row in rows]
        else:
            rows.clear()
        runs.older_version += 1

    monkeypatch.setattr(runs, "older", lambda: {"runs": [dict(row) for row in rows], "version": runs.older_version},
                        raising=False)
    monkeypatch.setattr(runs, "answer_older", answer_older, raising=False)
    monkeypatch.setattr(runs, "answer_all_older", answer_all_older, raising=False)
    assert client.get("/api/v1/runs/older", headers=auth()).json() == {"runs": rows, "version": 0}
    assert client.get("/api/v1/health", headers=auth()).json()["older"] == {"runs": rows, "version": 0}
    for body in ({}, {"answer": "stay"}, {"answer": None}, {"answer": ["add"]}, {"answer": "save"}):
        assert client.post("/api/v1/runs/older/old-1", json=body, headers=page(application)).status_code == 422, body
        assert client.post("/api/v1/runs/older", json=body, headers=page(application)).status_code == 422, body
    # Later is for every row at once, never for one.
    assert client.post("/api/v1/runs/older/old-1", json={"answer": "later"}, headers=page(application)).status_code == 422
    assert calls == []
    with client.websocket_connect("/api/v1/ws", headers=page_ws(application)) as socket:
        assert pushed(socket, lambda e: e["ev"] == "older")["runs"] == rows
        reply = client.post("/api/v1/runs/older/old-1", json={"answer": "add"}, headers=page(application))
        assert reply.status_code == 200 and [row["id"] for row in reply.json()["runs"]] == ["old-2"]
        assert [row["id"] for row in pushed(socket, lambda e: e["ev"] == "older")["runs"]] == ["old-2"]
        assert client.post("/api/v1/runs/older/nope", json={"answer": "discard"}, headers=auth()).status_code == 404
        later = client.post("/api/v1/runs/older", json={"answer": "later"}, headers=page(application))
        assert later.status_code == 200 and later.json()["runs"] == [{**rows[0], "later": True}]
        assert pushed(socket, lambda e: e["ev"] == "older")["runs"][0]["later"] is True
        assert client.post("/api/v1/runs/older", json={"answer": "discard"}, headers=auth()).json() == {
            "runs": [], "version": 3}
        assert pushed(socket, lambda e: e["ev"] == "older") == {"ev": "older", "runs": [], "version": 3}
    assert calls == [("old-1", "add"), ("nope", "discard"), ("all", "later"), ("all", "discard")]
    # While the engine closes: no new answer, but Later still stops a Save all after the row under way.
    runs.closing = True
    assert client.post("/api/v1/runs/older", json={"answer": "add"}, headers=auth()).status_code == 503
    assert client.post("/api/v1/runs/older/old-2", json={"answer": "discard"}, headers=auth()).status_code == 503
    assert client.post("/api/v1/runs/older", json={"answer": "later"}, headers=auth()).status_code == 200
    runs.closing = False
    assert calls[-1] == ("all", "later") and len(calls) == 5


def test_health_answers_when_the_runs_from_earlier_versions_cannot_be_listed(app, monkeypatch, capsys):
    """/health listed the runs from earlier versions unguarded, and the launcher takes an
    engine whose /health fails for a dead one. It answers with none listed; the failure is said once in the engine
    log (and noted for the Log), and the page's WebSocket goes on without an "older" push."""
    application, client = app

    def broken():
        raise OSError("disk gone")
    monkeypatch.setattr(application.state.runs, "older", broken, raising=False)
    for _ in range(3):
        health = client.get("/api/v1/health", headers=auth())
        assert health.status_code == 200 and health.json()["older"] == {"runs": [], "done": [], "count": 0}
    with client.websocket_connect("/api/v1/ws", headers=TOOL_WS) as socket:
        socket.send_json({"op": "ping"})
        seen = []
        while (event := socket.receive_json())["ev"] != "pong":
            seen.append(event["ev"])
        assert "older" not in seen
    said = capsys.readouterr().out
    assert said.count("SoPA companion: listing the runs from earlier versions failed: OSError: disk gone") == 1
    assert application.state.noted.list()[-1][2].startswith("listing the runs from earlier versions failed")


def test_older_runs_over_http_on_the_engines_own_rows(config, tmp_path):
    """The routes on Runs itself (engine/runs.py): a run from an earlier version (no one-run marker) is listed by
    GET /runs/older and health.older, outside the registry (no run is refused for it); Discard over HTTP removes it,
    and its end goes out as {ev: "older"}; Later sets the rest aside (health.older counts none, the folder stays for
    the next start), also while the engine closes."""
    from sopa_companion.engine.runs import Runs
    if not hasattr(Runs, "answer_older"):
        pytest.skip("needs engine/runs.py's runs from earlier versions (Runs.answer_older)")
    from test_runs_save import ended_run
    from sopa_companion.server import create_app
    sample = tmp_path / "older-sample.zarr"
    sample.mkdir()
    ended_run(config, "older-1", sample_path=str(sample))
    ended_run(config, "older-22", sample_path=str(sample))           # made a second later: listed first
    application = create_app(config, token="t" * 40)
    root = application.state.runs.root
    with TestClient(application, base_url=BASE) as client:
        listed = client.get("/api/v1/runs/older", headers=auth()).json()
        assert [row["id"] for row in listed["runs"]] == ["older-22", "older-1"] and listed["count"] == 2
        assert client.get("/api/v1/health", headers=auth()).json()["older"]["count"] == 2
        assert client.get("/api/v1/runs", headers=auth()).json() == []
        with client.websocket_connect("/api/v1/ws", headers=page_ws(application)) as socket:
            assert [row["id"] for row in pushed(socket, lambda e: e["ev"] == "older")["runs"]] == ["older-22", "older-1"]
            row = client.post("/api/v1/runs/older/older-1", json={"answer": "discard"}, headers=page(application)).json()
            assert row["id"] == "older-1" and row["answer"] == "discard"
            ended = pushed(socket, lambda e: e["ev"] == "older" and [r["id"] for r in e["runs"]] == ["older-22"])
            assert [(done["id"], done["stage"]) for done in ended["done"]] == [("older-1", "discarded")]
            later = client.post("/api/v1/runs/older", json={"answer": "later"}, headers=page(application))
            assert later.status_code == 200 and later.json()["count"] == 0
            assert [(r["id"], r["later"]) for r in later.json()["runs"]] == [("older-22", True)]
            assert pushed(socket, lambda e: e["ev"] == "older" and e["count"] == 0)["runs"][0]["later"] is True
        assert client.get("/api/v1/health", headers=auth()).json()["older"]["count"] == 0
        assert client.post("/api/v1/runs/older/older-1", json={"answer": "discard"}, headers=auth()).status_code == 404
        wait_for(lambda: not (root / "older-1").exists())
        application.state.runs.closing = True                        # (as Runs.close sets it)
        assert client.post("/api/v1/runs/older", json={"answer": "later"}, headers=auth()).status_code == 200
        assert client.post("/api/v1/runs/older", json={"answer": "discard"}, headers=auth()).status_code == 503
        application.state.runs.closing = False
    assert (root / "older-22" / "run.json").is_file()                 # left for the next start


def test_an_api_session_is_wired_into_the_app(tmp_path, monkeypatch):
    """The wiring: create_app(session=...) hands the session to Runs and attaches the API hooks (the desktop
    gets none: health.session {"mode": "desktop"}); health.session is the session's own and its changes go out on the
    page's WebSocket as {ev: "session"}; the page trades a launch code for the session engine's own key, a code lives
    10 minutes, and one used is answered with how to open the session again; a link's sign-in (never a window's) tells
    the kernel, which replaces the link it showed; /closing takes the runs, revision and withdraw there too."""
    import inspect
    from sopa_companion.api import protocol
    from sopa_companion.engine import runs as runs_module
    from sopa_companion.engine.session import Session
    from sopa_companion.server import app as server
    given = []

    class SessionRuns(runs_module.Runs):          # what create_app gives Runs, passed on once Runs takes it
        def __init__(self, *args, session=None, **kwargs):
            given.append(session)
            if "session" in inspect.signature(runs_module.Runs.__init__).parameters:
                kwargs["session"] = session
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(server, "Runs", SessionRuns)
    api = protocol.build_session_config(id="1f3a9c2e", token="t" * 43, data_dir=str(tmp_path / "sopa-companion-1f3a9c2e"),
                                        state_dir=str(tmp_path / "1f3a9c2e"), python=sys.executable,
                                        kernel_pid=os.getpid(), open="link")
    control_r, control_w = os.pipe()
    events_r, events_w = os.pipe()
    session = Session(api, control_fd=control_r, events_fd=events_w)
    engine = json.loads(json.dumps(api))
    engine["server"]["port"] = 8799                  # api_serve.serve puts the port it listens on here
    try:
        application = server.create_app(engine, session=session, shutdown=lambda: None)
        assert given == [session] and application.state.session is session
        assert application.state.session_hooks is not None
        bearer = {"Authorization": "Bearer " + "t" * 43, "Origin": BASE}
        with TestClient(application, base_url=BASE) as client:
            health = client.get("/api/v1/health", headers=bearer).json()
            assert health["session"]["mode"] == "api" and health["session"]["id"] == "1f3a9c2e"
            assert health["older"]["runs"] == [] and "threads" in health
            code = client.post("/api/v1/session/launch-code", headers=bearer).json()
            assert code["expires_in"] == 600
            assert client.get("/?code=" + code["code"], follow_redirects=False).status_code == 200    # the page
            signed_in = sign_in(client, code["code"])
            assert signed_in.json() == {"key": application.state.page_key} and "set-cookie" not in signed_in.headers
            used = sign_in(client, code["code"])                          # a link used twice
            assert used.status_code == 401 and used.json()["detail"].endswith("Open SoPA companion again from your notebook.")
            os.set_blocking(events_r, False)

            def told():
                """The kernel's events from the session since the last look: the link ones."""
                lines = b""
                while True:
                    try:
                        chunk = os.read(events_r, 1 << 16)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    lines += chunk
                events = [protocol.decode_event_line(line) for line in lines.splitlines() if line]
                return [event for event in events if event and event["ev"] == "link"]
            assert told() == [], "a code for a window opened at once: nothing to replace in the notebook"
            shown = client.post("/api/v1/session/launch-code?link=1", headers=bearer).json()["code"]
            assert sign_in(client, shown).status_code == 200
            assert told() == [{"ev": "link", "state": "used"}], "the link shown in the notebook is replaced"
            assert sign_in(client, shown).status_code == 401 and told() == [], "a refused one tells nothing"
            for stale in ({"Origin": BASE, "Cookie": "sc_1f3a9c2e=" + application.state.page_key},
                          {"Origin": BASE, "X-SoPA-Page-Key": "x" * 43}):
                assert client.get("/api/v1/health", headers=stale).status_code == 401
            own = page(application)
            with client.websocket_connect("/api/v1/ws", headers=page_ws(application)) as socket:
                first = socket.receive_json()
                assert socket.accepted_subprotocol == "sopa-companion"
                assert first["ev"] == "session" and first["mode"] == "api" and first["state"] == "running"
            say = lambda **body: client.post("/api/v1/closing", json=body, headers=own).json()
            assert say(answer="add", revision=2, page="w")["taken"] is True
            assert say(answer="discard", revision=1, page="w")["taken"] is False
            answer = say(answer="withdraw", revision=3, page="w")
            assert answer["taken"] is True and answer["answer"] is None and session.closing_policy() is None
    finally:
        session.stop()
        for fd in (control_r, control_w, events_r, events_w):
            try:
                os.close(fd)
            except OSError:
                pass


def test_leave_passes_a_windows_order_on_and_refuses_a_bad_one(app, monkeypatch):
    """The leave prompt: a late Discard landing after Stay deleted the run the window stayed on.
    POST /runs/{id}/leave takes the window's {revision, page} and hands them to Runs.leave, which keeps each window's
    answers in their order."""
    application, client = app
    seen = []
    monkeypatch.setattr(application.state.runs, "leave",
                        lambda run_id, answer, **order: seen.append((run_id, answer, order)) or {"id": run_id})
    monkeypatch.setattr(application.state.qc_tests, "cancel", lambda run_id: None)
    reply = client.post("/api/v1/runs/r1/leave", json={"answer": "stay", "revision": 4, "page": "w"}, headers=auth())
    assert reply.json() == {"id": "r1"}
    client.post("/api/v1/runs/r1/leave", json={"answer": "discard", "revision": None}, headers=auth())
    assert seen == [("r1", "stay", {"revision": 4, "page": "w"}), ("r1", "discard", {})]
    for bad in ({"revision": -1}, {"revision": 1.5}, {"revision": "4"}, {"revision": True}, {"page": ""}, {"page": 4},
                {"page": "p" * 101}):
        assert client.post("/api/v1/runs/r1/leave", json={"answer": "stay", **bad}, headers=auth()).status_code == 422
    assert len(seen) == 2


def test_a_late_discard_never_cancels_the_runs_test(config, monkeypatch):
    """The leave route cancelled the run's test of a setting before Runs.leave found the answer
    overtaken (Stay, revision 20, then the Discard sent before it, revision 10): the run stayed, its test did not. An
    answer older than the last one its window had taken (Run.leave_revisions) cancels nothing."""
    from sopa_companion.server import create_app
    application = create_app(config, token="t" * 40)
    run = this_sessions_run(application, config, "r1")
    run.leave_revisions = {}                       # page -> the newest revision taken (engine/runs.py Runs.leave)
    cancelled, left = [], []
    monkeypatch.setattr(application.state.qc_tests, "cancel", cancelled.append)

    def leave(run_id, answer, revision=None, page=None):     # Runs.leave's own order check
        last = run.leave_revisions.get(page)
        taken = revision is None or last is None or revision >= last
        if taken and revision is not None:
            run.leave_revisions[page] = revision
        left.append((answer, revision, taken))
        return {"id": run_id}
    monkeypatch.setattr(application.state.runs, "leave", leave)
    with TestClient(application, base_url=BASE) as client:
        def say(**body):
            reply = client.post("/api/v1/runs/r1/leave", json=body, headers=page(application))
            assert reply.status_code == 200, reply.text
        say(answer="stay", revision=20, page="w")
        say(answer="discard", revision=10, page="w")                 # overtaken on its way
        assert cancelled == [] and left[-1] == ("discard", 10, False)
        say(answer="discard", revision=10, page="other")             # another window counts on its own
        say(answer="add", revision=21, page="w")
        say(answer="discard")                                        # a page that sends no order: as before
    assert cancelled == ["r1"] * 3 and len(left) == 5


def test_a_late_discard_leaves_the_run_and_its_test_on_the_engines_own_order(config, monkeypatch):
    """The same on Runs.leave itself (engine/runs.py keeps each window's newest revision in Run.leave_revisions)."""
    import inspect
    from sopa_companion.engine.runs import Runs
    if "revision" not in inspect.signature(Runs.leave).parameters:
        pytest.skip("needs engine/runs.py's leave in a window's order (Runs.leave revision)")
    from sopa_companion.server import create_app
    application = create_app(config, token="t" * 40)
    this_sessions_run(application, config, "r1")
    cancelled = []
    monkeypatch.setattr(application.state.qc_tests, "cancel", cancelled.append)
    with TestClient(application, base_url=BASE) as client:
        def say(**body):
            return client.post("/api/v1/runs/r1/leave", json=body, headers=page(application))
        assert say(answer="stay", revision=20, page="w").status_code == 200
        late = say(answer="discard", revision=10, page="w")             # overtaken on its way
        assert late.status_code == 200 and not late.json().get("leaving") and cancelled == []
        assert application.state.runs.get("r1").dir.exists()


def test_a_notebook_window_gets_a_45_s_code_and_only_a_link_lives_ten_minutes(tmp_path, monkeypatch):
    """Every code of a notebook session lived 10 minutes, though in window mode the code is used at once. The 10
    minutes are for links a person opens by hand: a session that opens no window itself (link; none, whose handle
    shows app.url) and a code asked for as a link (?link=1: the window mode's printed fallback); the bearer alone
    asks."""
    import inspect
    from sopa_companion.api import protocol
    from sopa_companion.engine import runs as runs_module
    from sopa_companion.engine.session import Session
    from sopa_companion.server import app as server

    class SessionRuns(runs_module.Runs):          # Runs takes the session once engine/runs.py does
        def __init__(self, *args, session=None, **kwargs):
            if "session" in inspect.signature(runs_module.Runs.__init__).parameters:
                kwargs["session"] = session
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(server, "Runs", SessionRuns)
    for mode, plain in (("window", 45), ("none", 600), ("link", 600)):
        folder = tmp_path / mode
        api = protocol.build_session_config(id="1f3a9c2e", token="t" * 43, data_dir=str(folder / "sopa-companion-1f3a9c2e"),
                                            state_dir=str(folder / "1f3a9c2e"), python=sys.executable,
                                            kernel_pid=os.getpid(), open=mode)
        engine = json.loads(json.dumps(api))
        engine["server"]["port"] = 8799              # api_serve.serve puts the port it listens on here
        client = TestClient(server.create_app(engine, session=Session(api), shutdown=lambda: None), base_url=BASE)
        bearer = {"Authorization": "Bearer " + "t" * 43, "Origin": BASE}
        assert client.post("/api/v1/session/launch-code", headers=bearer).json()["expires_in"] == plain, mode
        link = client.post("/api/v1/session/launch-code?link=1", headers=bearer).json()
        assert link["expires_in"] == 600, mode
        assert client.get("/?code=" + link["code"], follow_redirects=False).status_code == 200
        key = sign_in(client, link["code"]).json()["key"]
        signed = {"Origin": BASE, "X-SoPA-Page-Key": key}
        assert client.get("/api/v1/health", headers=signed).status_code == 200                   # signed in: the page
        assert client.post("/api/v1/session/launch-code?link=1", headers=signed).status_code == 401


# ---- proseg-live: found, or what makes it ---------------------------------------------------------------------------
def app_folder(root, *, source_tree: bool):
    """An app folder running on the example's settings ([proseg] binary = "auto"): a source tree (pyproject.toml
    beside the code) or an installed copy (a wheel's package folder)."""
    import shutil
    from sopa_companion.engine.config import ROOT
    root.mkdir(parents=True)
    shutil.copy(ROOT / "config.example.toml", root / "config.example.toml")
    if source_tree:
        (root / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    return root


def program_place(tmp_path, monkeypatch):
    """This Python's scripts folders as empty tmp folders, and PATH emptied: no proseg-live of this computer's is found."""
    import types
    from sopa_companion import locate
    scripts = tmp_path / "env" / "bin"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(locate, "sysconfig", types.SimpleNamespace(
        get_path=lambda name, scheme=None: str(tmp_path / "user" / "bin" if scheme else scripts),
        get_preferred_scheme=lambda kind: "posix_user"))
    monkeypatch.setenv("PATH", "")
    return scripts


def health_of(client):
    return client.get("/api/v1/health", headers=auth()).json()


def test_a_proseg_live_built_after_the_start_shows_without_a_restart(tmp_path, xenium_root, monkeypatch):
    """[proseg] binary = "auto" with nothing found: the engine keeps the place its fix puts the program (a source
    tree's native/bin/proseg-live, an installed copy's environment), and looks there at every /health and every run's
    start, so a build or an install shows at the page's next reload, as the page's words say."""
    from sopa_companion.engine import config as C
    from sopa_companion.server import create_app
    scripts = program_place(tmp_path, monkeypatch)
    overrides = {"server": {"port": 8799}, "desktop": {"exit_when_closed": False},
                 "paths": {"data_dir": str(tmp_path / "var"), "state_dir": str(tmp_path / "state"),
                           "sample_roots": [str(xenium_root)]},
                 **{section: {"python": "off"} for section in ("analysis", "zarr", "qc", "stain")}}
    for name, tree, fix, program in (("clone", True, "build", tmp_path / "clone" / "native" / "bin" / "proseg-live"),
                                     ("installed", False, "install", scripts / "proseg-live")):
        monkeypatch.setattr(C, "ROOT", app_folder(tmp_path / name, source_tree=tree))
        application = create_app({**overrides, "paths": {**overrides["paths"], "data_dir": str(tmp_path / name / "v")}},
                                 token="t" * 40)
        with TestClient(application, base_url=BASE) as client:
            health = health_of(client)
            assert (health["proseg"], health["proseg_missing"], health["proseg_fix"]) == (False, fix, fix), name
            assert application.state.runs.binary == str(program)
            program.parent.mkdir(parents=True, exist_ok=True)
            program.write_text("#!/bin/sh\nexit 0\n")
            program.chmod(0o755)
            health = health_of(client)                                     # no restart: the next reload sees it
            assert (health["proseg"], health["proseg_missing"]) == (True, None), name
            assert health["proseg_fix"] == fix, name                       # an older one is updated the same way


def test_health_says_how_to_fix_a_missing_proseg_live(tmp_path, xenium_root, monkeypatch):
    """/health's proseg_missing names what makes a missing proseg-live (build, install, or the setting that names it),
    and a run's start refuses with the same words: never an index command."""
    from sopa_companion.engine import config as C
    from sopa_companion.engine.runs import MISSING_PROSEG
    from sopa_companion.server import create_app
    program_place(tmp_path, monkeypatch)
    assert MISSING_PROSEG == {"build": "proseg-live is not built yet: run native/build.sh",
                              "install": "proseg-live is not installed: install it as the README's Install section says",
                              "setting": "proseg-live is not where [proseg] binary says"}
    base = {"server": {"port": 8799}, "desktop": {"exit_when_closed": False},
            **{section: {"python": "off"} for section in ("analysis", "zarr", "qc", "stain")}}
    cases = (("clone", True, "auto", "build"), ("installed", False, "auto", "install"),
             ("named", True, str(tmp_path / "elsewhere" / "proseg-live"), "setting"),
             ("built", True, "native/bin/proseg-live", "build"))
    for name, tree, binary, fix in cases:
        monkeypatch.setattr(C, "ROOT", app_folder(tmp_path / name, source_tree=tree))
        overrides = {**base, "proseg": {"binary": binary},
                     "paths": {"data_dir": str(tmp_path / name / "var"), "state_dir": str(tmp_path / name / "state"),
                               "sample_roots": [str(xenium_root)]}}
        application = create_app(overrides, token="t" * 40)
        with TestClient(application, base_url=BASE) as client:
            assert health_of(client)["proseg_missing"] == health_of(client)["proseg_fix"] == fix, name
            sample = application.state.samples.list()[0]["id"]
            refused = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth())
            assert refused.status_code == 503 and refused.json()["detail"] == MISSING_PROSEG[fix], name
            assert application.state.runs.list() == []
            assert application.state.runs.binary_fix == fix


def test_a_used_launch_code_names_the_command_as_it_is_typed(config, monkeypatch):
    """A launch code that was used already says how to open the app again with the command as the user types it
    (cli.command_words): an installed copy's `sopa-companion`, a source folder's `python -m sopa_companion`."""
    from sopa_companion.server import app as server_app, create_app
    for words in ("sopa-companion", "python -m sopa_companion"):
        monkeypatch.setattr(server_app, "command_words", lambda words=words: words)
        with TestClient(create_app(config, token="t" * 40), base_url=BASE) as client:
            code = client.post("/api/v1/session/launch-code", headers=auth()).json()["code"]
            assert sign_in(client, code).status_code == 200
            used = sign_in(client, code)
        assert used.status_code == 401
        assert used.json()["detail"] == ("This launch code was used already or has expired. Relaunch SoPA companion "
                                         f"from the icon or run {words} launch."), words
