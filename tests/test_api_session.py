"""API sessions use private fake pipes and samples; no engine, display or user data."""
from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from sopa_companion.api import protocol
from sopa_companion.engine.session import MAX_CONTROL_BYTES, Session, read_session_config

ACTIVE_CLIENT = None


class AppClient:
    """Same-thread ASGI driver: this sandbox cannot wake TestClient's blocking portal.

    All managers here are fakes. Run their short methods directly, so asyncio's executor shutdown does not need that
    same broken thread wakeup. Real control and events pipes still use the session's actual reader thread.
    """

    def __init__(self, app):
        self.app = app
        self.portal = self

    def __enter__(self):
        import httpx
        from unittest.mock import patch

        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        global ACTIVE_CLIENT
        self.loop = asyncio.new_event_loop()
        self.patch = patch("asyncio.to_thread", inline)
        self.patch.start()
        self.lifespan = self.app.router.lifespan_context(self.app)
        self.call(self.lifespan.__aenter__)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        ACTIVE_CLIENT = self
        return self

    def __exit__(self, *exc):
        global ACTIVE_CLIENT
        try:
            self.call(self.client.aclose)
            self.call(self.lifespan.__aexit__, *exc)
        finally:
            ACTIVE_CLIENT = None
            self.patch.stop()
            self.loop.close()

    def call(self, function, *args, **kwargs):
        async def run():
            answer = function(*args, **kwargs)
            return await answer if inspect.isawaitable(answer) else answer

        return self.loop.run_until_complete(run())

    def request(self, method, url, **kwargs):
        content = kwargs.get("content")
        if content is not None and not isinstance(content, (str, bytes)):
            async def stream():
                for chunk in content:
                    yield chunk
            kwargs["content"] = stream()
        return self.call(self.client.request, method, url, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)


@pytest.fixture
def api_config(tmp_path):
    return protocol.build_session_config(id="1f3a9c2e", token="t" * 43, data_dir=str(tmp_path / "sopa-companion-1f3a9c2e"),
                                         state_dir=str(tmp_path / "1f3a9c2e"), python="/fake/python", kernel_pid=os.getpid())


@pytest.fixture
def pipes(api_config):
    control_r, control_w = os.pipe()
    events_r, events_w = os.pipe()
    session = Session(api_config, control_fd=control_r, events_fd=events_w)
    os.set_blocking(events_r, False)
    yield session, control_w, events_r
    session.stop()
    for fd in (control_r, control_w, events_r, events_w):
        try:
            os.close(fd)
        except OSError:
            pass


def send(fd, op, **fields):
    os.write(fd, protocol.encode_line(protocol.build_control(op, **fields)).encode())


def until(check, timeout=2):
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "The session did not change in time"
        if ACTIVE_CLIENT is None:
            time.sleep(.005)
        else:
            ACTIVE_CLIENT.call(asyncio.sleep, .005)


def events(fd):
    data = b""
    while True:
        try:
            part = os.read(fd, 65536)
        except BlockingIOError:
            break
        if not part:
            break
        data += part
    return [protocol.decode_event_line(line) for line in data.splitlines()]


def binding(**updates):
    return protocol.build_control("bind", **{"path": "/fake/view.zarr", "name": "Notebook crop", "view": "overlay",
                                             "backed": True, "zarr": "/fake/sample-1.zarr", "memory_only": ["extra"],
                                             "points_key": "rna", "linked": ["image"], "written": ["extra"], **updates})


def delivery(id="d1", **updates):
    return protocol.build_event("deliver", **{"id": id, "run": "r1", "kind": "proseg", "staged": "/fake/staged.zarr",
                                              "shapes_key": "cells", "table_key": "counts", "set_default": True,
                                              "overwrite": False, "replaced": [], "written_to": None, "cells": 40,
                                              "genes": 5, "provenance": {}, **updates})


def write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"[{section}]\n" + "\n".join(f"{key} = {json.dumps(value)}" for key, value in table.items())
                              for section, table in config.items()))


def test_i1_file_and_initial_i5_are_valid(api_config, pipes, tmp_path):
    path = tmp_path / "session.toml"
    write_config(path, api_config)
    assert read_session_config(path) == api_config
    session, _, _ = pipes
    assert isinstance(session, protocol.Session)
    public = protocol.validate_health_session(session.public())
    assert public["sample"]["state"] == "opening"
    assert public["sample"]["id"] is None
    assert public["writes"] == "memory"
    assert session.target() is None and session.names() == set()
    assert session.bound is None and session.view_path is None
    assert session.closing_policy() is None
    public["sample"]["progress"]["done"] = 1
    assert session.public()["sample"]["progress"]["done"] == 0
    # An earlier version's file named the page's cookie (sc_<id>), which no page has now (its key goes in a header):
    # such a file still reads, its cookie unread.
    api_config["session"]["cookie"] = "sc_1f3a9c2e"
    write_config(path, api_config)
    assert read_session_config(path)["session"]["id"] == "1f3a9c2e"
    assert "cookie" not in protocol.build_session_config(id="1f3a9c2e", token="t" * 43, data_dir="/d", state_dir="/s",
                                                         python="/p", kernel_pid=os.getpid())["session"]


def test_prepare_bind_and_versioned_page_push(pipes):
    session, writer, reader = pipes
    calls = []
    session.bind_sample = lambda path, **kwargs: calls.append((path, kwargs)) or {"id": "z1", "tools": ["proseg", "stain"]}
    session.start()
    send(writer, "hello", protocol=1)
    send(writer, "prepare", done=5, total=10)
    until(lambda: session.public()["sample"]["state"] == "preparing")
    version, event = session.event()
    assert event["sample"]["progress"] == {"done": 5, "total": 10}
    assert session.event(version) == (version, None)
    os.write(writer, protocol.encode_line(binding()).encode())
    until(lambda: session.public()["sample"]["state"] == "ready")
    assert calls == [("/fake/view.zarr", {"name": "Notebook crop", "points_key": "rna"})]
    sample = session.public()["sample"]
    assert sample["name"] == "Notebook crop" and sample["zarr_name"] == "sample-1.zarr"
    assert sample["backed"] and sample["view"] == "overlay" and sample["memory_only"] == ["extra"]
    assert session.bound == "z1" and session.view_path == "/fake/view.zarr"
    assert [event["state"] for event in events(reader)] == ["preparing", "starting", "ready"]
    assert session.event(version)[1]["sample"]["id"] == "z1"


def test_bind_failure_and_prepare_failure_are_visible(pipes):
    session, writer, _ = pipes
    session.start()
    send(writer, "prepare_failed", message="Not enough room")
    until(lambda: session.public()["sample"]["state"] == "failed")
    assert session.public()["sample"]["message"] == "Not enough room"
    session.bind_sample = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Cannot read the image\ntrace"))
    os.write(writer, protocol.encode_line(binding()).encode())
    until(lambda: session.public()["sample"]["message"] == "Cannot read the image")
    assert session.bound is None


def test_an_empty_bind_error_still_reaches_failed(pipes):
    session, writer, _ = pipes
    session.bind_sample = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
    session.start()
    os.write(writer, protocol.encode_line(binding()).encode())
    until(lambda: session.public()["sample"]["state"] == "failed")
    assert session.public()["sample"]["message"] == "The sample could not be opened"


def test_a_slow_bind_does_not_block_control_or_complete_after_close(pipes):
    session, writer, _ = pipes
    entered, release = threading.Event(), threading.Event()

    def bind(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return {"id": "z1", "tools": ["stain"]}

    session.bind_sample = bind
    session.start()
    try:
        os.write(writer, protocol.encode_line(binding()).encode())
        assert entered.wait(2)
        send(writer, "received", id="d1")
        assert session.received("d1", .5)
        send(writer, "close", pending="discard")
        until(lambda: session.closing_policy() == "discard")
    finally:
        release.set()
    until(lambda: not session._binding)
    assert session.bound is None and session.public()["state"] == "ending"


@pytest.mark.parametrize("zarr,auto_save,writes,target", [
    ("/fake/changed.zarr", True, "zarr", "/fake/changed.zarr"),
    ("/fake/changed.zarr", False, "memory_autosave_off", None),
    (None, True, "memory", None),
])
def test_object_names_and_live_target(pipes, zarr, auto_save, writes, target):
    session, writer, _ = pipes
    session.start()
    send(writer, "object", names=["new", "RNA"], zarr=zarr, auto_save=auto_save)
    until(lambda: session.names() == {"new", "RNA"})
    assert session.target() == target and session.public()["writes"] == writes
    session.names().clear()
    assert session.names() == {"new", "RNA"}
    send(writer, "notebook", busy=True)
    until(lambda: session.notebook_busy)
    assert session.public()["notebook"] == {"busy": True}


def test_bad_and_unknown_control_lines_do_not_end_the_reader(pipes, caplog):
    session, writer, _ = pipes
    session.start()
    os.write(writer, b'{"op":"future"}\nnot json\n{"op":"hello","protocol":9}\n{"op":"notebook","busy":1}\n')
    send(writer, "notebook", busy=True)
    until(lambda: session.notebook_busy)
    assert "unknown API op 'future'" in caplog.text
    assert "invalid API control message" in caplog.text
    assert not session.kernel_gone


def test_oversized_control_line_is_skipped_and_reader_recovers(pipes, caplog):
    session, writer, _ = pipes
    session.start()
    os.write(writer, b"x" * (MAX_CONTROL_BYTES + 1) + b'\n{"op":"notebook","busy":true}\n')
    until(lambda: session.notebook_busy)
    assert "larger than" in caplog.text


def test_deliver_received_failed_and_applied_are_independent(pipes):
    session, writer, reader = pipes
    session.start()
    event = delivery()
    session.deliver(event)
    assert events(reader) == [event]
    assert not session.received("d1", .01) and session.applied("d1") is None
    send(writer, "received", id="d1")
    assert session.received("d1", .5)
    assert session.applied("d1") is None
    send(writer, "applied", id="d1", state="added", keys=["cells", "counts"], clash=[], message=None)
    until(lambda: session.applied("d1") is not None)
    answer = session.applied("d1")
    assert answer["state"] == "added"
    answer["keys"].clear()
    assert session.applied("d1")["keys"] == ["cells", "counts"]
    send(writer, "receive_failed", id="d2", message="Could not read it")
    until(lambda: "d2" in session._received)
    assert not session.received("d2", 1)


@pytest.mark.parametrize("closed", [False, True])
def test_full_or_closed_events_pipe_does_not_block_the_engine(pipes, caplog, closed, monkeypatch):
    """A full pipe nobody reads holds an event EVENTS_WRITE_S at most, then it is not sent; a closed one, not at all."""
    from sopa_companion.engine import session as module
    monkeypatch.setattr(module, "EVENTS_WRITE_S", 0.1)
    session, _, reader = pipes
    if closed:
        os.close(reader)
    else:
        while True:
            try:
                os.write(session._events_fd, b"x" * 4096)
            except BlockingIOError:
                break
    start = time.monotonic()
    session.deliver(delivery())
    assert not session.received("d1", .5)
    assert time.monotonic() - start < .25
    assert "was not sent" in caplog.text
    session.handle(protocol.build_control("notebook", busy=True))
    assert session.notebook_busy


def test_a_large_event_waits_for_room_and_one_cut_off_ends_the_channel(pipes, caplog, monkeypatch):
    """An event larger than the pipe waits for the notebook to read and arrives whole, the next one after it. Cut off
    part-way with nobody reading for EVENTS_WRITE_S, the channel ends: never another message run into the first."""
    import fcntl
    from sopa_companion.engine import session as module

    session, _, reader = pipes
    capacity = fcntl.fcntl(session._events_fd, fcntl.F_GETPIPE_SZ)
    event, ready = delivery(provenance={"large": "x" * (capacity * 2)}), protocol.build_event("session", state="ready",
                                                                                                message=None)
    received = bytearray()

    def drain():
        deadline = time.monotonic() + 5
        while received.count(b"\n") < 2 and time.monotonic() < deadline:
            try:
                received.extend(os.read(reader, 65536))
            except BlockingIOError:
                time.sleep(.01)

    thread = threading.Thread(target=drain)
    thread.start()
    session.deliver(event)
    session.deliver(ready)
    thread.join(10)
    assert [protocol.decode_event_line(line) for line in bytes(received).splitlines()] == [event, ready]
    assert session._events_fd is not None and session.received("d1", .01) is False
    monkeypatch.setattr(module, "EVENTS_WRITE_S", 0.1)
    session.deliver(delivery(id="d2", provenance={"large": "x" * (capacity * 2)}))
    assert session._events_fd is None and not session.received("d2", 1)
    partial = os.read(reader, capacity * 2)
    assert partial and not partial.endswith(b"\n")
    session.deliver(ready)
    with pytest.raises(BlockingIOError):
        os.read(reader, 1)
    assert "stopped reading part-way" in caplog.text


def test_event_descriptor_from_environment_is_private(api_config, monkeypatch):
    reader, writer = os.pipe()
    monkeypatch.setenv(protocol.ENV_EVENTS_FD, str(writer))
    session = Session(api_config)
    try:
        assert not os.get_inheritable(session._events_fd)
        assert not os.get_blocking(session._events_fd)
        event = protocol.build_event("ready", port=49157, pid=os.getpid())
        session.deliver(event)
        assert protocol.decode_event_line(os.read(reader, 4096)) == event
    finally:
        session.stop()
        os.close(reader)
        os.close(writer)


def test_control_reader_accepts_fragmented_utf8_and_last_line_before_eof(pipes):
    session, writer, _ = pipes
    session.start()
    # The control writer can split at any byte, including inside a UTF-8 character.
    payload = '{"op":"prepare_failed","message":"Cannot read µm"}\n'.encode()
    index = payload.index("µ".encode()) + 1
    os.write(writer, payload[:index])
    os.write(writer, payload[index:])
    until(lambda: session.public()["sample"]["state"] == "failed")
    assert session.public()["sample"]["message"] == "Cannot read µm"
    os.write(writer, b'{"op":"notebook","busy":true}')
    os.close(writer)
    until(lambda: session.kernel_gone)
    assert session.notebook_busy


@pytest.mark.parametrize("pending", ["add", "discard"])
def test_handle_close_is_kept_and_eof_overrides_it(pipes, pending):
    session, writer, reader = pipes
    closed = threading.Event()
    session.on_close = closed.set
    session.start()
    send(writer, "close", pending=pending)
    assert closed.wait(2)
    assert session.closing_policy() == pending
    assert events(reader)[0] == {"ev": "closing", "pending": pending, "by": "handle"}
    os.close(writer)
    until(lambda: session.kernel_gone)
    assert session.closing_policy() == "discard"
    assert not session.received("never", 10)


def test_no_answer_discards_and_page_answer_does_not_expire(pipes):
    session, _, _ = pipes
    clock = [1.0]
    session.clock = lambda: clock[0]
    session.answer_closing(protocol.build_closing("save"))
    clock[0] += 3600
    assert session.closing()["applies"] and session.closing_policy() == "add"
    session.begin_close(by="window")
    session.answer_closing(protocol.build_closing("discard", end=True))
    assert session.closing_policy() == "add"


def test_no_answer_on_window_close_uses_protocol_default(pipes):
    session, _, _ = pipes
    session.begin_close(by="window")
    assert session.closing_policy() == protocol.DEFAULT_CLOSE_POLICY == "discard"


def test_finish_reports_deliveries_and_discards(pipes):
    session, writer, reader = pipes
    session.start()
    session.deliver(delivery())
    session.deliver(delivery("d2", run="r2"))
    send(writer, "received", id="d1")
    send(writer, "applied", id="d1", state="added", keys=["cells"], clash=[], message=None)
    send(writer, "applied", id="d2", state="failed", keys=[], clash=["cells"], message="The name is taken")
    until(lambda: session.applied("d2") is not None)
    session.note_not_added("r3", "discarded", "Discarded")
    session.finish()
    ended = events(reader)[-1]
    assert ended == {"ev": "ended", "added": ["d1"], "not_added": [
        {"run": "r3", "reason": "discarded", "message": "Discarded"},
        {"run": "r2", "reason": "clash", "message": "The name is taken"}]}


def test_stop_wakes_reader_without_claiming_kernel_gone(pipes):
    session, _, _ = pipes
    session.start()
    session.stop()
    session.stop()
    assert not session._reader.is_alive()
    assert not session.kernel_gone and session.closing_policy() is None


def fake_app(config, session, history, *, shutdown=None, cleanup=None):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from sopa_companion.server.api_serve import attach_session

    @asynccontextmanager
    async def desktop_lifespan(app):
        raise AssertionError("The desktop closing dialog must not run")
        yield

    app = FastAPI(lifespan=desktop_lifespan)

    @app.get("/api/v1/health")
    def old_health():
        return {"ok": True, "threads": 3}

    @app.get("/api/v1/closing")
    @app.post("/api/v1/closing")
    @app.get("/api/v1/samples")
    @app.post("/api/v1/samples/open")
    @app.delete("/api/v1/samples/{sample_id}")
    @app.get("/api/v1/browse")
    def old_route():
        raise AssertionError("An API route fell through to a desktop route")

    for name in ("qc_tests", "compare", "qc", "deeper", "analysis"):
        setattr(app.state, name, SimpleNamespace(close=lambda name=name: history.append(name)))

    class Samples:
        def __init__(self):
            self.details = {"other": {"id": "other", "name": "Other sample"}}

        def bind(self, path, *, name, points_key):
            history.append(("bind", path, name, points_key))
            self.details["z1"] = {"id": "z1", "name": name, "tools": ["proseg", "stain"]}
            return self.details["z1"]

        def list(self):
            return list(self.details.values())

        def detail(self, id):
            return self.details[id]

        def close(self):
            history.append("zarr")

    class Runs:
        def pending(self):
            return [{"id": "r1", "kind": "stain", "sample": "Notebook crop", "iter": None, "state": "done"}]

        def close_session(self, policy):
            history.append(("close_session", policy))

        def close(self, policy):
            history.append(("runs", policy))

    from sopa_companion.server.presence import Presence
    app.state.zarr, app.state.runs = Samples(), Runs()
    app.state.presence = Presence()
    attach_session(app, config, session=session, shutdown=shutdown or (lambda: history.append("shutdown")),
                   close_window=lambda: history.append("window"), cleanup=cleanup or (lambda: history.append("cleanup")))
    return app


@pytest.fixture
def hooked_app(pipes):
    session, writer, reader = pipes
    config = json.loads(json.dumps(session.config))
    config["server"]["port"] = 49157
    history = []
    return fake_app(config, session, history), history, session, writer, reader


def test_hook_keeps_health_fields_and_reports_ready_only_when_serving(hooked_app):

    app, history, session, _, reader = hooked_app
    hooks = app.state.session_hooks
    serving = threading.Event()
    hooks.ready_when = serving.is_set
    with AppClient(app) as client:
        answer = client.get("/api/v1/health").json()
        assert answer == {"ok": True, "threads": 3, "session": session.public()}
        assert events(reader) == []
        serving.set()
        found = []
        until(lambda: found.extend(events(reader)) or bool(found))
        assert found == [{"ev": "ready", "port": 49157, "pid": os.getpid()}]
    assert history == ["qc_tests", "compare", "qc", "deeper", ("close_session", "discard"), "analysis", "zarr",
                       ("runs", "discard"), "window"]


def test_hook_binds_one_sample_and_refuses_desktop_sample_actions(hooked_app):

    app, _, session, writer, _ = hooked_app
    with AppClient(app) as client:
        assert client.get("/api/v1/samples").json() == {"samples": []}
        for url in ("/api/v1/browse", "/api/v1/browse?path=/fake"):
            response = client.get(url)
            assert response.status_code == 403
            assert response.json()["detail"] == "This session works on one sample"
        assert client.delete("/api/v1/samples/other").status_code == 403
        # The training patches' folder dialog is no sample browser: a notebook's session keeps it (here no run: 404).
        assert client.get("/api/v1/runs/none/stain/patches/folders").status_code == 404
        os.write(writer, protocol.encode_line(binding()).encode())
        until(lambda: session.bound == "z1")
        assert client.get("/api/v1/samples").json() == {"samples": [app.state.zarr.details["z1"]]}
        assert client.post("/api/v1/samples/open", json={"path": "/fake/view.zarr"}).json()["id"] == "z1"
        assert client.post("/api/v1/samples/open", json={"path": "/fake/other.zarr"}).status_code == 403
        assert client.post("/api/v1/samples/open", content="bad").status_code == 403
        sample = client.get("/api/v1/health").json()["session"]["sample"]
        assert sample["id"] == "z1" and sample["name"] == "Notebook crop"


def test_http_closing_keeps_answer_and_lists_all_pending_results(hooked_app):

    app, history, session, _, _ = hooked_app
    with AppClient(app) as client:
        before = client.get("/api/v1/closing").json()
        assert before["answer"] is None and not before["applies"]
        pending = [{"id": "r1", "kind": "stain", "sample": "Notebook crop", "iter": None, "state": "done"}]
        assert before["pending"] == before["runs"] == pending
        answer = client.post("/api/v1/closing", content='{"answer": "save"}', headers={"Content-Type": "text/plain"})
        assert answer.status_code == 200
        assert answer.json()["answer"] == "add"
        assert client.get("/api/v1/closing").json()["applies"]
        assert session.public()["state"] == "running" and "shutdown" not in history
    assert ("close_session", "add") in history


@pytest.mark.parametrize("body", [{}, [], {"answer": "stay"}, {"answer": "add", "end": 1},
                                  {"answer": "add", "extra": float("nan")}])
def test_http_closing_rejects_invalid_answers_without_storing_them(hooked_app, body):

    app, _, session, _, _ = hooked_app
    with AppClient(app) as client:
        response = client.post("/api/v1/closing", content=json.dumps(body))
        assert response.status_code == 422
        assert session.closing_policy() is None


def test_http_closing_limits_declared_and_streamed_bodies(hooked_app):

    app, _, session, _, _ = hooked_app
    with AppClient(app) as client:
        assert client.post("/api/v1/closing", content="x" * 4097).status_code == 413
        assert client.post("/api/v1/closing", content=iter([b"x" * 2000, b"x" * 2097])).status_code == 413
        assert client.post("/api/v1/closing", content="{").status_code == 422
        assert session.closing_policy() is None


class FakeSocket:
    def __init__(self, history):
        self.history = history

    async def send_json(self, event):
        self.history.append(event)


def test_http_end_keeps_answer_and_pushes_ending_before_shutdown(hooked_app):

    app, history, session, _, _ = hooked_app
    hooks = app.state.session_hooks
    with AppClient(app) as client:
        socket = FakeSocket(history)
        version = client.portal.call(hooks.push, socket)
        assert history[-1]["state"] == "running"
        assert client.portal.call(hooks.push, socket, version) == version
        response = client.post("/api/v1/closing", json={"answer": "add", "end": True})
        assert response.status_code == 200 and response.json()["answer"] == "add"
        assert session.closing_policy() == "add"
        client.portal.call(hooks.push, socket, version)
        until(lambda: "shutdown" in history)
        ending = next(i for i, item in enumerate(history) if isinstance(item, dict) and item.get("state") == "ending")
        assert ending < history.index("shutdown")
    assert ("close_session", "add") in history


def test_shutdown_after_half_a_second_without_a_websocket(hooked_app):
    from sopa_companion.server.api_serve import ENDING_PUSH_TIMEOUT_S

    app, history, session, writer, _ = hooked_app
    with AppClient(app):
        start = time.monotonic()
        send(writer, "close", pending="discard")
        until(lambda: "shutdown" in history)
        elapsed = time.monotonic() - start
        assert ENDING_PUSH_TIMEOUT_S <= elapsed < 1.5
        assert session.public()["state"] == "ending"


def test_teardown_reads_received_after_http_has_stopped_and_finishes_with_ended(hooked_app):

    app, history, session, writer, reader = hooked_app

    def close_session(policy):
        history.append(("close_session", policy))
        assert session._reader.is_alive()
        session.deliver(delivery())
        send(writer, "received", id="d1")
        assert session.received("d1", .5)

    app.state.runs.close_session = close_session
    with AppClient(app) as client:
        assert client.post("/api/v1/closing", json={"answer": "add"}).status_code == 200
    final = events(reader)
    assert final[-1] == {"ev": "ended", "added": [], "not_added": []}
    assert final[-2] == {"ev": "session", "state": "closed", "message": None}
    assert not session._reader.is_alive() and "cleanup" not in history


def test_failed_close_save_still_removes_runs_and_closes_every_resource(hooked_app, caplog):

    app, history, _, _, reader = hooked_app

    def fail_save(policy):
        history.append(("close_session", policy))
        raise OSError("No space left")

    def fail_analysis():
        history.append("analysis")
        raise OSError("Worker stopped")

    app.state.runs.close_session = fail_save
    app.state.analysis.close = fail_analysis
    with AppClient(app) as client:
        client.post("/api/v1/closing", json={"answer": "add"})
    assert history == ["qc_tests", "compare", "qc", "deeper", ("close_session", "add"), "analysis", "zarr",
                       ("runs", "discard"), "window"]
    assert "pending results failed" in caplog.text and "analysis failed" in caplog.text
    assert events(reader)[-1]["ev"] == "ended"


def test_kernel_eof_during_delivery_discards_remainder_and_cleans_up_last(hooked_app):

    app, history, session, writer, _ = hooked_app

    def close_session(policy):
        history.append(("close_session", policy))
        session.deliver(delivery())
        os.close(writer)
        assert not session.received("d1", 1)
        assert session.kernel_gone and session.closing_policy() == "discard"

    app.state.runs.close_session = close_session
    with AppClient(app) as client:
        client.post("/api/v1/closing", json={"answer": "add"})
    assert history[-2:] == ["window", "cleanup"]
    assert history.index(("runs", "discard")) < history.index("cleanup")


def test_cleanup_removes_only_named_private_folders_and_never_follows_links(api_config, tmp_path):
    from sopa_companion.server.api_serve import cleanup_session_folders

    target = tmp_path / "outside"
    target.mkdir()
    (target / "keep").write_text("unchanged")
    for key in ("state_dir", "data_dir"):
        path = Path(api_config["paths"][key])
        path.mkdir()
        (path / "linked").symlink_to(target, target_is_directory=True)
    cleanup_session_folders(api_config)
    assert (target / "keep").read_text() == "unchanged"
    assert not Path(api_config["paths"]["data_dir"]).exists()
    assert not Path(api_config["paths"]["state_dir"]).exists()
    api_config["paths"].update(data_dir=str(target), state_dir=str(target))
    cleanup_session_folders(api_config)
    assert (target / "keep").read_text() == "unchanged"


def test_window_cleanup_matches_own_uid_and_exact_profile(api_config, monkeypatch):
    import psutil
    from sopa_companion.server.api_serve import close_session_window

    profile = str(Path(api_config["paths"]["state_dir"]) / "chrome-profile")
    closed = []

    def process(name, flag, uid):
        return SimpleNamespace(info={"cmdline": ["chrome", flag], "uids": SimpleNamespace(real=uid)},
                               terminate=lambda: closed.append(name))

    processes = [process("own", "--user-data-dir=" + profile, os.getuid()),
                 process("other", "--user-data-dir=" + profile + "-other", os.getuid()),
                 process("foreign", "--user-data-dir=" + profile, os.getuid() + 1),
                 SimpleNamespace(info={"cmdline": [], "uids": None})]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: processes)
    close_session_window(api_config)
    assert closed == ["own"]


def test_a_window_sessions_last_page_closes_it_on_a_goodbye_and_only_pauses_its_runs_when_away(pipes):
    """A session opened in a window (open="window": exit_when_closed): its last page gone without a goodbye is away,
    so its runs pause and the session goes on (the kernel owns it, never stopped by itself); its last page gone with a
    goodbye closes the session as a window close does (the page's answer, else discard)."""
    session, _, _ = pipes
    config = json.loads(json.dumps(session.config))
    config["server"]["port"] = 49157
    config["desktop"].update(exit_when_closed=True, close_grace_s=0.05, close_grace_pending_s=0.05)
    history, paused = [], []
    app = fake_app(config, session, history)
    app.state.runs.pause_for_away = lambda: paused.append("away") or ["r1"]
    pages = app.state.presence
    with AppClient(app):
        pages.connected("page-1")
        pages.disconnected("page-1")                      # a crash, a lost connection: no goodbye
        until(lambda: paused == ["away"], timeout=5)
        assert "shutdown" not in history and session.public()["state"] == "running"
        pages.connected("page-2")                         # opened again (app.open()), then closed for good
        pages.goodbye("page-2", "socket")
        pages.disconnected("page-2")
        until(lambda: "shutdown" in history, timeout=5)
        assert session.public()["state"] == "ending" and paused == ["away"]
    assert ("close_session", "discard") in history          # nobody answered: discard


def test_a_window_session_left_away_never_stops_by_itself_and_its_answer_with_bye_is_its_pages_goodbye(pipes,
                                                                                                    monkeypatch):
    """A session opened in a window, its page gone without a goodbye and nothing waiting for an answer: it pauses and
    waits however long, never stopping by itself (the kernel owns it: the watcher's away_stop_s is 0, where the
    desktop's engine stops after 30 minutes). The session's own POST /closing with bye is its page's goodbye, as the
    socket's bye is (but not one its page's earlier showing said, reaching the engine late), and the session closes."""
    from sopa_companion.server import api_serve
    session, _, _ = pipes
    config = json.loads(json.dumps(session.config))
    config["server"]["port"] = 49157
    config["desktop"].update(exit_when_closed=True, close_grace_s=0.05, close_grace_pending_s=0.05)
    watched, real = [], api_serve.watch
    # The watcher as the session starts it, looking every 10 ms instead of every second.
    monkeypatch.setattr(api_serve, "watch", lambda presence, **kw: watched.append(kw) or real(presence, **{**kw, "tick": .01}))
    history, paused = [], []
    app = fake_app(config, session, history)
    app.state.runs.pause_for_away = lambda: paused.append("away") or []
    app.state.runs.pending = lambda: []                       # nothing waits for an answer
    pages = app.state.presence
    with AppClient(app) as client:
        assert [kw["away_stop_s"] for kw in watched] == [0]
        pages.connected("page-1", 1)
        pages.disconnected("page-1")                          # gone without a goodbye
        until(lambda: paused == ["away"], timeout=5)
        client.call(asyncio.sleep, 1.5)                       # what a stop after a second's absence would have taken
        assert "shutdown" not in history and session.public()["state"] == "running"
        pages.connected("page-2", 2)                          # opened again (app.open())
        late = {"answer": "discard", "page": "page-2", "bye": True, "visit": 1}
        assert client.post("/api/v1/closing", json=late).json()["taken"] is False
        assert pages.pages["page-2"]["bye"] is None
        answer = client.post("/api/v1/closing", json={**late, "visit": 2})
        assert answer.status_code == 200 and pages.pages["page-2"]["bye"] == "closing"
        pages.disconnected("page-2")
        until(lambda: "shutdown" in history, timeout=5)
        assert session.public()["state"] == "ending"
    assert ("close_session", "discard") in history


class FakeChrome:
    """A Chrome process as desktop/window.py AppWindow lists it (psutil.process_iter's info) and reads its desktop."""

    def __init__(self, cmdline, environ):
        self.info = {"cmdline": cmdline, "uids": SimpleNamespace(real=os.getuid()), "status": "sleeping"}
        self._environ = environ

    def environ(self):
        return dict(self._environ)

    def is_running(self):
        return True

    def status(self):
        return "sleeping"


def test_the_sessions_engine_follows_the_window_the_kernel_opened_whatever_its_files_say(tmp_path, monkeypatch):
    """The kernel reads [desktop] browser (the user's own file over the app folder's); the session's engine reads the
    app folder's files and session.toml, never that setting. Whichever window the kernel opened, the engine follows it:
    the session's own Chrome window (its profile flag) is where the Cellpose GUI opens and what the session's end
    closes; in the user's own browser, the GUI opens on this desktop while a page is open, and there is nothing to
    close (a tab is the user's)."""
    import psutil
    from sopa_companion.server import app as server
    from sopa_companion.server.api_serve import close_session_window
    api = protocol.build_session_config(id="1f3a9c2e", token="t" * 43, data_dir=str(tmp_path / "sopa-companion-1f3a9c2e"),
                                        state_dir=str(tmp_path / "1f3a9c2e"), python=sys.executable,
                                        kernel_pid=os.getpid(), open="window")
    assert "browser" not in api["desktop"]
    control_r, control_w = os.pipe()
    events_r, events_w = os.pipe()
    session = Session(api, control_fd=control_r, events_fd=events_w)
    engine = json.loads(json.dumps(api))
    engine["server"]["port"] = 8799
    for name in ("PLH_TAG", "PLH_DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":7")
    try:
        application = server.create_app(engine, session=session, shutdown=lambda: None)
        patches = application.state.runs.stain.patches
        flag = "--user-data-dir=" + str(Path(api["paths"]["state_dir"]) / "chrome-profile")
        chrome = FakeChrome(["/opt/google/chrome/chrome", flag, "--class=SoPACompanion"], {"DISPLAY": ":42"})
        firefox = FakeChrome(["/usr/lib/firefox/firefox", "file:///run/open-x.html"], {"DISPLAY": ":7"})
        ended = []

        def processes(*listed):
            return lambda attrs: [SimpleNamespace(info=process.info, terminate=lambda p=process: ended.append(p))
                                  for process in listed]
        # The kernel opened the session's own Chrome window (the user's file said "chrome").
        patches.window.processes = lambda attrs: [chrome]
        assert patches.display() == {"DISPLAY": ":42"}
        monkeypatch.setattr(psutil, "process_iter", processes(chrome, firefox))
        close_session_window(engine)
        assert ended == [chrome]
        # The kernel opened the user's own browser (the user's file said "default"): no Chrome of the session's.
        patches.window.processes, patches.window.found, ended[:] = (lambda attrs: [firefox]), None, []
        assert patches.display() == {}                                    # no page open
        application.state.presence.connected("page-1")
        assert patches.display()["DISPLAY"] == ":7"                       # a page open, on this desktop
        monkeypatch.setattr(psutil, "process_iter", processes(firefox))
        close_session_window(engine)
        assert ended == []
    finally:
        session.stop()
        for fd in (control_r, control_w, events_r, events_w):
            try:
                os.close(fd)
            except OSError:
                pass


def test_desktop_hook_leaves_routes_and_lifespan_alone():
    from fastapi import FastAPI
    from sopa_companion.server.api_serve import attach_session, session_health

    app = FastAPI()
    routes, lifespan = list(app.routes), app.router.lifespan_context
    assert attach_session(app, {}) is None
    assert app.routes == routes and app.router.lifespan_context == lifespan
    assert app.state.session is None and app.state.session_hooks is None
    assert session_health() == {"mode": "desktop"}


def test_api_app_factory_receives_the_same_session_and_hook_is_idempotent(api_config, pipes):
    from sopa_companion.server.api_serve import attach_session, create_api_app

    session, _, _ = pipes
    history = []
    api_config["server"]["port"] = 49157

    def factory(config, **kwargs):
        assert kwargs["session"] is session
        return fake_app(config, session, history, shutdown=kwargs["shutdown"])

    app = create_api_app(api_config, session=session, app_factory=factory)
    count, lifespan = len(app.routes), app.router.lifespan_context
    hooks = attach_session(app, api_config, session=session)
    assert len(app.routes) == count and app.router.lifespan_context == lifespan
    assert hooks is app.state.session_hooks and app.state.session is session


def test_serve_uses_port_zero_and_releases_listener_and_lock(api_config, pipes, monkeypatch, tmp_path):
    import sopa_companion.cli as cli
    import sopa_companion.desktop.logs as logs
    import sopa_companion.server.runtime_file as runtime
    from sopa_companion.server.api_serve import serve

    session, _, reader = pipes
    history = []
    engine_lock = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR, 0o600)

    class Listener:
        def getsockname(self):
            return ("127.0.0.1", 49157)

        def close(self):
            history.append("listener closed")

    listener = Listener()

    def listen(host, port):
        assert host == "127.0.0.1" and port == 0
        return listener

    def factory(config, **kwargs):
        assert config["server"]["port"] == 49157
        return fake_app(config, kwargs["session"], history, shutdown=kwargs["shutdown"])

    class Server:
        def __init__(self, config):
            self.config, self.started, self.should_exit = config, False, False

        def run(self, *, sockets):
            assert sockets == [listener]
            with AppClient(self.config.app) as client:
                self.started = True
                found = []
                until(lambda: found.extend(events(reader)) or bool(found))
                assert found == [{"ev": "ready", "port": 49157, "pid": os.getpid()}]
                client.post("/api/v1/closing", json={"answer": "discard", "end": True})
                client.portal.call(self.config.app.state.session_hooks.push, FakeSocket(history))
                until(lambda: self.should_exit)

    monkeypatch.setattr(cli, "listen", listen)
    monkeypatch.setattr(logs, "install_rotation", lambda config: None)
    monkeypatch.setattr(runtime, "acquire_engine_lock", lambda config: engine_lock)
    monkeypatch.setattr(runtime, "install_lifecycle", lambda *args: history.append("runtime"))
    assert serve(api_config, session=session, app_factory=factory, server_factory=Server) == 0
    assert history[0] == "runtime" and history[-1] == "listener closed"
    with pytest.raises(OSError):
        os.fstat(engine_lock)


@pytest.mark.parametrize("host", ["0.0.0.0", "example.com"])
def test_serve_refuses_non_loopback_hosts_before_startup(api_config, pipes, monkeypatch, host):
    import sopa_companion.desktop.logs as logs
    from sopa_companion.server.api_serve import serve

    api_config["server"]["host"] = host
    monkeypatch.setattr(logs, "install_rotation", lambda config: pytest.fail("No startup on a non-loopback host"))
    with pytest.raises(ValueError, match="loopback"):
        serve(api_config, session=pipes[0])


def test_serve_releases_resources_when_app_construction_fails(api_config, pipes, monkeypatch, tmp_path):
    import sopa_companion.cli as cli
    import sopa_companion.desktop.logs as logs
    import sopa_companion.server.runtime_file as runtime
    from sopa_companion.server.api_serve import serve

    engine_lock = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    closed = []
    listener = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 49157), close=lambda: closed.append("listener"))
    monkeypatch.setattr(cli, "listen", lambda *args: listener)
    monkeypatch.setattr(logs, "install_rotation", lambda config: None)
    monkeypatch.setattr(runtime, "acquire_engine_lock", lambda config: engine_lock)

    def fail(config, **kwargs):
        raise ValueError("App construction failed")

    with pytest.raises(ValueError, match="App construction failed"):
        serve(api_config, session=pipes[0], app_factory=fail)
    assert closed == ["listener"]
    assert pipes[0]._events_fd is None
    with pytest.raises(OSError):
        os.fstat(engine_lock)


def test_entry_reads_i1_before_loading_effective_config_and_never_reexecs(api_config, tmp_path, monkeypatch):
    import sopa_companion.engine.config as engine_config
    from sopa_companion.server import api_serve

    path = tmp_path / "session.toml"
    write_config(path, api_config)
    calls = []
    monkeypatch.delenv(protocol.ENV_EVENTS_FD, raising=False)
    monkeypatch.setattr(engine_config, "load_config", lambda filename: calls.append(filename) or api_config)

    def serve(config, *, session):
        assert session.config == config == api_config
        session.stop()
        return 7

    monkeypatch.setattr(api_serve, "serve", serve)
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("API mode must use the current interpreter"))
    assert api_serve.main(["--config", str(path)]) == 7
    assert calls == [str(path)]


# ---- whose answer closes the session ----------------------------------------------------------------------------------
def test_an_answer_is_for_its_runs_in_its_windows_order_and_can_be_taken_back(pipes):
    session, _, _ = pipes
    taken = session.answer_closing({"answer": "add", "revision": 2, "page": "a"})
    assert taken and session.closing()["covers"] is None              # every run pending, until covered
    session.cover_closing(taken, ["r1"])
    assert session.closing()["covers"] == ["r1"] and session.closing_policy() == "add"
    # Discard and close, sent before Keep working (revision 3), arrives after it: it changes nothing.
    assert session.answer_closing({"answer": "withdraw", "revision": 3, "page": "a"})
    assert session.answer_closing({"answer": "discard", "revision": 2, "page": "a"}) == 0
    assert session.closing_policy() is None and session.closing()["applies"] is False
    assert session.answer_closing({"answer": "discard", "revision": 0, "page": "b"})   # another window: its own order
    named = session.answer_closing({"answer": "add", "runs": ["r1", "gone"], "revision": 4, "page": "a"})
    session.cover_closing(taken, ["r1", "r2"])                        # an older answer's runs never narrow this one
    assert session.closing()["covers"] == ["gone", "r1"]
    session.cover_closing(named, ["r1", "r2"])
    assert session.closing()["covers"] == ["r1"] and session.closing()["applies"] is True
    # An answer whose runs are all gone covers none: it applies to nothing, as on the desktop (ClosingAnswer.public).
    none = session.answer_closing({"answer": "discard", "runs": ["gone"], "revision": 5, "page": "a"})
    session.cover_closing(none, ["r1"])
    assert session.closing()["covers"] == [] and session.closing()["applies"] is False
    session.withdraw_closing()                                        # a page connected again (a reload)
    assert session.closing_policy() is None
    with pytest.raises(ValueError):
        session.answer_closing({"answer": "stay"})


def test_the_handles_close_outranks_a_kept_beacon_and_the_popup_outranks_both(pipes):
    session, _, _ = pipes
    session.answer_closing({"answer": "discard"})                     # a beacon: the window went without the popup
    session.begin_close("add", by="handle")                          # app.close(pending="add"): the caller's code
    assert session.closing_policy() == "add" and session.closing_covers() is None


def test_the_popups_answer_with_end_closes_for_its_runs(pipes):
    session, _, _ = pipes
    taken = session.answer_closing({"answer": "add", "end": True})
    session.cover_closing(taken, ["r1"])
    session.begin_close("discard", by="handle")                       # too late: the popup decided
    assert session.closing_policy() == "add" and session.closing_covers() == frozenset({"r1"})
    assert session.answer_closing({"answer": "withdraw"}) == 0        # closing: nothing more is taken


def test_http_closing_takes_runs_revision_and_withdraw(hooked_app):
    app, history, session, _, _ = hooked_app
    with AppClient(app) as client:
        first = client.post("/api/v1/closing", json={"answer": "add", "revision": 5, "page": "w"}).json()
        assert first["taken"] is True and first["covers"] == ["r1"]          # every run pending when it came
        late = client.post("/api/v1/closing", json={"answer": "discard", "revision": 4, "page": "w"}).json()
        assert late["taken"] is False and late["answer"] == "add"
        back = client.post("/api/v1/closing", content='{"answer": "withdraw", "revision": 6, "page": "w"}',
                           headers={"Content-Type": "text/plain"}).json()
        assert back["taken"] is True and back["answer"] is None and session.closing_policy() is None
        for body in ({"answer": "add", "runs": "r1"}, {"answer": "add", "revision": "1"}, {"answer": "add", "page": 1},
                     {"answer": "withdraw", "end": 1}):
            assert client.post("/api/v1/closing", json=body).status_code == 422, body
        named = client.post("/api/v1/closing", json={"answer": "add", "runs": ["r1", "nope"]}).json()
        assert named["covers"] == ["r1"]
    assert ("close_session", "add") in history


def test_a_run_started_after_the_pages_add_is_discarded_at_close(hooked_app):
    """The page answered Add for r1 (its popup's runs) and went; r2 started later in another window: at close only r1
    is added; r2, which nobody answered for, is discarded (Runs.close_session takes {run id: answer})."""
    app, history, session, _, _ = hooked_app
    pending = [{"id": "r1", "kind": "proseg", "sample": "Notebook crop", "iter": 3, "state": "paused"}]
    app.state.runs.pending = lambda: list(pending)
    with AppClient(app) as client:
        assert client.post("/api/v1/closing", json={"answer": "add", "runs": ["r1"]}).json()["covers"] == ["r1"]
        pending.append({"id": "r2", "kind": "stain", "sample": "Notebook crop", "iter": None, "state": "running"})
    assert ("close_session", {"r1": "add"}) in history           # Runs.close_session: any other run is discarded


def test_serve_with_a_session_config_runs_the_session_in_this_interpreter(api_config, tmp_path, monkeypatch):
    """`python -m sopa_companion serve --config <session.toml>` (how api/handle.py starts the engine) serves the notebook's
    session in the kernel's own interpreter: never the app's .venv, whatever the folder holds."""
    from sopa_companion import cli
    from sopa_companion.server import api_serve
    path = tmp_path / "session.toml"
    write_config(path, api_config)
    served = []
    monkeypatch.setattr(api_serve, "main", lambda argv: served.append(argv) or 0)
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("API mode must use the current interpreter"))
    assert cli.main(["serve", "--config", str(path)]) == 0
    assert cli.main(["--config", str(path), "serve"]) == 0
    assert served == [["--config", str(path)]] * 2
    args = cli.parser().parse_args(["--config", "desktop.toml", "serve", "--port", "0"])
    assert args.config == "desktop.toml" and args.port == 0               # the top-level --config is kept
    assert not cli.api_session(tmp_path / "missing.toml")
    (tmp_path / "desktop.toml").write_text('[server]\nport = 8810\n')
    assert not cli.api_session(tmp_path / "desktop.toml")


def test_serve_port_zero_is_any_free_port_not_the_configured_one(tmp_path, monkeypatch):
    from sopa_companion import cli
    import sopa_companion.desktop.logs as logs
    import sopa_companion.server.runtime_file as runtime
    asked = []
    lock = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    monkeypatch.setattr(logs, "install_rotation", lambda config: None)
    monkeypatch.setattr(runtime, "acquire_engine_lock", lambda config: lock)

    def listen(host, port):
        asked.append(port)
        raise RuntimeError("stop here")
    monkeypatch.setattr(cli, "listen", listen)
    # Its own state folder: serve first looks there for another copy's engine (none here).
    config = {"server": {"port": 8777, "host": "127.0.0.1"},
              "paths": {"state_dir": str(tmp_path / "state"), "data_dir": str(tmp_path / "var")}}
    for port in (0, None, 8811):
        with pytest.raises(RuntimeError, match="stop here"):
            cli.serve(config, "127.0.0.1", port)
        lock = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    os.close(lock)
    assert asked == [0, 8777, 8811]
