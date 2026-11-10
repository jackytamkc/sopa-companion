"""Closing without a Chrome process to watch (server/presence.py): the pages signed in, the last one that leaves decides.
It said goodbye (its socket's bye, its log's or its answer's last request): a real close, and a run nobody answered for
is discarded with no dialog. It said nothing (a crash, a tab the browser put away, a lost connection): it is away, its
runs pause, and the engine waits for the app to be opened again, which finds them with a notice. The same in every
browser; the engine never asks which one."""
from __future__ import annotations

import asyncio
import threading
import types

import pytest
from fastapi.testclient import TestClient

from conftest import wait_for
from sopa_companion.cli import command_words
from sopa_companion.server import presence as P

BASE = "http://127.0.0.1:8799"
TOKEN = "t" * 40


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


# ---- who is here -----------------------------------------------------------------------------------------------------
def test_the_last_page_that_leaves_decides_closed_or_away():
    clock = Clock()
    pages = P.Presence(clock=clock)
    assert pages.verdict(6) is None and pages.public() == {"pages": 0, "state": "none", "since_s": 0.0, "connects": 0,
                                                           "windows": 0}
    pages.connected("a")
    assert pages.open() == 1 and pages.verdict(6) is None and pages.public()["state"] == "here"
    pages.disconnected("a")                                         # gone without a word ...
    clock.now += 3
    assert pages.verdict(6) is None and pages.public()["state"] == "leaving"     # ... a reload may be on its way
    clock.now += 4
    assert pages.verdict(6) == "away" and pages.public()["state"] == "away"
    pages.goodbye("a", "page-log")                                  # its log's goodbye, after its socket closed
    assert pages.verdict(6) == "closed"
    pages.connected("b")                                            # another page: its goodbye before it goes
    pages.goodbye("b", "socket")
    assert pages.verdict(0) is None                                 # still here
    pages.disconnected("b")
    clock.now += 6
    assert pages.verdict(6) == "closed" and pages.public()["connects"] == 2


def test_a_reload_within_the_grace_and_a_page_shown_again_are_here():
    clock = Clock()
    pages = P.Presence(clock=clock)
    pages.connected("old")
    pages.goodbye("old", "socket")
    pages.disconnected("old")
    clock.now += 2
    pages.connected("new")                                          # the reloaded page: a new id, signed in again
    clock.now += 60
    assert pages.verdict(6) is None and pages.open() == 1
    pages.goodbye("new", "closing")
    pages.connected("new")                                          # a page the browser kept and showed again
    pages.disconnected("new")
    pages.disconnected("new")
    clock.now += 7
    assert pages.verdict(6) == "away", "its goodbye was taken back as it came back"


def test_a_goodbye_counts_only_for_the_showing_that_said_it():
    """A page the browser kept aside and showed again (pageshow persisted) connects again as its next showing. A goodbye
    of the showing before it (its log's or its answer's last request, or its old socket's bye) that reaches the engine
    only then changes nothing, so a crash of the page shown now is away, never a close that discards."""
    clock = Clock()
    pages = P.Presence(clock=clock)
    pages.connected("a", 1)
    pages.goodbye("a", "socket", 1)
    pages.disconnected("a")
    pages.connected("a", 2)                                         # shown again
    for via in GOODBYES_LATE:
        pages.goodbye("a", via, 1)                                  # the first showing's goodbyes, arriving late
    assert pages.pages["a"]["bye"] is None and pages.verdict(0) is None
    pages.disconnected("a")                                         # then the page shown now crashes
    clock.now += 7
    assert pages.verdict(6) == "away"
    pages.goodbye("a", "page-log", 2)                               # its own goodbye counts, after its socket closed too
    assert pages.verdict(6) == "closed"
    # A later showing's goodbye counts even before that showing connected (shown again and closed at once); a socket of
    # an earlier showing connecting late changes nothing of it.
    pages.connected("b", 1)
    pages.goodbye("b", "closing", 2)
    pages.connected("b", 1)
    pages.disconnected("b")
    pages.disconnected("b")
    clock.now += 7
    assert pages.verdict(6) == "closed" and pages.pages["b"]["bye"] == "closing"
    # A socket or a goodbye that names no showing is the page's newest, and what it says is never stale.
    pages.connected("c")
    pages.goodbye("c", "socket")
    pages.disconnected("c")
    clock.now += 7
    assert pages.verdict(6) == "closed"
    pages.connected("d", 2)
    pages.goodbye("d", "page-log")
    assert pages.pages["d"]["bye"] == "page-log" and pages.pages["d"]["visit"] == 2
    assert not pages.stale("d", None) and not pages.stale("d", 2) and pages.stale("d", 1) and not pages.stale("new", 1)


GOODBYES_LATE = ("page-log", "closing", "socket")


def test_a_showing_is_a_small_whole_number_or_none():
    for value, number in (("2", 2), (3, 3), ("0", 0), ("999999999", 999999999)):
        assert P.visit_number(value) == number, value
    for value in (None, "", "-1", "1e3", "2.0", "²", " 2", "1000000000", True, 2.0, -1, 10 ** 9, [2]):
        assert P.visit_number(value) is None, value


def test_the_last_page_decides_not_the_others():
    clock = Clock()
    pages = P.Presence(clock=clock)
    for page in ("a", "b"):
        pages.connected(page)
    pages.goodbye("a", "socket")
    pages.disconnected("a")
    clock.now += 1
    pages.disconnected("b")                                         # b went last, without a word
    clock.now += 7
    assert pages.verdict(6) == "away"
    pages.connected("c")
    pages.disconnected("c")
    pages.goodbye("c", "page-log")                                  # c went last, and said goodbye
    clock.now += 7
    assert pages.verdict(6) == "closed"


def test_goodbyes_that_name_no_page_change_nothing_and_at_most_64_pages_are_kept():
    clock = Clock()
    pages = P.Presence(clock=clock)
    pages.connected("a")
    pages.disconnected("a")
    for page, via in ((None, "socket"), ("", "page-log"), ("a b", "closing"), ("x" * 101, "socket"), ("a/b", "socket"),
                      (3, "socket"), ("a", "elsewhere")):
        pages.goodbye(page, via)
    clock.now += 7
    assert pages.verdict(6) == "away" and set(pages.pages) == {"a"}
    assert P.page_id("0b3f2c4e-6a1d-4c1e-9a7b-2f8e1d0c9b6a") and P.page_id("k3x9a_-") and P.page_id("é") is None
    pages.connected("open-one")
    for n in range(100):
        pages.connected(f"p{n}")
        pages.disconnected(f"p{n}")
    assert len(pages.pages) == P.PAGES_KEPT and "open-one" in pages.pages and "p99" in pages.pages


# ---- the watcher -----------------------------------------------------------------------------------------------------
class Watched:
    """watch() on a fake clock (tick 1 ms), its calls recorded: the closes, the aways (paused run ids), the idle stops."""

    def __init__(self, *, grace=6, pending_grace=20, away_stop_s=1800, waiting=False, paused=("r1",)):
        self.clock, self.logs, self.calls, self.waiting = Clock(), [], [], waiting
        self.pages = P.Presence(clock=self.clock)
        self.paused, self.pending_threads = list(paused), []
        self.task = asyncio.create_task(P.watch(
            self.pages, grace=grace, pending_grace=pending_grace, away_stop_s=away_stop_s, pending=self.pending,
            on_closed=lambda: self.calls.append("closed"), on_away=self.away, on_idle=lambda: self.calls.append("idle"),
            tick=0.001, clock=self.clock, log=self.logs.append))

    def pending(self):
        self.pending_threads.append(threading.current_thread() is threading.main_thread())
        if isinstance(self.waiting, BaseException):
            raise self.waiting
        return self.waiting

    def away(self):
        self.calls.append("away")
        return list(self.paused)

    async def settle(self):
        for _ in range(30):
            await asyncio.sleep(0.002)

    async def until(self, predicate, timeout=5.0):
        """Until `predicate()` holds (the watcher ticks every 1 ms; its calls run in threads)."""
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            assert asyncio.get_running_loop().time() < deadline, "condition not reached in time"
            await asyncio.sleep(0.002)


def test_the_watcher_closes_once_after_a_goodbye_and_waits_longer_while_a_run_waits():
    async def scenario():
        w = Watched(waiting=True)
        w.pages.connected("a")
        w.pages.goodbye("a", "socket")
        w.pages.disconnected("a")
        w.clock.now += 7                                            # past close_grace_s: a run waits, so 20 s
        await w.until(lambda: len(w.pending_threads) >= 2)
        assert w.calls == [] and not any(w.pending_threads)          # asked in a thread
        w.clock.now += 14
        await asyncio.wait_for(w.task, 5)
        assert w.calls == ["closed"] and w.logs == [P.CLOSED_LINE] and w.pages.public()["state"] == "closed"
        assert P.CLOSED_LINE.startswith("SoPA companion: the last window was closed")   # what log readers match
    asyncio.run(scenario())


def test_away_pauses_once_per_absence_and_a_page_coming_back_is_said():
    async def scenario():
        w = Watched(paused=("r1", "r2"))
        w.pages.connected("a")
        await w.settle()
        w.pages.disconnected("a")                                    # crashed
        w.clock.now += 7
        await w.until(lambda: w.logs)
        w.clock.now += 100
        await w.settle()
        assert w.calls == ["away"] and w.logs == [P.AWAY_LINE.format(n=2, command=command_words())]
        assert "2 run(s) paused" in w.logs[0]
        assert w.pages.public()["state"] == "away"
        w.pages.connected("b")                                       # the app opened again
        await w.until(lambda: len(w.logs) == 2)
        assert w.logs[1] == "SoPA companion: a page is back after 107 s away" and w.pages.public()["state"] == "here"
        w.pages.disconnected("b")                                    # gone again without a word: paused again
        w.clock.now += 7
        await w.until(lambda: len(w.calls) == 2)
        assert w.calls == ["away", "away"] and not w.task.done()
        w.task.cancel()
    asyncio.run(scenario())


def test_an_engine_left_away_stops_only_with_nothing_waiting_and_never_with_away_stop_0():
    async def scenario(waiting, away_stop_s, hours):
        w = Watched(waiting=waiting, away_stop_s=away_stop_s, paused=())
        w.pages.connected("a")
        w.pages.disconnected("a")
        w.clock.now += 25
        await w.until(lambda: w.calls == ["away"] and len(w.pending_threads) >= 2)    # away, and its idle time counts
        for _ in range(int(hours * 12)):                             # five minutes at a time
            w.clock.now += 300
            looked = len(w.pending_threads)
            await w.until(lambda: w.task.done() or len(w.pending_threads) > looked)
            if w.task.done():
                break
        calls, logs = list(w.calls), list(w.logs)
        if not w.task.done():
            w.task.cancel()
        return calls, logs
    calls, logs = asyncio.run(scenario(False, 1800, 1))
    assert calls == ["away", "idle"] and logs[-1] == ("SoPA companion: no window for 30 minutes and nothing waiting for "
                                                     "an answer: shutting down")
    assert asyncio.run(scenario(True, 1800, 2))[0] == ["away"]       # a run waits: never by itself
    assert asyncio.run(scenario(False, 0, 2))[0] == ["away"]         # 0: never
    assert asyncio.run(scenario(RuntimeError("cannot say"), 1800, 2))[0] == ["away"]   # unknown: as if one waits


def test_what_fails_in_the_watchers_calls_is_said_and_the_watcher_goes_on():
    async def scenario():
        w = Watched()
        def broken():
            raise OSError("the disk is full")
        w.away = broken
        w.task.cancel()
        w.task = asyncio.create_task(P.watch(w.pages, grace=6, pending_grace=20, away_stop_s=0, pending=lambda: False,
                                             on_closed=lambda: w.calls.append("closed"), on_away=broken, tick=0.001,
                                             clock=w.clock, log=w.logs.append))
        w.pages.connected("a")
        w.pages.disconnected("a")
        w.clock.now += 7
        await w.until(lambda: len(w.logs) == 2)
        assert w.logs == ["SoPA companion: pausing the runs failed: OSError: the disk is full",
                          P.AWAY_LINE.format(n=0, command=command_words())]
        w.pages.goodbye("a", "page-log")                             # its goodbye, late: a close all the same
        await asyncio.wait_for(w.task, 5)
        assert w.calls == ["closed"]
    asyncio.run(scenario())


# ---- the engine ------------------------------------------------------------------------------------------------------
def auth():
    return {"Authorization": "Bearer " + TOKEN, "Origin": BASE}


def page_headers(application):
    """A page's request: this engine's page credential and the page's own Origin (as tests/test_server.py page())."""
    from test_server import page
    return page(application)


def page_socket(client, application, page="page-1"):
    """A page's socket, signed in as a page, naming itself `page` in its address (None: naming nothing)."""
    from test_server import page_ws
    return client.websocket_connect("/api/v1/ws" + (f"?page={page}" if page is not None else ""),
                                    headers=page_ws(application))


TOOL_WS = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + TOKEN}


def closing_config(config, **desktop):
    config["desktop"] = {"exit_when_closed": True, "close_grace_s": 0.2, "close_grace_pending_s": 0.4,
                         "away_stop_s": 0, **desktop}
    return config


def start_run(client, application):
    runs = application.state.runs
    sample = runs.samples.list()[0]["id"]
    rid = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()["id"]
    client.post(f"/api/v1/runs/{rid}/control", json={"action": "resume"}, headers=auth())
    wait_for(lambda: len(runs.get(rid).frames) >= 2)
    return rid


def test_a_page_that_goes_without_a_goodbye_is_away_its_run_pauses_and_the_next_page_finds_it(config, capsys):
    """A crashed page (a sad tab, a killed browser, a lost tunnel): the engine neither stops nor deletes its run, which
    pauses as its Pause button would, marked so that the next page can say why; that mark goes once it is resumed."""
    from sopa_companion.server import create_app
    shut = []
    application = create_app(closing_config(config), token=TOKEN, shutdown=lambda: shut.append(1))
    with TestClient(application, base_url=BASE) as client:
        runs = application.state.runs
        rid = start_run(client, application)
        with page_socket(client, application) as ws:
            assert ws.receive_json()["ev"] == "runs"
            assert application.state.presence.open() == 1
        wait_for(lambda: runs.get(rid).meta.get("paused_away"), timeout=10)
        run = runs.get(rid).public()
        assert run["paused"] is True and run["paused_away"].endswith("+00:00")
        health = client.get("/api/v1/health", headers=auth()).json()
        assert health["presence"]["state"] == "away" and health["presence"]["pages"] == 0 and shut == []
        assert "browser" not in health                                # the engine never reads where the app opens
        with page_socket(client, application, "page-2") as ws:     # the app opened again
            pushed = ws.receive_json()
            assert pushed["ev"] == "runs" and pushed["runs"][0]["paused_away"] == run["paused_away"]
            wait_for(lambda: "a page is back after" in capsys.readouterr().out, timeout=10)
            client.post(f"/api/v1/runs/{rid}/control", json={"action": "resume"}, headers=auth())
            assert "paused_away" not in runs.get(rid).meta and runs.get(rid).meta["paused"] is False
            ws.send_json({"op": "bye"})
        wait_for(lambda: shut == [1], timeout=10)
    log = capsys.readouterr().out
    assert "the last window was closed; shutting down (its page said goodbye)" in log
    assert "discarding (dialog answer: not asked, the page closed without an answer)" in log


def test_a_page_that_said_goodbye_closes_the_engine_and_its_unanswered_run_is_discarded_without_the_dialog(
        config, monkeypatch, capsys):
    from sopa_companion.server import app as server
    monkeypatch.setattr(server, "ask_to_save", lambda *a, **k: pytest.fail("a page that said goodbye has no dialog"))
    shut = []
    application = server.create_app(closing_config(config), token=TOKEN, shutdown=lambda: shut.append(1))
    with TestClient(application, base_url=BASE) as client:
        runs = application.state.runs
        rid = start_run(client, application)
        with page_socket(client, application) as ws:
            ws.receive_json()
            ws.send_json({"op": "bye"})
        wait_for(lambda: shut == [1], timeout=10)
        assert application.state.page_closed is True and "paused_away" not in runs.get(rid).meta
        run = runs.get(rid)
    assert run.meta["state"] == "stopped" and not run.dir.exists()
    line = next(line for line in capsys.readouterr().out.splitlines() if "unsaved run(s)" in line)
    assert line == ("SoPA companion: closing with 1 unsaved run(s): discarding (dialog answer: not asked, the page closed "
                    "without an answer)")


def test_a_goodbye_that_arrives_after_its_page_came_back_never_turns_a_later_crash_into_a_close(config, capsys):
    """The page said goodbye and its socket closed; the browser showed it again (it connects as its next showing); only
    then does its first showing's last log upload and answer to closing reach the engine. The page shown now then
    vanishes without a word: it is away, its run paused and the engine waiting, never a close that discards the run."""
    from sopa_companion.server import create_app
    shut = []
    # A run waits for an answer, so a page has 3 s to come back: time enough for this test's reconnect on a busy disk.
    application = create_app(closing_config(config, close_grace_s=1, close_grace_pending_s=3), token=TOKEN,
                             shutdown=lambda: shut.append(1))
    pages = application.state.presence
    with TestClient(application, base_url=BASE) as client:
        runs = application.state.runs
        rid = start_run(client, application)
        with page_socket(client, application, "page-a&visit=1") as ws:
            ws.receive_json()
            ws.send_json({"op": "bye"})
            wait_for(lambda: pages.pages["page-a"]["bye"] == "socket")
        with page_socket(client, application, "page-a&visit=2") as ws:      # shown again (pageshow persisted)
            ws.receive_json()
            assert pages.pages["page-a"]["bye"] is None
            late = {"text": "  12:00:00.000  page      the page was put aside\n", "session": "page-a",
                    "at": "2026-10-25T12:00:00.000Z", "pagehide": True, "visit": 1}
            assert client.post("/api/v1/page-log", json=late, headers=page_headers(application)).status_code == 200
            answer = {"answer": "discard", "runs": [rid], "revision": 3, "page": "page-a", "bye": True, "visit": 1}
            taken = client.post("/api/v1/closing", json=answer, headers=page_headers(application)).json()
            assert pages.pages["page-a"]["bye"] is None, "said before the page came back: no goodbye of the page shown now"
            # Nor is its answer kept: a page that connects again takes its answer back, and this one came before that.
            assert taken["taken"] is False and taken["answer"] is None
            # The page shown now answers; a socket of its first showing connecting late takes that answer back no more
            # than its goodbyes count.
            mine = {"answer": "discard", "runs": [rid], "revision": 4, "page": "page-a", "visit": 2}
            assert client.post("/api/v1/closing", json=mine, headers=page_headers(application)).json()["taken"] is True
            with page_socket(client, application, "page-a&visit=1") as late_socket:
                late_socket.receive_json()
                answer = client.get("/api/v1/closing", headers=page_headers(application)).json()["answer"]
                assert answer == "discard" and pages.pages["page-a"]["visit"] == 2
            # An answer that names no showing is the page's newest: taken, as every answer was.
            unnamed = {"answer": "discard", "runs": [rid], "revision": 5, "page": "page-a"}
            assert client.post("/api/v1/closing", json=unnamed, headers=page_headers(application)).json()["taken"] is True
        said = []
        wait_for(lambda: said.append(capsys.readouterr().out) or "the last window went without closing" in "".join(said),
                 timeout=15)
        assert runs.get(rid).meta.get("paused_away") and runs.get(rid).dir.exists() and shut == []
        assert client.get("/api/v1/health", headers=auth()).json()["presence"]["state"] == "away"
        assert "the last window was closed" not in "".join(said)


def test_each_goodbye_is_enough_and_tools_never_count_as_pages(config):
    """The socket's bye, the /page-log upload with pagehide and the /closing answer with bye each name the page that
    goes (the id its socket's address gives); a bearer's socket (a tool, the harness's watch) is no page, and a page's
    socket that names no usable id counts under a name of its own, which only that socket's own bye names."""
    from sopa_companion.server import create_app
    application = create_app(config, token=TOKEN)                   # exit_when_closed false: nothing closes here
    pages = application.state.presence
    with TestClient(application, base_url=BASE) as client:
        with client.websocket_connect("/api/v1/ws", headers=TOOL_WS) as ws:
            ws.receive_json()
            ws.send_json({"op": "bye"})
            assert pages.open() == 0 and pages.public()["connects"] == 0
        log = {"text": "  12:00:00.000  page      bye\n", "session": "p-log", "at": "2026-10-23T12:00:00.000Z",
               "pagehide": True}
        client.post("/api/v1/page-log", json=log, headers=page_headers(application))
        client.post("/api/v1/closing", json={"answer": "discard", "page": "p-closing", "bye": True},
                    headers=page_headers(application))
        client.post("/api/v1/closing", json={"answer": "discard", "page": "p-stays"}, headers=page_headers(application))
        assert (pages.pages["p-log"]["bye"], pages.pages["p-closing"]["bye"]) == ("page-log", "closing")
        assert "p-stays" not in pages.pages
        assert client.post("/api/v1/closing", json={"answer": "discard", "bye": "yes"},
                           headers=page_headers(application)).status_code == 422
        with page_socket(client, application, None) as ws:
            ws.receive_json()
            assert list(pages.pages)[-1] == "socket-1" and pages.open() == 1
        with page_socket(client, application, "not a page id!") as ws:
            ws.receive_json()
            assert list(pages.pages)[-1] == "socket-2"
        with page_socket(client, application, "p-socket") as ws:
            ws.receive_json()
            ws.send_json({"op": "bye"})
            wait_for(lambda: pages.pages["p-socket"]["bye"] == "socket")
        wait_for(lambda: pages.open() == 0)
        presence = client.get("/api/v1/health", headers=auth()).json()["presence"]
        # No watcher here (exit_when_closed false): its state says what the pages did, the last one a moment ago.
        assert presence["connects"] == 3 and presence["pages"] == 0 and presence["state"] == "leaving"


def test_the_engine_counts_the_pages_of_browsers_it_opened_apart_from_other_tabs(config):
    """/health's presence.windows: the window codes traded (the page of a browser the launcher or a notebook opened),
    which the launcher and a notebook wait for; a link's sign-in, or another tab's socket connecting, is not one."""
    from sopa_companion.server import create_app
    from sopa_companion.server.security import CODE_HEADER, SIGN_IN
    application = create_app(config, token=TOKEN)
    with TestClient(application, base_url=BASE) as client:
        def trade(link: bool) -> None:
            code = client.post("/api/v1/session/launch-code" + ("?link=1" if link else ""), headers=auth()).json()["code"]
            assert client.post(SIGN_IN, headers={"Origin": BASE, CODE_HEADER: code}).status_code == 200
        presence = lambda: client.get("/api/v1/health", headers=auth()).json()["presence"]
        trade(link=True)
        with page_socket(client, application, "another-tab") as ws:
            ws.receive_json()
            assert presence()["windows"] == 0 and presence()["connects"] == 1
        trade(link=False)
        wait_for(lambda: presence()["windows"] == 1)                 # told once the page has its key
        trade(link=True)
        assert presence()["windows"] == 1
    pages = P.Presence()
    for kind in ("window", "link", "something else", "window"):
        pages.signed_in(kind)
    assert pages.public()["windows"] == 2


def test_an_engine_whose_setting_names_no_browser_starts_and_answers(config):
    """[desktop] browser is the launcher's and the kernel's to read; a value they would refuse never stops an engine."""
    from sopa_companion.server import create_app
    config["desktop"]["browser"] = "firefox"
    application = create_app(config, token=TOKEN)
    with TestClient(application, base_url=BASE) as client:
        assert client.get("/api/v1/health", headers=auth()).status_code == 200
        assert client.get("/api/v1/runs", headers=auth()).status_code == 200


def test_away_pauses_what_still_samples_and_leaves_the_rest_as_it_is(config):
    """Runs.pause_for_away: a proseg run running or preparing and an active stain run pause (marked paused_away); a run
    finishing, being left or saved, a test's own run, a paused run and a run that has ended are left alone."""
    from sopa_companion.engine.runs import RunError
    from sopa_companion.server import create_app
    runs = create_app(config, token=TOKEN).state.runs
    metas = {
        "running": {"state": "running"}, "preparing": {"state": "preparing"},
        "stain": {"kind": "stain", "state": "running"}, "stain-preparing": {"kind": "stain", "state": "preparing"},
        "finishing": {"state": "running", "finish_requested": True}, "leaving": {"state": "running", "leaving": {"answer": "add"}},
        "saving": {"state": "running", "zarr_save": {"state": "saving"}}, "test": {"state": "running", "qc_test": {"of": "p"}},
        "paused": {"state": "paused", "paused": True}, "done": {"state": "done"},
        "stain-stopping": {"kind": "stain", "state": "finishing", "stain_stop": True},
        "patches": {"kind": "patches", "state": "running"}, "gone": {"state": "running"},
        "parent": {"state": "running", "qc_test": {"state": "testing", "progress": 0.5}},
    }
    saved, controlled = [], []

    class FakeRun:
        def __init__(self, id):
            self.lock, self.meta = threading.RLock(), {"id": id, "created": 0, **metas[id]}

        def save(self):
            saved.append(self.meta["id"])

        def touch(self):
            pass

    fake = {id: FakeRun(id) for id in metas}
    runs.list = lambda: [dict(run.meta) for run in fake.values()]

    def control(run_id, action):
        controlled.append((run_id, action))
        if run_id == "gone":
            raise RunError("Unknown run", 404)
        fake[run_id].meta["paused"] = True
        return dict(fake[run_id].meta)
    runs.control = control
    runs.get = lambda run_id: fake[run_id]
    version = runs.version
    assert runs.pause_for_away() == ["running", "preparing", "stain", "stain-preparing", "parent"]
    assert [id for id, _ in controlled] == ["running", "preparing", "stain", "stain-preparing", "gone", "parent"]
    assert {action for _, action in controlled} == {"pause"} and saved == ["running", "preparing", "stain",
                                                                           "stain-preparing", "parent"]
    assert all(fake[id].meta["paused_away"].endswith("+00:00") for id in saved)
    assert not any("paused_away" in fake[id].meta for id in metas if id not in saved)
    assert runs.version == version + 1
    controlled.clear()
    assert runs.pause_for_away() == [] and controlled == [("gone", "pause")]   # paused already: nothing twice


def test_a_stain_run_carried_on_or_ended_loses_its_away_mark():
    """StainRuns.control: Resume, Stop, Finish and Abort take paused_away off (the page says it no more); Pause, and a
    press it refuses, leave it."""
    from sopa_companion.engine.runs import RunError
    from sopa_companion.engine.stain import StainRuns

    def press(action, **meta):
        run = types.SimpleNamespace(lock=threading.RLock(), ack=None, worker=None, process=None, stop_requested=False,
                                    meta={"state": "running", "paused": True, "paused_away": "2026-10-25T01:00:00+00:00",
                                          "stain": {"method": "cellpose"}, "progress": {"done": 1}, **meta},
                                    save=lambda: None, touch=lambda: None, public=lambda: {})
        stain = types.SimpleNamespace(offered=lambda method: None, write_control=lambda run: None, launch=lambda run: None,
                                      runs=types.SimpleNamespace(closing=False, changed=lambda: None))
        try:
            StainRuns.control(stain, run, action)
        except RunError:
            pass
        return "paused_away" in run.meta
    assert not press("resume") and not press("stop") and not press("finish") and not press("abort")
    assert press("pause") and press("dance")
    assert press("resume", stain_stop=True)                           # refused (its helper is stopping): kept
