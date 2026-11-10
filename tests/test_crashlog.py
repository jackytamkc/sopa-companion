"""Crash reports (engine/crashlog.py, server/app.py): a session that ended unexpectedly (an engine that stopped without
shutting down, a page that closed without unloading) leaves a plain-text report, which the page offers once at the next
start (frontend/src/crash-report.mjs). What a report holds, when one is written, offered, kept quietly or not written at
all, the routes, the page logs kept per page session, and that nothing secret is ever in one."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

import psutil
import pytest
from fastapi.testclient import TestClient

from conftest import HERE, wait_for

BASE = "http://127.0.0.1:8799"


def shaped(seed: str) -> str:
    """43 URL-safe characters, as secrets.token_urlsafe(32) makes the session's secret."""
    return base64.urlsafe_b64encode(hashlib.sha256(seed.encode()).digest()).decode().rstrip("=")


TOKEN = shaped("this session")           # the session's secret
OLD = shaped("an earlier session")       # another session's: blanked by its shape alone


def page(client, content_type: str = "application/json") -> dict:
    """A page's headers: the engine's page key in its header (new at each start, never the bearer: server/security.py)."""
    return {"Origin": BASE, "X-SoPA-Page-Key": client.app.state.page_key, "Content-Type": content_type}


def beacon(client) -> dict:
    return page(client, "text/plain;charset=UTF-8")             # an earlier page's beacon body: read as JSON all the same


def auth(extra=None):
    return {"Authorization": "Bearer " + TOKEN, "Origin": BASE, **(extra or {})}


def row(name: str, value: str) -> str:
    """A report's header row, as engine/crashlog.py writes it."""
    from sopa_companion.engine.crashlog import ROW
    return f"{name:<{ROW}}{value}"


def iso(t: datetime) -> str:
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


@pytest.fixture
def one_gpu(monkeypatch):
    from sopa_companion.engine import crashlog
    monkeypatch.setattr(crashlog, "gpus", lambda timeout=crashlog.GPU_TIMEOUT_S: ["GPU 0: A test GPU"])


def folders(config) -> tuple[Path, Path]:
    return Path(config["paths"]["state_dir"]), Path(config["paths"]["data_dir"])


def reports(state: Path) -> list[Path]:
    return sorted((state / "crash-reports").glob("*.txt"))


def dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def write_marker(state: Path, pid: int, started: datetime, data: Path, version: str = "0.0.9") -> Path:
    marker = state / "run" / "engine.alive"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"pid": pid, "started": started.isoformat(), "version": version, "data_dir": str(data)}))
    return marker


def write_run(data: Path, run_id: str, created: datetime, iterations=(), **meta) -> Path:
    folder = data / "runs" / run_id
    (folder / "live").mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"id": run_id, "created": created.isoformat(), **meta}))
    if iterations:
        (folder / "live" / "frames.jsonl").write_text("".join(json.dumps({"seq": i, "iter": it}) + "\n"
                                                              for i, it in enumerate(iterations)))
    return folder


def write_page_log(state: Path, session: str, sent: datetime, body: str, closed: bool = False) -> Path:
    """A page log as an engine keeps it (logs/page-logs/<session>.txt)."""
    path = state / "logs" / "page-logs" / f"{session}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Page session {session}, sent {iso(sent)}, first {iso(sent)}{', closed' if closed else ''}\n{body}")
    return path


def test_the_secrets_of_these_tests_are_shaped_as_the_apps():
    import secrets
    from sopa_companion.engine.crashlog import SHAPED, redact
    for token in (TOKEN, OLD, secrets.token_urlsafe(32)):
        assert SHAPED.fullmatch(token), token
    assert redact(f"x {OLD} y") == "x [redacted] y"


def test_nothing_secret_is_written_and_names_keep_theirs():
    """The session's secret wherever it stands, the cookie's value, tokens and codes in addresses, after Bearer and in
    JSON, and anything shaped like the secret are blanked; a run's id, a sample's name and a GPU's id are not."""
    from sopa_companion.engine.crashlog import redact
    query, bearer, cookie, quoted = OLD[:20], shaped("bearer")[:12], shaped("cookie")[:16], shaped("json")[:10]
    text = redact(f"a{TOKEN}b GET /?token={query}&x=1 /?code=abc Authorization: Bearer {bearer} "
                  f"Cookie: sopa_companion_session={cookie}; theme=dark {json.dumps({'token': quoted, 'code': 'c-1'})} "
                  f"{OLD}. Cookie: proseg_live_session={cookie[::-1]}", [TOKEN])
    for hidden in (TOKEN, OLD, query, "abc", bearer, cookie, cookie[::-1], quoted, '"c-1"'):
        assert hidden not in text, (hidden, text)
    assert "theme=dark" in text and "x=1" in text
    run_id = "20261023-140304-slide_region_1_xenium_crop2"          # 43 characters, but a name
    capitals = "20261023-140304-Slide_Region_One_XeniumCrop"          # a run's id, its sample's name in capitals
    sample = "XETG00000__0000000__region_1__20240101__abc"           # 43 characters, 4 capitals
    names = (f"{run_id} {capitals} {sample} sample-1 GPU-0a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d "
             "/data/output-XETG00000__0000000__TEST-1")
    assert len(run_id) == len(capitals) == len(sample) == 43 and redact(names) == names


def test_every_secret_the_app_makes_is_blanked_by_its_shape():
    """A secret of an earlier session (session.key made again since) is blanked by its shape alone: of 100,000 made as
    the app makes them, at most a handful are not (the rule asked for 8 capitals, 8 small letters and 2 digits before,
    and missed about 680 of them)."""
    from sopa_companion.engine.crashlog import redact
    rng = random.Random(20261023)
    made = [base64.urlsafe_b64encode(rng.randbytes(32)).decode().rstrip("=") for _ in range(100_000)]
    kept = [word for word in redact(" ".join(made)).split(" ") if word != "[redacted]"]
    assert len(kept) <= 5, len(kept)


@pytest.mark.parametrize("prefix, cookie", [("SoPA companion", "sopa_companion_session"),
                                            ("Proseg Live", "proseg_live_session")])
def test_a_marker_left_by_a_dead_engine_gives_one_report_before_its_runs_go(config, one_gpu, prefix, cookie):
    """The engine that starts and finds the marker of a process that is gone writes that session's report first: the
    header (what ended, when it started and was found, the version it ran, python, the platform, the GPUs), the runs it
    left (before Runs removes the unsaved ones), its own lines of server.log (from its first line: not an earlier
    session's), the page's last log; no secret. The marker goes: the next start writes no second report. The same for a
    session of the previous version, whose log lines begin "Proseg Live: " and whose cookie was proseg_live_session (its
    state folder, log included, is this version's after the first start)."""
    from sopa_companion import __version__
    from sopa_companion.server import create_app
    state, data = folders(config)
    pid, started = dead_pid(), (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
    marker = write_marker(state, pid, started, data)
    runtime = "runtime-file-" + shaped("the runtime file")[:20]      # not shaped: only the runtime file says it is one
    (state / "run" / "server.json").write_text(json.dumps({"schema": 1, "pid": pid, "token": runtime}))
    logs = state / "logs"
    logs.mkdir()
    (logs / "server.log.1").write_text(f"{prefix}: engine starting: pid 111, SoPA companion 0.0.8\n"
                                       f"INFO:     Started server process [111]\n{prefix}: an earlier session's line\n"
                                       "INFO:     Finished server process [111]\n")
    (logs / "server.log").write_text("\n".join([
        f"{prefix}: engine starting: pid {pid}, SoPA companion 0.0.9, {started.isoformat()}",
        f"{prefix}: http://127.0.0.1:8799/",
        f"INFO:     Started server process [{pid}]",
        f"{prefix}: GET /api/v1/runs/x/log?token={TOKEN} failed (500): KeyError: 'x'",
        f"Cookie: {cookie}={OLD}",
        f"a line naming the runtime file's: {runtime}",
        f"an earlier session's, alone: {OLD}",
        "Traceback (most recent call last):",
        '  File "engine/runs.py", line 1, in worker',
        "MemoryError: the last line before the end",
    ]) + "\n")
    sampling = write_run(data, "20261023-120000-sample-1", started + timedelta(minutes=1), (0, 5, 42), kind="proseg",
                         state="running", sample_name="sample-1")
    stain = write_run(data, "20261023-110000-sample-2", started - timedelta(minutes=5), kind="stain", state="running",
                      sample_name="sample-2", progress={"done": 3, "total": 12, "revision": 4})
    write_run(data, "20261023-100000-sample-1", started - timedelta(minutes=50), (0, 450), kind="proseg", state="done",
              sample_name="sample-1")
    # The page that was open (no goodbye), and one that closed cleanly before it: the open one's log goes in.
    write_page_log(state, "s-1", started + timedelta(minutes=59), "Page noted      nothing went wrong\n"
                                                                  "  13:59:58.000  action    Resume\n")
    write_page_log(state, "s-0", started + timedelta(minutes=2), "  12:02:00.000  action    a page closed\n", closed=True)

    create_app(config, token=TOKEN)
    [report] = reports(state)
    assert re.fullmatch(r"\d{8}-\d{6}-engine\.txt", report.name)
    assert stat.S_IMODE(report.stat().st_mode) == 0o600 and stat.S_IMODE(report.parent.stat().st_mode) == 0o700
    text = report.read_text()
    head = text.split("\n\n")[0]
    assert head.startswith("SoPA companion closed unexpectedly\n==================================\n")
    local = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started.timestamp()))
    for line in (row("What ended", "the engine: it stopped without shutting down"),
                 row("Session started", f"{local} ({started.isoformat()})\n"),
                 row("That engine", f"pid {pid}, SoPA companion 0.0.9, data folder {data}\n"),
                 row("App version", f"SoPA companion {__version__} (protocol "),
                 row("Python", f"{platform.python_version()} ({sys.executable})\n"),
                 row("Platform", f"{platform.platform()}\n"), row("GPUs", "GPU 0: A test GPU\n"),
                 row("Written", ""), "as the engine started again\n", row("Log last written", "")):
        assert line in head + "\n", line
    assert "Offered" not in head and "Earlier reports" not in head
    # The runs it left, newest first, then gone or stopped (after the report: Runs only removes them as it starts).
    rows = text[text.index("Runs then"):text.index("server.log:")]
    assert re.search(r"20261023-120000-sample-1 +proseg +sample-1 +running +iteration 42\n", rows), rows
    assert re.search(r"20261023-110000-sample-2 +stain +sample-2 +running +3 of 12 tiles\n", rows), rows
    assert re.search(r"20261023-100000-sample-1 +proseg +sample-1 +done +iteration 450\n", rows), rows
    assert rows.index("120000") < rows.index("110000") < rows.index("100000")
    assert not sampling.exists() and json.loads((stain / "run.json").read_text())["state"] == "stopped"
    # Its own lines of the engine log, from its first; not the earlier session's.
    part = text[text.index("server.log:"):text.index("The page's last log")]
    assert part.split("\n")[2] == f"{prefix}: engine starting: pid {pid}, SoPA companion 0.0.9, {started.isoformat()}"
    assert "MemoryError: the last line before the end" in part and "(10 lines)" in part.split("\n")[0]
    assert "an earlier session's line" not in text and "[111]" not in text
    assert "The page's last log (page session s-1, sent " in text and "13:59:58.000  action    Resume" in text
    assert "a page closed" not in text                         # the page that said its goodbye: not the one open
    # No secret in it: this session's, the runtime file's, the cookie's, one shaped like the secret.
    for hidden in (TOKEN, OLD, runtime):
        assert hidden not in text
    assert text.count("[redacted]") >= 4
    assert not marker.exists()                                 # gone: this engine puts its own once it is up

    application = create_app(config, token=TOKEN)              # the next start: no second report
    assert len(reports(state)) == 1
    with TestClient(application, base_url=BASE) as client:
        summary = client.get("/api/v1/crash", headers=auth()).json()["report"]
        assert summary["id"] == report.stem and summary["reason"] == "engine" and summary["bytes"] == report.stat().st_size
        assert summary["started"] == started.isoformat()
        assert abs(datetime.fromisoformat(summary["at"]).timestamp() - time.time()) < 60
        served = client.get(f"/api/v1/crash/{report.stem}.txt", headers=auth())
        assert served.status_code == 200 and served.text == text
        assert served.headers["content-type"] == "text/plain; charset=utf-8" and served.headers["cache-control"] == "no-store"
        assert client.post(f"/api/v1/crash/{report.stem}/seen", headers=page(client)).json() == {}
        assert client.get("/api/v1/crash", headers=auth()).json() == {"report": None}       # offered once
        assert client.get(f"/api/v1/crash/{report.stem}.txt", headers=auth()).status_code == 200   # still there


def test_a_clean_shutdown_leaves_no_marker_and_no_report(config, one_gpu):
    """The marker is written once the server is up ({pid, started, version, data_dir}) and removed by its clean
    shutdown; this very process's marker (an earlier app of it) is no crash."""
    from sopa_companion import __version__
    from sopa_companion.engine.crashlog import CrashReports
    from sopa_companion.server import create_app
    state, data = folders(config)
    marker = state / "run" / "engine.alive"
    application = create_app(config, token=TOKEN)
    assert not marker.exists()
    with TestClient(application, base_url=BASE) as client:
        value = json.loads(marker.read_text())
        assert set(value) == {"pid", "started", "version", "data_dir"}
        assert value["pid"] == os.getpid() and value["version"] == __version__ and value["data_dir"] == str(data)
        assert abs(datetime.fromisoformat(value["started"]).timestamp() - time.time()) < 60
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600
        assert client.get("/api/v1/crash", headers=auth()).json() == {"report": None}
        assert CrashReports(config).check() is None                  # this process's own marker: no crash
        assert marker.exists() and not reports(state)
    assert not marker.exists()
    create_app(config, token=TOKEN)
    assert not reports(state) and not marker.exists()


def test_the_marker_of_an_engine_still_up_is_left_to_it(config, one_gpu):
    """Another engine with the same state folder, still up: no report, its marker untouched, and this engine keeps none
    (it never removes the other's). Its pid come round again (a process started after the marker): a crash."""
    from sopa_companion.engine.crashlog import CrashReports
    state, data = folders(config)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        marker = write_marker(state, other.pid, datetime.now(timezone.utc) + timedelta(seconds=1), data)
        before = marker.read_bytes()
        crashes = CrashReports(config)
        assert crashes.check() is None and not reports(state)
        crashes.mark_alive()
        crashes.mark_closed()
        assert marker.read_bytes() == before
        made = datetime.fromtimestamp(psutil.Process(other.pid).create_time(), timezone.utc)
        write_marker(state, other.pid, made - timedelta(hours=1), data)    # written an hour before that process began
        assert CrashReports(config).check() is not None and len(reports(state)) == 1 and not marker.exists()
    finally:
        other.kill()
        other.wait()


def test_an_engine_killed_is_reported_at_the_next_start_and_one_stopped_is_not(config, tmp_path, one_gpu):
    """Real engines (python -m sopa_companion serve, its output in server.log as the desktop launcher gives it): one
    stopped by SIGTERM shuts down and removes its marker; one killed (SIGKILL) leaves it, and the next start writes one
    report with that engine's own log lines (from its first line: not the first engine's)."""
    from sopa_companion.server import create_app
    state, data = folders(config)
    marker, log = state / "run" / "engine.alive", state / "logs" / "server.log"
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    toml = tmp_path / "engine.toml"
    toml.write_text(f"[server]\nport = {port}\ntoken = {json.dumps(TOKEN)}\n"
                    f"[paths]\ndata_dir = {json.dumps(str(data))}\nstate_dir = {json.dumps(str(state))}\nsample_roots = []\n"
                    f"[proseg]\nbinary = {json.dumps(config['proseg']['binary'])}\nthreads = 2\n"
                    f"[analysis]\npython = \"\"\n[zarr]\npython = \"\"\nbrowse_start = {json.dumps(str(tmp_path))}\n"
                    "[qc]\npython = \"\"\n[stain]\npython = \"\"\ninstanseg_python = \"\"\npatch_python = \"\"\n"
                    "[desktop]\nexit_when_closed = false\nprofile_dir = \"\"\n")
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env.update(PYTHONPATH=str(HERE.parents[1]), SOPA_COMPANION_DESKTOP_LOG=str(log), PYTHONUNBUFFERED="1")

    def serve() -> subprocess.Popen:
        child = subprocess.Popen([sys.executable, "-m", "sopa_companion", "--config", str(toml), "serve", "--port", str(port)],
                                 cwd=str(HERE.parents[1]), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        try:
            wait_for(lambda: marker.exists() and json.loads(marker.read_text())["pid"] == child.pid, timeout=90)
        except AssertionError:
            child.kill()
            child.wait()
            raise AssertionError(log.read_text()[-3000:] if log.exists() else "no server.log") from None
        return child

    stopped = serve()
    try:
        stopped.send_signal(signal.SIGTERM)
        stopped.wait(timeout=90)
    finally:
        if stopped.poll() is None:
            stopped.kill()
            stopped.wait()
    assert not marker.exists() and not reports(state), log.read_text()[-3000:]
    killed = serve()
    killed.kill()
    killed.wait()
    assert json.loads(marker.read_text())["pid"] == killed.pid

    create_app(config, token=TOKEN)
    [report] = reports(state)
    text = report.read_text()
    assert row("That engine", f"pid {killed.pid}, SoPA companion ") in text
    part = text[text.index("server.log:"):text.index("The page's last log")]
    assert part.split("\n")[2].startswith(f"SoPA companion: engine starting: pid {killed.pid}, ")
    assert f"Started server process [{killed.pid}]" in part
    assert f"[{stopped.pid}]" not in part and not re.search(rf"\bpid {stopped.pid}\b", part)
    assert TOKEN not in text and not marker.exists()


def test_a_page_that_did_not_unload_gets_its_report_once(config, one_gpu):
    """POST /crash/page {at, session, found}: that page session's report (reason "page"), with its own last log even
    when the new page sent its own first, answered as GET /crash; once per page session; refused bodies."""
    from sopa_companion.server import create_app
    state, _ = folders(config)
    started = "2026-10-23T12:00:00.000Z"
    application = create_app(config, token=TOKEN)
    with TestClient(application, base_url=BASE) as client:
        sent = {"text": "Page noted      1 thing went wrong\n  12:00:05.000  action    Start\n", "session": "s-old",
                "at": "2026-10-23T12:00:10.000Z"}
        assert client.post("/api/v1/page-log", content=json.dumps(sent), headers=beacon(client)).json() == {}
        assert client.post("/api/v1/page-log", json={"text": "the new page", "session": "s-new", "at": started},
                           headers=page(client)).json() == {}
        answer = client.post("/api/v1/crash/page", json={"at": started, "session": "s-old", "found": "tab"}, headers=page(client))
        assert answer.status_code == 200
        summary = answer.json()["report"]
        assert re.fullmatch(r"\d{8}-\d{6}-page", summary["id"]) and summary["reason"] == "page"
        assert summary["started"] == "2026-10-23T12:00:00+00:00" and summary["bytes"] > 0
        text = client.get(f"/api/v1/crash/{summary['id']}.txt", headers=auth()).text
        assert row("What ended", "the page: it closed without unloading") in text
        assert row("Page session", "s-old\n") in text and "as the next page in the same window started" in text
        assert "The page's last log (page session s-old, sent " in text and "12:00:05.000  action    Start" in text
        assert "the new page" not in text and TOKEN not in text and "Offered" not in text
        again = client.post("/api/v1/crash/page", content=json.dumps({"at": started, "session": "s-old"}), headers=beacon(client))
        assert again.json() == {"report": summary} and len(reports(state)) == 1          # written once
        assert client.post(f"/api/v1/crash/{summary['id']}/seen", headers=page(client)).json() == {}
        assert client.get("/api/v1/crash", headers=auth()).json() == {"report": None}
        assert json.loads((state / "crash-reports" / "seen.json").read_text()) == {"seen": [summary["id"]]}
        assert client.post("/api/v1/crash/page", json={"at": started, "session": "s-old"},
                           headers=page(client)).json() == {"report": None}                  # seen, and still written once
        for body, status in ((b"", 422), (b"not json", 422), (b"[]", 422), (b'{"at": 1, "session": "s"}', 422),
                             (b'{"at": "x"}', 422), (b'{"at": "x", "session": " "}', 422),
                             (b'{"at": "x", "session": "' + b"s" * 5000 + b'"}', 413)):
            assert client.post("/api/v1/crash/page", content=body, headers=beacon(client)).status_code == status, body[:30]
        assert client.post("/api/v1/crash/page", content=b"{" + b" " * 5000 + b"}",
                           headers=beacon(client)).json()["detail"] == "This is limited to 4 KB"
    assert len(reports(state)) == 1


def test_a_page_found_by_its_flag_alone_is_kept_quietly(config, one_gpu):
    """A page found only by its flag went with its browser (a logout, the desktop restarting, the browser's own end):
    its report is written but kept quietly, never offered, and it hides no other report; at most 3 such are kept, and
    they go before the others."""
    from sopa_companion.engine.crashlog import KEEP, QUIET_KEEP
    from sopa_companion.server import create_app
    state, _ = folders(config)
    at = "2026-10-23T12:00:00.000Z"
    application = create_app(config, token=TOKEN)
    with TestClient(application, base_url=BASE) as client:
        client.post("/api/v1/page-log", json={"text": "  12:00:01.000  action    Start\n", "session": "s-1", "at": at},
                    headers=page(client))
        assert client.post("/api/v1/crash/page", json={"at": at, "session": "s-1", "found": "flag"},
                           headers=page(client)).json() == {"report": None}
        [quiet] = reports(state)
        text = quiet.read_text()
        assert row("Offered", "no: the browser closed with the page open (a logout, the desktop restarting") in text
        assert "as a page started again and found its flag" in text and "12:00:01.000  action    Start" in text
        assert client.get("/api/v1/crash", headers=auth()).json() == {"report": None}
        assert client.get(f"/api/v1/crash/{quiet.stem}.txt", headers=auth()).status_code == 200      # kept here
        # A report to offer, then a quiet one after it: the first is still the one offered.
        offered = client.post("/api/v1/crash/page", json={"at": at, "session": "s-2", "found": "tab"},
                              headers=page(client)).json()["report"]
        assert offered["reason"] == "page"
        for i in range(3, 3 + QUIET_KEEP + 1):
            assert client.post("/api/v1/crash/page", json={"at": at, "session": f"s-{i}", "found": "flag"},
                               headers=page(client)).json() == {"report": offered}
        names = [p.stem for p in reports(state)]
        assert quiet.stem not in names and offered["id"] in names and len(names) == QUIET_KEEP + 1 <= KEEP
        assert client.get("/api/v1/crash", headers=auth()).json() == {"report": offered}


def test_a_page_whose_goodbye_came_gets_no_report_and_one_brought_back_does(config, one_gpu):
    """Its last log was its goodbye ("pagehide"): it closed cleanly, and a flag of it is an old copy the browser kept
    (it writes its storage to disk only now and then). A page brought back after its goodbye says it is open again."""
    from sopa_companion.server import create_app
    state, _ = folders(config)
    application = create_app(config, token=TOKEN)
    with TestClient(application, base_url=BASE) as client:
        bye = {"text": "  12:00:01.000  page      the page closed or reloaded\n", "session": "s-1",
               "at": "2026-10-23T12:00:02.000Z", "pagehide": True}
        assert client.post("/api/v1/page-log", content=json.dumps(bye), headers=beacon(client)).json() == {}
        kept = (state / "logs" / "page-logs" / "s-1.txt").read_text()
        assert kept.startswith("Page session s-1, sent 2026-10-23T12:00:02.000Z, first 2026-10-23T12:00:02.000Z, closed\n")
        for found in ("flag", "tab", None):
            body = {"at": "2026-10-23T12:00:00.000Z", "session": "s-1", **({"found": found} if found else {})}
            assert client.post("/api/v1/crash/page", json=body, headers=page(client)).json() == {"report": None}
        assert not reports(state)
        back = {"text": "  12:05:00.000  page      the page was brought back\n", "session": "s-1",
                "at": "2026-10-23T12:05:00.000Z"}
        client.post("/api/v1/page-log", json=back, headers=page(client))
        assert not (state / "logs" / "page-logs" / "s-1.txt").read_text().split("\n")[0].endswith(", closed")
        report = client.post("/api/v1/crash/page", json={"at": "2026-10-23T12:00:00.000Z", "session": "s-1",
                                                         "found": "tab"}, headers=page(client)).json()["report"]
        assert report is not None and len(reports(state)) == 1
    # Another engine (the next start) knows it from the page log kept: a goodbye there is no crash either.
    from sopa_companion.engine.crashlog import CrashReports
    crashes = CrashReports(config)
    crashes.save_page_log("bye", "s-2", "2026-10-23T12:10:00.000Z", pagehide=True)
    assert CrashReports(config).page("2026-10-23T12:09:00.000Z", "s-2", "flag") == report and len(reports(state)) == 1


def test_a_page_that_went_while_the_engine_runs_is_reported_by_the_next_page_never_by_the_engine(config, one_gpu):
    """The engine writes no report of its own for a page gone without a goodbye while it runs: such a page is away
    (server/presence.py: its runs pause), as a tab the browser put away or a lost connection is, and neither is a crash.
    The page that starts next in that window says so ("tab"): that page's report, written once and offered."""
    from sopa_companion.engine import crashlog
    from sopa_companion.server import create_app
    state, _ = folders(config)
    assert not hasattr(crashlog.CrashReports, "pages_lost") and "window" not in crashlog.FOUND
    application = create_app(config, token=TOKEN)
    with TestClient(application, base_url=BASE) as client:
        client.post("/api/v1/page-log", json={"text": "  12:00:05.000  action    Start\n", "session": "s-1",
                                              "at": "2026-10-23T12:00:01.000Z"}, headers=page(client))
        client.post("/api/v1/page-log", json={"text": "  12:00:30.000  action    Pause\n", "session": "s-1",
                                              "at": "2026-10-23T12:00:31.000Z"}, headers=page(client))
        client.post("/api/v1/page-log", json={"text": "another window", "session": "s-2",
                                              "at": "2026-10-23T12:00:02.000Z"}, headers=page(client))
        assert not reports(state)                                     # the engine says nothing of its own
        answer = client.post("/api/v1/crash/page", json={"at": "2026-10-23T12:00:01.000Z", "session": "s-1",
                                                         "found": "tab"}, headers=page(client)).json()["report"]
        text = (state / "crash-reports" / f"{answer['id']}.txt").read_text()
        assert row("Page session", "s-1\n") in text and "Offered" not in text
        assert "as the next page in the same window started" in text and "12:00:30.000  action    Pause" in text
        assert "another window" not in text and len(reports(state)) == 1
        assert client.get("/api/v1/crash", headers=auth()).json()["report"]["id"] == answer["id"]


def test_a_page_that_ended_with_the_engine_brings_the_engines_report_along(config, one_gpu):
    """The computer stopped with the page open: the engine's report (written as it started) and the page's, found by
    its flag. The page's, asked for first, holds the engine's, which then counts as offered: one popup, one file (not
    kept quietly: the engine went down with it)."""
    from sopa_companion.server import create_app
    state, data = folders(config)
    started = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
    write_marker(state, dead_pid(), started, data)
    write_page_log(state, "s-1", started + timedelta(minutes=30), "  12:30:00.000  action    Resume\n")
    application = create_app(config, token=TOKEN)
    [engine] = reports(state)
    with TestClient(application, base_url=BASE) as client:
        client.post("/api/v1/page-log", json={"text": "the new page", "session": "s-2", "at": "x"}, headers=page(client))
        summary = client.post("/api/v1/crash/page", json={"at": (started + timedelta(minutes=1)).isoformat(),
                                                          "session": "s-1", "found": "flag"}, headers=page(client)).json()["report"]
        assert summary["reason"] == "page" and summary["id"] != engine.stem
        text = client.get(f"/api/v1/crash/{summary['id']}.txt", headers=auth()).text
        assert f"The engine's own report, written as this engine started ({engine.stem})" in text
        assert engine.read_text().rstrip("\n") in text and text.count("12:30:00.000  action    Resume") == 2
        assert "Offered" not in text and "Earlier reports" not in text
        client.post(f"/api/v1/crash/{summary['id']}/seen", headers=page(client))
        assert client.get("/api/v1/crash", headers=auth()).json() == {"report": None}


def test_a_report_older_than_one_offered_is_never_offered(config, one_gpu):
    """Two sessions ended unexpectedly before a page offered either: the newer is offered (and names the older); once
    it is answered the older one is not offered at the next start ("last time" it is not), only a newer one would be."""
    from sopa_companion.engine.crashlog import CrashReports
    from sopa_companion.server import create_app
    state, data = folders(config)
    write_marker(state, dead_pid(), datetime.now(timezone.utc) - timedelta(hours=5), data)
    create_app(config, token=TOKEN)                                   # the first crash's report (no page opened)
    write_marker(state, dead_pid(), datetime.now(timezone.utc) - timedelta(hours=1), data)
    application = create_app(config, token=TOKEN)                     # the second, before the first was offered
    first, second = [p.stem for p in reports(state)]
    assert row("Earlier reports", f"{first}: not offered yet (in the same folder)") in (state / "crash-reports" /
                                                                                       f"{second}.txt").read_text()
    with TestClient(application, base_url=BASE) as client:
        assert client.get("/api/v1/crash", headers=auth()).json()["report"]["id"] == second
        client.post(f"/api/v1/crash/{second}/seen", headers=page(client))
    application = create_app(config, token=TOKEN)                     # after a clean shutdown
    with TestClient(application, base_url=BASE) as client:
        assert client.get("/api/v1/crash", headers=auth()).json() == {"report": None}
    now = [time.time() + 120]
    crashes = CrashReports(config, clock=lambda: now[0])
    third = crashes.page("2026-10-23T12:00:00Z", "s-1")["id"]
    assert third > second and crashes.newest()["id"] == third


def test_a_page_report_holds_the_engine_session_the_page_lived_in(config, one_gpu):
    """A page of an earlier engine session that ended cleanly: its report's engine lines are that session's, from its
    "engine starting" line to the next session's (the page was not heard from after), not this session's."""
    from sopa_companion.engine.crashlog import CrashReports, starting_line
    state, _ = folders(config)
    began = datetime.now(timezone.utc) - timedelta(hours=3)
    logs = state / "logs"
    logs.mkdir(parents=True)
    (logs / "server.log").write_text("\n".join([
        starting_line(111, began.timestamp()), "a line of the session the page lived in",
        starting_line(222, (began + timedelta(hours=1)).timestamp()), "a line of the next session",
        starting_line(os.getpid(), (began + timedelta(hours=2)).timestamp()), "a line of this session"]) + "\n")
    write_page_log(state, "s-1", began + timedelta(minutes=20), "  page lines\n")
    crashes = CrashReports(config)
    page = crashes.page(iso(began + timedelta(minutes=10)), "s-1", "tab")
    text = (state / "crash-reports" / f"{page['id']}.txt").read_text()
    part = text[text.index("server.log:"):text.index("The page's last log")]
    assert part.split("\n")[0].endswith("(2 lines)") and part.split("\n")[2].startswith("SoPA companion: engine starting: pid 111")
    assert "a line of the session the page lived in" in part and "next session" not in part and "this session" not in part
    assert "is not in server.log" not in part
    # A page heard from in the next session too (its window stayed open as the engine started again): both.
    write_page_log(state, "s-2", began + timedelta(hours=1, minutes=5), "  page lines\n")
    page = crashes.page(iso(began + timedelta(minutes=10)), "s-2", "tab")
    text = (state / "crash-reports" / f"{page['id']}.txt").read_text()
    part = text[text.index("server.log:"):text.index("The page's last log")]
    assert "the page lived in" in part and "a line of the next session" in part and "this session" not in part


def test_the_page_logs_are_kept_per_page_session(config, one_gpu):
    """POST /page-log {text, session, at, pagehide}: logs/page-logs/<session>.txt (private, written whole) holds the last
    512 KB of that page's text under a line naming it, when it was sent and when its first was; the newest sent is also
    logs/page-log.txt; one sent before the one kept changes nothing; the newest 6 page sessions are kept."""
    from sopa_companion.engine.crashlog import PAGE_LOG_BYTES, PAGE_LOGS_KEPT
    from sopa_companion.server import create_app
    state, _ = folders(config)
    logs = state / "logs"
    def kept(session):
        return (logs / "page-logs" / f"{session}.txt").read_text()
    application = create_app(config, token=TOKEN)
    with TestClient(application, base_url=BASE) as client:
        lines = [f"  12:00:{i % 60:02d}.000  action    line {i:06d} " + "x" * 90 for i in range(6000)]
        text = "\n".join(lines) + "\n"
        assert len(text.encode()) > PAGE_LOG_BYTES == 512 * 1024
        body = json.dumps({"text": text, "session": "s-1", "at": "2026-10-23T12:00:00.000Z"})
        assert client.post("/api/v1/page-log", content=body, headers=beacon(client)).json() == {}
        head, _, body = (logs / "page-logs" / "s-1.txt").read_bytes().partition(b"\n")
        assert head == b"Page session s-1, sent 2026-10-23T12:00:00.000Z, first 2026-10-23T12:00:00.000Z"
        assert len(body) <= PAGE_LOG_BYTES and body.endswith(lines[-1].encode() + b"\n")
        assert body.startswith(b"(the page's older lines were left out)\n  12:00:")       # from a whole line
        assert len(body) > PAGE_LOG_BYTES - 200
        assert (logs / "page-log.txt").read_bytes() == (logs / "page-logs" / "s-1.txt").read_bytes()
        for path in (logs / "page-log.txt", logs / "page-logs" / "s-1.txt"):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE((logs / "page-logs").stat().st_mode) == 0o700
        # No temporary file (the mixing check's own log may hold its probe's answer by now, written at the start).
        assert sorted(p.name for p in logs.rglob("*") if p.name != "qc-trainer.log") == ["page-log.txt", "page-logs",
                                                                                          "s-1.txt"]
        assert client.post("/api/v1/page-log", json={"text": "short", "session": "s-1", "at": "2026-10-23T12:00:10Z"},
                           headers=page(client)).json() == {}
        assert kept("s-1") == "Page session s-1, sent 2026-10-23T12:00:10Z, first 2026-10-23T12:00:00.000Z\nshort"
        client.post("/api/v1/page-log", json={"text": "late", "session": "s-1", "at": "2026-10-23T12:00:05Z"},
                    headers=page(client))                                    # sent before the one kept, arriving after it
        assert kept("s-1").endswith("\nshort")
        client.post("/api/v1/page-log", json={"text": "bye", "session": "s-1", "at": "2026-10-23T12:00:15Z",
                                              "pagehide": True}, headers=page(client))
        client.post("/api/v1/page-log", json={"text": "with the goodbye", "session": "s-1", "at": "2026-10-23T12:00:15Z"},
                    headers=page(client))                                    # sent with it, after it: still closed
        assert kept("s-1") == "Page session s-1, sent 2026-10-23T12:00:15Z, first 2026-10-23T12:00:00.000Z, closed\nbye"
        client.post("/api/v1/page-log", json={"text": "another page", "session": "s-2", "at": "2026-10-23T12:00:20Z"},
                    headers=page(client))
        assert (logs / "page-log.txt").read_text() == kept("s-2") == ("Page session s-2, sent 2026-10-23T12:00:20Z, "
                                                                     "first 2026-10-23T12:00:20Z\nanother page")
        assert kept("s-1").endswith("\nbye")
        for body, status in ((b'{"text": 5}', 422), (b'{"session": "s-1"}', 422), (b"[]", 422), (b"{", 422),
                             (b'{"text": "' + b"x" * (2 * 1024 * 1024) + b'"}', 413)):
            assert client.post("/api/v1/page-log", content=body, headers=page(client)).status_code == status, body[:30]

        def chunked():                                       # no Content-Length: still cut off at 2 MB
            yield b'{"text": "'
            for _ in range(2100):
                yield b"x" * 1000
            yield b'"}'
        refused = client.post("/api/v1/page-log", content=chunked(), headers=page(client))
        assert refused.status_code == 413 and refused.json()["detail"] == "This is limited to 2 MB"
        assert (logs / "page-log.txt").read_text().endswith("\nanother page")              # refused: unchanged
        # A session id that is no plain file name gets a file of its own all the same.
        client.post("/api/v1/page-log", json={"text": "odd", "session": "../a b", "at": "2026-10-23T12:00:25Z"},
                    headers=page(client))
        odd = "s-" + hashlib.sha256(b"..ab").hexdigest()[:32]                          # ("../a b" as a header word)
        assert kept(odd) == "Page session ..ab, sent 2026-10-23T12:00:25Z, first 2026-10-23T12:00:25Z\nodd"
        for i in range(3, 3 + PAGE_LOGS_KEPT):
            client.post("/api/v1/page-log", json={"text": f"page {i}", "session": f"s-{i}",
                                                  "at": f"2026-10-23T12:01:{i:02d}Z"}, headers=page(client))
        assert sorted(p.stem for p in (logs / "page-logs").glob("*.txt")) == [f"s-{i}" for i in range(3, 3 + PAGE_LOGS_KEPT)]


def test_a_goodbye_cut_to_fit_keeps_the_longer_log_of_its_page():
    """The page's goodbye is small (a beacon): cut to fit, it leaves out the older lines a longer log of the same page
    has. That log stays, with the goodbye's lines from its last line on (that line's count as the goodbye has it)."""
    from sopa_companion.engine.crashlog import merge_goodbye
    head = "Page (this window)\n==================\nShowing         no sample open\n\nWhat the user did\n" + "-" * 74 + "\n"
    steps = [f"  12:00:{i:02d}.000  action    step {i}" for i in range(50)]
    kept = head + "\n".join(steps) + "\n"
    bye = (head + "  … 45 older lines left out\n" + "\n".join(steps[45:49]) + "\n" + steps[49] +
           " (2 times, last 12:00:59.000)\n  12:01:00.000  page      the page closed or reloaded\n")
    merged = merge_goodbye(kept, bye)
    assert merged == head + "\n".join(steps[:49]) + "\n" + steps[49] + " (2 times, last 12:00:59.000)\n" \
                                                                         "  12:01:00.000  page      the page closed or reloaded\n"
    assert merge_goodbye(kept, head + "\n".join(steps) + "\n") is None                  # a whole goodbye: as it is
    assert merge_goodbye("short", bye) is None                                           # longer than what is kept
    apart = merge_goodbye(head + "\n".join(steps[:10]) + "\n" + "z" * 2000 + "\n", bye)  # no line in common
    assert apart.split("\n")[len(head.split("\n")) + 10:len(head.split("\n")) + 12] == \
        ["  … (lines between these were not kept)", steps[45]]


def test_the_goodbye_of_a_page_keeps_its_longer_log(config, one_gpu):
    from sopa_companion.engine.crashlog import CrashReports
    state, _ = folders(config)
    crashes = CrashReports(config)
    steps = [f"  12:00:{i % 60:02d}.{i // 60:03d}  action    step {i}" for i in range(400)]
    crashes.save_page_log("\n".join(steps) + "\n", "s-1", "2026-10-23T12:00:00Z")
    crashes.save_page_log("  … 390 older lines left out\n" + "\n".join(steps[390:]) + "\n  12:01:00.000  page      gone\n",
                          "s-1", "2026-10-23T12:00:10Z", pagehide=True)
    text = (state / "logs" / "page-logs" / "s-1.txt").read_text()
    assert text.startswith("Page session s-1, sent 2026-10-23T12:00:10Z, first 2026-10-23T12:00:00Z, closed\n")
    assert all(step in text for step in steps) and text.endswith("page      gone\n") and text.count("step 399") == 1


def test_at_most_ten_reports_are_kept_and_seen_forgets_those_gone(config, one_gpu):
    from sopa_companion.engine.crashlog import KEEP, CrashReports
    state, _ = folders(config)
    now = [time.time()]
    crashes = CrashReports(config, clock=lambda: now[0])
    made = []
    for i in range(KEEP + 2):
        made.append(crashes.page("2026-10-23T12:00:00Z", f"s-{i}")["id"])
        if i in (0, 5):
            assert crashes.seen(made[-1])
        now[0] += 61
    assert KEEP == 10 and len(set(made)) == KEEP + 2 and made == sorted(made)
    assert [p.stem for p in reports(state)] == made[-KEEP:] == crashes.ids()
    assert json.loads((state / "crash-reports" / "seen.json").read_text()) == {"seen": [made[5]]}
    assert crashes.newest()["id"] == made[-1] and crashes.text(made[0]) is None and not crashes.seen(made[0])


def test_two_reports_in_one_second_get_a_name_each(config, one_gpu):
    from sopa_companion.engine.crashlog import CrashReports
    state, _ = folders(config)
    crashes = CrashReports(config, clock=lambda: 1_790_000_000.0)
    first = crashes.page("2026-10-23T12:00:00Z", "s-1")["id"]
    second = crashes.page("2026-10-23T12:00:00Z", "s-2")["id"]
    assert first != second and first.endswith("-page") and second.endswith("-page") and len(reports(state)) == 2


def test_the_gpus_are_nvidia_smis_list_or_unknown(tmp_path, monkeypatch):
    """The header's GPUs: nvidia-smi -L's lines without their UUIDs (a report is made to be shared); "unknown" when it
    fails, is not there, or has not answered in time."""
    from sopa_companion.engine.crashlog import gpus
    folder = tmp_path / "bin"
    folder.mkdir()
    smi = folder / "nvidia-smi"
    monkeypatch.setenv("PATH", str(folder))

    def script(body: str):
        smi.write_text(f"#!{sys.executable}\nimport sys, time\n{body}\n")
        smi.chmod(0o755)
    script("print('GPU 0: A test GPU (UUID: GPU-0a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d)')\n"
           "print('GPU 1: A test GPU (UUID: GPU-1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d)')")
    assert gpus() == ["GPU 0: A test GPU", "GPU 1: A test GPU"]                  # a GPU's UUID is its serial number
    script("print('NVIDIA-SMI has failed', file=sys.stderr)\nsys.exit(9)")
    assert gpus() == ["unknown"]
    script("time.sleep(30)")
    began = time.monotonic()
    assert gpus(timeout=0.5) == ["unknown"] and time.monotonic() - began < 10
    smi.unlink()
    assert gpus() == ["unknown"]


def test_an_nvidia_smi_that_cannot_be_killed_is_never_waited_for(tmp_path, monkeypatch):
    """One stuck in the driver (a GPU gone wrong) does not die when killed: gpus() answers "unknown" in its time all the
    same (subprocess.run waited for it as long as it lived), and a thread collects it whenever it ends."""
    from sopa_companion.engine.crashlog import gpus
    folder = tmp_path / "bin"
    folder.mkdir()
    smi = folder / "nvidia-smi"
    smi.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", str(folder))
    monkeypatch.setattr(subprocess.Popen, "kill", lambda self: None)          # it cannot be killed
    began = time.monotonic()
    try:
        assert gpus(timeout=0.5) == ["unknown"] and time.monotonic() - began < 5
        assert any(t.name == "nvidia-smi -L" and t.daemon for t in threading.enumerate())
    finally:
        for process in psutil.Process().children():
            if str(smi) in " ".join(process.cmdline()):
                process.kill()


def test_the_gpus_are_looked_up_once_and_never_while_the_reports_wait(config, monkeypatch):
    """A slow nvidia-smi holds up only the report that needs it: the page's log and its answers go on meanwhile (they
    share the reports' lock, which the lookup never holds); the GPUs are looked up once per engine."""
    from sopa_companion.engine import crashlog
    calls = []

    def slow(timeout=crashlog.GPU_TIMEOUT_S):
        calls.append(1)
        time.sleep(1.5)
        return ["GPU 0: A slow GPU"]
    monkeypatch.setattr(crashlog, "gpus", slow)
    crashes = crashlog.CrashReports(config)
    first = crashes.page("2026-10-23T12:00:00Z", "s-1", "tab")["id"]
    writing = threading.Thread(target=crashes.page, args=("2026-10-23T12:00:00Z", "s-2", "tab"))
    crashes.gpu_names = None                                                  # as before the first lookup
    writing.start()
    time.sleep(0.2)
    began = time.monotonic()
    crashes.save_page_log("hello", "s-3", "2026-10-23T12:00:01Z")
    assert crashes.seen(first) and time.monotonic() - began < 1.0
    writing.join()
    assert len(calls) == 2                                   # once per lookup (and the test asked for a second)
    crashes.page("2026-10-23T12:00:00Z", "s-4", "tab")
    assert len(calls) == 2 and "GPU 0: A slow GPU" in crashes.text(crashes.ids()[-1])
