"""API-mode entry point and the app.py session hook; no desktop interpreter re-exec.

server/app.py create_app takes session=None, passes it to Runs, and calls attach_session just before returning the
app; its WebSocket loop calls hooks.push(socket, version) when app.state.session_hooks is not None. The hook owns the
API routes and lifespan, so the desktop closing dialog and its save fallback are never entered in API mode.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import inspect
import json
import logging
import os
from pathlib import Path
import shutil

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import portable
from ..api import protocol
from ..engine.session import Session, read_session_config
from .presence import VISITS, watch

LOG = logging.getLogger(__name__)
CLOSING_MAX_BYTES = 4 * 1024
CLOSING_MAX_RUNS = 64
ENDING_PUSH_TIMEOUT_S = .5
BASE = "/api/v1"


def session_health(session=None) -> dict:
    return protocol.build_health_session("desktop") if session is None else session.public()


def _not_json(constant):
    raise ValueError(f"{constant} is not JSON")


def read_closing(raw) -> dict:
    """POST /closing's body, both modes: {answer: "add" | "discard" | "withdraw" ("save" is "add"), runs?: [the ids of
    the runs the page showed, at most 64], revision?: a whole number that grows with each choice in the page's window
    (Keep working too; the beacon repeats the last one), page?: the window's own id, end?: true (API mode: close now),
    bye?: true (sent as the page goes for good: its goodbye, server/presence.py), visit?: a whole number, the page's
    showing that says it}, as {answer, runs, revision, page, end, bye, visit} with None for what was not sent (bye:
    false). JSON whatever the Content-Type said (a text/plain body too). ValueError for anything else."""
    try:
        body = json.loads(bytes(raw), parse_constant=_not_json)
    except (ValueError, RecursionError) as exc:     # UnicodeDecodeError is a ValueError
        raise ValueError("An answer to closing is a JSON object") from exc
    if not isinstance(body, dict) or body.get("answer") not in ("add", "discard", "save", "withdraw"):
        raise ValueError("Unknown answer to closing")
    runs, revision, page, end, bye, visit = (body.get(key) for key in ("runs", "revision", "page", "end", "bye", "visit"))
    if runs is not None and not (isinstance(runs, list) and len(runs) <= CLOSING_MAX_RUNS and
                                 all(isinstance(run, str) and 0 < len(run) <= 200 for run in runs)):
        raise ValueError(f"runs: at most {CLOSING_MAX_RUNS} run ids")
    if revision is not None and (type(revision) is not int or revision < 0):
        raise ValueError("revision: a whole number")
    if page is not None and not (isinstance(page, str) and 0 < len(page) <= 100):
        raise ValueError("page: a short id")
    if end is not None and not isinstance(end, bool):
        raise ValueError("end: true or false")
    if bye is not None and not isinstance(bye, bool):
        raise ValueError("bye: true or false")
    if visit is not None and (type(visit) is not int or not 0 <= visit < VISITS):
        raise ValueError("visit: a whole number")
    return {"answer": "add" if body["answer"] == "save" else body["answer"], "runs": runs, "revision": revision,
            "page": page, "end": end, "bye": bye is True, "visit": visit}


def problem(detail, status):
    return JSONResponse({"detail": detail, "status": status}, status_code=status, media_type="application/problem+json")


def close_session_window(config):
    """Close only this user's Chrome processes carrying this session's exact profile flag: the session's own Chrome
    window, when its kernel opened one (browser = "chrome"). No other browser carries that flag, so a session shown in
    the user's own browser has nothing to close here (a tab cannot be closed from outside: its page says the session
    has ended)."""
    import psutil
    from ..desktop.window import AppWindow, chrome_profile

    if not portable.LINUX:                      # the app's own Chrome window is Linux's alone (desktop/browser.py)
        return
    window = AppWindow(chrome_profile(config))
    for process in psutil.process_iter(["cmdline", "uids"]):
        try:
            uids = process.info.get("uids")
            if uids is not None and uids.real == os.getuid() and window.carries(process.info["cmdline"]):
                process.terminate()
        except (psutil.Error, ProcessLookupError):
            pass


def cleanup_session_folders(config):
    """Only the handle's named private folders can be removed after the kernel goes."""
    id = config["session"]["id"]
    for key, expected in (("data_dir", f"sopa-companion-{id}"), ("state_dir", id)):
        path = Path(config["paths"][key])
        if not path.is_absolute() or path.name != expected or path.is_symlink():
            LOG.warning("Leaving an unexpected API %s folder in place", key)
            continue
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            LOG.warning("Removing the API %s folder failed: %s", key, exc)


class SessionHooks:
    def __init__(self, app, config, session, *, shutdown, close_window=None, cleanup=None):
        self.app, self.config, self.session, self.shutdown = app, config, session, shutdown
        self.close_window = close_window or (lambda: close_session_window(config))
        self.cleanup = cleanup or (lambda: cleanup_session_folders(config))
        self.ready_when = lambda: True
        self._loop = None
        self._shutdown_task = self._watcher = self._ready_task = None
        self._ending_sent = asyncio.Event()
        self._tearing_down = False
        self._closed = False

    async def push(self, socket, since=-1) -> int:
        version, event = self.session.event(since)
        if event is not None:
            await socket.send_json(event)
            if event["state"] == "ending":
                self._ending_sent.set()
        return version

    def _schedule_shutdown(self):
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._start_shutdown)

    def _start_shutdown(self):
        if not self._tearing_down and self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._stop_after_push())

    async def _stop_after_push(self):
        try:
            await asyncio.wait_for(self._ending_sent.wait(), timeout=ENDING_PUSH_TIMEOUT_S)
        except asyncio.TimeoutError:
            pass
        result = self.shutdown()
        if inspect.isawaitable(result):
            await result

    async def _ready(self):
        while not self.ready_when():
            await asyncio.sleep(.01)
        self.session.deliver(protocol.build_event("ready", port=self.config["server"]["port"], pid=os.getpid()))

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self.session.bind_sample = lambda path, **kwargs: self.app.state.zarr.bind(path, **kwargs)
        self.session.on_close = self._schedule_shutdown
        self.session.start()
        self._ready_task = asyncio.create_task(self._ready())
        desktop = self.config.get("desktop", {})
        if desktop.get("exit_when_closed", False):
            # A session opened in a window (open="window"): its last page's goodbye closes the session with the page's
            # answer, else discard (Session.begin_close). A page gone without one pauses its runs, and the session waits
            # for a page to come back however long that takes: the kernel owns it (its close(), or the kernel's end).
            runs = self.app.state.runs
            self._watcher = asyncio.create_task(watch(
                self.app.state.presence, grace=float(desktop.get("close_grace_s", 6)),
                pending_grace=float(desktop.get("close_grace_pending_s", 20)), away_stop_s=0,
                pending=lambda: bool(runs.pending()), on_closed=lambda: self.session.begin_close(by="window"),
                on_away=runs.pause_for_away))
        if self.session.public()["state"] == "ending":
            self._start_shutdown()

    async def _close(self, name, action, *args, **kwargs):
        try:
            await asyncio.to_thread(action, *args, **kwargs)
        except Exception:
            LOG.exception("Closing the API session's %s failed", name)

    def close_policy(self):
        """The session's answer as Runs.close_session takes it. The page's "add" is for the runs it was given for:
        with a pending run it did not cover (started after it), {run id: "add"} for those it
        did, any other run discarded as nobody answered for it. Otherwise one answer for every run:
        the handle's close, the kernel gone, no answer, or a page's answer that covers every pending run."""
        policy, covers = self.session.closing_policy(), self.session.closing_covers()
        if policy != "add" or covers is None:
            return policy
        uncovered = [run["id"] for run in self.app.state.runs.pending()
                     if run["id"] not in covers and not run.get("decided")]
        return {run_id: "add" for run_id in sorted(covers)} if uncovered else policy

    async def teardown(self):
        if self._closed:
            return
        self._tearing_down = True
        self.session.begin_close(by="window")
        tasks = [task for task in (self._watcher, self._ready_task, self._shutdown_task) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        state = self.app.state
        # The pipe reader remains alive until all Adds and their received acknowledgements have completed.
        for name in ("qc_tests", "compare", "qc", "deeper"):
            await self._close(name, getattr(state, name).close)
        try:
            await asyncio.to_thread(state.runs.close_session, await asyncio.to_thread(self.close_policy))
        except Exception:
            LOG.exception("Finishing the API session's pending results failed")
        finally:
            await self._close("analysis", state.analysis.close)
            await self._close("sample helpers", state.zarr.close)
            await self._close("run folders", state.runs.close, policy="discard")
        await self._close("window", self.close_window)
        self.session.finish()
        await asyncio.to_thread(self.session.stop)
        if self.session.kernel_gone:
            await self._close("folders", self.cleanup)
        self._closed = True

    @asynccontextmanager
    async def lifespan(self, app):
        try:
            await self.start()
            yield
        finally:
            await self.teardown()

    async def closing_state(self):
        pending = await asyncio.to_thread(self.app.state.runs.pending)
        return {**self.session.closing(), "runs": pending,
                "pending": [{key: run.get(key) for key in ("id", "kind", "sample", "iter", "state")} for run in pending]}

    async def closing_answer(self, request: Request):
        """POST /closing as the desktop takes it, runs, revision, page, withdraw, bye and visit (read_closing): an
        answer covers the runs the page showed (else every run pending now), never one started later; one overtaken by
        a later choice of its window, or said before its page was shown again, changes nothing; "withdraw" (Keep
        working, a reload) takes a kept answer back; with bye it is also that showing's goodbye (server/presence.py)."""
        declared = request.headers.get("content-length", "")
        too_big = "An answer to closing is limited to 4 KB"
        if declared.isdigit() and int(declared) > CLOSING_MAX_BYTES:
            return problem(too_big, 413)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > CLOSING_MAX_BYTES:
                return problem(too_big, 413)
        try:
            body = read_closing(raw)
            if body["answer"] != "withdraw":
                protocol.validate_closing(json.loads(raw))       # as the protocol reads it
        except (ValueError, RecursionError, UnicodeError):
            return problem('Send {"answer": "add"}, {"answer": "discard"} or {"answer": "withdraw"}, with an optional '
                           '"end": true', 422)
        presence = self.app.state.presence
        # Kept before anything is awaited, in the order answers came; one said before the page came back is not.
        taken = 0 if presence.stale(body["page"], body["visit"]) else self.session.answer_closing(body)
        if body["bye"]:
            presence.goodbye(body["page"], "closing", body["visit"])
        state = await self.closing_state()
        if taken and body["answer"] != "withdraw":
            self.session.cover_closing(taken, [run["id"] for run in state["runs"]])
            state.update(self.session.closing())
        return {**state, "taken": bool(taken)}

    def install_routes(self):
        app, session = self.app, self.session
        existing = next((route.endpoint for route in app.router.routes if getattr(route, "path", None) == BASE + "/health"
                         and "GET" in getattr(route, "methods", set())), None)
        router = APIRouter(prefix=BASE)

        @router.get("/health")
        async def health():
            original = {} if existing is None else (
                await existing() if inspect.iscoroutinefunction(existing) else await asyncio.to_thread(existing))
            return {**original, "session": session.public()}

        router.add_api_route("/closing", self.closing_state, methods=["GET"])
        router.add_api_route("/closing", self.closing_answer, methods=["POST"])

        @router.get("/browse")
        @router.delete("/samples/{sample_id}")
        async def refuse():
            return problem("This session works on one sample", 403)

        @router.get("/samples")
        async def samples():
            listed = await asyncio.to_thread(app.state.zarr.list)
            return {"samples": [sample for sample in listed if sample["id"] == session.bound]}

        @router.post("/samples/open")
        async def open_sample(request: Request):
            try:
                body = await request.json()
                path = body.get("path") if isinstance(body, dict) else None
                same = isinstance(path, str) and bool(path) and session.view_path is not None and (
                    Path(path).expanduser().resolve() == Path(session.view_path).expanduser().resolve())
            except (ValueError, OSError, RuntimeError):
                same = False
            if not same or session.bound is None:
                return problem("This session works on one sample", 403)
            return await asyncio.to_thread(app.state.zarr.detail, session.bound)

        # API routes take precedence over the desktop routes and the final static-files mount.
        app.router.routes[:0] = router.routes


def attach_session(app, config, *, session=None, shutdown=None, close_window=None, cleanup=None):
    """Called at the end of create_app. Desktop gets None; API replaces the desktop lifespan and closing routes."""
    if config.get("session", {}).get("mode") != "api":
        app.state.session = app.state.session_hooks = None
        return None
    existing = getattr(app.state, "session_hooks", None)
    if existing is not None:
        return existing
    session = session if session is not None else Session(config)
    shutdown = shutdown or portable.shut_down_this_process
    hooks = SessionHooks(app, config, session, shutdown=shutdown, close_window=close_window, cleanup=cleanup)
    app.state.session, app.state.session_hooks = session, hooks
    hooks.install_routes()
    app.router.lifespan_context = hooks.lifespan
    return hooks


def create_api_app(config, *, session=None, app_factory=None, shutdown=None):
    """The normal FastAPI app (server/app.py create_app, which hands the session to Runs) with the session's hooks."""
    if app_factory is None:
        from .app import create_app
        app_factory = create_app
    session = session if session is not None else Session(config)
    app = app_factory(config, session=session, shutdown=shutdown)
    attach_session(app, config, session=session, shutdown=shutdown)
    return app


def serve(config, *, session=None, app_factory=None, server_factory=None):
    from ..cli import listen
    from ..desktop.logs import install_rotation
    from .runtime_file import acquire_engine_lock, install_lifecycle
    import uvicorn

    session = session if session is not None else Session(config)
    listener = engine_lock = None
    portable.more_open_files()            # macOS: 256 by default, for the engine and every helper it starts
    try:
        host = config["server"].get("host", "127.0.0.1")
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("SoPA companion requires a loopback bind address")
        install_rotation(config)
        engine_lock = acquire_engine_lock(config)
        listener = listen(host, 0)
        port = listener.getsockname()[1]
        config["server"].update(host=host, port=port)
        server = None

        def shutdown():
            server.should_exit = True

        app = create_api_app(config, session=session, app_factory=app_factory, shutdown=shutdown)
        install_lifecycle(app, config, host, port)
        server = (server_factory or uvicorn.Server)(uvicorn.Config(app, host=host, port=port, log_level="info", access_log=False))
        app.state.session_hooks.ready_when = lambda: server.started
        server.run(sockets=[listener])
    finally:
        session.stop()
        if listener is not None:
            listener.close()
        if engine_lock is not None:
            os.close(engine_lock)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve one SoPA companion notebook session")
    parser.add_argument("--config", required=True, help="The private session.toml")
    args = parser.parse_args(argv)
    raw = read_session_config(args.config)
    from ..engine.config import load_config
    config = load_config(args.config)
    if located := config.get("locate"):       # where each helper runs, and proseg-live (load_config found them)
        from .. import locate
        print("SoPA companion: " + locate.summary(located), flush=True)
    return serve(config, session=Session(raw))


if __name__ == "__main__":
    raise SystemExit(main())
