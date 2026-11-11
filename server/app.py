"""REST API, live WebSocket and the app's own files (public: server/security.py)."""
from __future__ import annotations

import asyncio
import bisect
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import faulthandler
import hmac
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
import re
import secrets

from fastapi import Body, Depends, FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import PROTOCOL, __version__, locate, portable
from ..api import protocol
from ..cli import command_words
from ..desktop import browser
from ..desktop.window import AppWindow, chrome_profile
from ..engine.config import ROOT, load_config
from ..engine.analysis import AnalysisError, AnalysisManager
from ..engine import crashlog, runlog
from ..engine.deeper import DeeperManager
from ..engine.qc import QcError, QcManager, QcModels
from ..engine.qc_pool import QcPools
from ..engine.qc_test import QcTests, parse_preview
from ..engine.runs import RunError, Runs, SPEEDS, _options_module, has_transcripts, page_frame
from ..engine.samples import SampleError, Samples
MAX_STAIN_BYTES = 1024 * 1024
from ..engine.zarr_samples import UNREAD, HelperFailed, ZarrSamples
from ..engine.tiles import Container
from ..engine.vectors import VERSION as VECTORS_VERSION
from .presence import Presence, visit_number, watch as watch_pages
from .runtime_file import private_directory
from .security import AUTH, LAUNCH_CODE_S, LINK_CODE_S, SOCKET_PROTOCOL, LaunchCodes, LocalSecurity, page_id, reopen_words

DIALOG_MARGIN_S = 30.0                     # no answer after the timeout discards
CLOSING_ANSWER_S = 60.0                    # the page's answer to closing (POST /closing) is carried out by a shutdown
                                           # that begins within this many seconds of it; later, the dialog asks
CLOSING_MAX_BYTES = 4 * 1024               # that answer's body (413 beyond)
CLOSING_MAX_RUNS = 64                      # the runs one answer may name
TELEMETRY_MAX_BYTES = 16 * 1024            # one report from the page (413 beyond)
TELEMETRY_KEEP_BYTES = 5 * 1024 * 1024     # telemetry.jsonl is renamed to telemetry.jsonl.1 before it would pass this
TELEMETRY_LOG_EVERY_S = 30.0               # at most one telemetry summary line in the engine log per this many seconds
TELEMETRY_LINE_CHARS = 2000                # that line's length limit: the page's whole report (frontend/src/telemetry.mjs) fits,
                                           # with a long animation frame's script and a few plv: measures (v2: ~1,500)
CRASH_PAGE_MAX_BYTES = 4 * 1024            # the page's word that its last session did not unload (413 beyond)
PAGE_LOG_MAX_BYTES = 2 * 1024 * 1024       # one upload of the page's own log (413 beyond; its last 512 KB are kept)


def problem(detail, status=400, **extra):
    """A refusal or failure as problem JSON: {detail: its words, status} and `extra`, what else the answer says (a
    save's name clash: {error, clash, existing}, engine/runs.py save_to_zarr)."""
    return JSONResponse({"detail": str(detail), "status": status, **extra}, status_code=status,
                        media_type="application/problem+json")


def _has_display() -> bool:
    return browser.desktop_session()


def ask_to_save(unsaved: list[dict], *, timeout_s: float = 600, has_display=_has_display) -> bool:
    """Ask about undecided runs. Only the Save button saves; no answer means discard."""
    asked = [r for r in unsaved if not r.get("decided")]
    kept = [r for r in unsaved if r.get("decided")]
    if not asked:
        return False
    zenity = shutil.which("zenity")
    if not zenity or not has_display():
        # Keep this log line's form: tools that read the engine log parse it.
        also = f"; waiting for {len(kept)} run(s) already being saved or closed" if kept else ""
        print(f"SoPA companion: closing with {len(asked)} unsaved run(s): discarding (dialog answer: not asked, no "
              f"{'desktop' if not has_display() else 'zenity'} to ask on){also}", flush=True)
        return False
    listed = lambda runs: "\n".join(f"  • {r.get('sample_name') or r['sample']} ({r.get('state') or 'run'}" +
                                    (f", iteration {r['iter']}" if r.get('iter') is not None else "") + ")" for r in runs)
    plural, it = ("these runs", "them") if len(asked) > 1 else ("this run", "it")
    wait = f" or not answering within {timeout_s / 60:g} minutes" if timeout_s >= 120 else \
        f" or not answering within {timeout_s:g} seconds" if timeout_s > 0 else ""
    saving = (f"\n\nAlready being saved or closed (your earlier choice still applies):\n{listed(kept)}") if kept else ""
    text = (f"SoPA companion is closing.\n\nSave {plural} to the sample first?\n{listed(asked)}{saving}\n\n"
            f"Save writes the cells to the sample under new names, finishing the run first. "
            f"Closing this dialog{wait} discards {it}.")
    command = [*(["setpriv", "--pdeathsig", "TERM"] if shutil.which("setpriv") else []),
               zenity, "--warning", "--no-markup", "--title=SoPA companion", "--width=460", "--ok-label=Save to the sample",
               "--extra-button=Discard", *([f"--timeout={max(1, round(timeout_s))}"] if timeout_s > 0 else []),
               "--text=" + text]
    try:
        answer = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
                                timeout=timeout_s + DIALOG_MARGIN_S if timeout_s > 0 else None)
        code, said, detail = answer.returncode, (answer.stdout or "").strip(), (answer.stderr or "").strip()[-300:]
    except subprocess.TimeoutExpired:
        code, said, detail = None, "", f"no answer within {timeout_s + DIALOG_MARGIN_S:g} s"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        code, said, detail = None, "", str(exc)
    discard = code != 0
    if said == "Discard" or code == 0:
        meaning = "Discard" if discard else "Save to the sample"
    elif code == 1 and not said:                   # also what zenity does when the display has gone (session ended)
        meaning = f"could not open: {detail}" if "display" in detail.lower() else "closed without choosing"
    elif code == 5:
        meaning = "no answer before its timeout"
    else:
        meaning = f"{'failed' if code is None else f'exit {code}'}: {said or detail or 'no detail'}"
    also = f"; waiting for {len(kept)} run(s) already being saved or closed" if kept else ""
    print(f"SoPA companion: closing with {len(asked)} unsaved run(s): {'discarding' if discard else 'saving'} "
          f"(dialog answer: {meaning}){also}", flush=True)
    return not discard


class ClosingAnswer:
    """The page's answer to closing with pending results (POST /closing): "add" or "discard", the runs it is for, and
    when it came. A shutdown that begins within `within_s` seconds of it carries it out for those runs instead of the
    dialog (carry_out_answer); without one (a crash, a page that could not answer) the dialog asks as before.
    - Its runs: the ids the page showed, kept as far as they were pending when it came (a page
      that names none answers for every run pending then). A run started afterwards is never covered by it: at
      shutdown that run takes the dialog, else is discarded, as with no answer at all.
    - Its order: `revision` grows with each choice in a window (Keep working too) and `page` names
      the window. An answer, or a withdraw, with a lower revision than the last one taken from the same window (from
      any window, when the page names none) was overtaken on its way and changes nothing: a discard sent before Keep
      working never lands after it.
    - "withdraw" (Keep working) takes the kept answer back; so does a page that connects again (a reload)."""

    def __init__(self, *, within_s: float = CLOSING_ANSWER_S, clock=time.monotonic, log=None):
        self.within_s, self.clock = within_s, clock
        self.log = log or (lambda message: print(message, flush=True))
        self.lock = threading.Lock()
        self.answer, self.at, self.runs = None, 0.0, frozenset()
        self.revisions: dict = {}      # page (None: a page that names none) -> the last revision taken from it
        self.taken = 0                 # how many answers were taken: names the kept one for cover()

    def give(self, answer: str, runs=None, revision=None, page=None) -> int:
        """Keep `answer` for `runs` (None: every run pending now), or take the kept one back ("withdraw"). -> the taken
        answer's number for cover(); 0, and nothing changes, when a later choice of the same window was taken first."""
        answer = "add" if answer == "save" else answer
        with self.lock:
            last = self.revisions.get(page)
            if revision is not None and last is not None and revision < last:
                return 0
            if revision is not None:
                self.revisions.pop(page, None)
                self.revisions[page] = revision
                while len(self.revisions) > CLOSING_MAX_RUNS:          # a few windows per session, bounded anyway
                    self.revisions.pop(next(iter(self.revisions)))
            if answer == "withdraw":
                self.answer, self.runs = None, frozenset()
            else:
                self.answer, self.at, self.runs = answer, self.clock(), None if runs is None else frozenset(runs)
            self.taken += 1
            return self.taken

    def cover(self, taken: int, pending):
        """Answer number `taken` (give's), if it is still the kept one, is for the runs pending when it came (`pending`,
        their ids) that it named, or every one of them when it named none: never for a run started later."""
        with self.lock:
            if self.answer is not None and self.taken == taken:
                self.runs = frozenset(pending) if self.runs is None else self.runs & frozenset(pending)

    def withdraw(self):
        """A page connected: an answer given before no longer counts (and a page that names no window starts its
        revisions again)."""
        with self.lock:
            answer, self.answer, self.runs = self.answer, None, frozenset()
            self.revisions.pop(None, None)
        if answer is not None:
            self.log(f"SoPA companion: a page connected again: the answer to closing ({answer}) no longer counts")

    def public(self) -> dict:
        """{answer, age_s, applies, covers}: `applies` while a shutdown would carry it out, for the runs in `covers`
        (null: every run pending, for a moment after an answer that named none)."""
        with self.lock:
            answer, age = self.answer, max(0.0, self.clock() - self.at)
            covers = None if self.runs is None else sorted(self.runs)
        if answer is None:
            return {"answer": None, "age_s": None, "applies": False, "covers": []}
        return {"answer": answer, "age_s": round(age, 1), "applies": age <= self.within_s and covers != [],
                "covers": covers}

    def at_shutdown(self) -> tuple[str, float, frozenset | None] | None:
        """(answer, age in seconds, the runs it covers or None for every one) to carry out as the shutdown begins;
        None without one, or with one too old (logged)."""
        state = self.public()
        if state["answer"] is not None and state["age_s"] > self.within_s:
            self.log(f"SoPA companion: the answer to closing ({state['answer']}, {state['age_s']:g} s before the "
                     f"shutdown) is too old to count")
        if not state["applies"]:
            return None
        return state["answer"], state["age_s"], None if state["covers"] is None else frozenset(state["covers"])


def carry_out_answer(unsaved: list[dict], answer: str, age_s: float) -> bool:
    """The page's answer instead of the desktop dialog; already decided leaves are waited for.
    Keep the log line's form: tools that read the engine log parse it."""
    asked = [r for r in unsaved if not r.get("decided")]
    kept = len(unsaved) - len(asked)
    if not asked:
        return True
    discard = answer == "discard"
    also = f"; waiting for {kept} run(s) already being saved or closed" if kept else ""
    print(f"SoPA companion: closing with {len(asked)} unsaved run(s): {'discarding' if discard else 'saving'} "
          f"(dialog answer: not asked, the page answered {answer} {age_s:.0f} s before){also}", flush=True)
    return not discard


def decide_closing(unsaved: list[dict], answered=None, *, forced: bool = False, timeout_s: float = 600,
                   page_closed: bool = False) -> dict:
    """{run id: "add" | "discard"} for each pending run nobody has decided yet (decided ones are completed whatever
    happens: a leave or save under way). The page's answer (ClosingAnswer.at_shutdown: answer, age, the runs it covers)
    decides its runs; the others are asked in the desktop dialog, which discards when nobody answers; `forced` (POST
    /shutdown, stop --force) discards them without asking, and so does `page_closed`: the last page said goodbye as it
    went (server/presence.py), so a run it did not answer for is one nobody answered for. The log lines keep the form
    tools that read the engine log parse."""
    decided = [r for r in unsaved if r.get("decided")]
    open_runs = [r for r in unsaved if not r.get("decided")]
    answer, age, covers = answered if answered is not None else (None, 0.0, frozenset())
    covered = [r for r in open_runs if answer is not None and (covers is None or r["id"] in covers)]
    rest = [r for r in open_runs if r not in covered]
    decisions = {}
    if covered:
        add = carry_out_answer(covered + decided, answer, age)
        decisions.update((r["id"], "add" if add else "discard") for r in covered)
    if rest and forced:
        also = f"; waiting for {len(decided)} run(s) already being saved or closed" if decided else ""
        print(f"SoPA companion: closing with {len(rest)} unsaved run(s): discarding (dialog answer: not asked, stop "
              f"--force){also}", flush=True)
    elif rest and page_closed:
        also = f"; waiting for {len(decided)} run(s) already being saved or closed" if decided else ""
        print(f"SoPA companion: closing with {len(rest)} unsaved run(s): discarding (dialog answer: not asked, the "
              f"page closed without an answer){also}", flush=True)
    elif rest:
        try:
            add = ask_to_save(rest + decided, timeout_s=timeout_s)
        except Exception:          # no answer: discard, including a lost desktop or terminal
            add = False
        decisions.update((r["id"], "add" if add else "discard") for r in rest)
    decisions.update((r["id"], "discard") for r in rest if r["id"] not in decisions)
    return decisions


def hang_up_like_term(signum=None, frame=None, terminals=(1, 2)):
    """SIGHUP, the terminal of a `serve` gone (an SSH session that dropped), shuts the engine down as SIGTERM does, so
    its runs follow the same save-or-discard answer as at any close (uvicorn handles SIGINT and SIGTERM only, and a
    hang-up killed the engine with no teardown; the next start deleted its runs). What the engine prints to that
    terminal from now on has nowhere to go, and must not fail the shutdown (a write to it fails: EIO): `terminals`,
    the outputs that were that terminal when the handler was set (hang_up_handler: a terminal that has hung up is no
    terminal to isatty any more), go to /dev/null; an output into a file or a pipe keeps it."""
    for fd in terminals:
        try:
            null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, fd)
            os.close(null)
        except OSError:
            pass
    os.kill(os.getpid(), signal.SIGTERM)


def hang_up_handler(getsignal=signal.getsignal):
    """SIGHUP's handler for this engine (hang_up_like_term, told which outputs are a terminal now), or None to leave
    SIGHUP as the engine was started with it: ignored, as nohup starts a `serve` so that it outlives its terminal (a
    handler set over that shut it down at the logout nohup was there for)."""
    if getsignal(signal.SIGHUP) is not signal.SIG_DFL:
        return None
    terminals = []
    for fd in (1, 2):
        try:
            if os.isatty(fd):
                terminals.append(fd)
        except OSError:
            continue
    return lambda signum=None, frame=None: hang_up_like_term(signum, frame, terminals=tuple(terminals))


def _not_json(constant):
    raise ValueError(f"{constant} is not JSON")


def _finite(text):
    """json.loads parse_float: a literal that overflows (1e999) would become inf, which JSON cannot carry."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is not a finite number")
    return value


def page_frames(records: list[dict]) -> list[dict]:
    """Frame records as pages get them (/runs/{id}/frames, the socket's frames event): each without the keys only the
    engine reads (engine/runs.py page_frame: ENGINE_FRAME_KEYS), as the run's `last` is (Run.public)."""
    return [page_frame(record) for record in records]


async def stain_body(request: Request):
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_STAIN_BYTES:
        raise RunError("This request is limited to 1 MB", 413)
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > MAX_STAIN_BYTES:
            raise RunError("This request is limited to 1 MB", 413)
        raw.extend(chunk)
    try:
        body = json.loads(raw or b"{}", parse_constant=_not_json, parse_float=_finite)
    except (ValueError, RecursionError):
        raise RunError("Send settings as JSON") from None
    if not isinstance(body, dict):
        raise RunError("Send settings as a JSON object")
    return body


async def json_body(request: Request, limit: int):
    """The body as JSON whatever its Content-Type says (a beacon's is text/plain), at most `limit` bytes:
    413 beyond (a body sent without Content-Length too), 422 for anything but JSON."""
    too_big = f"This is limited to {limit / 2 ** 20:g} MB" if limit >= 2 ** 20 else f"This is limited to {limit // 1024} KB"
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise RunError(too_big, 413)
    raw = bytearray()
    async for chunk in request.stream():
        raw += chunk
        if len(raw) > limit:
            raise RunError(too_big, 413)
    try:
        return json.loads(bytes(raw), parse_constant=_not_json)
    except (ValueError, RecursionError):
        raise RunError("Send JSON", 422) from None


def _log_key(key) -> str:
    """A report key as it may appear in the one-line log: whitespace and control characters (a newline would forge a
    line of its own) become _."""
    return re.sub(r"[\s\x00-\x1f\x7f]+", "_", str(key))


def telemetry_summary(report: dict, limit: int = TELEMETRY_LINE_CHARS) -> str:
    """One compact line for the engine log: the report's numbers, flags and (quoted, shortened, ASCII-escaped) strings
    as key=value, nested objects flattened as a.b (two levels), short lists of numbers inline ([1920,1080]), other lists
    and deeper objects by their length, floats to 3 decimals (no exponent below a million: a 12 s stall reads 12345.6).
    Every key at every level is cleaned (_log_key), so nothing a page sends can break the line."""
    parts = []

    def add(key, value, depth=0):
        if isinstance(value, dict) and depth < 2:
            for k, v in value.items():
                add(f"{key}.{_log_key(k)}", v, depth + 1)
        elif isinstance(value, dict):
            parts.append(f"{key}={{{len(value)}}}")
        elif isinstance(value, list) and len(value) <= 4 and all(v is None or isinstance(v, (int, float)) for v in value):
            parts.append(f"{key}={json.dumps(value, separators=(',', ':'))}")
        elif isinstance(value, list):
            parts.append(f"{key}=[{len(value)}]")
        elif isinstance(value, float):
            parts.append(f"{key}={round(value, 3):g}" if abs(value) < 1e6 else f"{key}={value:.3g}")
        else:
            parts.append(f"{key}={json.dumps(value[:60] if isinstance(value, str) else value)}")
    for key, value in report.items():
        if key != "t":
            add(_log_key(key), value)
    text = " ".join(parts)
    return text if len(text) <= limit else text[:limit - 1] + "…"


class TelemetryLog:
    """The page's telemetry reports, one JSON line each ({"t": the engine's UTC time, ...report}; a "t" sent by the
    page is kept as "page_t") in <state_dir>/logs/telemetry.jsonl, renamed to telemetry.jsonl.1 before it would pass
    `keep_bytes`; and a compact summary in the engine log at most once per `every_s` seconds."""

    def __init__(self, path, *, keep_bytes=TELEMETRY_KEEP_BYTES, every_s=TELEMETRY_LOG_EVERY_S, clock=time.monotonic,
                 log=None):
        self.path, self.keep_bytes, self.every_s, self.clock = Path(path), keep_bytes, every_s, clock
        self.log = log or (lambda message: print(message, flush=True))
        self.lock = threading.Lock()
        self.logged_at, self.unlogged = -math.inf, 0

    def write(self, report: dict) -> dict:
        record = {"t": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}
        for key, value in report.items():
            record["page_t" if key == "t" else key] = value
        line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        with self.lock:
            private_directory(self.path.parent)
            try:
                if self.path.stat().st_size + len(line) > self.keep_bytes:
                    portable.replace(self.path, self.path.with_name(self.path.name + ".1"))
            except FileNotFoundError:
                pass
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | portable.O_NOFOLLOW | portable.O_BINARY,
                         0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
            self.unlogged += 1
            now = self.clock()
            if now - self.logged_at >= self.every_s:
                count, self.logged_at, self.unlogged = self.unlogged, now, 0
                self.log(f"SoPA companion: telemetry at {record['t'][11:19]} UTC ({count} report{'s' if count > 1 else ''} "
                         f"since the last line): {telemetry_summary(record)}")
        return record


def fresh_token(cfg) -> str:
    """The bearer token of an engine with [server] token = "auto": new at every start, as config.example.toml says, and
    kept only in the private runtime file (the command line reads it there). It used to be kept in <state_dir>/run/
    session.key across restarts so that an open window stayed signed in, as the cookie was the token itself; the page
    has a key of its own now (server/security.py), so that file only kept an old secret on disk."""
    try:
        (Path(cfg["paths"]["state_dir"]) / "run" / "session.key").unlink(missing_ok=True)
    except OSError:
        pass
    return secrets.token_urlsafe(32)


# The type each of the page's own files is served with, by its ending. Never left to the computer: Python's mimetypes
# reads Windows' registry, where .js is often text/plain (an editor or an installer set it), and a browser runs no
# module served as text: the page would show its bare frame for ever, nothing started.
PAGE_FILE_TYPES = {".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
                   ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8",
                   ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".webp": "image/webp",
                   ".ico": "image/x-icon", ".wasm": "application/wasm", ".woff2": "font/woff2"}


class FreshStaticFiles(StaticFiles):
    """Revalidate the app's own files on every load so a stale module never breaks start-up, and serve each with the
    type its ending names (PAGE_FILE_TYPES), whatever the computer's own table of file types says."""

    def file_response(self, full_path, *args, **kwargs):
        response = super().file_response(full_path, *args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        kind = PAGE_FILE_TYPES.get(Path(full_path).suffix.lower())
        if kind is not None and "content-type" in response.headers:      # a 304 names no type
            response.headers["Content-Type"] = kind
        return response


# The largest image view a page may ask for (px across). A whole tissue on the widest desktop this app runs on (5120 px)
# fits at one image pixel per screen pixel; the page also stays under its GPU's texture size (frontend tiled-loader.mjs).
IMAGE_VIEW_MAX_PX = 6144


def create_app(config=None, *, token=None, shutdown=None, session=None):
    """`shutdown` (tests): instead of SIGTERM to this process. `session`: an API session's engine.session.Session
    ([session] mode = "api", server/api_serve.py), None on the desktop."""
    from .api_serve import attach_session, read_closing, session_health
    cfg = load_config(overrides=config)
    api = (cfg.get("session") or {}).get("mode") == "api"
    session_token = token or (fresh_token(cfg) if cfg["server"]["token"] == "auto" else cfg["server"]["token"])
    # The page's own key, which a launch code is traded for: new at each start, in memory only (server/security.py).
    page_key = secrets.token_urlsafe(32)
    # A session that ended without shutting down (engine/crashlog.py): its crash report comes first, before Runs below
    # removes the runs it left unsaved; then this session's first line in the engine log (a report finds it there).
    crashes = crashlog.CrashReports(cfg, secrets=(session_token, page_key, cfg["server"]["token"]))
    crashes.check()
    print(crashlog.starting_line(), flush=True)
    samples = Samples(cfg)
    zarr = ZarrSamples(cfg)
    # What went wrong that the engine noted itself (failed requests below, failed runs and saves), for the Log.
    noted = runlog.Noted()
    runs = Runs(cfg, samples, zarr, noted=noted, **({"session": session} if session is not None else {}))
    analysis = AnalysisManager(cfg, runs)
    # The mixing check: each sample's model (QcModels, GPU 0) and each run's results (QcManager), which is the analysis
    # scheduler's QC provider (it asks qc_model(run) and hands over every result through qc_result(record)).
    qc_models = QcModels(cfg, zarr)
    qc = QcManager(cfg, runs, qc_models, analysis=analysis)
    if hasattr(analysis, "attach_qc"):
        analysis.attach_qc(qc)
    # Go deeper: one cell's RNA explained by the run's model; the scheduler's
    # deeper provider (it attaches itself), its answers' caches and the WebSocket's "deeper" events.
    deeper = DeeperManager(cfg, runs, qc, qc_models, analysis)
    # Go deeper's "where it spills" (engine/deeper_where.py; its route below), which also finds the cell a test's corner
    # view is centred on; it tells the scheduler the box a followed cell's counted frames export (follow_where).
    from ..engine.deeper_where import DeeperWhere
    where = DeeperWhere(runs, analysis)
    # Test a setting here: the cell's pool and every reason a test cannot answer (engine/qc_pool.py), then the test's
    # runs, made, read and judged here (engine/qc_test.py); its corner view.
    pools = QcPools(runs, qc, analysis, where)
    qc_tests = QcTests(runs, qc, analysis, where=where, pools=pools)
    # Compare (engine/compare.py): a run's frame and the sample's other segmentations
    # read by its mixing check side by side, counted by the sopa helper and scored by the GPU worker.
    from ..engine.compare import CompareManager
    compare = CompareManager(runs, zarr, qc, analysis)
    telemetry_log = TelemetryLog(Path(cfg["paths"]["state_dir"]) / "logs" / "telemetry.jsonl")
    # The page's Log button (engine/runlog.py): what this engine knows, as text, the crash reports' secrets blanked.
    log_sources = runlog.Sources(cfg, runs, analysis, qc, qc_models, tuple(crashes.secrets), deeper=deeper, noted=noted)

    desktop = cfg.get("desktop", {})
    # Where the Cellpose GUI opens (engine/stain_patches.py display): on the desktop of the app's own Chrome window when
    # there is one (browser = "chrome"), else of this engine while a page is open in any browser. The engine never reads
    # which browser the app opens in.
    runs.stain.patches.window = AppWindow(chrome_profile(cfg))
    runs.stain.patches.page_here = lambda: app.state.presence.open() > 0 and browser.desktop_session(os.environ)
    dialog_timeout = float(desktop.get("save_dialog_timeout_s", 600))   # checked now: a bad value fails at start-up
    # Closing as the pages go (server/presence.py), its times checked now too.
    closing_times = {"grace": float(desktop.get("close_grace_s", 6)),
                     "pending_grace": float(desktop.get("close_grace_pending_s", 20)),
                     "away_stop_s": float(desktop.get("away_stop_s", 1800))}
    stop_engine = shutdown or portable.shut_down_this_process

    def closed_by_page():
        """The last page said goodbye and none came back: the engine shuts down, and a run that page did not answer
        for is discarded without the dialog (decide_closing page_closed)."""
        app.state.page_closed = True
        stop_engine()

    @asynccontextmanager
    async def lifespan(app):
        watcher = None
        if desktop.get("exit_when_closed", True):
            # Once the last page said goodbye: shut down. A page gone without one (away): its runs pause, and an engine
            # left away with nothing waiting for an answer stops after away_stop_s.
            watcher = asyncio.create_task(watch_pages(
                app.state.presence, **closing_times, pending=lambda: bool(runs.pending()), on_closed=closed_by_page,
                on_away=runs.pause_for_away, on_idle=stop_engine))
        # A signal's handler is set from the main thread only (uvicorn sets SIGINT and SIGTERM's); Windows has no SIGHUP.
        if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGHUP"):
            hang_up = hang_up_handler()
            if hang_up is not None:
                signal.signal(signal.SIGHUP, hang_up)
        # For a crash report: a crash in native code prints every thread's stack to the engine log (descriptor 2, which
        # follows its rotation: desktop/logs.py), and the marker says the engine is up until its shutdown has run to the
        # end (below): a session that ends any other way is reported at the next start (engine/crashlog.py).
        if not faulthandler.is_enabled():
            try:
                faulthandler.enable(file=2, all_threads=True)
            except (OSError, RuntimeError, ValueError):
                pass
        await asyncio.to_thread(crashes.mark_alive)
        yield
        # The page's answer to closing, taken as the shutdown begins; after POST /shutdown, the one it was taken as there
        # (its reply said which runs are saved: an answer that turned a minute old since must not change that).
        answered = app.state.shutdown_answer if app.state.stop_forced else app.state.closing.at_shutdown()
        if watcher is not None:
            watcher.cancel()
        try:                                         # a test's runs go before anything could save them as results
            await asyncio.to_thread(qc_tests.close)
        except Exception as exc:
            print(f"SoPA companion: stopping a setting's test failed: {type(exc).__name__}: {exc}", flush=True)
        try:                                         # then its pools (the ones the tests use: tests replace them)
            await asyncio.to_thread(qc_tests.pools.close)
        except Exception as exc:
            print(f"SoPA companion: stopping a test's pools failed: {type(exc).__name__}: {exc}", flush=True)
        try:                                         # comparisons stop (their counting helpers killed)
            await asyncio.to_thread(compare.close)
        except Exception as exc:
            print(f"SoPA companion: stopping a comparison failed: {type(exc).__name__}: {exc}", flush=True)
        # The mixing check stops first, before anything waits (the save dialog below can wait 10 minutes): a model fit
        # has nothing to save, and must neither go on nor start on GPU 0 (shared with other work) once the
        # app is closed.
        try:
            await asyncio.to_thread(qc.close)
        except Exception as exc:
            print(f"SoPA companion: stopping the mixing check failed: {type(exc).__name__}: {exc}", flush=True)
        try:
            await asyncio.to_thread(deeper.close)    # Go deeper's answers still queued for the disk
        except Exception as exc:
            print(f"SoPA companion: stopping Go deeper failed: {type(exc).__name__}: {exc}", flush=True)
        # Pending results include finished runs and stain runs. A leave or save already underway is decided. The page's
        # answer decides the runs it was given for; the dialog the others (none on stop --force, nor once the last page
        # said goodbye), else discard.
        forced = app.state.stop_forced
        decisions = {}
        try:
            unsaved = await asyncio.to_thread(runs.pending)
            if not (forced and runs.keep_unsaved):
                decisions = await asyncio.to_thread(decide_closing, unsaved, answered, forced=forced,
                                                    timeout_s=dialog_timeout, page_closed=app.state.page_closed)
        except Exception:          # no answer: discard, including a lost desktop or terminal
            pass
        # A test engine's stop --force finishes its runs into their folders, as before (it never writes into a sample at
        # close: Runs.close).
        test_add = runs.keep_unsaved and (forced or "add" in decisions.values())
        # The user's choice (save or discard the runs) is carried out whatever fails before it.
        try:
            await asyncio.to_thread(analysis.close)
        finally:
            try:
                await asyncio.to_thread(zarr.close)
            finally:
                await asyncio.to_thread(close_runs, "add" if test_add else "discard", decisions)
        await asyncio.to_thread(crashes.mark_closed)   # shut down to the end: the next start writes no crash report

    def close_runs(policy: str, decisions: dict):
        """Runs.close with each pending run's decision: {run id: "add" | "discard"} once one is added, which Runs.close
        takes for the runs named and discards any other run (one nobody answered for, as it became pending after the
        answers were taken); else "discard". A decided run is completed whatever it says. A test
        engine closes with `policy`."""
        if not runs.keep_unsaved and decisions:
            policy = dict(decisions) if "add" in decisions.values() else "discard"
        runs.close(policy)

    # No schema route either: outside /api/ the engine answers with the app's own files alone, which need no credential.
    app = FastAPI(title="SoPA companion", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.token, app.state.runs, app.state.samples, app.state.analysis = session_token, runs, samples, analysis
    app.state.zarr, app.state.telemetry = zarr, telemetry_log
    app.state.qc, app.state.qc_models, app.state.deeper = qc, qc_models, deeper
    app.state.qc_tests = qc_tests
    app.state.compare = compare
    app.state.crashes = crashes
    # The pages signed in now, how the last one went (server/presence.py), and whether the shutdown is the last page's
    # goodbye; a page's socket that names no usable id counts under a name of its own (socket-<n>), which only that
    # socket's own bye names.
    app.state.presence, app.state.page_closed = Presence(), False
    app.state.socket_ids = itertools.count(1)
    app.state.stop_forced = False        # set by POST /shutdown (sopa_companion stop --force)
    app.state.shutdown_answer = None     # ... with the page's answer as it took it (ClosingAnswer.at_shutdown)
    app.state.closing = ClosingAnswer()  # set by POST /closing (the page's popup or beacon)
    app.state.noted = noted
    app.state.page_key = page_key        # (tests sign a page in with it; it is never written anywhere)
    # A code for a window the app opens at once lives 45 s: it is traded as the window opens. One for a link a person
    # opens by hand lives 10 minutes: every code of an API session that opens no window itself (open "link", and
    # "none", whose handle shows app.url), and a code asked for as a link (?link=1: launch --print-url, serve in a
    # terminal).
    session_config = cfg.get("session") or {}
    link_s = session_config.get("link_code_ttl_s", LINK_CODE_S) if api else LINK_CODE_S
    app.state.launch_codes = LaunchCodes(ttl=link_s, limit=link_s, link_ttl=link_s) \
        if api and session_config.get("open") != "window" else LaunchCodes(ttl=LAUNCH_CODE_S, link_ttl=link_s)

    def signed_in(kind: str) -> None:
        """A page signed in: counted (presence's windows: the launcher and a notebook wait for the page of the browser
        they opened). A link signed a browser in: a notebook's session tells its kernel, which replaces the link it
        showed with words that hold no code (api/handle.py), so a notebook saved afterwards keeps none."""
        app.state.presence.signed_in(kind)
        if kind == "link" and app.state.session is not None:
            app.state.session.deliver(protocol.build_event("link", state="used"))

    app.add_middleware(LocalSecurity, token=session_token, port=cfg["server"]["port"],
                       hosts=cfg["server"]["allowed_hosts"], page_key=page_key, launch_codes=app.state.launch_codes,
                       on_sign_in=signed_in,
                       reopen="Open SoPA companion again from your notebook." if api else reopen_words(command_words()))
    base = "/api/v1"

    def failed(request: Request, text: str, status: int):
        """A request the engine failed (the page saw a refusal or a 500, and neither the engine log nor the
        Log's summary said so): one line in the engine log (a helper's own lines, its traceback, come before it:
        engine/zarr_samples.py helper) and an event the Log counts, the run's when the request is about one."""
        text = f"{request.method} {request.url.path} failed ({status}): {text}"
        # The same failure again, back to back (a page polling a route that keeps failing), is counted, not printed
        # again: one line per poll pushed the earlier lines out of the 150 the Log shows.
        if not noted.add(text, request.path_params.get("run_id")):
            return
        try:
            print(f"SoPA companion: {text}", flush=True)
        except (OSError, ValueError):               # no terminal any more: the event still counts
            pass

    @app.exception_handler(RunError)
    async def run_error(request, exc):
        if exc.status >= 500:
            failed(request, str(exc), exc.status)
        return problem(exc, exc.status, **exc.extra)
    @app.exception_handler(AnalysisError)
    async def analysis_error(request, exc): return problem(exc, exc.status)
    @app.exception_handler(QcError)
    async def qc_error(request, exc): return problem(exc, exc.status, **getattr(exc, "extra", {}))
    @app.exception_handler(SampleError)
    async def sample_error(request, exc):
        # A helper that failed (a sample that cannot be opened or read) went wrong whatever its status; other refusals
        # are answers (an unknown sample, a path that is not a zarr).
        if exc.status >= 500 or isinstance(exc, HelperFailed) and str(exc) != "Stopped":
            failed(request, str(exc), exc.status)
        return problem(exc, exc.status, **getattr(exc, "extra", {}))
    @app.exception_handler(FileNotFoundError)
    async def missing(request, exc): return problem("Requested file is unavailable", 404)
    @app.exception_handler(OSError)
    async def storage(request, exc):
        failed(request, f"Storage error: {exc}", 503)
        return problem(f"Storage error: {exc}", 503)
    @app.exception_handler(Exception)
    async def unexpected(request, exc):
        # Anything else: said here, then raised on to the server, which logs its traceback after this line.
        failed(request, f"{type(exc).__name__}: {exc}", 500)
        return problem(f"The engine could not do this ({type(exc).__name__}). The Log has the details.", 500)

    # health.session: {"mode": "desktop"}; an API session's health route puts its own beside these (api_serve.py).
    desktop_session = session_health(None)

    def older_state() -> dict:
        """The runs made before the one-run model that wait for an answer (Runs.older): {runs:
        [row], newest first, ...what else Runs says of them}; no rows on a test engine and in an API session."""
        value = runs.older()
        return dict(value) if isinstance(value, dict) else {"runs": list(value or [])}

    def older_or_none() -> dict | None:
        """older_state, or None when listing them failed (said once in the engine log and noted for the Log): /health
        must answer whatever happens there, or the launcher takes the engine for dead (desktop/launcher.py healthy)."""
        try:
            return older_state()
        except Exception as exc:
            text = f"listing the runs from earlier versions failed: {type(exc).__name__}: {exc}"
            if noted.add(text):
                print(f"SoPA companion: {text}", flush=True)
            return None

    def qc_missing() -> str | None:
        """What the mixing check is missing on this engine, None when nothing: "analysis" where Clusters & UMAP cannot
        run ([analysis] python), which score every frame (whatever [qc] says, nothing is scored without them), else why
        it cannot learn here, as its probe found (engine/qc.py capability: "no_gpu", "apple_gpu" (a Mac's GPU that
        failed its test), "torch_cpu", "driver", or "software" for its packages or [qc] python). The page has words for each (qc-copy.mjs UNAVAILABLE_REASONS)."""
        if not analysis.available:
            return "analysis"
        return None if qc_models.available else (qc_models.capability()["reason"] or "software")

    @app.get(base + "/health")
    def health():
        older = older_or_none()
        return {"ok": True, "version": __version__, "protocol": PROTOCOL,
                "proseg": Path(cfg["proseg"]["binary"]).is_file(),
                # What makes a missing proseg-live (locate.py: "build", "install" or "setting"), looked at at every
                # call: the engine keeps the place that fix puts the program, so a build or an install shows at a reload.
                "proseg_missing": None if Path(cfg["proseg"]["binary"]).is_file() else
                (((cfg.get("locate") or {}).get("proseg") or {}).get("fix") or "build"),
                # What makes a newer one ("build", "install", "setting": locate.program_fix), for the page's words when
                # it cannot keep proseg's own count (a run's proseg_counts.reason "old_proseg"); always there.
                "proseg_fix": locate.program_fix((cfg.get("locate") or {}).get("proseg")),
                "threads": cfg["proseg"]["threads"],
                "default_region_um": cfg["proseg"]["default_region_um"], "max_region_um": cfg["proseg"]["max_region_um"],
                "speeds": list(SPEEDS), "options": _options_module() is not None,
                "max_active_runs": runs.max_active, "one_run": runs.one_run, "leftovers": runs.leftovers,
                "older": {"runs": [], "done": [], "count": 0} if older is None else older, "session": desktop_session,
                # The methods this engine offers (StainRuns.methods): InstanSeg is listed only while it is switched on.
                "zarr": zarr.available, "analysis": analysis.available, "stain": {"available": runs.stain.available,
                    "methods": {m: runs.stain.method_available(m) for m in runs.stain.methods}},
                # qc.state: what the check's probe says ("checking" until its first answer, "ready", "unavailable",
                # "unknown": no answer, the check stays offered); qc.reason: what is missing, None when nothing.
                "qc": {"available": qc_models.available and analysis.available,
                       "state": qc_models.capability()["state"], "reason": qc_missing()},
                # The pages signed in (Presence.public): the launcher waits for a sign-in, status shows it.
                "presence": app.state.presence.public()}

    @app.get(base + "/stain/models")
    def stain_models(method: str = "cellpose"):
        # A method this engine does not offer (InstanSeg while it is switched off) is refused: 422 (StainRuns.offered).
        # `resolution`: the level a custom model's patches were cut at (StainRuns._model_labels), which the picker names.
        fields = ("name", "kind", "date", "sample", "source", "created", "resolution")
        return {"models": [{k: m.get(k) for k in fields} for m in runs.stain.models(method)]}

    @app.post(base + "/shutdown")
    def shutdown_engine(request: Request):
        """`sopa_companion stop --force`: stop without the desktop dialog. On a user engine the runs nobody answered for
        are discarded; a run being saved (a leave or save under way) is finished first, and so is a page's answer from
        the last minute for its runs. A test engine finishes its runs into their
        folders. Only with the command line's bearer token (the private runtime file's), never with the page's key.
        -> {discarding: [run id], saving: [run id]}, which the launcher's words use."""
        bearer = request.headers.get("authorization", "")
        if not bearer.startswith("Bearer ") or not hmac.compare_digest(bearer[7:].encode(), session_token.encode()):
            return problem("Only the command line (sopa_companion stop --force) can stop the engine", 403)
        # The page's answer as the shutdown carries it out (the lifespan takes this one, not the answer as it is by then).
        app.state.shutdown_answer = app.state.closing.at_shutdown()
        app.state.stop_forced = True
        pending = runs.pending()
        answered, _, covers = app.state.shutdown_answer or (None, 0.0, frozenset())
        saving = [r["id"] for r in pending if runs.keep_unsaved or r.get("decided") or answered == "add" and (
            covers is None or r["id"] in covers)]
        discarding = [r["id"] for r in pending if r["id"] not in saving]
        print(f"SoPA companion: stop requested from the command line: discarding {len(discarding)} run(s) nobody saved, "
              f"saving {len(saving)}, then exiting", flush=True)
        stop_engine()
        return {"discarding": discarding, "saving": saving}

    @app.post(base + "/closing")
    async def closing_answer(request: Request):
        """The page's answer to closing (api_serve.read_closing): {answer: "add" | "discard" | "withdraw" ("save" also
        means "add"), runs?: [the run ids the page showed], revision?: a whole number that grows with each choice in its
        window, page?: the window's id, bye?: true as the page goes for good, visit?: the page's showing}, as JSON
        whatever its Content-Type says (the page's keepalive request as it goes too; the same page key and Origin rules
        as every POST). Kept with its time and its runs (ClosingAnswer) until a page connects again or withdraws it; a
        shutdown within CLOSING_ANSWER_S carries it out for those runs without the dialog. An answer overtaken by a
        later one of its window, or said before its page was shown again (a page that connects again takes its answer
        back), changes nothing (`taken`: false). With bye it is also that showing's goodbye (server/presence.py).
        Answers as GET, with `taken`."""
        too_big = f"An answer to closing is limited to {CLOSING_MAX_BYTES // 1024} KB"
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > CLOSING_MAX_BYTES:
            return problem(too_big, 413)
        raw = bytearray()
        async for chunk in request.stream():         # also bounds a body sent without Content-Length
            raw += chunk
            if len(raw) > CLOSING_MAX_BYTES:
                return problem(too_big, 413)
        try:
            body = read_closing(raw)                 # whatever its Content-Type says
        except ValueError:
            return problem('Send {"answer": "add"}, {"answer": "discard"} or {"answer": "withdraw"}', 422)
        answer, named = body["answer"], body["runs"]
        # Kept before anything is awaited, so answers are kept in the order they came; then narrowed to the runs
        # pending.
        taken = 0 if app.state.presence.stale(body["page"], body["visit"]) else \
            app.state.closing.give(answer, named, body["revision"], body["page"])
        if body["bye"]:
            app.state.presence.goodbye(body["page"], "closing", body["visit"])
        unsaved = await asyncio.to_thread(runs.pending)
        if taken and answer != "withdraw":
            app.state.closing.cover(taken, [r["id"] for r in unsaved])
        covered = len(unsaved) if named is None else sum(r["id"] in named for r in unsaved)
        late = "(a later answer of that window came first: this one changes nothing)"
        if answer == "withdraw":
            said = "the page took its answer to closing back" if taken else f"the page took an answer back {late}"
        else:
            said = f"the page answered how to close: {answer} " + (
                f"({sum(not r.get('decided') for r in unsaved)} run(s) still sampling, {covered} answered for)" if taken
                else late)
        print(f"SoPA companion: {said}", flush=True)
        return {**app.state.closing.public(), "taken": bool(taken), "runs": unsaved}

    @app.get(base + "/closing")
    async def closing_state():
        """{answer, age_s, applies, covers, runs}: the page's answer to closing as kept (`applies`: a shutdown now
        carries it out, for the runs in `covers`) and pending results (Runs.pending)."""
        return {**app.state.closing.public(), "runs": await asyncio.to_thread(runs.pending)}

    # ---- runs made before the one-run model --------------------------------------------------------------------------
    def older_answer(body, *answers) -> str:
        answer = body.get("answer") if isinstance(body, dict) else None
        if answer not in answers:
            raise RunError("Choose Save all, Discard all or Later" if "later" in answers else
                           "Choose Save to the sample or Discard")
        if runs.closing and answer != "later":
            raise RunError("The engine is shutting down", 503)
        return answer

    @app.get(base + "/runs/older")
    def runs_older():
        """{runs: [row], ...}: the runs from earlier versions never saved to their samples (Runs.older), newest first,
        the rows set aside by Later included (row.later; `count` leaves them out)."""
        return older_state()

    @app.post(base + "/runs/older")
    def runs_older_answer(body: dict = Body(...)):
        """{answer: "add" | "discard" | "later"}: Save all to their samples, Discard all, or Later (Runs.answer_all_older):
        Save all and Discard all go one at a time, in the engine, never blocking new runs; Later sets aside, for this
        engine session, every row not under way (they stay on disk and are listed again at the next start) and stops a
        Save all after the row under way, also while the engine closes. Progress and the rows' ends come as {ev:
        "older", runs, ...} pushes. Answers as GET."""
        runs.answer_all_older(older_answer(body, "add", "discard", "later"))
        return older_state()

    @app.post(base + "/runs/older/{run_id}")
    def run_older_answer(run_id: str, body: dict = Body(...)):
        """{answer: "add" | "discard"} for one row (Runs.answer_older): Save to the sample (never as the sample's
        segmentation) or Discard, carried out in the engine. 404 for a run that is not listed. Answers the row (as
        Runs gives it), else as GET."""
        row = runs.answer_older(run_id, older_answer(body, "add", "discard"))
        return row if isinstance(row, dict) else older_state()

    @app.post(base + "/telemetry")
    async def telemetry(request: Request):
        """A performance report from the page: any JSON object up to 16 KB (see TelemetryLog)."""
        too_big = f"A telemetry report is limited to {TELEMETRY_MAX_BYTES // 1024} KB"
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > TELEMETRY_MAX_BYTES:
            return problem(too_big, 413)
        raw = bytearray()
        async for chunk in request.stream():         # also bounds a body sent without Content-Length
            raw += chunk
            if len(raw) > TELEMETRY_MAX_BYTES:
                return problem(too_big, 413)
        try:
            report = json.loads(bytes(raw), parse_constant=_not_json, parse_float=_finite)
            # What is kept must be text UTF-8 can carry: a lone surrogate (\ud800) is refused here, not a 500 later.
            json.dumps(report, ensure_ascii=False, allow_nan=False).encode()
        except (ValueError, RecursionError):        # UnicodeEncodeError is a ValueError
            report = None
        if not isinstance(report, dict):
            return problem("A telemetry report must be a JSON object of finite numbers and text", 422)
        await asyncio.to_thread(telemetry_log.write, report)
        return Response(status_code=204)

    def log_text(run_id: str | None):
        return Response(runlog.build(log_sources, run_id), media_type="text/plain; charset=utf-8",
                        headers={"Cache-Control": "no-store"})

    @app.get(base + "/log")
    def session_log():
        """The Log button with no run open: this engine session as plain text (engine/runlog.py)."""
        return log_text(None)

    @app.get(base + "/runs/{run_id}/log")
    def run_log(run_id: str):
        """The Log button: this engine session and everything it knows of the run, as plain text (engine/runlog.py)."""
        runs.get(run_id)                           # 404 for an unknown run
        return log_text(run_id)

    # ---- crash reports (engine/crashlog.py): a session that ended unexpectedly, offered by the page at its next start
    @app.get(base + "/crash")
    def crash_newest():
        """{report: null | {id, at, reason, bytes, started}}: the newest crash report the page has not offered yet."""
        return {"report": crashes.newest()}

    @app.get(base + "/crash/{report_id}.txt")
    def crash_text(report_id: str):
        """A crash report as plain text, by the id GET /crash gave (a report's own name: nothing else is read here)."""
        text = crashes.text(report_id)
        if text is None:
            return problem("No such crash report", 404)
        return Response(text, media_type="text/plain; charset=utf-8", headers={"Cache-Control": "no-store"})

    @app.post(base + "/crash/page")
    async def crash_page(request: Request):
        """The page found a page that did not unload ({at: when that page started, session: its id, found: "tab" (the
        page before it in the same window) | "flag" (its flag alone; such a report is kept quietly)}): that session's
        crash report, written once; answers as GET /crash."""
        body = await json_body(request, CRASH_PAGE_MAX_BYTES)
        at, session = (body.get("at"), body.get("session")) if isinstance(body, dict) else (None, None)
        if not (isinstance(at, str) and isinstance(session, str) and session.strip()):
            return problem('Send {"at": when that page started, "session": its id}', 422)
        found = body.get("found") if body.get("found") in ("tab", "flag") else None
        return {"report": await asyncio.to_thread(crashes.page, at, session, found)}

    @app.post(base + "/crash/{report_id}/seen")
    def crash_seen(report_id: str):
        """The page offered the report (Save a copy… or Not now): it is not offered again."""
        if not crashes.seen(report_id):
            return problem("No such crash report", 404)
        print(f"SoPA companion: the page offered crash report {report_id}", flush=True)
        return {}

    @app.post(base + "/page-log")
    async def page_log(request: Request):
        """The page's own log ({text, session, at, pagehide: true for its goodbye, visit: the showing that says it}:
        JSON, the goodbye a keepalive request with the page's key), kept for a crash report: its last 512 KB, per page
        session (<state_dir>/logs/page-logs/; the newest also as logs/page-log.txt). The one sent with pagehide is that
        showing's goodbye (server/presence.py): its session is the id its socket's address names."""
        body = await json_body(request, PAGE_LOG_MAX_BYTES)
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            return problem('Send {"text": the page\'s log, "session": its id, "at": when it was sent}', 422)
        await asyncio.to_thread(crashes.save_page_log, text, body.get("session"), body.get("at"),
                                body.get("pagehide") is True)
        if body.get("pagehide") is True:
            app.state.presence.goodbye(body.get("session"), "page-log", visit_number(body.get("visit")))
        return {}

    @app.get(base + "/options")
    def options():
        module = _options_module()
        return module.OPTIONS if module is not None else []

    @app.get(base + "/options/guide")
    def options_guide():
        module = _options_module()
        return module.GUIDE if module is not None else []

    def parse_box(box: str | None, limit: float | None = None):
        if box is None:
            return None
        try:
            x0, y0, x1, y1 = (float(v) for v in box.split(","))
        except ValueError:
            raise SampleError("box must be x0,y0,x1,y1 in µm") from None
        if not (x1 > x0 and y1 > y0):
            raise SampleError("The box is empty")
        if limit is not None and max(x1 - x0, y1 - y0) > limit:
            raise SampleError("Zoom in further")
        return [x0, y0, x1, y1]

    @app.get(base + "/samples")
    def sample_list():
        return zarr.list()

    @app.get(base + "/browse")
    def browse(path: str | None = None):
        return zarr.browse(path)

    def qc_hook(hook, value):
        """The mixing check follows what happened; whatever fails there never fails the request itself."""
        try:
            hook(value)
        except Exception as exc:
            print(f"SoPA companion: mixing check: {type(exc).__name__}: {exc}", flush=True)

    @app.post(base + "/samples/open")
    def sample_open(body: dict = Body(...)):
        detail = zarr.open(str(body.get("path") or ""))
        if has_transcripts(detail):                 # [qc] learn = "open": learn it now if it has no model yet; a sample
            qc_hook(qc.sample_opened, detail)       # of images only has nothing to learn
        return detail

    @app.delete(base + "/samples/{sample_id}")
    def sample_forget(sample_id: str):
        zarr.forget(sample_id)
        return {"forgotten": sample_id}

    # ---- a sample the routes answer for --------------------------------------------------------------------
    def run_sample(sample_id: str, store=None) -> bool:
        """Whether the sample routes answer for a sample: one the engine knows, or (forgotten since) a run's, opened
        again where the run found it, as its image and prior shapes are, and left off the open dialog's list (the
        detail said "open it again" while the layer routes opened it again, putting it back on top of the list).
        `store`: the samples to ask (runs.zarr for the stain routes)."""
        store = store if store is not None else zarr
        if store.known(sample_id):
            return True
        found = next((r for r in runs.list() if r.get("sample") == sample_id and r.get("sample_kind") == "zarr"
                      and r.get("sample_path")), None)
        if found is None:
            return False
        store.open(found["sample_path"], recent=False)
        return True

    def known_sample(sample_id: str, store=None):
        if not run_sample(sample_id, store):
            raise SampleError("Unknown sample: open it again", 404)

    @app.get(base + "/samples/{sample_id}")
    def sample_detail(sample_id: str):
        if run_sample(sample_id):
            return zarr.detail(sample_id)
        value = dict(samples.detail(sample_id))      # a Xenium bundle (runs made by older versions)
        value["overview"] = samples.overview_meta(sample_id) if value.get("channels") else None
        return value

    @app.get(base + "/samples/{sample_id}/overview.png")
    def sample_overview(sample_id: str):
        path = zarr.overview_path(sample_id) if run_sample(sample_id) else samples.overview(sample_id)
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @app.get(base + "/samples/{sample_id}/shapes/{key}")
    def sample_shapes(sample_id: str, key: str, box: str | None = None, limit: int = 20000):
        """The picker's prior outlines: JSON built in a worker process and sent as bytes (engine/shape_query.py), so a
        whole sample's box never stalls the event loop (the health check, WebSocket pushes, frames)."""
        known_sample(sample_id)
        body = zarr.shapes_json(sample_id, key, parse_box(box), max(0, min(limit, 400_000)))
        return Response(body, media_type="application/json")

    # ---- segmentation layers (engine/zarr_samples.py) ------------------------------
    def segmentation_sample(sample_id: str):
        """A sample the segmentation routes answer for (known_sample): one the engine knows, or (forgotten since) a
        run's, opened again where the run found it, as its image and prior shapes are."""
        known_sample(sample_id)

    @app.get(base + "/samples/{sample_id}/segmentations")
    def sample_segmentations(sample_id: str):
        """{segmentations: [{key, name, source, model, count, extent, table, coverage, version, count_by,
        count_table, outlines}]}: every cell-like shapes element (polygons, more than one), named by what made it when
        the zarr says so (describe_segmentation), Xenium's first; count_by and count_table: whose count Compare reads
        for it (zarr_samples.count_source); `outlines` is its outlines' URL at its current version."""
        segmentation_sample(sample_id)
        listed = zarr.segmentations(sample_id)
        for entry in listed:
            # A version read as UNREAD (its folder was briefly unreadable) names no file: no ?v= for the browser to keep.
            entry["outlines"] = f"{base}/samples/{sample_id}/segmentations/{entry['key']}/outlines" + (
                f"?v={entry['version']}" if entry.get("version") not in (None, UNREAD) else "")
        return {"segmentations": listed}

    @app.get(base + "/samples/{sample_id}/segmentations/{key}/outlines")
    def sample_segmentation_outlines(sample_id: str, key: str, request: Request, v: str | None = None):
        """The element's every outline as the prior vectors binary with the cells layer only (decodeVectors reads it),
        sent as the zstd file it is cached as: immutable when `v` is its current version (the catalogue's URL), else
        revalidated (an element saved again under the same key changes it). 404 not a segmentation of the sample."""
        segmentation_sample(sample_id)
        path, version = zarr.segmentation_outlines(sample_id, key)
        return encoded(path, request, immutable=v is not None and v != UNREAD and v == version)

    @app.get(base + "/samples/{sample_id}/tissue")
    def sample_tissue(sample_id: str, image: str | None = None):
        """The tissue stain runs and training patches stay inside: region_of_interest, or detected in memory."""
        if runs.zarr is None or not run_sample(sample_id, runs.zarr):
            raise SampleError("Unknown sample: open it again", 404)
        return runs.zarr.tissue(sample_id, image)

    @app.get(base + "/samples/{sample_id}/cellpose-settings")
    def sample_cellpose_settings(sample_id: str):
        """The settings the Cellpose GUI saved with the newest corrected patch of this sample."""
        if runs.zarr is None or not run_sample(sample_id, runs.zarr):
            raise SampleError("Unknown sample: open it again", 404)
        return runs.stain.gui_settings(sample_id)

    @app.post(base + "/samples/{sample_id}/patches/preview")
    def sample_patch_preview(sample_id: str, body: dict = Depends(stain_body)):
        """The boxes Training patches would start with for the picker's region and settings, checked as creating the
        item is; no item or folder is made."""
        return runs.stain.patches.preview(sample_id, body.get("region"), body.get("stain"))

    @app.get(base + "/samples/{sample_id}/image/{key}/c{channel}.u16")
    def sample_image(sample_id: str, key: str, channel: int, box: str, request: Request, px: int = 2048):
        """One channel over the box from the pyramid level that fits `px` pixels: raw little-endian uint16, zstd."""
        known_sample(sample_id)
        plane, meta = zarr.read_view(sample_id, key, channel, parse_box(box), max(64, min(px, IMAGE_VIEW_MAX_PX)))
        return raw_u16(plane, meta, request)

    def raw_u16(plane, meta: dict, request: Request):
        import json as _json
        import zstandard
        headers = {"Cache-Control": "private, max-age=3600", "X-Image": _json.dumps(meta), "Vary": "Accept-Encoding",
                   "Access-Control-Expose-Headers": "X-Image"}
        data = plane.tobytes()
        if "zstd" in request.headers.get("accept-encoding", "").lower():
            return Response(zstandard.ZstdCompressor(level=1, threads=-1).compress(data), media_type="application/octet-stream",
                            headers={**headers, "Content-Encoding": "zstd"})
        return Response(data, media_type="application/octet-stream", headers=headers)

    @app.get(base + "/runs")
    def run_list():
        return runs.list()

    @app.post(base + "/runs")
    def run_create(body: dict = Body(...)):
        # Every run follows proseg's own schedule and starts paused on its prior: "schedule" and "paused", which older
        # pages sent, are ignored.
        meta = runs.create(str(body.get("sample", "")), body.get("box"), body.get("options") or {},
                           region=body.get("region"), prior=body.get("prior"),
                           kind=body.get("kind", "proseg"), stain=body.get("stain"))
        if meta.get("kind") not in ("stain", "patches"):
            qc_hook(qc.run_created, meta)          # stain runs have no RNA counts to check or cluster
            try:                                   # it starts with the Clusters & UMAP settings last applied
                analysis.run_created(meta)
            except Exception as exc:               # never the request's failure: the run keeps the defaults
                print(f"SoPA companion: Clusters & UMAP settings: {type(exc).__name__}: {exc}", flush=True)
        return meta

    @app.post(base + "/runs/{run_id}/zarr")
    def run_save_zarr(run_id: str, body: dict = Body(...)):
        """{shapes_key, table_key, set_default?, overwrite?}: without set_default, the run's own default (its save
        defaults say it: never for a stain run of part of the sample). A name clash answers 409 {detail, error, clash,
        existing} (engine/runs.py save_to_zarr)."""
        set_default = body.get("set_default")
        return runs.save_to_zarr(run_id, body.get("shapes_key"), body.get("table_key"),
                                 None if set_default is None else bool(set_default), bool(body.get("overwrite", False)))

    @app.get(base + "/runs/{run_id}")
    def run_detail(run_id: str):
        # A run that can be saved has its save defaults worked out against the sample as it is now: the Save dialog asks
        # each time it opens.
        runs.current_save_defaults(run_id)
        return qc_tests.detail(runs.get(run_id).public())     # with "qc_test" {state, progress} while it has a test

    @app.get(base + "/runs/{run_id}/stain/vectors")
    def stain_vectors(run_id: str, request: Request, box: str | None = None):
        run = runs.get(run_id)
        if run.meta.get("kind") != "stain":
            raise RunError("This is not an image run", 404)
        return packed(runs.stain.outline_bytes(run, parse_box(box)), request, {}, immutable=False)

    @app.get(base + "/runs/{run_id}/stain/exclusions")
    def stain_exclusions(run_id: str):
        run = runs.get(run_id)
        if run.meta.get("kind") != "stain":
            raise RunError("This is not an image run", 404)
        return runs.stain.exclusions.public(run)

    @app.post(base + "/runs/{run_id}/stain/exclusions")
    def stain_exclusion_add(run_id: str, body: dict = Depends(stain_body)):
        return change_exclusions(run_id, body)

    def change_exclusions(run_id, body, *args, **kwargs):
        run = runs.get(run_id)
        with run.lock:
            if run.meta.get("leaving"):
                raise RunError("Wait for this run to finish leaving, or press Stay.", 409)
            return runs.stain.exclusions.change(run, body, *args, **kwargs)

    @app.post(base + "/runs/{run_id}/stain/exclusions/size")
    def stain_exclusion_size(run_id: str, body: dict = Depends(stain_body)):
        return change_exclusions(run_id, body, size=True)

    @app.post(base + "/runs/{run_id}/stain/exclusions/brightness")
    def stain_exclusion_brightness(run_id: str, body: dict = Depends(stain_body)):
        return change_exclusions(run_id, body, brightness=True)

    @app.get(base + "/runs/{run_id}/stain/cells")
    def stain_cells(run_id: str, request: Request, set: str, channel: str | None = None):
        return packed(runs.stain.exclusions.cell_bytes(runs.get(run_id), set, channel), request, {}, immutable=False)

    @app.get(base + "/runs/{run_id}/stain/brightness")
    def stain_brightness(run_id: str, channel: str | None = None):
        return JSONResponse(runs.stain.exclusions.brightness(runs.get(run_id), channel),
                            headers={"Cache-Control": "no-store"})

    @app.delete(base + "/runs/{run_id}/stain/exclusions/{area_id}")
    def stain_exclusion_remove(run_id: str, area_id: int, version: int):
        return change_exclusions(run_id, {"version": version}, area_id)

    @app.get(base + "/runs/{run_id}/stain/patches")
    def stain_patches(run_id: str):
        return JSONResponse(runs.stain.patches.status(runs.get(run_id)), headers={"Cache-Control": "no-store"})

    @app.post(base + "/runs/{run_id}/stain/patches")
    def stain_patch_boxes(run_id: str, body: dict = Depends(stain_body)):
        return runs.stain.patches.change(runs.get(run_id), body)

    @app.post(base + "/runs/{run_id}/stain/patches/propose")
    def stain_patch_propose(run_id: str, body: dict = Depends(stain_body)):
        return runs.stain.patches.change(runs.get(run_id), body, propose=True)

    @app.post(base + "/runs/{run_id}/stain/patches/export")
    def stain_patch_export(run_id: str, body: dict = Depends(stain_body)):
        return runs.stain.patches.export(runs.get(run_id), body)

    @app.post(base + "/runs/{run_id}/stain/patches/open")
    def stain_patch_open(run_id: str, body: dict = Depends(stain_body)):
        return runs.stain.patches.open_gui(runs.get(run_id), body)

    @app.get(base + "/runs/{run_id}/stain/patches/folders")
    def stain_patch_folders(run_id: str, path: str | None = None):
        # The export dialog's view of a folder (any path answers; its problems are in `refused`). Nothing is written.
        return JSONResponse(runs.stain.patches.folder_listing(runs.get(run_id), path),
                            headers={"Cache-Control": "no-store"})

    @app.get(base + "/runs/{run_id}/frames")
    def run_frames(run_id: str, since: int = 0):
        run = runs.get(run_id)
        with run.lock:
            records = run.frames[max(0, since):]
        return {"from": max(0, since), "frames": page_frames(records)}

    def encoded(path: Path, request: Request, immutable: bool = True):
        """A zstd/gzip file sent as-is with Content-Encoding (the browser inflates it), or inflated for clients that
        do not accept that encoding."""
        if not path.exists():
            raise FileNotFoundError(path)
        headers = {"Cache-Control": "private, max-age=31536000, immutable" if immutable else "private, no-cache",
                   "Vary": "Accept-Encoding"}
        encoding = "zstd" if path.suffix == ".zst" else "gzip"
        if encoding in request.headers.get("accept-encoding", "").lower():
            return FileResponse(path, media_type="application/octet-stream", headers={**headers, "Content-Encoding": encoding})
        import gzip
        import zstandard
        raw = path.read_bytes()
        data = zstandard.ZstdDecompressor().decompressobj().decompress(raw) if encoding == "zstd" else gzip.decompress(raw)
        return Response(data, media_type="application/octet-stream", headers=headers)

    @app.get(base + "/runs/{run_id}/frames/{seq}")
    def run_frame(run_id: str, seq: int, request: Request):
        return encoded(runs.frame_path(run_id, seq), request)

    def packed(data: bytes, request: Request, headers: dict, immutable: bool = True):
        """Raw bytes, zstd-encoded for clients that accept it (fetch() always sees raw bytes)."""
        import zstandard
        headers = {"Cache-Control": "private, max-age=31536000, immutable" if immutable else "private, no-cache",
                   "Vary": "Accept-Encoding", "Access-Control-Expose-Headers": ", ".join(headers), **headers}
        if "zstd" in request.headers.get("accept-encoding", "").lower():
            return Response(zstandard.ZstdCompressor(level=1, threads=-1).compress(data),
                            media_type="application/octet-stream", headers={**headers, "Content-Encoding": "zstd"})
        return Response(data, media_type="application/octet-stream", headers=headers)

    def raster_window(geometry: dict, box: list[float], stride: int, max_px: int):
        """Raster columns/rows covering the box, the stride that keeps it under max_px, snapped to the stride."""
        import math
        voxel, nx, ny = float(geometry["voxel"]), int(geometry["nx"]), int(geometry["ny"])
        c0 = max(0, math.floor((box[0] - geometry["x0"]) / voxel))
        r0 = max(0, math.floor((box[1] - geometry["y0"]) / voxel))
        c1 = min(nx, math.ceil((box[2] - geometry["x0"]) / voxel))
        r1 = min(ny, math.ceil((box[3] - geometry["y0"]) / voxel))
        if c1 <= c0 or r1 <= r0:
            raise RunError("The box is outside the run", 404)
        stride = max(1, int(stride), math.ceil(math.sqrt((c1 - c0) * (r1 - r0) / max(1, max_px))))
        return (c0 // stride) * stride, (r0 // stride) * stride, c1, r1, stride

    dense_cache: dict = {}

    def dense_planes(path: Path, frame: dict):
        """consensus and last_change of a dense frame (decoded once, the latest few kept)."""
        import gzip
        import numpy as np
        import zstandard
        key = str(path)
        if key not in dense_cache:
            raw = path.read_bytes()
            data = zstandard.ZstdDecompressor().decompressobj().decompress(raw) if path.suffix == ".zst" else gzip.decompress(raw)
            values = np.frombuffer(data, "<u4")
            plane = frame["nx"] * frame["ny"]
            dense_cache[key] = {"consensus": values[:plane].reshape(frame["ny"], frame["nx"]),
                                "last_change": values[plane:2 * plane].reshape(frame["ny"], frame["nx"])}
            while len(dense_cache) > 6:
                dense_cache.pop(next(iter(dense_cache)))
        return dense_cache[key]

    @app.get(base + "/runs/{run_id}/frames/{seq}/view")
    def run_frame_view(run_id: str, seq: int, box: str, request: Request, stride: int = 1, max_px: int = 2_400_000,
                       planes: str = "consensus,last_change"):
        """Labels of a frame over the box, sampled every `stride` columns (raised to stay under max_px): uint32 planes
        back to back (consensus, last_change); geometry in the X-View header."""
        import json as _json
        names = [p for p in planes.split(",") if p]
        if not names or any(p not in ("consensus", "last_change") for p in names):
            raise RunError("planes: consensus and/or last_change")
        frame = runs.frame_record(run_id, seq)
        path = runs.frame_path(run_id, seq)
        c0, r0, c1, r1, step = raster_window(frame, parse_box(box), stride, max(4096, min(max_px, 16_000_000)))
        if frame.get("format") == "tiles":
            container = Container(path)
            arrays = [container.compose(name, c0, r0, c1, r1, step) for name in names]
        else:
            source = dense_planes(path, frame)
            arrays = [source[name][r0:r1:step, c0:c1:step] for name in names]
        voxel = float(frame["voxel"])
        meta = {"seq": seq, "iter": frame.get("iter"), "x0": frame["x0"] + c0 * voxel, "y0": frame["y0"] + r0 * voxel,
                "nx": int(arrays[0].shape[1]), "ny": int(arrays[0].shape[0]), "stride": step, "voxel": voxel * step,
                "planes": names}
        return packed(b"".join(a.astype("<u4", copy=False).tobytes() for a in arrays), request, {"X-View": _json.dumps(meta)})

    @app.get(base + "/runs/{run_id}/frames/{seq}/window")
    def run_frame_window(run_id: str, seq: int, request: Request):
        """Per-layer labels of the window a tiled frame carries (the view at that iteration), or of the paused state
        after the view moved: uint32 layers[nz][ny][nx]; geometry in X-Window."""
        import json as _json
        import zstandard
        run = runs.get(run_id)
        frame = runs.frame_record(run_id, seq)
        if frame.get("format") != "tiles":
            raise RunError("Dense frames carry every layer", 404)
        side = run.window
        if side and side.get("seq") == seq and re.fullmatch(r"window-[0-9]+\.bin\.zst", str(side.get("file", ""))) \
                and (run.live / side["file"]).exists():
            data = zstandard.ZstdDecompressor().decompressobj().decompress((run.live / side["file"]).read_bytes())
            info, source, origin = side, "paused", (side["x0"], side["y0"])
        else:
            found = Container(runs.frame_path(run_id, seq)).window()
            if found is None:
                raise RunError("No window in this frame: zoom in while the run is going, or pause", 404)
            info, layers = found
            data, source, origin = layers.tobytes(), "frame", (frame["x0"], frame["y0"])
        voxel = float(frame["voxel"])
        meta = {"seq": seq, "iter": frame.get("iter"), "x0": origin[0] + info["i0"] * voxel, "y0": origin[1] + info["j0"] * voxel,
                "nx": info["nx"], "ny": info["ny"], "nz": info["nz"], "voxel": voxel, "source": source}
        return packed(data, request, {"X-Window": _json.dumps(meta)}, immutable=False)

    @app.get(base + "/runs/{run_id}/prior/view")
    def run_prior_view(run_id: str, box: str, request: Request, stride: int = 1, max_px: int = 2_400_000):
        """Tiled runs: proseg's prior projected to 2-D over the box: uint32 label then permille; X-View geometry."""
        import json as _json
        run = runs.get(run_id)
        prior = (run.info or {}).get("prior") or {}
        if prior.get("format") != "tiles":
            raise RunError("This run's prior is dense: GET /runs/{id}/prior", 404)
        container = Container(run.live / "prior.tiles")
        c0, r0, c1, r1, step = raster_window(prior, parse_box(box), stride, max(4096, min(max_px, 16_000_000)))
        arrays = [container.compose(name, c0, r0, c1, r1, step) for name in ("label", "permille")]
        voxel = float(prior["voxel"])
        meta = {"x0": prior["x0"] + c0 * voxel, "y0": prior["y0"] + r0 * voxel, "nx": int(arrays[0].shape[1]),
                "ny": int(arrays[0].shape[0]), "stride": step, "voxel": voxel * step, "planes": ["label", "permille"],
                "nucleus_prob": prior.get("nucleus_prob"), "cell_prob": prior.get("cell_prob")}
        return packed(b"".join(a.tobytes() for a in arrays), request, {"X-View": _json.dumps(meta)})

    @app.get(base + "/runs/{run_id}/transcripts/view")
    def run_transcripts_view(run_id: str, box: str, request: Request, limit: int = 3_000_000):
        """The modelled transcripts inside the box (16-byte records as /transcripts), 413 when more than `limit`."""
        run = runs.get(run_id)
        if not (run.live / "transcripts.bin.gz").exists():
            raise RunError("Proseg has not written the transcripts yet", 404)
        index = runs.transcript_index(run_id)      # kept by Runs, which lets go of it when the run is deleted
        records, count = index.query(parse_box(box), max(1, min(limit, 8_000_000)))
        if records is None:
            raise RunError(f"More than {max(1, min(limit, 8_000_000)):,} transcripts in view: zoom in to show them", 413)
        return packed(records.tobytes(), request, {"X-Count": str(count)})

    @app.get(base + "/runs/{run_id}/image/view/c{channel}.u16")
    def run_image_view(run_id: str, channel: int, box: str, request: Request, px: int = 2048):
        """Tiled runs: one channel over the box from the sample's image pyramid (raw uint16; X-Image geometry)."""
        run = runs.get(run_id)
        if run.meta.get("sample_kind") != "zarr":
            raise RunError("Only runs on a SpatialData sample stream their image", 404)
        if not zarr.known(run.meta["sample"]):
            zarr.open(run.meta["sample_path"], recent=False)     # the run's sample, left off the open dialog's list
        key = (run.meta.get("image") or {}).get("image")
        plane, meta = zarr.read_view(run.meta["sample"], key, channel, parse_box(box), max(64, min(px, IMAGE_VIEW_MAX_PX)))
        return raw_u16(plane, meta, request)

    @app.post(base + "/runs/{run_id}/view")
    def run_view(run_id: str, body: dict = Body(...)):
        return runs.set_view(run_id, body.get("box"))

    @app.get(base + "/runs/{run_id}/cellstats/{seq}")
    def run_cellstats(run_id: str, seq: int, request: Request):
        """Per-cell statistics of a frame (see native/proseg/src/live.rs): 7 x ncells little-endian u32/f32."""
        run = runs.get(run_id)
        with run.lock:
            if not 0 <= seq < len(run.frames) or "cells_file" not in run.frames[seq]:
                raise RunError("No cell statistics for this frame", 404)
            name = run.frames[seq]["cells_file"]
        if not re.fullmatch(r"frames/[0-9]{6,9}\.cells\.zst", name):
            raise RunError("Malformed frame index", 500)
        return encoded(run.live / name, request)

    @app.get(base + "/runs/{run_id}/counted/{seq}")
    def run_counted(run_id: str, seq: int, request: Request):
        """Proseg's own count of each cell in a counted frame (its sidecar, as proseg-live writes it): f32 LE x ncells,
        each the sum of the cell's averaged count over its genes (frontend/src/cellstats.mjs decodeCounted)."""
        run = runs.get(run_id)
        with run.lock:
            if not 0 <= seq < len(run.frames) or "counted_file" not in run.frames[seq]:
                raise RunError("No proseg count for this frame", 404)
            name = run.frames[seq]["counted_file"]
        if not re.fullmatch(r"frames/[0-9]{6,9}\.counted\.zst", str(name)):
            raise RunError("Malformed frame index", 500)
        return encoded(run.live / name, request)

    @app.get(base + "/runs/{run_id}/cellhistory")
    def run_cellhistory(run_id: str, cell: int, points: int = 400):
        """One cell's area/transcripts across the run (at most `points` frames, evenly spaced): instant sparklines."""
        import numpy as np
        import zstandard
        run = runs.get(run_id)
        ncells = (run.info or {}).get("cells", 0)
        if not 0 <= cell < ncells:
            raise RunError("Unknown cell", 404)
        with run.lock:
            frames = [f for f in run.frames if "cells_file" in f]
        step = max(1, len(frames) // max(points, 1))
        picked = frames[::step] + ([frames[-1]] if frames and frames[-1] is not frames[::step][-1] else [])
        decoder = zstandard.ZstdDecompressor()
        out = {"seq": [], "iter": [], "area_um2": [], "voxels": [], "transcripts": [], "foreground": []}
        for f in picked:
            stats = np.frombuffer(decoder.decompressobj().decompress((run.live / f["cells_file"]).read_bytes()), "<u4")
            stats = stats.reshape(7, ncells)
            out["seq"].append(f["seq"]); out["iter"].append(f["iter"])
            out["area_um2"].append(float(stats[0, cell]) * f["voxel"] ** 2); out["voxels"].append(int(stats[1, cell]))
            out["transcripts"].append(int(stats[2, cell])); out["foreground"].append(int(stats[3, cell]))
        return out

    @app.get(base + "/runs/{run_id}/prior")
    def run_prior(run_id: str, request: Request):
        return encoded(runs.get(run_id).live / "prior.bin.zst", request)

    @app.get(base + "/runs/{run_id}/xenium")
    def run_xenium(run_id: str, request: Request, fmt: str | None = Query(None, alias="format"), v: int | None = None):
        """The run's prior shapes, ?format=bin&v=1: one little-endian binary (docs/ARCHITECTURE.md, "Prior vectors"),
        sent as the zstd file it is cached as; immutable per URL (the version is in it). The JSON form no page asks
        for is not served (Runs.xenium_boundaries keeps it as the tests' reference for the binary)."""
        if fmt != "bin" or v != VECTORS_VERSION:
            raise RunError(f"Unknown prior vectors format: this engine serves format=bin&v={VECTORS_VERSION} (reload the page)", 404)
        return encoded(runs.prior_vectors(run_id), request)

    @app.get(base + "/runs/{run_id}/xenium/masks")
    def run_xenium_masks(run_id: str):
        return runs.xenium_masks(run_id)

    @app.get(base + "/runs/{run_id}/xenium/masks.bin")
    def run_xenium_masks_bin(run_id: str, request: Request):
        runs.xenium_masks(run_id)
        return encoded(runs.get(run_id).dir / "xenium" / "masks.u32.zst", request)

    @app.post(base + "/runs/{run_id}/analysis")
    def run_analysis(run_id: str, body: dict = Body(...)):   # configure() checks seq and params (422/404)
        return analysis.configure(run_id, body.get("enabled"), body.get("seq"), body.get("params"))

    @app.get(base + "/runs/{run_id}/analysis")
    def run_analysis_status(run_id: str):
        runs.get(run_id)
        return analysis.describe(run_id)

    # `generation`: the run's settings generation the page holds; a result of any other is a 404 (it is outdated).
    @app.get(base + "/runs/{run_id}/analysis/{seq}")
    def run_analysis_summary(run_id: str, seq: int, generation: int | None = Query(None, ge=0)):
        return analysis.summary(run_id, seq, generation)

    @app.get(base + "/runs/{run_id}/analysis/{seq}/cells")
    def run_analysis_cells(run_id: str, seq: int, generation: int | None = Query(None, ge=0)):
        return FileResponse(analysis.cells_path(run_id, seq, generation), media_type="application/octet-stream",
                            headers={"Cache-Control": "private, no-cache"})

    # ---- the mixing check -----------------------------------------------------------------------------------------
    @app.get(base + "/samples/{sample_id}/qc")
    def sample_qc(sample_id: str):
        """The sample's model, and `new_runs`: the switch new runs start with (the picker speaks only while on)."""
        return {**qc_models.describe(sample_id), "new_runs": qc.new_runs()}

    @app.post(base + "/samples/{sample_id}/qc")
    def sample_qc_learn(sample_id: str, body: dict = Body(...)):
        """{learn: true}: learn the sample now (also again after a failure). {detail: true}: learn the details Go deeper
        needs (deep.npz) for a sample learned before Go deeper existed: the same recipe again, kept only if it matches
        the model it has (whose key, numbers and results stay); also again after a failure. Answers as GET."""
        if body.get("detail") is True:
            return qc_models.ensure(sample_id, retry=True, detail=True)
        if body.get("learn") is not True:
            raise QcError("Send {learn: true} to learn this sample, or {detail: true} to learn the details Go deeper "
                          "needs")
        return qc_models.ensure(sample_id, retry=True)

    def qc_cells(path: Path, request: Request, size: int | None = None, pinned: bool = False):
        """One byte per cell (share x 250, 255 = not judged); sent zstd-encoded as written when the client accepts it.
        `size`: only the first `size` values (start.bin). Cached as immutable only when the URL names the model
        (`pinned`: ?model=<id>): the same URL without it answers with another model's bytes after a relearn."""
        import zstandard
        raw = path.read_bytes()
        packed_zstd = raw[:4] == b"\x28\xb5\x2f\xfd"
        headers = {"Cache-Control": "private, max-age=31536000, immutable" if pinned else "private, no-cache",
                   "Vary": "Accept-Encoding"}
        if packed_zstd and size is None and "zstd" in request.headers.get("accept-encoding", "").lower():
            return Response(raw, media_type="application/octet-stream", headers={**headers, "Content-Encoding": "zstd"})
        data = zstandard.ZstdDecompressor().decompressobj().decompress(raw) if packed_zstd else raw
        return Response(data if size is None else data[:size], media_type="application/octet-stream", headers=headers)

    def qc_bins(data: bytes, grid: dict, request: Request, pinned: bool = False):
        """The overview layer's grid: `planes` planes of one byte per bin, row-major (by x nx + bx): the share of judged
        cells that look mixed in the bin's 3 x 3 window, sum M / sum J over the window's bins, so a bin weighs what it
        holds and the page averages nothing (floor(share x 250), 255 where the window judges nothing), then the bin's
        own judged cells (rounded, at most 254: how full it is); engine/qc_areas.py overview_grid, over the smoothed sums
        of a live result and the frame's own of a single one. Results stored by older versions hold the first
        plane only, each bin's own share (no "planes"). The grid in the X-Bins header ({x0, y0, bin, nx, ny[, planes]},
        µm). Sent zstd-encoded as stored when the client accepts it; cached for good only when the URL names the
        model."""
        import zstandard
        packed = data[:4] == b"\x28\xb5\x2f\xfd"
        header = {k: grid[k] for k in ("x0", "y0", "bin", "nx", "ny")}
        if isinstance(grid.get("planes"), int) and grid["planes"] > 1:
            header["planes"] = grid["planes"]
        headers = {"Cache-Control": "private, max-age=31536000, immutable" if pinned else "private, no-cache",
                   "Vary": "Accept-Encoding", "X-Bins": json.dumps(header, separators=(",", ":"))}
        if packed and "zstd" in request.headers.get("accept-encoding", "").lower():
            return Response(data, media_type="application/octet-stream", headers={**headers, "Content-Encoding": "zstd"})
        raw = zstandard.ZstdDecompressor().decompressobj().decompress(data) if packed else data
        return Response(raw, media_type="application/octet-stream", headers=headers)

    @app.get(base + "/runs/{run_id}/qc")
    def run_qc(run_id: str):
        return qc.describe(run_id)

    @app.post(base + "/runs/{run_id}/qc")
    def run_qc_configure(run_id: str, body: dict = Body(...)):   # configure() checks enabled and seq (422/404)
        return qc.configure(run_id, body.get("enabled"), body.get("seq"))

    @app.get(base + "/runs/{run_id}/qc/start")
    def run_qc_start(run_id: str):
        return qc.start_info(run_id)

    @app.get(base + "/runs/{run_id}/qc/start/cells")
    def run_qc_start_cells(run_id: str, request: Request, model: str | None = None):
        path, cells = qc.start_cells(run_id, model)
        return qc_cells(path, request, cells or None, pinned=model is not None)

    @app.get(base + "/runs/{run_id}/qc/start/bins")
    def run_qc_start_bins(run_id: str, request: Request, model: str | None = None):
        data, grid = qc.start_grid(run_id, model)
        return qc_bins(data, grid, request, pinned=model is not None)

    @app.get(base + "/runs/{run_id}/qc/series")
    def run_qc_series(run_id: str, points: int = 600):
        return qc.series(run_id, points)

    @app.post(base + "/runs/{run_id}/qc/trial")
    def run_qc_trial(run_id: str, body: dict = Body(...)):
        """{rule, area, flag, from, to}: the change the page just made (or is about to make) through POST /control set,
        to be judged; 409 while another trial runs or when the run is not sampling."""
        return qc.trial(run_id, body)

    @app.post(base + "/runs/{run_id}/qc/trial/end")
    def run_qc_trial_end(run_id: str, body: dict = Body(...)):
        """{outcome: kept | undone, rule?: "U"}: end the running trial, or answer the Undo card."""
        return qc.trial_end(run_id, body.get("outcome"), body.get("rule"))

    # ---- Test a setting here (engine/qc_test.py) ----------------------------------
    @app.post(base + "/runs/{run_id}/qc/test")
    def run_qc_test(run_id: str, body: dict = Body(...)):
        """{cell (label - 1), flag: cell-compactness | prior-weight | diffusion-probability, value} -> 202 {test}: the
        test on its way ("preparing"): the cell's pool fixed at the run's last frame, then fresh runs of the area to
        the end of the schedule, as now and with the value (3 + 3, or 4 + 4 for a start from cells only), all made
        from one shared input, read and judged once they end; the run waits paused at its end meanwhile (one sampling
        on past it is paused). While a wait holds (the run not at its end yet; the mixing check off, learning or
        scoring; Go deeper's details to learn; another run sampling), or while the cell's pool is still being worked out
        after a few seconds (a whole sample's can take minutes), 202 {test} "waiting": the app does what it can itself
        (resumes the run, switches the check on) and the test starts once nothing holds and its pool is there. Where the
        measurements did not cover it, it runs all the same and its answer's caveats say so. 409 while a test of this
        run is waiting or on its way, and where a test cannot run ({detail, status, reasons}: the refusals GET
        .../qc/test/pool lists; nothing is paused or made); 422 for a body that is not {cell, flag, value} (changes,
        another flag, a value that is no number or out of range: far drift 0.1 to 0.3), the run's own value, or far
        drift while drift is off; 404 for an unknown run or cell; 503 while the app closes. suspects and sources are
        ignored: the pool is the engine's. A body that is not an object gets FastAPI's own 422."""
        return JSONResponse(qc_tests.start(run_id, body), status_code=202)

    @app.get(base + "/runs/{run_id}/qc/test")
    def run_qc_test_get(run_id: str):
        """{test}: the run's test, waiting, on its way or as it ended (its answer in `result`, v 2); 404 when it has
        none."""
        return qc_tests.get(run_id)

    @app.delete(base + "/runs/{run_id}/qc/test")
    def run_qc_test_cancel(run_id: str):
        """{test}: the run's test waiting or on its way is cancelled (every run of it aborted at once, then deleted;
        DELETE waits a few seconds for that; a run a waiting test resumed is paused again) and stays as "cancelled";
        one that ended is dismissed; 404 when it has none."""
        return qc_tests.cancel(run_id)

    @app.get(base + "/runs/{run_id}/qc/test/view")
    def run_qc_test_view(run_id: str):
        """{state, box, suspects, runs: [{role: A1 | B1, side: A | B, seq, iter, voxel, x0, y0, w, h, labels, cell,
        sources, held, held_by, unsure, cell_total, parents, ms}]}: the test's corner view: runs A1 (as now) and B1
        (with the value), each one's newest counted frame over the crop around the cell, its cell and neighbours there
        and proseg's share of each of the cell's suspects (the pool's) it holds (held: 0..1 per suspect, null where
        unsure or not found; held_by "exact" from a where export of the view's box, else "approx"; cell_total:
        proseg's count of the cell); the last frames' views stay after the runs are deleted; a kept view only moves
        forward. Under /qc, so the page keeps its refusals quiet; 404 when the run has no test."""
        return JSONResponse(qc_tests.view(run_id), headers={"Cache-Control": "private, no-cache"})

    @app.get(base + "/runs/{run_id}/qc/test/pool")
    def run_qc_test_pool(run_id: str, cell: str | None = None, wait: str | None = None):
        """?cell=&wait= -> the pool a test of the cell would judge and its reasons by kind: {v, state: ready | working |
        failed, run, cell, cell_name, seq, iter, count, model, constants, start, runs_per_side, k, area, offer, reasons
        (the refusals: [{code, words, ...}]), pattern, line, line_shown, this_cell, pool, estimate_s, ms, message, waits
        (what the test would wait for), caveats (where it was not measured), row ({start, situation}: the measured row
        it is judged with), hold ({value, effective, scale, seq}: the run's hold at the frame a test reads, which a test
        of the hold is against)}. Its query is read by the engine (qc_test.parse_preview: 422 in its own words, as the
        where route's); it waits at most `wait` seconds (0 to 10) for a pool being worked out. Under /qc, so the page
        keeps its refusals quiet; 404 for an unknown run or cell or a run that is not proseg's; 503 while the app
        closes."""
        runs.get(run_id)
        at, seconds = parse_preview(cell, wait)
        return JSONResponse(qc_tests.preview(run_id, at, wait=seconds), headers={"Cache-Control": "private, no-cache"})

    # ---- Go deeper (engine/deeper.py) -----------------------------------
    @app.post(base + "/runs/{run_id}/qc/deeper")
    def run_qc_deeper(run_id: str, body: dict = Body(...)):
        """{seq: n | "start", cell (label - 1), follow?: bool} -> {state, seq, cell, model, deep, reason?, progress?,
        eta_s?, estimate_s?, iter?, result?}: "ready" with the kept answer, "working" (it comes by WebSocket), or what
        stands in the way (off, waiting, learning, failed, unavailable, needs_detail, detail, detail_failed). 404 for an
        unknown run, frame or cell; 422 for a bad body; 503 where Clusters & UMAP cannot run."""
        return deeper.request(run_id, body)

    @app.delete(base + "/runs/{run_id}/qc/deeper")
    def run_qc_deeper_stop(run_id: str):
        """The window closed: stop following its cell."""
        deeper.stop(run_id)
        return Response(status_code=204)

    @app.get(base + "/runs/{run_id}/qc/deeper/history")
    def run_qc_deeper_history(run_id: str, cell: int, points: int = 200):
        """{model, start: byte | null, points: [[iter, byte]]}: the cell's stored byte over the run (its small line)."""
        return deeper.history(run_id, cell, points)

    # ---- Go deeper: where it spills (engine/deeper_where.py; `where` above) --------
    from ..engine.deeper_where import givers_of, parse as parse_where

    @app.get(base + "/runs/{run_id}/qc/deeper/where")
    def run_qc_deeper_where(run_id: str, seq: str | None = None, cell: str | None = None, sources: str = "",
                            genes: str = "", own: str = ""):
        """?seq=n|start&cell=&sources=&genes=&own= -> where the cell's and its sources' transcripts of those genes sit
        in that frame, with what each gene's place says. Each gene's giver is the
        one the window's kept answer names. 404 not a proseg run, no such frame or cell; 409 not written yet; 422 bad
        parameters."""
        runs.get(run_id)
        asked = parse_where(seq, cell, sources, genes, own)
        where.check(run_id, *asked[:3])         # refused before anything else is asked of the run
        try:
            givers = givers_of(deeper.cached(run_id, asked[0], asked[1]))
        except Exception:                       # not worked out (or no check): each gene's densest source instead
            givers = None
        return Response(where.reply(run_id, *asked, givers=givers), media_type="application/json",
                        headers={"Cache-Control": "private, no-cache"})

    @app.get(base + "/runs/{run_id}/qc/deeper/{seq}/{cell}")
    def run_qc_deeper_result(run_id: str, seq: str, cell: int, model: str | None = None, count: str | None = None):
        """A kept answer (the worker's result), on the count its frame is read on now; 404 until worked out, and when
        `count` ("averaged" | "sampled") names another one (the frame was counted again: a pause's, the hold's, a
        finish's or the end's count). Cached for good only when the URL names the model and the count and the answer is
        the frame's own: without the model the same URL answers with a relearned model's, without the count with a
        frame's count again (the harness's and the tests' route: the page reads answers as they are pushed), and a
        frame answered as the counted frame it snaps to answers with its own once its count is in."""
        result, own = deeper.kept_answer(run_id, seq, cell, model, count)
        return JSONResponse(result, headers={"Cache-Control": "private, max-age=31536000, immutable"
                                             if model is not None and count is not None and own
                                             else "private, no-cache"})

    @app.get(base + "/runs/{run_id}/qc/{seq:int}")
    def run_qc_summary(run_id: str, seq: int):
        return qc.summary(run_id, seq)

    @app.get(base + "/runs/{run_id}/qc/{seq:int}/cells")
    def run_qc_cells(run_id: str, seq: int, request: Request, model: str | None = None, count: str | None = None):
        """The frame's bytes on the count it is read on (`count` names the one the page means: "averaged" or
        "sampled"; 404 when the frame is read on another now: counted again). Cached for good only when the URL names
        the model and the count: a frame counted again (a pause's, the hold's, a finish's or the end's count) has other
        bytes."""
        return qc_cells(qc.cells_path(run_id, seq, model, count), request,
                        pinned=model is not None and count is not None)

    @app.get(base + "/runs/{run_id}/qc/{seq:int}/bins")
    def run_qc_bins(run_id: str, seq: int, request: Request, model: str | None = None):
        path, grid = qc.bins_grid(run_id, seq, model)
        return qc_bins(path.read_bytes(), grid, request, pinned=model is not None)

    # ---- Compare (engine/compare.py) ------------------------------------------------
    @app.post(base + "/runs/{run_id}/compare")
    def run_compare(run_id: str, body: dict = Body(...)):
        """{columns: ["run" | a segmentation's key, ...] (1-4, each once), seq?: the run's frame (default its newest)}
        -> 202 the comparison as GET answers it: started, the same one while it holds, or one found beside the run. 404
        unknown run, frame or segmentation, or not a proseg run on a SpatialData sample; 409 the sample has no model yet,
        or no frame to compare; 422 bad columns; 503 where Clusters & UMAP cannot run."""
        return JSONResponse(compare.start(run_id, body), status_code=202)

    @app.get(base + "/runs/{run_id}/compare")
    def run_compare_state(run_id: str):
        """{job, state: working | ready | failed, step, progress {done, of}, model, seq, iter, area {box, size_um, tiles,
        tile_um} | null, kinds [{index, name, color}] | null, columns [{key, name, state, cells_in_area, tiles, judged,
        mixed_per100, like100_per100, histogram (7 bins), median_tx, under100_share, umap_points, sampled, error,
        ready_s}], layout (how many layouts the points carried so far: fetch them again when it grows), error,
        seconds}; 404 before the run has one."""
        return JSONResponse(compare.describe(run_id), headers={"Cache-Control": "private, no-cache"})

    @app.get(base + "/runs/{run_id}/compare/points/{index:int}.bin")
    def run_compare_points(run_id: str, index: int, job: str | None = None, layout: int | None = None, own: int = 0):
        """Column `index`'s UMAP points as the newest layout placed them: per point f32 x, f32 y, u8 kind (255 none), u8
        share (floor(share x 250), 255 not judged), little-endian, nothing between; 404 before it has any. ?job= and
        ?layout= (the comparison and layout the page draws): 409 once the run's comparison or its layout is another.
        ?own=1: the column's own map instead (the same points in the same order, laid out alone and lined up with the
        run's: Compare's "Each its own"), 404 for a column with none."""
        return FileResponse(compare.points(run_id, index, job, layout, own=bool(own)), media_type="application/octet-stream",
                            headers={"Cache-Control": "private, no-cache"})

    @app.get(base + "/runs/{run_id}/compare/clusters/{index:int}.bin")
    def run_compare_clusters(run_id: str, index: int, job: str | None = None, layout: int | None = None):
        """Column `index`'s own clusters (Compare's "Each its own"): one u16 per point, in its points' order, matched
        to the run's column's ids where their profiles match (65535 none); 404 for a column with none, 409 as points."""
        return FileResponse(compare.points(run_id, index, job, layout, what="clusters"), media_type="application/octet-stream",
                            headers={"Cache-Control": "private, no-cache"})

    @app.get(base + "/runs/{run_id}/image/{name}")
    def run_image(run_id: str, name: str, request: Request):
        if re.fullmatch(r"c[0-9]{1,2}\.u16", name):   # full bit depth: raw uint16, zstd
            return encoded(runs.get(run_id).dir / "image" / (name + ".zst"), request)
        if not re.fullmatch(r"c[0-9]{1,2}\.png", name):
            raise RunError("Unknown image", 404)
        return FileResponse(runs.get(run_id).dir / "image" / name, media_type="image/png",
                            headers={"Cache-Control": "private, max-age=31536000, immutable"})

    @app.get(base + "/runs/{run_id}/transcripts")
    def run_transcripts(run_id: str):
        # x, y, normalised z (f32) and gene (u32) per modelled transcript; gzip on disk, inflated by the browser.
        path = runs.get(run_id).live / "transcripts.bin.gz"
        if not path.exists():
            raise RunError("Proseg has not written the transcripts yet", 404)
        return FileResponse(path, media_type="application/octet-stream",
                            headers={"Content-Encoding": "gzip", "Cache-Control": "private, no-cache"})

    @app.get(base + "/runs/{run_id}/cells")
    def run_cells(run_id: str):
        """Original Xenium cell ids: entry k names label k + 1 in the frames."""
        path = runs.get(run_id).live / "cells.txt"
        return path.read_text().split("\n")[:-1] if path.exists() else []

    @app.post(base + "/runs/{run_id}/control")
    def run_control(run_id: str, body: dict = Body(...)):
        action = str(body.get("action", ""))
        if runs.get(run_id).meta.get("leaving"):
            raise RunError("Wait for this run to finish leaving, or press Stay.", 409)
        # While a test runs (engine/qc_test.py), its own runs sample alone: no other run resumes or steps.
        if action in ("resume", "step") and qc_tests.going() and not runs.get(run_id).meta.get("qc_test"):
            raise RunError("A test of a setting is running: wait for its answer, or cancel it first", 409)
        return runs.control(run_id, action, body.get("speed"), body.get("values"))

    def forget_run(run):
        run_id, folder = run.meta["id"], str(run.dir) + os.sep
        deeper.forget(run_id)
        compare.forget(run_id)
        qc.forget(run_id)
        qc_tests.forget(run_id)
        qc_tests.pools.forget(run_id)
        with analysis.lock:
            analysis._forget(run_id)
        for key in [k for k in list(dense_cache) if k.startswith(folder)]:
            dense_cache.pop(key, None)

    runs.on_delete = forget_run

    def overtaken(run_id: str, order: dict) -> bool:
        """Whether a later answer of the same window was taken first, so that Runs.leave changes nothing for this one:
        it must not cancel the run's test of a setting either. Read from the run's record of each
        window's newest revision taken (Run.leave_revisions, page -> revision, which Runs.leave keeps and checks again
        as it takes an answer)."""
        if "revision" not in order:
            return False
        try:
            run = runs.get(run_id)
        except RunError:
            return False                                 # Runs.leave answers for it
        with run.lock:
            last = (getattr(run, "leave_revisions", None) or {}).get(order.get("page"))
        return last is not None and order["revision"] < last

    @app.post(base + "/runs/{run_id}/leave")
    def run_leave(run_id: str, body: dict = Body(...)):
        """{answer: "add" | "discard" | "stay", revision?, page?}: Runs.leave. `revision` (a whole number that grows
        with each answer of the window `page` names) keeps a window's answers in their order: one overtaken by a later
        one changes nothing (a late Discard after Stay), the run's test included."""
        answer = body.get("answer")
        if answer not in ("add", "discard", "stay"):
            raise RunError("Choose Save to the sample, Discard or Stay")
        order = {key: body[key] for key in ("revision", "page") if body.get(key) is not None}
        if "revision" in order and (type(order["revision"]) is not int or order["revision"] < 0):
            raise RunError("revision must be a whole number")
        if "page" in order and not (isinstance(order["page"], str) and 0 < len(order["page"]) <= 100):
            raise RunError("page must be a short id")
        if runs.closing:
            raise RunError("The engine is shutting down", 503)
        if answer != "stay" and not overtaken(run_id, order):
            try:
                qc_tests.cancel(run_id)
            except QcError as exc:
                if exc.status != 404:
                    raise
        return runs.leave(run_id, answer, **order)

    @app.delete(base + "/runs/{run_id}")
    def run_delete(run_id: str):
        runs.delete(run_id)
        return {"deleted": run_id}

    @app.websocket(base + "/ws")
    async def live(socket: WebSocket):
        # Signed in already (server/security.py checks a socket's credential in its handshake), so nothing here waits for
        # a first message. A browser that offered subprotocols fails a handshake whose answer names none.
        offered = socket.scope.get("subprotocols") or ()
        await socket.accept(subprotocol=SOCKET_PROTOCOL if SOCKET_PROTOCOL in offered else None)
        page, visit = None, None
        if socket.scope.get(AUTH) == "page":            # tools, the harness's watch, use the bearer: never a page
            # A page names itself in its socket's address (?page=, its log's session, and ?visit=, which showing of it
            # this is); one that does not (or not usably) still counts, under a name of its own (socket-<n>) that only
            # this socket's own bye can say goodbye for, so without that its going is "away" (runs paused), never
            # "closed" (runs discarded).
            page = page_id(socket.query_params.get("page")) or f"socket-{next(app.state.socket_ids)}"
            visit = visit_number(socket.query_params.get("visit"))
            app.state.presence.connected(page, visit)
            # A page connected again (a reload): an answer to closing no longer counts. A socket of an earlier showing
            # than one heard of since (it was on its way as the page went) takes nothing back.
            if not app.state.presence.stale(page, visit):
                if app.state.session is not None:
                    app.state.session.withdraw_closing()
                else:
                    app.state.closing.withdraw()
        try:
            await live_session(socket, page, visit)
        finally:
            if page is not None:
                app.state.presence.disconnected(page)

    def run_update(run, since: int):
        """The frames after `since` (at most 400) and the run's state, for a push. Run in a thread: both take the run's
        lock, which a worker holds while it writes run.json on a slow shared disk (stalls of 100 ms and more while
        a whole sample prepares), and the event loop must not wait for it."""
        with run.lock:
            fresh = run.frames[since:since + 400]
        return page_frames(fresh), qc_tests.detail(run.public())

    def counted_before(run, since: int) -> tuple[list[int], list[int]]:
        """(The run's counted frames before `since` (those a page that watches from there holds already), those of them
        read on the count proseg saves): a frame's counts line can merge after its record went out, and a page that was
        away meanwhile (a lost connection) or read the records just before (it opened the run) hears of it only here;
        proseg_counts.recent and .sampled name the 64 newest alone."""
        with run.lock:
            counted = list(getattr(run, "counted", None) or [])
            sampled = list(getattr(run, "sampled", None) or [])
        return counted[:bisect.bisect_left(counted, since)], sampled[:bisect.bisect_left(sampled, since)]

    def result_counts(run_id: str) -> dict:
        """{seq: the count its Clusters & UMAP result was read on} of the results the scheduler keeps for the run
        (AnalysisManager.result_counts): a frame analysed again on another count is announced again. Without them, none
        (every result as read on the averaged count)."""
        found = getattr(analysis, "result_counts", None)
        try:
            return found(run_id) if found is not None else {}
        except Exception:
            return {}

    async def live_session(socket: WebSocket, page: str | None = None, visit: int | None = None):
        """What one socket is sent: the runs, the watched run's frames and state (a watch from a frame on, the page
        holding the frames before it, first names the counted ones among those: {ev: "counted", seqs}), the mixing
        check's and Go deeper's events, an API session's state. It asks with {op: "watch", run, since}, keeps itself
        open with {op: "ping"}, and
        a page says it goes for good with {op: "bye"} (`page`: its id from its address, `visit` the showing it names
        there; a bearer's socket has none)."""
        watched, sent_frames, run_version, list_version = None, 0, -1, -1
        counted_due = False                      # a watch: the counted frames the page holds go out once
        frames_version, list_sent_at = -1, 0.0
        analysis_version, analysis_sent, analysis_known = -1, None, set()
        analysis_gen = None                      # the watched run's settings generation this page has heard of
        qc_version, qc_cursor = -1, {}           # what this page has had of the mixing check (QcManager.events)
        deeper_version, deeper_cursor = -1, {}   # ... and of Go deeper (DeeperManager.events)
        counted = None                           # the run this connection counts as watching (DeeperManager.watching)
        hooks, session_version = app.state.session_hooks, -1    # an API session's state, pushed as it changes
        older_sent, older_mark = None, None                     # what this page has of the runs from earlier versions
        try:
            while True:
                try:
                    message = await asyncio.wait_for(socket.receive_json(), timeout=.05)
                except asyncio.TimeoutError:
                    message = None
                if message is not None:
                    if message.get("op") == "watch":
                        watched = message.get("run") or None
                        sent_frames, run_version = max(0, int(message.get("since", 0))), -1
                        counted_due = sent_frames > 0
                        analysis_version, analysis_sent, analysis_known = -1, None, set()
                        analysis_gen = None
                        qc_version = -1
                        qc_cursor.pop("runs", None)
                        qc_cursor.pop("switch", None)    # the first look after a watch pushes the switch with its count
                        deeper_version = -1
                        deeper_cursor.pop("runs", None)
                    elif message.get("op") == "bye":
                        app.state.presence.goodbye(page, "socket", visit)   # a bearer's socket (page None): ignored
                    elif message.get("op") == "ping":
                        await socket.send_json({"ev": "pong"})
                if watched != counted:
                    deeper.watching(counted, watched)
                    counted = watched
                if hooks is not None:                   # {ev: "session"}: its sample, the notebook, ending
                    session_version = await hooks.push(socket, session_version)
                # The mixing check: samples being learned (at most one event per second each while waiting or
                # learning) and the watched run's switch (flipped elsewhere), start line, new results (at most 4 per
                # tick) and errors.
                if qc.version != qc_version or qc_cursor.get("pending"):
                    qc_version = qc.version
                    clock = asyncio.get_running_loop().time()
                    for event in await asyncio.to_thread(qc.events, qc_cursor, watched, clock):
                        await socket.send_json(event)
                # Go deeper: the watched run's answers (every one asked for, the newest followed one) and each sample's
                # details job (at most one event per second while it learns).
                if deeper.version != deeper_version or deeper_cursor.get("pending"):
                    deeper_version = deeper.version
                    clock = asyncio.get_running_loop().time()
                    for event in await asyncio.to_thread(deeper.events, deeper_cursor, watched, clock):
                        await socket.send_json(event)
                # Lifecycle changes go out at once; frame counts of running runs at most every 1.5 s.
                clock = asyncio.get_running_loop().time()
                if runs.version != list_version or (runs.frames_version != frames_version and clock - list_sent_at > 1.5):
                    list_version, frames_version, list_sent_at = runs.version, runs.frames_version, clock
                    await socket.send_json({"ev": "runs", "runs": await asyncio.to_thread(runs.list)})
                # Runs from earlier versions: {ev: "older", runs, ...} whenever they change (Runs.older_version;
                # without one, looked at with each change of the runs and every second while some are listed), until
                # none is left. Nothing is pushed when there never were any.
                mark = getattr(runs, "older_version", None)
                if mark is None:
                    mark = (list_version, int(clock) if older_sent and older_sent.get("runs") else 0)
                if mark != older_mark:
                    older_mark = mark
                    state = await asyncio.to_thread(older_or_none)
                    if state is None:                   # listing them failed (logged): nothing pushed this time
                        pass
                    elif older_sent is None and not state.get("runs") and not state.get("done"):
                        older_sent = state
                    elif state != older_sent:
                        await socket.send_json({"ev": "older", **state})
                        older_sent = state
                if watched:
                    try:
                        run = runs.get(watched)
                    except RunError:
                        if watched in runs.departed:
                            # A fast save and leave can finish between socket ticks. Send its final names first.
                            await socket.send_json({"ev": "run", "run": runs.departed[watched]})
                        await socket.send_json({"ev": "gone", "run": watched,
                                                **({"reason": runs.gone[watched]} if watched in runs.gone else {})})
                        watched = None
                        continue
                    if run.meta.get("kind") not in ("stain", "patches") and analysis.version != analysis_version:
                        analysis_version = analysis.version
                        state = await asyncio.to_thread(analysis.describe, watched)
                        status, generation = state["status"], state.get("generation")
                        # The run's settings changed (here or in another window, or a tool): the page hears it first,
                        # drops what it shows of the earlier settings, and every frame recomputed (the same seq
                        # numbers) is announced again. The first look after a watch says nothing: the page asked.
                        if analysis_gen is not None and generation != analysis_gen:
                            await socket.send_json({"ev": "analysis", "run": watched, "enabled": state["enabled"],
                                                    "state": "settings", "generation": generation,
                                                    **{k: state[k] for k in ("settings", "defaults", "limits",
                                                                             "changed")}})
                            analysis_known = set()
                        analysis_gen = generation
                        # Every completed result is announced (a result is "ready" only for an instant before the
                        # next frame starts, so the transient status alone would miss most of them), and again once a
                        # frame is analysed on another count (counted again: a pause's, the hold's, a finish's or the
                        # end's).
                        counts = await asyncio.to_thread(result_counts, watched)
                        known = [(s, counts.get(s, "averaged")) for s in state["seqs"]]
                        fresh = [s for s, c in known if (s, c) not in analysis_known]
                        analysis_known.update(known)
                        for seq in fresh[-4:]:
                            try:                        # reads the disk; another generation by now: a 404, skipped
                                summary = await asyncio.to_thread(analysis.summary, watched, seq, generation)
                            except (AnalysisError, RunError):
                                continue
                            await socket.send_json({"ev": "analysis", "run": watched, "enabled": state["enabled"],
                                                    "state": "ready", "seq": seq, "iter": summary.get("iter"),
                                                    "summary": summary, "generation": generation})
                        key = (status.get("state"), status.get("seq"), state["enabled"], generation)
                        if key != analysis_sent and status.get("state") != "ready":
                            analysis_sent = key
                            await socket.send_json({"ev": "analysis", "run": watched, "enabled": state["enabled"],
                                                    **status, "generation": generation})
                    if counted_due:                     # {ev: "counted"}: every counted frame the page holds
                        counted_due = False
                        if run.meta.get("kind") not in ("stain", "patches"):
                            seqs, sampled = await asyncio.to_thread(counted_before, run, sent_frames)
                            if seqs:
                                await socket.send_json({"ev": "counted", "run": watched, "seqs": seqs,
                                                        "sampled": sampled})
                    if run.version != run_version:
                        run_version = run.version
                        fresh, public = await asyncio.to_thread(run_update, run, sent_frames)
                        if fresh:
                            await socket.send_json({"ev": "frames", "run": watched, "from": sent_frames, "frames": fresh})
                            sent_frames += len(fresh)
                            if len(fresh) == 400:
                                run_version = -1  # more to send on the next tick
                        await socket.send_json({"ev": "run", "run": public})
        except (WebSocketDisconnect, RuntimeError, ValueError, TypeError):
            return
        finally:
            deeper.watching(counted, None)

    app.mount("/", FreshStaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    # An API session ([session] mode = "api") puts its own lifespan and its /health, /closing and sample routes ahead of
    # these (server/api_serve.py); on the desktop it only sets app.state.session and session_hooks to None.
    attach_session(app, cfg, session=session, shutdown=stop_engine)
    return app
