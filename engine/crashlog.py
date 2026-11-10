"""Crash reports: a session of the app that ended unexpectedly leaves a plain-text report, which the page offers at the
next start ("SoPA companion closed unexpectedly last time": Save a copy… or Not now; frontend/src/crash-report.mjs).

Two kinds of session can end so:
  * the engine's. While it is up an engine keeps a marker, <state_dir>/run/engine.alive ({pid, started, version,
    data_dir}): written once the server is up, removed when it shuts down (as asked, or after SIGTERM, SIGINT or SIGHUP;
    server/app.py's lifespan). An engine that starts and finds the marker of another process, one that is gone, writes
    that session's report (reason "engine") before anything else (Runs then removes the runs it left unsaved), and puts
    its own marker in place once it is up. The marker of an engine still running (another engine on the same state
    folder) is left to it, and this engine keeps none;
  * a page's (reason "page"). Each page sends its own log every 10 s while it changes and once more as it goes (POST
    /page-log; the last one says "pagehide"), so the engine knows which pages it has seen and which said goodbye. A page
    that went without one ended unexpectedly, found
      - by the next page in the same window ("tab": the page before it there did not unload) -> offered;
      - by a page that found the flag of one that did not unload ("flag": the browser's shared storage, of which it
        writes a copy to disk only now and then). The page then went with its browser: a logout, the desktop restarting
        and the browser's own end look the same, so its report is kept quietly, not offered, unless the engine went
        down with it.
    A page whose goodbye came closed cleanly (its flag was an old copy): no report. One report per page session. The
    report this engine wrote as it started goes into a page's when it has not been offered yet (both ended together, the
    computer stopped: one popup, one file). The engine writes none of its own for a page that went while it runs: a
    page gone without a goodbye is "away" (server/presence.py: its runs pause), which a tab the browser put away or a
    lost connection is too, and neither is a crash.
The page logs are kept per page session, <state_dir>/logs/page-logs/<session>.txt (the newest PAGE_LOGS_KEPT: each the
last 512 KB of that page's log under a line naming it), and the newest one sent also as <state_dir>/logs/page-log.txt.

A report, <state_dir>/crash-reports/<YYYYMMDD-HHMMSS>-<reason>.txt (this computer's time), is plain text: what ended,
when it started and when it was found, the app's version, python, the platform and the GPUs (nvidia-smi -L without their
UUIDs, once per engine); the runs there were then (id, kind, sample, state, iteration); the engine's own log of that
session (server.log.1 and server.log, from the session's first line where it can be found, at most 3000 lines); then the
page's last log. The newest 10 are kept, at most 3 of them kept quietly. seen.json lists those the page has offered: the
newest report not offered yet is offered, once, and never one older than a report offered already (it is not "last time"
any more). Nothing secret is written: the session's secret wherever it appears, cookie values, tokens in addresses and
anything shaped like the secret are blanked.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import threading
import time

import psutil

from .. import PROTOCOL, __version__, portable
from ..names import LOG_PREFIX, LOG_PREFIXES

KEEP = 10                          # reports kept, the newest
QUIET_KEEP = 3                     # ... of which kept quietly (never offered) at most: the older quiet ones go first
LOG_LINES = 3000                   # server.log lines in a report: the session's last
LOG_READ_BYTES = 8 << 20           # read from the end of server.log.1 and of server.log, each
LOG_KEEP_BYTES = 2 << 20           # a report's server.log lines at most (the newest)
LINE_CHARS = 4000                  # a longer line is cut
PAGE_LOG_BYTES = 512 * 1024        # a page's log as kept: its last 512 KB of text
PAGE_LOGS_KEPT = 6                 # page sessions whose logs are kept (logs/page-logs/), the newest sent
PAGE_LOGS_IN_REPORT = 3            # an engine report's page logs at most
GPU_TIMEOUT_S = 3.0                # nvidia-smi -L's time to answer
GPU_UUID = re.compile(r"\s*\(UUID: [^)]*\)")   # "GPU 0: <name> (UUID: GPU-…)": the UUID goes, the GPU's line stays
SLACK_S = 2.0                      # clocks compared to the second: a process's start and its marker's, log times
REPORT_ID = re.compile(r"\d{8}-\d{6}-(?:engine|page)")   # a report's name: its time and reason ("engine" or "page")
MARKER, PAGE_LOG, PAGE_LOGS, SEEN = "engine.alive", "page-log.txt", "page-logs", "seen.json"
TITLE = "SoPA companion closed unexpectedly"
WHAT = {"engine": "the engine: it stopped without shutting down (it was killed or crashed, or the computer stopped)",
        "page": "the page: it closed without unloading (the browser tab crashed or was killed, or the computer stopped)"}
# How a page's end was found: POST /crash/page's "found" ("tab", "flag"; none from an older page). The report's
# "Written" row says it.
FOUND = {"tab": "as the next page in the same window started", "flag": "as a page started again and found its flag",
         None: "as a page started again"}
QUIET_ROW = "Offered"              # a report kept quietly says so in this header row ("no: ..."); the others have none
QUIET = ("no: the browser closed with the page open (a logout, the desktop restarting or the browser's own end look "
         "the same), so it is only kept here")
ROW = 18                           # the width of a header row's name
# The engine's first line in its log (server/app.py prints it as it starts), and uvicorn's: where a session begins. The
# previous version's prefix too: the first start after the state folder moved reads that version's log.
_PREFIX = "(?:" + "|".join(map(re.escape, LOG_PREFIXES)) + ")"
STARTING = re.compile(rf"^{_PREFIX}: engine starting: pid (\d+)\b")
STARTING_AT = re.compile(rf"^{_PREFIX}: engine starting: pid \d+\b.*, (\d{{4}}-\d\d-\d\dT\S+)$")   # ... and when
UVICORN_STARTED = re.compile(r"Started server process \[(\d+)\]")
STAMP = re.compile(r"^\[?(\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:[.,]\d+)?(?:Z|[+-]\d\d:?\d\d)?)")   # a line with its time
ISO_IN = re.compile(r"\((\d{4}-\d\d-\d\dT[^)\s]+)\)")
# A page log's first line: its page session, when the newest part was sent, when the first was, and whether that was
# the page's goodbye.
PAGE_HEAD = re.compile(r"^Page session ([^\s,]+), sent ([^\s,]+)(?:, first ([^\s,]+))?(, closed)?$")
LOG_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}")      # a page session that is its own log's file name
LEFT_OUT = b"(the page's older lines were left out)\n"
# The page's own lines (frontend/src/log-export.mjs): a log cut to fit says how many older lines it left out, and a
# line repeated counts itself at its end.
CUT = re.compile(r"^  … [\d,]+ older lines? left out$")
COUNTED = re.compile(r" \([\d,]+ times?, last [\d:.]+\)$")
# The session's secret is secrets.token_urlsafe(32): 43 URL-safe characters. Also blanked: an earlier version's session
# cookie value (in an older log: this version sets no cookie), tokens and codes in JSON.
SHAPED = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])")
RUN_ID = re.compile(r"\d{8}-\d{6}-")                         # a run's id begins with its time (engine/runs.py)
OLD_COOKIE = re.compile(r"(?i)((?:sopa_companion|proseg_live)_session=)[^\s;,\"'<>]+")   # earlier versions' cookies
JSON_TOKEN = re.compile(r'(?i)("(?:token|code)"\s*:\s*")[^"]+(")')


def starting_line(pid: int | None = None, now: float | None = None) -> str:
    """The engine's first line in its log, which tells where a session begins and when (STARTING, STARTING_AT)."""
    return (f"{LOG_PREFIX}: engine starting: pid {os.getpid() if pid is None else pid}, SoPA companion {__version__}, "
            f"{_iso(time.time() if now is None else now)}")


# ---- small helpers -------------------------------------------------------------------------------------------------------
def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def _local(t: float | None) -> str:
    """This computer's time: "2026-10-23 14:03:04"."""
    return "not known" if t is None else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def _when(t: float | None) -> str:
    """This computer's time and the exact one: "2026-10-23 14:03:04 (2026-10-23T13:03:04+00:00)"."""
    return "not known" if t is None else f"{_local(t)} ({_iso(t)})"


def parse_time(value) -> float | None:
    """An ISO 8601 time (with Z or an offset; without one, this computer's time) or a number (ms since 1970 when that
    large, else s) -> unix time; None for anything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return (value / 1000 if abs(value) > 1e11 else float(value)) if math.isfinite(value) else None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00").replace(",", ".")).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def _word(value, limit: int = 100) -> str:
    """A page's session or time as one word for a header line: letters, digits and . _ : + - only."""
    return re.sub(r"[^A-Za-z0-9._:+-]", "", str(value or ""))[:limit] or "unknown"


def _row(name: str, value) -> list[str]:
    lines = str(value).split("\n")
    return [f"{name:<{ROW}}{lines[0]}", *(" " * ROW + line for line in lines[1:])]


def _title(text: str) -> list[str]:
    return ["", text, "-" * len(text)]


def _private(folder: Path) -> Path:
    """A folder only this user can open (as server/runtime_file.py private_directory: never a link, never another's)."""
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = folder.lstat()
    if not stat.S_ISDIR(info.st_mode) or not portable.mine(info):
        raise ValueError(f"{folder} must be a folder of this user's own")
    if not portable.WINDOWS and stat.S_IMODE(info.st_mode) != 0o700:
        folder.chmod(0o700)
    return folder


def _atomic(path: Path, data: bytes) -> None:
    """`data` through a private temporary file beside `path`, on disk before it takes the name: a crash leaves the old
    file or the new one, never a torn one."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | portable.O_NOFOLLOW | portable.O_BINARY, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        portable.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _shaped(match) -> str:
    """43 URL-safe characters mixed as a random secret is (5 capitals and 5 small letters at least: all but about one
    in 80,000 of the secrets the app makes) are blanked. A run's id (it begins with its time) and a name mostly of one
    case keep their own."""
    word = match.group(0)
    if RUN_ID.match(word):
        return word
    mixed = sum(c.isupper() for c in word) >= 5 and sum(c.islower() for c in word) >= 5
    return "[redacted]" if mixed else word


def redact(text: str, secrets=()) -> str:
    """`text` without the session's secret (`secrets`, wherever they stand), an earlier version's session cookie value,
    tokens and codes in addresses, after Bearer and in JSON, and anything shaped like the secret."""
    from ..desktop.logs import redact as blank      # desktop/logs.py imports server/, whose app imports this module
    text = OLD_COOKIE.sub(r"\1[redacted]", blank(text, secrets))
    return SHAPED.sub(_shaped, JSON_TOKEN.sub(r"\1[redacted]\2", text))


def gpus(timeout: float = GPU_TIMEOUT_S) -> list[str]:
    """The GPUs as `nvidia-smi -L` lists them, without their UUIDs (a report is made to be shared, and a GPU's UUID is
    its serial number); ["unknown"] without it, when it fails or when it has not answered within `timeout` seconds.
    Never longer: one that has not answered is killed and left to a thread to collect, never waited for here (stuck in
    the driver, as on a GPU gone wrong, it cannot end until the driver lets it go, and subprocess.run would wait for it
    as long)."""
    try:
        child = subprocess.Popen(["nvidia-smi", "-L"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True, errors="replace", **portable.no_window())
    except (OSError, ValueError, subprocess.SubprocessError):
        return ["unknown"]
    try:
        out, _ = child.communicate(timeout=timeout)
    except (OSError, ValueError, subprocess.SubprocessError):     # TimeoutExpired among them
        try:
            child.kill()
        except OSError:
            pass
        try:
            child.stdout.close()
        except (OSError, ValueError):
            pass
        threading.Thread(target=_collect, args=(child,), name="nvidia-smi -L", daemon=True).start()
        return ["unknown"]
    lines = [GPU_UUID.sub("", line).strip() for line in out.splitlines() if line.strip()]
    return lines if child.returncode == 0 and lines else ["unknown"]


def _collect(child: subprocess.Popen) -> None:
    try:
        child.wait()
    except Exception:                                  # noqa: BLE001 - a thread of its own: nothing to tell
        pass


# ---- what a report reads -------------------------------------------------------------------------------------------------
def _tail(path: Path, limit: int) -> bytes:
    with open(path, "rb") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        stream.seek(max(0, end - limit))
        data = stream.read()
    if end > limit:                                   # the first line is cut
        data = data[data.find(b"\n") + 1:] if b"\n" in data else b""
    return data


def log_lines(logs: Path) -> tuple[list[str], float | None]:
    """server.log.1 then server.log (the last LOG_READ_BYTES of each) as lines (carriage-return progress bars split into
    lines, blank lines dropped, as engine/runlog.py tail), and when the log was last written (None without one)."""
    lines, written = [], None
    for path in (logs / "server.log.1", logs / "server.log"):
        try:
            data, mtime = _tail(path, LOG_READ_BYTES), path.stat().st_mtime
        except OSError:
            continue
        written = mtime if written is None else max(written, mtime)
        lines += [line for line in data.decode("utf-8", "replace").replace("\r", "\n").split("\n") if line.strip()]
    return lines, written


def session_start(lines: list[str], pid) -> int | None:
    """Where the session of process `pid` begins in the log: its "engine starting" line, else uvicorn's "Started server
    process [pid]" (an engine of an earlier version); the last such line, as pids come round again. None: neither."""
    if type(pid) is not int:
        return None
    for pattern in (STARTING, UVICORN_STARTED):
        found = [i for i, line in enumerate(lines) if (m := pattern.search(line)) and int(m.group(1)) == pid]
        if found:
            return found[-1]
    return None


def engine_starts(lines: list[str]) -> list[tuple[int, float]]:
    """Where each engine session begins in the log and when: its "engine starting" line (those with a time)."""
    return [(i, t) for i, line in enumerate(lines)
            if (m := STARTING_AT.match(line)) and (t := parse_time(m.group(1))) is not None]


def session_at(lines: list[str], when: float | None) -> int | None:
    """Where the engine session running at `when` begins in the log: the last "engine starting" line timed at or before
    it. None without a time, or without such a line."""
    if when is None:
        return None
    return max((i for i, t in engine_starts(lines) if t <= when + SLACK_S), default=None)


def stamped_start(lines: list[str], started: float | None) -> int | None:
    """The first line stamped with a time at or after `started` (a line that starts with its own time), if any is."""
    if started is None:
        return None
    for i, line in enumerate(lines):
        if (m := STAMP.match(line)) and (t := parse_time(m.group(1).replace(" ", "T", 1))) is not None \
                and t >= started - SLACK_S:
            return i
    return None


def log_part(lines: list[str], first: int | None, unknown: str) -> tuple[list[str], str]:
    """The lines from `first` (else from the start, saying `unknown`): the last LOG_LINES of them, each cut at
    LINE_CHARS, LOG_KEEP_BYTES at most; -> (lines, a note saying what was left out, "" for nothing)."""
    notes = [unknown] if first is None else []
    part = lines[first or 0:]
    if len(part) > LOG_LINES:
        notes.append(f"its first {len(part) - LOG_LINES:,} lines left out")
        part = part[-LOG_LINES:]
    part = [line if len(line) <= LINE_CHARS else line[:LINE_CHARS - 1] + "…" for line in part]
    size, keep = 0, len(part)
    while keep and size + len(part[keep - 1].encode("utf-8", "replace")) + 1 <= LOG_KEEP_BYTES:
        keep -= 1
        size += len(part[keep].encode("utf-8", "replace")) + 1
    if keep:
        notes.append(f"{keep:,} more lines left out (a report keeps {LOG_KEEP_BYTES >> 20} MB of them)")
        part = part[keep:]
    return part, "; ".join(notes)


def log_section(title: str, lines: list[str], written: float | None, first: int | None, unknown: str) -> list[str]:
    """A report's part of server.log under `title`: the lines from `first` (log_part), what was left out in brackets."""
    if written is None:
        return [*_title(title), "  none: there is no server.log"]
    part, note = log_part(lines, first, unknown)
    return [*_title(f"{title} ({len(part):,} lines)"), *([f"({note})"] if note else []), *(part or ["  none"])]


def _last_iteration(frames: Path) -> int | None:
    """The iteration of a run's last snapshot (the last whole line of live/frames.jsonl)."""
    try:
        data = _tail(frames, 1 << 16)
    except OSError:
        return None
    for line in reversed(data.split(b"\n")):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and type(value.get("iter")) is int:
            return value["iter"]
    return None


def runs_text(root: Path, limit: int = 50) -> list[str]:
    """The runs in `root` (var/runs/*/run.json), newest first: id, kind, sample, state and how far each got (its last
    iteration; a stain run's tiles)."""
    rows = []
    for path in root.glob("*/run.json"):
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            meta = None
        if not isinstance(meta, dict):
            rows.append(("", [path.parent.name, "?", "?", "its run.json cannot be read", ""]))
            continue
        progress = meta.get("progress") if isinstance(meta.get("progress"), dict) else {}
        if meta.get("kind") == "stain" and type(progress.get("total")) is int and progress["total"] > 0:
            where = f"{progress.get('done', 0)} of {progress['total']} tiles"
        elif meta.get("kind", "proseg") == "proseg":
            iteration = _last_iteration(path.parent / "live" / "frames.jsonl")
            where = f"iteration {iteration}" if iteration is not None else "no iteration yet"
        else:
            where = "-"
        rows.append((str(meta.get("created") or ""),
                     [str(meta.get("id") or path.parent.name), str(meta.get("kind") or "proseg"),
                      str(meta.get("sample_name") or meta.get("sample") or "-"), str(meta.get("state") or "-"), where]))
    rows.sort(key=lambda row: row[0], reverse=True)
    if not rows:
        return ["  none"]
    shown = [cells for _, cells in rows[:limit]]
    widths = [min(60, max(len(cells[i]) for cells in shown)) for i in range(4)]
    out = ["  " + "  ".join(cell.ljust(width) for cell, width in zip(cells, [*widths, 0])).rstrip() for cells in shown]
    if len(rows) > limit:
        out.append(f"  ({len(rows) - limit} older runs not listed)")
    return out


# ---- the page logs -------------------------------------------------------------------------------------------------------
def _log_name(session: str) -> str:
    """The file of a page session's log: its own name when it is a plain one (a page's sessions are), else a hash."""
    return f"{session}.txt" if LOG_NAME.fullmatch(session) else f"s-{hashlib.sha256(session.encode()).hexdigest()[:32]}.txt"


def _head(line: str) -> dict | None:
    """A page log's first line -> {session, sent (as the page said), at (unix time or None), first, closed}."""
    match = PAGE_HEAD.match(line)
    if not match:
        return None
    return {"session": match.group(1), "sent": match.group(2), "at": parse_time(match.group(2)),
            "first": match.group(3) or match.group(2), "closed": bool(match.group(4))}


def _page_head(path: Path) -> dict | None:
    try:
        with open(path, "rb") as stream:
            return _head(stream.read(512).split(b"\n", 1)[0].decode("utf-8", "replace"))
    except OSError:
        return None


def _page_logs(logs: Path) -> list[dict]:
    """The page logs kept (logs/page-logs/), the newest sent first: [{session, sent, at, first, closed, text}]."""
    out = []
    for path in (logs / PAGE_LOGS).glob("*.txt"):
        try:
            text, mtime = path.read_text(errors="replace"), path.stat().st_mtime
        except OSError:
            continue
        head, _, body = text.partition("\n")
        log = _head(head) or {"session": None, "sent": "unknown", "at": None, "first": None, "closed": False}
        out.append({**log, "text": body if _head(head) else text, "order": log["at"] if log["at"] is not None else mtime})
    return sorted(out, key=lambda log: log["order"], reverse=True)


def _cut(data: bytes) -> bytes:
    """A page's log as kept: its last PAGE_LOG_BYTES, from a whole line where one starts soon, saying so."""
    if len(data) <= PAGE_LOG_BYTES:
        return data
    cut = data[-(PAGE_LOG_BYTES - len(LEFT_OUT)):]
    newline = cut.find(b"\n", 0, 4096)
    return LEFT_OUT + cut[newline + 1 if newline >= 0 else 0:].decode("utf-8", "ignore").encode()


def merge_goodbye(kept: str, goodbye: str) -> str | None:
    """A page's goodbye cut to fit (its older lines left out: the beacon is small) after a longer log of the same page:
    that log, with the goodbye's lines from its last line on (that line with its count as the goodbye has it). None for
    a goodbye that is whole, or no longer than what is kept: it is kept as it is."""
    lines = goodbye.rstrip("\n").split("\n")
    at = next((i for i, line in enumerate(lines) if CUT.match(line)), None)
    if at is None or len(goodbye) >= len(kept):
        return None
    newest, old = lines[at + 1:], kept.rstrip("\n").split("\n")
    def key(line):                                           # a line without its count
        return COUNTED.sub("", line)
    last = key(old[-1])
    same = [i for i, line in enumerate(newest) if key(line) == last]
    if same:
        return "\n".join([*old[:-1], *newest[same[-1]:]]) + "\n"
    return "\n".join([*old, "  … (lines between these were not kept)", *newest]) + "\n"


# ---- what a report reads -------------------------------------------------------------------------------------------------
def _header(path: Path) -> dict[str, str]:
    """A report's header rows (name -> its first line), from its first 8 kB."""
    try:
        with open(path, "rb") as stream:
            head = stream.read(8192).decode("utf-8", "replace")
    except OSError:
        return {}
    rows = {}
    for line in head.split("\n")[2:]:
        if not line.strip():
            break                                         # the header ends at its first blank line
        if not line.startswith(" "):
            rows[line[:ROW].strip()] = line[ROW:].strip()
    return rows


# ---- the reports ---------------------------------------------------------------------------------------------------------
def marker_owner(marker: dict, me: int | None = None) -> str:
    """Whose engine marker (<state_dir>/run/engine.alive): "this" process's (`me`: an earlier app of the same process),
    another engine's that is still "running", or one "gone" (no such process, or its pid came round again: a process
    started after the marker). Also how the state folder's move tells an engine of the previous version still up
    (desktop/migrate.py)."""
    pid = marker.get("pid") if isinstance(marker, dict) else None
    if type(pid) is not int or pid <= 0:
        return "gone"
    started = parse_time(marker.get("started"))
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return "gone"
        created = process.create_time()
    except psutil.Error:                              # gone, or another user's: never this user's engine
        return "gone"
    if started is not None and created > started + SLACK_S:
        return "gone"
    return "this" if pid == me else "running"


class CrashReports:
    """One engine's crash reports (see the module's docstring): check() as it starts, before anything else; mark_alive()
    once it is up and mark_closed() when it has shut down; save_page_log() for each page log sent; and what the page
    asks for (server/app.py's /crash routes). `secrets`: the session's secret(s), never written; `clock` (tests): unix
    time."""

    def __init__(self, config: dict, *, secrets=(), clock=time.time):
        paths = config["paths"]
        self.state, self.data_dir = Path(paths["state_dir"]), Path(paths["data_dir"])
        self.folder, self.logs = self.state / "crash-reports", self.state / "logs"
        self.marker = self.state / "run" / MARKER
        self.secrets = [s for s in secrets if isinstance(s, str) and len(s) >= 8]   # never "auto" (config: a new one)
        self.clock, self.pid = clock, os.getpid()
        self.created = clock()                           # this engine's start, near enough
        self.lock = threading.RLock()                    # the reports, seen.json and the page logs
        self.gpu_lock, self.gpu_names = threading.Lock(), None    # the GPUs, looked up once (never under self.lock)
        self.found: str | None = None                    # the report this engine wrote as it started
        self.crashed: dict | None = None                 # ... of that session: {pid, started}
        self.foreign = None                              # the pid of another engine, still up, that keeps the marker
        self.alive: dict | None = None                   # this engine's marker as written
        # The page sessions this engine has had a log from: {session: {first (as the page said), closed}}.
        self.pages: dict[str, dict] = {}

    # -- the engine's marker --
    def check(self) -> str | None:
        """As the engine starts: a marker left by a process that is gone -> its session's report (reason "engine"), and
        the marker goes (mark_alive puts this engine's in its place). -> the report's id, or None. Never fails the
        start: what goes wrong is said in the engine log."""
        try:
            try:
                raw = self.marker.read_bytes()
            except FileNotFoundError:
                return None
            try:
                marker = json.loads(raw)
            except ValueError:
                marker = None
            marker = marker if isinstance(marker, dict) else {}         # torn or not ours: a session ended all the same
            owner = self._owner(marker)
            if owner == "this":
                return None
            if owner == "running":
                self.foreign = marker["pid"]
                print(f"SoPA companion: another engine (pid {self.foreign}) is up with this state folder ({self.state}): "
                      "it keeps its crash marker, and this engine keeps none", flush=True)
                return None
            self.secrets += self._runtime_secret()
            pid, started = marker.get("pid"), parse_time(marker.get("started"))
            self.gpu_list()
            with self.lock:
                report = self._engine_report(marker)
            self.found, self.crashed = report, {"pid": pid, "started": started}
            try:
                if json.loads(self.marker.read_bytes()) == marker:      # still that one (never another engine's)
                    self.marker.unlink()
            except (OSError, ValueError):
                pass
            print(f"SoPA companion: the last session (pid {pid if type(pid) is int else 'unknown'}, started "
                  f"{_local(started)}) ended without shutting down: its crash report is {self.folder / report}.txt",
                  flush=True)
            return report
        except Exception as exc:                          # noqa: BLE001 - the engine starts whatever this finds
            print(f"SoPA companion: could not check how the last session ended: {type(exc).__name__}: {exc}", flush=True)
            return None

    def _owner(self, marker: dict) -> str:
        return marker_owner(marker, self.pid)

    def _runtime_secret(self) -> list[str]:
        """The token in the runtime file a session that ended unexpectedly leaves (server/runtime_file.py): its secret.
        (The desktop launcher removes that file before it starts an engine: then the secret's shape blanks it.)"""
        try:
            token = json.loads((self.state / "run" / "server.json").read_text()).get("token")
        except (OSError, ValueError, AttributeError):
            return []
        return [token] if isinstance(token, str) and len(token) >= 8 else []

    def mark_alive(self) -> None:
        """The server is up: the marker says so until mark_closed (or the next start finds it)."""
        if self.foreign is not None:
            return
        value = {"pid": self.pid, "started": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(
            timespec="milliseconds"), "version": __version__, "data_dir": str(self.data_dir)}
        try:
            _private(self.marker.parent)
            _atomic(self.marker, json.dumps(value).encode())
            self.alive = value
        except (OSError, ValueError) as exc:
            print(f"SoPA companion: could not write the crash marker: {type(exc).__name__}: {exc}", flush=True)

    def mark_closed(self) -> None:
        """A clean shutdown: this engine's marker goes (only its own: never one another engine wrote since)."""
        try:
            if self.alive is not None and json.loads(self.marker.read_bytes()) == self.alive:
                self.marker.unlink()
        except (OSError, ValueError):
            pass
        self.alive = None

    # -- writing --
    def gpu_list(self) -> list[str]:
        """The GPUs (gpus()), looked up once per engine: at most GPU_TIMEOUT_S, and never while self.lock is held (a
        page's log and its answers wait for nothing of it)."""
        with self.gpu_lock:
            if self.gpu_names is None:
                self.gpu_names = gpus()
            return self.gpu_names

    def _new_id(self, reason: str, now: float) -> str:
        for step in range(3600):                          # two reports in one second: the next free second
            report = time.strftime("%Y%m%d-%H%M%S", time.localtime(now + step)) + "-" + reason
            if not (self.folder / f"{report}.txt").exists():
                return report
        raise OSError(f"no free crash report name in {self.folder}")

    def _common(self, report: str, reason: str, now: float, found: str, started: float | None,
                inside: str | None = None) -> list[str]:
        earlier = [r for r in self._waiting() if r != inside]        # (`inside`: a report that goes into this one)
        return [TITLE, "=" * len(TITLE), *_row("What ended", WHAT[reason]), *_row("Report", f"{report}, in {self.folder}"),
                *_row("Written", f"{_when(now)}, {found}"), *_row("Session started", _when(started)),
                *(_row("Earlier reports", f"{', '.join(earlier)}: not offered yet (in the same folder)") if earlier else [])]

    def _system(self) -> list[str]:
        return [*_row("App version", f"SoPA companion {__version__} (protocol {PROTOCOL})"),
                *_row("Python", f"{platform.python_version()} ({sys.executable})"),
                *_row("Platform", platform.platform()), *_row("GPUs", "\n".join(self.gpu_list()))]

    def _page_part(self, logs: list[dict], note: str = "", none: str = "none kept") -> list[str]:
        if not logs:
            return [*_title("The page's last log"), f"  {none}"]
        out = []
        for log in logs:
            gone = ", then it closed" if log["closed"] else ""
            out += _title(f"The page's last log (page session {log['session'] or 'unknown'}, sent "
                          f"{_when(log['at']) if log['at'] is not None else log['sent']}{gone}){note}")
            out += log["text"].rstrip("\n").split("\n")
        return out

    def _engine_report(self, marker: dict) -> str:
        now, pid = self.clock(), marker.get("pid")
        started, version = parse_time(marker.get("started")), str(marker.get("version") or "not known")
        data_dir = Path(marker["data_dir"]) if isinstance(marker.get("data_dir"), str) and \
            Path(marker["data_dir"]).is_absolute() else self.data_dir
        lines, written = log_lines(self.logs)
        title = "server.log: that session's output"
        if written is None:
            last, section = "no server.log", log_section(title, lines, None, None, "")
        elif started is not None and written < started - SLACK_S:
            last = f"{_when(written)}, before that session started"
            section = [*_title(title), "  none: it was not written during that session (its output went elsewhere: "
                                       "a terminal?)"]
        else:
            first = session_start(lines, pid)
            first = stamped_start(lines, started) if first is None else first
            last = f"{_when(written)}: the engine stopped after this"
            section = log_section(title, lines, written, first, "its first line is not in server.log or server.log.1: "
                                                                "the lines may begin in an earlier session")
        logs = _page_logs(self.logs)
        during = [page for page in logs if started is None or (page["at"] is not None and page["at"] >= started - SLACK_S)]
        # The pages open as it ended (no goodbye) first; else the newest that page sent then.
        shown = [page for page in during if not page["closed"]][:PAGE_LOGS_IN_REPORT] or during[:1]
        report = self._new_id("engine", now)
        text = [*self._common(report, "engine", now, "as the engine started again", started),
                *_row("Log last written", last),
                *_row("That engine", f"pid {pid if type(pid) is int else 'unknown'}, SoPA companion {version}, data "
                                     f"folder {data_dir}"),
                *self._system(),
                *_title("Runs then, as that session left them"), *runs_text(data_dir / "runs"), *section,
                *self._page_part(shown or logs[:1], "" if shown or not logs else ": sent before that session started")]
        self._write(report, text)
        return report

    def _page_report(self, at: str, session: str, how: str | None) -> str:
        """The report of page session `session` (it started at `at`), found as `how` says (FOUND): kept quietly when only
        its flag was found, unless the engine's own report from this start goes into it."""
        now, started = self.clock(), parse_time(at)
        lines, written = log_lines(self.logs)
        logs = [log for log in _page_logs(self.logs) if log["session"] == session]
        first = session_at(lines, started)                              # the engine session the page began in
        if first is None and started is not None and started >= self.created - SLACK_S:
            first = session_start(lines, self.pid)
        elif first is None and started is not None and self.crashed and self.crashed["started"] is not None \
                and started >= self.crashed["started"] - SLACK_S:
            first = session_start(lines, self.crashed["pid"])
        # ... to the end of the one it was last heard from in (its last log sent; else as it began): the engine may have
        # started again while its window stayed open.
        heard = max([t for t in (started, *(log["at"] for log in logs)) if t is not None], default=None)
        if first is not None and heard is not None:
            end = next((i for i, t in engine_starts(lines) if i > first and t > heard + SLACK_S), None)
            lines = lines if end is None else lines[:end]
        engine = None
        if self.found and self.found in self._waiting():
            engine = self.text(self.found)                # it goes into this one: one popup, one file
        quiet = how == "flag" and engine is None
        report = self._new_id("page", now)
        text = [*self._common(report, "page", now, FOUND.get(how, FOUND[None]), started,
                              self.found if engine is not None else None),
                *(_row(QUIET_ROW, QUIET) if quiet else []),
                *_row("Page session", session),
                *_row("This engine", f"pid {self.pid}, SoPA companion {__version__}, data folder {self.data_dir}"),
                *self._system(),
                *_title("Runs now"), *runs_text(self.data_dir / "runs"),
                *log_section("server.log: the engine's output while that page was open", lines, written, first,
                             "the engine session that page began in is not in server.log or server.log.1: the lines "
                             "may begin in an earlier one"),
                *self._page_part(logs, none="none from that page was kept (a page sends its log every 10 s)")]
        if engine is not None:
            text += [*_title(f"The engine's own report, written as this engine started ({self.found})"),
                     *engine.rstrip("\n").split("\n")]
        self._write(report, text)
        if engine is not None:
            self._mark_seen(self.found)                   # offered with this one
        return report

    def _write(self, report: str, lines: list[str]) -> None:
        """Keep a report (its secrets blanked), then only the newest KEEP, QUIET_KEEP of them kept quietly at most (the
        older quiet ones go first); seen.json forgets the reports gone."""
        _private(self.folder)
        _atomic(self.folder / f"{report}.txt", redact("\n".join(lines).rstrip() + "\n", self.secrets).encode(
            "utf-8", "replace"))
        ids = self.ids()
        quiet = [r for r in ids if self._quiet(r)]
        gone = set(quiet[:-QUIET_KEEP])
        kept = [r for r in ids if r not in gone]
        gone |= set(kept[:-KEEP])
        for old in gone:
            (self.folder / f"{old}.txt").unlink(missing_ok=True)
        seen = self._seen()
        if [s for s in seen if s not in gone] != seen:
            self._save_seen([s for s in seen if s not in gone])

    # -- the page logs --
    def save_page_log(self, text: str, session=None, at=None, pagehide: bool = False) -> int:
        """A page's own log (POST /page-log), kept for its session: its last PAGE_LOG_BYTES of text under a line naming
        the page session, when this was sent and when its first was, and whether it was the page's goodbye
        ("pagehide": a later log of the same page, one brought back, says it is open again). A log sent before the one
        kept (one late on its way) changes nothing; a goodbye cut to fit adds its newest lines to the longer log kept
        (merge_goodbye). The newest log sent is also page-log.txt. -> the bytes kept."""
        session, sent, when = _word(session), _word(at), parse_time(at)
        with self.lock:
            folder = _private(_private(self.logs) / PAGE_LOGS)
            path = folder / _log_name(session)
            kept = _page_head(path)
            if kept is not None and kept["session"] != session:
                kept = None                               # (another session's of the same hash: never seen in use)
            if kept is not None and when is not None and kept["at"] is not None and \
                    (when < kept["at"] or when == kept["at"] and kept["closed"] and not pagehide):
                return 0                                  # (a log sent with the goodbye, after it: no news either)
            text = str(text)
            if pagehide and kept is not None:
                try:
                    merged = merge_goodbye(path.read_text(errors="replace").partition("\n")[2], text)
                except OSError:
                    merged = None
                text = text if merged is None else merged
            data = _cut(text.encode("utf-8", "replace"))
            first = kept["first"] if kept is not None else sent
            blob = f"Page session {session}, sent {sent}, first {first}{', closed' if pagehide else ''}\n".encode() + data
            _atomic(path, blob)
            _atomic(self.logs / PAGE_LOG, blob)
            self.pages[session] = {**self.pages.get(session, {}), "first": first, "closed": bool(pagehide)}
            self._prune_page_logs(folder, path)
        return len(data)

    @staticmethod
    def _prune_page_logs(folder: Path, keep: Path) -> None:
        """The newest PAGE_LOGS_KEPT page logs stay (by when they were sent, else written), `keep` among them."""
        def sent(path):
            try:
                head, mtime = _page_head(path), path.stat().st_mtime
            except OSError:
                return 0.0
            return head["at"] if head is not None and head["at"] is not None else mtime
        logs = sorted((p for p in folder.glob("*.txt") if p != keep), key=sent, reverse=True)
        for old in logs[PAGE_LOGS_KEPT - 1:]:
            old.unlink(missing_ok=True)

    def _closed(self, session: str) -> bool:
        """Whether page session `session` said its goodbye (the last log it sent was its pagehide's)."""
        if session in self.pages:
            return self.pages[session]["closed"]
        head = _page_head(self.logs / PAGE_LOGS / _log_name(session))
        return bool(head and head["session"] == session and head["closed"])

    def _reported(self, session: str) -> bool:
        return any(_header(self.folder / f"{r}.txt").get("Page session") == session for r in self.ids())

    def _page(self, at, session: str, how: str | None) -> str | None:
        """Page session `session`'s report, written once (None: there is one already). The GPUs are looked up first,
        outside the lock."""
        if self._reported(session):
            return None
        self.gpu_list()
        with self.lock:
            if self._reported(session):
                return None
            report = self._page_report(at, session, how)
            if session in self.pages:
                self.pages[session]["reported"] = True
            return report

    # -- what the page asks for --
    def ids(self) -> list[str]:
        """The reports kept, oldest first (their names sort by time)."""
        try:
            return sorted(p.stem for p in self.folder.glob("*.txt") if REPORT_ID.fullmatch(p.stem))
        except OSError:
            return []

    def _quiet(self, report: str) -> bool:
        return _header(self.folder / f"{report}.txt").get(QUIET_ROW, "").startswith("no")

    def _waiting(self) -> list[str]:
        """The reports kept that would still be offered (newest()'s choice among them), oldest first."""
        seen, ids = set(self._seen()), self.ids()
        latest = max((r for r in seen if r in ids), default="")
        return [r for r in ids if r > latest and r not in seen and not self._quiet(r)]

    def summary(self, report: str) -> dict | None:
        """{id, at, reason, bytes, started}: when it was written and when the session started (ISO, None when not
        known); None for no such report."""
        if not REPORT_ID.fullmatch(str(report)):
            return None
        path = self.folder / f"{report}.txt"
        try:
            size = path.stat().st_size
        except OSError:
            return None
        rows = _header(path)
        written = ISO_IN.search(rows.get("Written", ""))
        started = ISO_IN.search(rows.get("Session started", ""))
        at = written.group(1) if written else datetime.strptime(report[:15], "%Y%m%d-%H%M%S").astimezone().isoformat()
        return {"id": report, "at": at, "reason": report.rsplit("-", 1)[1], "bytes": size,
                "started": started.group(1) if started else None}

    def newest(self) -> dict | None:
        """The report to offer (GET /crash), or None: the newest not offered yet, never one kept quietly, nor one older
        than a report offered already (that one was "last time")."""
        for report in reversed(self._waiting()):
            if (summary := self.summary(report)) is not None:
                return summary
        return None

    def text(self, report: str) -> str | None:
        """A report's text; None for no such report (only a report's own name is ever read: no paths)."""
        if not REPORT_ID.fullmatch(str(report)):
            return None
        try:
            return (self.folder / f"{report}.txt").read_text(errors="replace")
        except OSError:
            return None

    def seen(self, report: str) -> bool:
        """The page offered it (Save a copy… or Not now): never offered again, nor any older one. False for no such
        report."""
        if self.summary(report) is None:
            return False
        with self.lock:
            self._mark_seen(report)
        return True

    def page(self, at: str, session: str, found: str | None = None) -> dict | None:
        """A page found that page session `session` ({at: when it started}) did not unload: as the page before it in
        the same window ("tab"), or by its flag ("flag"; None: a page that does not say). That session's report (reason
        "page"), once per session, kept quietly when only its flag was found (see the module's docstring); none when its
        goodbye came (the flag was an old copy). -> the report to offer now (newest())."""
        session = _word(session)
        if self._closed(session):
            print(f"SoPA companion: a page found the flag of page session {session}, which closed cleanly (its goodbye "
                  "came): no crash report", flush=True)
        elif (report := self._page(at, session, found)) is not None:
            print(f"SoPA companion: a page closed without unloading (it started {_local(parse_time(at))}): its crash "
                  f"report is {self.folder / report}.txt{' (kept quietly)' if self._quiet(report) else ''}", flush=True)
        return self.newest()

    def _seen(self) -> list[str]:
        try:
            value = json.loads((self.folder / SEEN).read_text())
        except (OSError, ValueError):
            return []
        seen = value.get("seen") if isinstance(value, dict) else None
        return [s for s in seen if isinstance(s, str)] if isinstance(seen, list) else []

    def _mark_seen(self, report: str) -> None:
        seen = self._seen()
        if report not in seen:
            self._save_seen([*seen, report])

    def _save_seen(self, seen: list[str]) -> None:
        _atomic(_private(self.folder) / SEEN, json.dumps({"seen": seen}).encode())
