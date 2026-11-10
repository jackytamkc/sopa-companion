"""The page's Log button: a plain-text log of this engine session and, for a run, everything the engine knows of it.

GET /api/v1/log and GET /api/v1/runs/{id}/log (server/app.py) answer with `build(sources, run_id)`, in this order:
  * a short Summary first (the sample, region, prior, schedule and options, where the run stands, the mixing check, and
    whether anything went wrong): the page adds its own line to it (frontend/src/log-export.mjs);
  * the engine (version, the code it runs, proseg-live, folders, the analysis worker, the mixing check);
  * the run (sample, region, prior, proseg's schedule with its own temperature moves, options, state, its phases, the
    knobs the user changed, Clusters & UMAP with each step's typical time, the mixing check, what the engine noted about
    the run, proseg's own log), when a run is open;
  * the page's latest telemetry in words and the tails of the engine's logs, kept to what concerns the run;
  * an Appendix with every Clusters & UMAP result kept (its clusters, cells and step times: the line the page used to
    show under the clusters), which the page moves to the very end, after its own part (APPENDIX).
Iterations name the moments everywhere (never snapshot numbers). Long lines are wrapped (proseg's command line with
shell continuations, so it can be pasted), never cut short of LINE_CHARS.

Reading only: nothing is started, learned or written (the mixing check is read from what this session holds, never
described, which could queue a model fit), every part that cannot be read says so instead of failing the log, and
nothing secret appears: it is blanked as a crash report is (engine/crashlog.py redact, with the same secrets). Each part
is bounded, so a whole-sample run's log stays a few hundred kB.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import threading
import time

from .. import PROTOCOL, __version__, locate
from .config import ROOT
from .shipping import SHIP_EXCLUDES       # what a deploy ships: the code hash is over these files (engine/shipping.py)

RESULT_LINES = 300          # Clusters & UMAP results listed one per line in the appendix (the newest; all are in the times)
PROSEG_LINES = 150          # the tail of the run's proseg.log
ENGINE_LINES = 150          # the tail of server.log (the engine's own output)
RUN_ENGINE_LINES = 80       # server.log lines naming the run (from its last few MB)
WORKER_LINES = 40           # the tails of analysis.log (the analysis worker) and qc-trainer.log (the check's fits)
WARM_KEPT = 5               # the analysis worker's warm-up lines kept (its newest warm-up, one line per step: the GPU's
                            # five, the CPU's four)
TELEMETRY_REPORTS = 3       # the page's newest performance reports
NOTED_KEPT = 200            # what went wrong that the engine noted itself (Noted), the newest kept
GAP_LIST = 20               # the longest gaps between snapshots (pauses), listed
KNOB_LIST = 200             # live knob changes listed
WRAP = 150                  # lines are wrapped at this width (APPENDIX_WRAP for the appendix's one line per result) ...
APPENDIX_WRAP = 200
LINE_CHARS = 4000           # ... and only a line longer than this is cut (a pathological one)
TEMPERATURE_TOLERANCE = 1e-3   # a snapshot's temperature within 0.1 % of the schedule's is the schedule's (float32)
SUMMARY = "Summary"
APPENDIX = "Appendix: each Clusters & UMAP result"   # the page moves this section to the very end (log-export.mjs)
STEPS = ("load", "counts", "pca", "knn", "leiden", "markers", "umap", "write")   # the worker's own order
SCHEDULE_FLAGS = ("burnin-samples", "samples", "recorded-samples", "hillclimb")
ACTIVE = ("preparing", "running", "paused", "finishing")
SAMPLE_STATES = {"off": "not learned (switching the check on learns it)", "waiting": "waiting to learn",
                 "learning": "learning", "ready": "learned", "failed": "learning failed", "unavailable": "cannot be learned"}
# A traceback or note of the analysis worker names a run by its folder (".../runs/<id>/...") or by its id.
RUN_NAMED = re.compile(r"/runs/([^/\s]+)/|\brun (\S+) was deleted")
ISO_START = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:[+-]\d\d:\d\d|Z))\s")
TRAINER_KEEP = re.compile(r"\bqc:|Traceback|Error\b|Exception\b|\berror\b|\bfailed\b")


class Noted:
    """What went wrong that the engine noted itself, for the summary's "Engine noted" and the engine events (else,
    after three failed requests, the session Log said nothing went wrong): a request the engine failed (server/app.py: a
    helper that failed, a 5xx answer, an error it did not expect) and a run that failed or could not be saved
    (engine/runs.py note). (unix time, run id or None, text), the newest NOTED_KEPT; the same text again for the same
    run, back to back (a page polling a route that keeps failing), is kept once with how many times."""

    def __init__(self, keep: int = NOTED_KEPT):
        self.events: deque = deque(maxlen=keep)       # [unix time, run id or None, text, times]
        self.lock = threading.Lock()

    def add(self, text: str, run_id: str | None = None) -> bool:
        """Note it; False when it is the last one again (counted, not added: its caller need not say it again)."""
        with self.lock:
            last = self.events[-1] if self.events else None
            if last is not None and last[1] == run_id and last[2] == text:
                last[0], last[3] = time.time(), last[3] + 1
                return False
            self.events.append([time.time(), run_id, text, 1])
            return True

    def list(self, run_id: str | None = None) -> list[tuple]:
        """(unix time, run id or None, text) oldest first: every event, or with `run_id` the run's own."""
        with self.lock:
            events = [tuple(e) for e in self.events]
        return [(t, run, text + (f" ({n} times, the last at {when(t)})" if n > 1 else ""))
                for t, run, text, n in events if run_id is None or run == run_id]


@dataclass
class Sources:
    """What the log reads: the app's own objects (server/app.py makes one per engine)."""
    config: dict
    runs: object
    analysis: object
    qc: object = None
    qc_models: object = None
    secrets: tuple = ()                      # the session's secrets, the crash reports' own (CrashReports.secrets)
    started: float = field(default_factory=time.time)
    deeper: object = None                    # Go deeper's DeeperManager (engine/deeper.py), when the engine has one
    noted: Noted | None = None               # what went wrong that the engine noted itself (Noted)


# ---- small formatting helpers ------------------------------------------------------------------------------------------
def when(t: float | None) -> str:
    """Unix time as this computer's local time."""
    if t is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def local(iso: str | None) -> str:
    """An ISO 8601 time (the engine writes UTC) as this computer's local time; anything else as it is."""
    if not iso:
        return "-"
    try:
        return when(datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return str(iso)


def span(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "-"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60} min"


def num(value) -> str:
    """A count with thousands separators; anything else as text ("-" for nothing)."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"
    return str(value)


def value(v) -> str:
    """A knob or option value as proseg takes it: floats to 6 significant digits (the snapshots hold float32)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}"
    return "automatic" if v is None else str(v)


def size(nbytes) -> str:
    try:
        nbytes = float(nbytes)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "kB", "MB", "GB"):
        if nbytes < 1000 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1000
    return "-"


def is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def cut(text: str, limit: int = LINE_CHARS) -> str:
    text = str(text).replace("\t", "    ")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def wrap(text: str, width: int = WRAP, first: str = "", rest: str = "", shell: bool = False) -> list[str]:
    """`text` (cut at LINE_CHARS) as lines of at most `width` characters, broken at spaces where it can: the first line
    starts with `first`, the others with `rest`. `shell`: a command line, each broken line ending in " \\" so the lines
    paste back as one command."""
    text = cut(text)
    if len(first) + len(text) <= width:
        return [first + text]
    end = " \\" if shell else ""
    lines, line, empty = [], first, True              # empty: nothing on the line yet but its indent
    for word in text.split(" "):
        while True:
            room = width - len(end) - len(line) - (0 if empty else 1)
            if len(word) <= room:
                line, empty = line + ("" if empty else " ") + word, False
                break
            if not empty:                                 # the word starts the next line
                lines.append(line + end)
                line, empty = rest, True
                continue
            take = max(1, room)                           # a word longer than a whole line: broken inside it
            lines.append(line + word[:take] + end)
            word, line = word[take:], rest
            if not word:
                break
    if not empty or line.strip():
        lines.append(line)
    return lines


def ms_line(ms: dict | None) -> str:
    """The step times of one Clusters & UMAP result, as the page showed them: "load 6 ms · counts 40 ms · ..."."""
    if not isinstance(ms, dict) or not ms:
        return "no step times"
    parts = []
    for key, v in ms.items():
        try:
            parts.append(f"{key} {float(v):.0f} ms")
        except (TypeError, ValueError):
            continue
    return " · ".join(parts) or "no step times"


class Text:
    """The log being written: headed sections, aligned "name  value" rows and indented lines, wrapped at WRAP."""

    def __init__(self):
        self.lines: list[str] = []

    def title(self, text: str, mark: str = "="):
        if self.lines:
            self.lines.append("")
        self.lines += [text, mark * len(text)]

    def heading(self, text: str):
        self.lines += ["", text]

    def row(self, name: str, text, width: int = 16):
        rows = str(text).split("\n")
        pad = " " * width
        self.lines += wrap(rows[0], first=f"{name:<{width}}", rest=pad)
        for r in rows[1:]:
            self.lines += wrap(r, first=pad, rest=pad)

    def line(self, text: str = "", indent: int = 2, width: int = WRAP):
        if not text:
            self.lines.append("")
            return
        self.lines += wrap(text, width, first=" " * indent, rest=" " * (indent + 2))

    def command(self, text: str, indent: int = 2):
        """A shell command line, wrapped with continuations (it pastes back as one command)."""
        self.lines += wrap(text, first=" " * indent, rest=" " * (indent + 4), shell=True)

    def part(self, name: str, fn, *args):
        """Write one part; one that cannot be read says so and the log goes on."""
        try:
            fn(self, *args)
        except Exception as exc:                          # noqa: BLE001 - a log must come out whatever is broken
            self.line(f"({name}: this part could not be read: {type(exc).__name__}: {exc})")

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


# ---- files -------------------------------------------------------------------------------------------------------------
def tail(path, lines: int, max_bytes: int = 1 << 20) -> list[str] | None:
    """The last `lines` lines of a text file (from its last `max_bytes`; carriage-return progress bars split into lines,
    blank lines dropped), or None when it cannot be read."""
    try:
        with open(path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            stream.seek(max(0, end - max_bytes))
            data = stream.read()
    except OSError:
        return None
    text = data.decode("utf-8", errors="replace").replace("\r", "\n")
    rows = [r for r in text.split("\n") if r.strip()]
    if end > max_bytes and rows:
        rows = rows[1:]                                   # the first one is cut
    return rows[-lines:]


def grep_tail(path, needle: str, lines: int, max_bytes: int = 4 << 20) -> list[str] | None:
    rows = tail(path, 10 ** 9, max_bytes)
    return None if rows is None else [r for r in rows if needle in r][-lines:]


def _excluded(rel: str, name: str) -> bool:
    for pattern in SHIP_EXCLUDES:
        if pattern.startswith("/"):
            if fnmatch.fnmatchcase(rel, pattern[1:]):
                return True
        elif fnmatch.fnmatchcase(name, pattern):
            return True
    return False


def code_hash(root: Path = ROOT) -> dict:
    """sha256 over the sorted "path NUL sha256" lines of every file a deploy ships (a deploy's tree-hash rule, with
    deploy's exclusions): the same code gives the same hash in the dev copy and in production. ~30 ms."""
    files: dict[str, str] = {}
    pending = [""]
    while pending:
        rel_dir = pending.pop()
        try:
            with os.scandir(root / rel_dir if rel_dir else root) as entries:
                listing = sorted(entries, key=lambda e: e.name)
        except OSError:
            continue
        for entry in listing:
            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            if _excluded(rel, entry.name):
                continue
            try:
                if entry.is_symlink():
                    files[rel] = "link:" + os.readlink(entry.path)
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(rel)
                elif entry.is_file(follow_symlinks=False):
                    digest = hashlib.sha256()
                    with open(entry.path, "rb") as stream:
                        while chunk := stream.read(1 << 20):
                            digest.update(chunk)
                    files[rel] = digest.hexdigest()
            except OSError:
                continue
    total = hashlib.sha256("".join(f"{rel}\0{files[rel]}\n" for rel in sorted(files)).encode()).hexdigest()
    return {"sha256": total, "files": len(files)}


def file_hash(path, limit: int = 256 << 20) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            read = 0
            while read < limit and (chunk := stream.read(1 << 20)):
                digest.update(chunk)
                read += len(chunk)
        return digest.hexdigest()
    except OSError:
        return None


# ---- the run's snapshots -----------------------------------------------------------------------------------------------
def iteration_ms(frame: dict) -> float | None:
    """proseg's own time for the iteration of a snapshot (morphology + repositioning + parameters, as "Iteration time"
    on the page)."""
    ms = frame.get("ms")
    if not isinstance(ms, dict):
        return None
    try:
        return float(ms.get("morphology", 0)) + float(ms.get("repositioning", 0)) + float(ms.get("parameters", 0))
    except (TypeError, ValueError):
        return None


def phases(frames: list[dict]) -> list[dict]:
    """One row per stretch of snapshots in the same phase, in order: {phase, snapshots, first, last (iterations), seconds
    (proseg's clock, first to last snapshot of the stretch), median_ms, slowest_ms (its iterations), voxel, changed
    (median voxels changed)}."""
    rows: list[dict] = []
    for f in frames:
        if not rows or rows[-1]["phase"] != f.get("phase"):
            rows.append({"phase": f.get("phase"), "frames": [], "voxel": f.get("voxel")})
        rows[-1]["frames"].append(f)
    out = []
    for r in rows:
        fs = r.pop("frames")
        times = [t for t in (iteration_ms(f) for f in fs) if t is not None and t > 0]
        clock = [f["t"] for f in fs if isinstance(f.get("t"), (int, float))]
        changed = [f["changed"] for f in fs if isinstance(f.get("changed"), (int, float))]
        out.append({**r, "snapshots": len(fs), "first": fs[0].get("iter"), "last": fs[-1].get("iter"),
                    "seconds": (max(clock) - min(clock)) if clock else None,
                    "median_ms": statistics.median(times) if times else None, "slowest_ms": max(times) if times else None,
                    "changed": statistics.median(changed) if changed else None})
    return out


def gaps(frames: list[dict], limit: int = GAP_LIST) -> list[dict]:
    """The waits between consecutive snapshots on proseg's clock longer than 5 s and than 10 iterations (a pause, a slow
    step or proseg waiting): the `limit` longest, in iteration order."""
    found = []
    for a, b in zip(frames, frames[1:]):
        try:
            dt = float(b["t"]) - float(a["t"])
        except (KeyError, TypeError, ValueError):
            continue
        it = iteration_ms(b) or 0.0
        if dt > max(5.0, 10 * it / 1000):
            found.append({"seconds": dt, "before": b.get("iter"), "seq": b.get("seq")})
    found = sorted(found, key=lambda g: -g["seconds"])[:limit]
    return sorted(found, key=lambda g: g["seq"] if isinstance(g["seq"], int) else 0)


def temperature_schedule(meta: dict) -> dict | None:
    """What proseg's own temperature schedule needs (schedule_temperature): {burnin, samples} from the run's info, and
    `temperature` when one was set at the start (Advanced options: proseg's --temperature holds for every iteration);
    None without the info (a run still loading, or made before it)."""
    sched = (meta.get("info") or {}).get("schedule")
    if not isinstance(sched, dict):
        return None
    out = {"burnin": sched.get("burnin"), "samples": sched.get("samples")}
    fixed = (meta.get("options") or {}).get("temperature")
    if is_number(fixed):
        out["temperature"] = float(fixed)
    return out


def schedule_temperature(frame: dict, schedule: dict | None) -> float | None:
    """The temperature proseg's own schedule gives a snapshot's iteration (proseg 3.2.0, main.rs): 1 in burn-in and
    sampling; in the hill-climb cooling^k at its k-th iteration (k from 0; cooling = 0.01^(1 / burn-in iterations));
    1 again past the end of the standard run (phase "continued"); a temperature set at the start everywhere. None when it
    cannot be told (no schedule, an unknown phase)."""
    if not isinstance(schedule, dict):
        return None
    if is_number(schedule.get("temperature")):
        return float(schedule["temperature"])
    phase = frame.get("phase")
    if phase in ("init", "burnin", "upscale", "sampling", "recorded", "continued"):
        return 1.0
    burnin, samples, it = schedule.get("burnin"), schedule.get("samples"), frame.get("iter")
    if phase != "hillclimb" or not all(is_number(v) for v in (burnin, samples, it)) or burnin <= 0:
        return None
    return math.exp(math.log(0.01) / burnin * max(0, it - burnin - samples - 1))


def scheduled_temperature(frame: dict, previous: dict | None, schedule: dict | None = None) -> bool:
    """Whether a temperature move between two snapshots is proseg's own schedule, never a knob change (the timeline's rule,
    frontend/src/timeline.js `scheduledTemperature`): the snapshot's temperature is the one the schedule gives its
    iteration (schedule_temperature, within TEMPERATURE_TOLERANCE). A temperature set live is a knob change wherever it is
    set, the hill-climb and the first iteration past the standard run included. Without a schedule (no info): the
    hill-climb and the step after it count as the schedule's, the old rule."""
    expected = schedule_temperature(frame, schedule)
    t = (frame.get("live") or {}).get("temperature", frame.get("temperature"))
    if expected is None or not is_number(t):
        return frame.get("phase") == "hillclimb" or (previous or {}).get("phase") == "hillclimb"
    return abs(t - expected) <= TEMPERATURE_TOLERANCE * max(expected, 1e-3)


def knob_changes(frames: list[dict], moves: list | None = None, schedule: dict | None = None) -> list[dict]:
    """Each live knob change as the snapshots show it (a snapshot's `live` values against the previous one's, as the
    timeline marks them): {seq, iter, t, flag, before, after}. The schedule's own temperature moves (scheduled_temperature
    with `schedule`, temperature_schedule's) are left out; a `moves` list gets them instead, as stretches of one phase:
    {phase, first, last, seq, before, after} (iterations)."""
    out, previous = [], None
    for f in frames:
        live = f.get("live")
        if not isinstance(live, dict):
            continue
        if previous is not None:
            before, skip = previous["live"], scheduled_temperature(f, previous, schedule)
            for flag in sorted(set(live) | set(before)):
                if live.get(flag) == before.get(flag):
                    continue
                if skip and flag == "temperature":
                    if moves is not None:
                        last = moves[-1] if moves else None
                        if last and last["phase"] == f.get("phase") and last["seq"] == previous.get("seq"):
                            last.update(last=f.get("iter"), seq=f.get("seq"), after=live.get(flag))
                        else:
                            moves.append({"phase": f.get("phase"), "first": f.get("iter"), "last": f.get("iter"),
                                          "seq": f.get("seq"), "before": before.get(flag), "after": live.get(flag)})
                    continue
                out.append({"seq": f.get("seq"), "iter": f.get("iter"), "t": f.get("t"), "flag": flag,
                            "before": before.get(flag), "after": live.get(flag)})
        previous = f
    return out


def move_text(moves: list[dict], schedule: dict | None = None) -> str:
    """proseg's own temperature moves in a few words: "temperature 1 -> 0.323594 over iterations 402-450 (the hill-climb
    cools it); 0.323594 -> 1 at iteration 451 (sampling on past the standard run)"."""
    parts = []
    for i, m in enumerate(moves):
        where = f"at iteration {m['first']}" if m["first"] == m["last"] else f"over iterations {m['first']}-{m['last']}"
        why = ("the hill-climb cools it" if m["phase"] == "hillclimb" else
               "sampling on past the standard run" if m["phase"] == "continued" else
               "the temperature set at the start" if is_number((schedule or {}).get("temperature")) else
               f"proseg's {m['phase']}")
        parts.append(f"{'temperature ' if not i else ''}{value(m['before'])} -> {value(m['after'])} {where} ({why})")
    return "; ".join(parts)


def option_labels() -> dict[str, str]:
    try:
        from . import proseg_options
        return {o["flag"]: o["label"] for o in proseg_options.OPTIONS}
    except Exception:                                     # noqa: BLE001 - labels are a nicety
        return {}


def knob_name(flag: str, labels: dict) -> str:
    return f"{labels[flag]} ({flag})" if flag in labels else flag


def short_name(flag: str, labels: dict) -> str:
    """An option's label without its hint ("Cell compactness" for "Cell compactness (lower = rounder cells)")."""
    return labels[flag].split(" (")[0] if flag in labels else flag


def step_stats(results: list) -> list[str]:
    """Median and slowest time of each step over the results kept (ms), in the worker's order."""
    by_step: dict[str, list[float]] = {}
    for _, summary in results:
        for key, v in (summary.get("ms") or {}).items() if isinstance(summary, dict) else ():
            try:
                by_step.setdefault(key, []).append(float(v))
            except (TypeError, ValueError):
                continue
    order = [s for s in STEPS if s in by_step] + sorted(set(by_step) - set(STEPS))
    return [f"{s} {statistics.median(by_step[s]):.0f} ({max(by_step[s]):.0f})" for s in order]


def iteration_of(frames: list[dict], seq) -> object:
    """The iteration of snapshot `seq` (the frames are indexed by it), else "?"."""
    if isinstance(seq, int) and 0 <= seq < len(frames) and frames[seq].get("seq") == seq:
        return frames[seq].get("iter", "?")
    for f in frames:
        if f.get("seq") == seq:
            return f.get("iter", "?")
    return "?"


def phase_of(frames: list[dict], seq) -> str | None:
    if isinstance(seq, int) and 0 <= seq < len(frames) and frames[seq].get("seq") == seq:
        return frames[seq].get("phase")
    return None


# ---- the parts ---------------------------------------------------------------------------------------------------------
def header(out: Text, src: Sources, run_id: str | None):
    out.title("SoPA companion log")
    zone = time.strftime("%Z, UTC%z")
    out.row("Written", f"{when(time.time())} by the engine ({zone})")
    out.row("About", f"run {run_id}" if run_id else "this engine session (no run was open)")


def prepared_text(prepared: dict) -> str:
    """The run's input as the zarr helper prepared it. A prior without nuclei seeds proseg from every transcript in a
    prior cell (zarr_bridge/helper.py prepare), which the log says instead of a nuclear count."""
    seeds = ("no nuclei chosen, so every one seeds its cell (proseg's rule without nuclei)"
             if prepared.get("seeded_from") == "cells" else f"{num(prepared.get('nuclear'))} nuclear")
    return (f"{num(prepared.get('transcripts'))} transcripts in the region, {num(prepared.get('assigned'))} in a prior "
            f"cell, {seeds}, {num(prepared.get('prior_cells'))} prior cells, {prepared.get('seconds', '-')} s")


def region_text(meta: dict) -> str:
    region, box = meta.get("region") or {}, meta.get("box") or [0, 0, 0, 0]
    x0, y0, x1, y1 = (float(v) for v in box)
    dims = f"{x1 - x0:,.0f} x {y1 - y0:,.0f} µm"
    kind = region.get("kind", "box")
    if kind == "full":
        return f"the full sample ({dims})"
    if kind == "polygon":
        return f"a lasso of {len(region.get('points') or [])} points, {dims} across, from ({x0:,.0f}, {y0:,.0f}) µm"
    return f"a rectangle of {dims} from ({x0:,.0f}, {y0:,.0f}) to ({x1:,.0f}, {y1:,.0f}) µm"


def prior_text(prior) -> str:
    if not isinstance(prior, dict):
        return "the Xenium bundle's own cell_id and overlaps_nucleus"

    def one(spec):
        if not spec:
            return "none"
        if spec.get("kind") == "shapes":
            return f"shapes {spec.get('key')}"
        return f"column {spec.get('column')}"
    return f"cells: {one(prior.get('cells'))}; nuclei: {one(prior.get('nuclei'))}"


def standing(meta: dict, frames: list[dict]) -> str:
    """Where the run stands, in a line: how far it got (and proseg's standard run), what it is doing, what is saved."""
    state = meta.get("state")
    last = frames[-1].get("iter") if frames else (meta.get("last") or {}).get("iter")
    done = meta.get("schedule_done") if isinstance(meta.get("schedule_done"), dict) else None
    held = bool(done) and not meta.get("continued")
    parts = []
    if done:
        end = done.get("iter")
        on = meta.get("continued")
        parts.append(f"proseg's standard run completed at iteration {end}"
                     + (f", then sampled on to iteration {last}" if on and last is not None and last != end else
                        ", then sampled on" if on else ""))
    elif last is not None:
        parts.append(f"at iteration {last}")
    parts.append({"preparing": "preparing the region", "running": "sampling",
                  "paused": "paused there (Finish & save keeps this result)" if held else "paused",
                  "finishing": "saving its results", "done": "finished",
                  "failed": "failed" + (f": {meta.get('message')}" if meta.get("message") else ""),
                  "stopped": "stopped without saving"}.get(state, str(state)))
    if state in ACTIVE:
        parts.append("not saved yet")
    elif state == "done":
        parts.append("its results saved" if meta.get("outputs") else "no results saved")
        save = meta.get("zarr_save") or {}
        if save.get("state") == "saved":
            parts.append(f"saved into the sample as {save.get('shapes_key')} + {save.get('table_key')}")
    return "; ".join(parts)


def options_brief(options: dict, labels: dict) -> str:
    shown = {k: v for k, v in (options or {}).items() if k != "nthreads"}
    if not shown:
        return "none changed (proseg's defaults)"
    return f"{len(shown)} changed: " + "; ".join(f"{short_name(k, labels)} {value(v)}" for k, v in sorted(shown.items()))


def schedule_brief(meta: dict) -> str:
    sched = (meta.get("info") or {}).get("schedule") or {}
    changed = [k for k in SCHEDULE_FLAGS if k in (meta.get("options") or {})]
    if all(is_number(sched.get(k)) for k in ("burnin", "samples", "hillclimb")):
        total = sched["burnin"] + sched["samples"] + sched["hillclimb"]
        text = (f"{num(int(total))} iterations: burn-in {sched['burnin']}, sampling {sched['samples']} (the last "
                f"{sched.get('recorded')} averaged), hill-climb {sched['hillclimb']}")
    else:
        text = "not reported yet (proseg has not started)"
    return ("set under Advanced options: " if changed else "proseg's default: ") + text


def noted_events(src: Sources, run_id: str | None) -> list[tuple]:
    """What the engine noted itself (Noted): the run's own, or without a run every one; none without a Noted (a log
    built from fewer sources)."""
    noted = getattr(src, "noted", None)
    return noted.list(run_id) if noted is not None else []


def run_noted(src: Sources, run_id: str | None) -> list[str]:
    """What went wrong that the engine noted for the run (its events) and for the analysis worker (failures only), and
    what the engine noted itself (Noted): the run's own, or without a run every one (a failed request, a failed run).
    Oldest first, so the last is the newest."""
    events = src.analysis.log_info(run_id)["events"]
    worker = [e for e in events if e[1] is None and any(w in e[2] for w in ("could not", "stopped", "timed out"))]
    own = [e for e in events if run_id is not None and e[1] == run_id]
    engine = noted_events(src, run_id)
    return [text for _, _, text in sorted(worker + own + engine, key=lambda e: e[0])]


def summary_part(out: Text, src: Sources, run_id: str | None):
    """The few lines that say it all (the page adds its own "Page noted" row under them)."""
    out.title(SUMMARY, "-")
    if run_id is None:
        listed = src.runs.list()
        active = [r for r in listed if r.get("state") in ACTIVE]
        out.row("Runs", f"{len(listed)} ({len(active)} active)")
    else:
        run = src.runs.get(run_id)
        meta = run.public()
        with run.lock:
            frames = [dict(f) for f in run.frames]
        labels = option_labels()
        out.row("Sample", f"{meta.get('sample_name')}: {region_text(meta)}, "
                          + ("whole-sample mode" if meta.get("mode") == "tiled" else "full detail"))
        if meta.get("kind") == "stain":
            method = "InstanSeg" if meta.get("stain", {}).get("method") == "instanseg" else "Cellpose"
            out.row("Kind", f"Stain segmentation · {method}")
            resolution = (meta.get("stain") or {}).get("resolution") or {}
            if resolution:
                from .stain_levels import level_name, resolution_name
                out.row("Resolution", f"{resolution_name(resolution) or level_name(0)} · "
                                      f"{resolution.get('um_per_px')} µm per pixel")
            out.row("Settings", json.dumps(meta.get("stain")))
            out.row("Where it stands", f"{meta['state']}; tiles {meta['progress']['done']} of {meta['progress']['total']}")
            out.row("Cells and sizes", json.dumps(meta.get("summary")))
            noted = [text for _, _, text in noted_events(src, run_id)]
            out.row("Engine noted", noted_text(noted) if noted else meta.get("message") or "nothing went wrong")
            return
        out.row("Prior", prior_text(meta.get("prior")))
        out.row("Schedule", schedule_brief(meta))
        out.row("Options", options_brief(meta.get("options"), labels))
        out.row("Where it stands", standing(meta, frames))
        state = getattr(src.qc, "states", {}).get(run_id) if src.qc is not None else None
        if src.qc is None:
            check = "not part of this engine"
        elif state is not None:
            check = "on" if state.enabled else "off"
        else:
            check = stored_check(run, src)
        out.row("Mixing check", check)
    noted = run_noted(src, run_id)
    out.row("Engine noted", noted_text(noted) if noted else "nothing went wrong")


def noted_text(noted: list[str]) -> str:
    """The summary's "Engine noted" for what went wrong: how many things, and the newest."""
    return (f"{len(noted)} thing{'s' if len(noted) != 1 else ''}: {noted[-1]}"
            + (" (and more below)" if len(noted) > 1 else ""))


def stored_check(run, src: Sources) -> str:
    """A run's mixing check switch as its settings.json keeps it, read without loading the check (engine/qc.py
    stored_switch: a file from before the check was opt-in follows [qc] enabled_by_default)."""
    try:
        from .qc import stored_switch
        settings = json.loads((run.dir / "qc" / "settings.json").read_text())
        default = bool(getattr(getattr(src.qc, "s", None), "enabled_by_default", False))
        return ("on" if stored_switch(settings, default)[0] else "off") + " (not looked at in this session)"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return "never switched on (new runs start off)"


def engine_part(out: Text, src: Sources):
    cfg = src.config
    out.title("Engine", "-")
    out.row("Version", f"SoPA companion {__version__} (protocol {PROTOCOL})")
    code = code_hash()
    out.row("Code", f"sha256 {code['sha256'][:16]} over {code['files']} files (the files a deploy ships), in {ROOT}")
    out.row("proseg-live", program_row(cfg))
    out.row("Process", f"pid {os.getpid()}, up {span(time.time() - src.started)} (since {when(src.started)}), "
                       f"Python {platform.python_version()}, {platform.system()} {platform.release()}")
    paths = cfg.get("paths", {})
    out.row("Folders", f"runs and caches: {paths.get('data_dir')}\nprivate state and logs: {paths.get('state_dir')}")
    server = cfg.get("server", {})
    out.row("Address", f"http://{server.get('host', '127.0.0.1')}:{server.get('port')} (loopback only)")
    proseg = cfg.get("proseg", {})
    # One run at a time (engine/runs.py one_run), unless a test engine keeps its unsaved runs and their limit.
    runs = f"at most {proseg.get('max_active_runs', 2)} active runs" if proseg.get("keep_unsaved") else "one run at a time"
    out.row("Threads", f"{proseg.get('threads')} for detail runs, {proseg.get('tiled_threads', proseg.get('threads'))} "
                       f"for whole-sample runs; {runs}")
    listed = src.runs.list()
    active = [r["id"] for r in listed if r.get("state") in ACTIVE]
    out.row("Runs", f"{len(listed)} ({len(active)} active{': ' + ', '.join(active) if active else ''})")
    spool = getattr(src.runs, "spool", None)
    if spool is not None:              # where proseg's counts the app asks for wait (engine/counts_spool.py)
        out.row("Counts spool", f"{spool.folder}: {spool.usage_mb():,.1f} MB now; at most {spool.run_cap_mb:,} MB a "
                                f"run, {spool.engine_cap_mb:,} MB in all")
    out.part("Analysis worker", worker_part, src)
    out.part("mixing check", qc_engine_part, src)
    out.part("engine events", engine_events_part, src)


# How the program's place was found, for one that appeared after the start (its fix put it where the engine looks).
FIXED_BY = {"build": "source", "install": "environment", "setting": "setting"}


def program_row(cfg: dict) -> str:
    """The proseg-live row: the file as it is now (looked at again, as a run's start does), and how it was found
    (locate.py); not there, what makes it."""
    binary = Path(cfg["proseg"]["binary"])
    found = (cfg.get("locate") or {}).get("proseg") or {}
    try:
        info = binary.stat()
        digest = file_hash(binary)
    except OSError:
        fix = found.get("fix") or "build"
        if fix == "install":
            return "missing: install it as the README's Install section says"
        if fix == "setting":
            return f"missing: [proseg] binary names {binary}, which is not there"
        return "missing: native/build.sh builds it"
    how = FIXED_BY.get(found.get("fix") or "build", "source") if found.get("how") == "missing" else found.get("how")
    said = f"{binary}: {size(info.st_size)}, built {when(info.st_mtime)}, sha256 {(digest or '-')[:16]}"
    return said + (f" ({locate.WHERE[how]})" if how in locate.WHERE else "")


def why_not(a: dict) -> str:
    """Why Clusters & UMAP cannot run here, as the engine says it (AnalysisManager.unavailable_why)."""
    return a.get("why_not") or "the analysis worker cannot run here"


def worker_part(out: Text, src: Sources):
    a = src.analysis.log_info()
    w = a["worker"]
    if not a["available"]:
        out.row("Analysis worker", f"not available: {why_not(a)}")
        out.row("Clusters & UMAP", f"cannot run ({why_not(a)})")
        return
    cpu = a.get("backend") == "cpu"
    if not w["pid"]:
        out.row("Analysis worker", f"{a['python']}: not running now (it starts with the first run)")
    elif cpu:
        why = f" ({a['backend_why']})" if a.get("backend_why") else ""
        out.row("Analysis worker", f"{a['python']} on the CPU with {a.get('threads') or '?'} threads{why}: running "
                                   f"(pid {w['pid']}, ready since {when(w['since'])})")
    else:
        out.row("Analysis worker", f"{a['python']} on GPU {a['gpu']}: running (pid {w['pid']}, ready since "
                                   f"{when(w['since'])})")
    interval = a["min_interval"]
    cadence = ("every run's cells / 15,000 s apart, 0.5-6 s (the CPU's spacing for min_interval_s = 0)"
               if cpu and interval == 0 else
               "back to back, as fast as the worker goes (min_interval_s = 0)" if interval == 0 else
               "every run's cells / 15,000 s apart, 0.5-6 s (min_interval_s = \"auto\")" if interval is None else
               f"at most one automatic analysis per run every {interval:g} s (min_interval_s)")
    off = a["runs"]["switched_off"]
    out.row("Clusters & UMAP", f"on for every active run; {cadence}"
                               + (f"\nswitched off through the API: {', '.join(off)}" if off else ""))
    defaults, fixed = a.get("defaults"), a.get("fixed") or {}
    if defaults:                       # the settings runs start from, and what new runs start with (the last applied)
        out.row("Settings", f"{settings_text(defaults)} by default"
                            + (f" ({settings_text(fixed)} from config.toml)" if fixed else ""))
        last = {k: v for k, v in (a.get("new_runs") or defaults).items() if defaults.get(k) != v}
        out.row("New runs", f"start with {settings_text(last)} (the settings last applied)" if last else
                "start with the defaults")


def settings_text(values: dict) -> str:
    """Clusters & UMAP settings as the log says them, in the engine's order: "resolution 1.5, n_neighbors 30"."""
    return ", ".join(f"{k} {value(v)}" for k, v in values.items())


def engine_events_part(out: Text, src: Sources):
    events = [e for e in src.analysis.log_info()["events"] if e[1] is None]
    out.heading("Engine events (the analysis worker)")
    if not events:
        out.line("none")
    for t, _, text in events[-60:]:
        out.line(f"{when(t)}  {text}")
    if getattr(src, "noted", None) is None:
        return
    failed = noted_events(src, None)
    out.heading("Engine events (what failed: requests, helpers, runs and saves)")
    if not failed:
        out.line("none")
    for t, _, text in failed[-60:]:
        out.line(f"{when(t)}  {text}")


def qc_engine_part(out: Text, src: Sources):
    qc, models = src.qc, src.qc_models
    if qc is None or models is None:
        out.row("Mixing check", "not part of this engine")
        return
    prefs = getattr(qc, "prefs", {}) or {}
    on = bool(prefs.get("enabled"))
    out.row("Mixing check", offered_text(models, src.analysis)
            + f"; on for new runs: {'yes' if on else 'no'} (the last choice"
            + ("" if on else "; nothing is learned until it is switched on") + ")")
    learning = [s for s in models.snapshot().values() if s.get("state") in ("waiting", "learning", "failed")]
    for s in learning:
        out.row("", f"sample {s['sample']}: {s['state']}" + (f" ({s['reason']})" if s.get("reason") else "")
                + (f": {s['message']}" if s.get("message") else ""))


def offered_text(models, analysis) -> str:
    """Whether the mixing check is offered here, and why (its probe's answer: engine/qc.py capability); without Clusters
    & UMAP, which score its frames, it is not, whatever the probe says."""
    if not getattr(analysis, "available", False):
        why = analysis.unavailable_why() if hasattr(analysis, "unavailable_why") else "Clusters & UMAP cannot run here"
        return f"not offered here: it needs Clusters & UMAP ({why})"
    cap = models.capability() if hasattr(models, "capability") else \
        {"state": "ready" if models.available else "unavailable", "why": None}
    state, why = cap.get("state"), cap.get("why") or "no reason given"
    if state == "unknown":
        error = (cap.get("probe") or {}).get("error") or "no answer"
        return f"offered; its Python could not be asked ({error}); asked again later"
    return {"ready": f"offered ({why})", "checking": "being checked"}.get(state, f"not offered here: {why}")


def schedule_text(meta: dict, moves: list[dict] | None = None) -> str:
    """proseg's schedule as it reported it (info.json), else as the run asked for it (runs of older versions chose
    one), whether proseg completed it, and its own temperature moves (`moves`: knob_changes' stretches)."""
    info, params = meta.get("info") or {}, meta.get("schedule_params") or {}
    sched = info.get("schedule") or {}
    if sched:
        text = (f"burn-in {sched.get('burnin')}, samples {sched.get('samples')}, recorded {sched.get('recorded')}, "
                f"hill-climb {sched.get('hillclimb')} iterations ({num(sched.get('morphology_steps'))} morphology steps each)")
    elif params:
        text = (f"burn-in {params.get('burnin-samples')}, samples {params.get('samples')}, recorded "
                f"{params.get('recorded-samples')}, hill-climb {params.get('hillclimb')} iterations")
    else:
        text = "not reported yet (proseg has not started)"
    if params.get("label"):
        text = f"{params['label']}: {text}"
    if info.get("indefinite") or params.get("indefinite"):
        text += "; open-ended (until finished)"
    done = meta.get("schedule_done")
    if isinstance(done, dict):
        text += f"\nproseg completed its schedule at iteration {done.get('iter')}"
        text += ", then sampled on" if meta.get("continued") else " and holds there"
    if moves:
        text += "\n" + move_text(moves, temperature_schedule(meta))
    return text


def run_part(out: Text, src: Sources, run_id: str):
    run = src.runs.get(run_id)
    meta = run.public()
    if meta.get("kind") == "stain":
        out.title(f"Stain segmentation {run_id}", "-")
        for key in ("sample_path", "region", "stain", "stain_runtime", "stain_fallback", "progress", "summary", "zarr_save"):
            out.row(key.replace("_", " "), json.dumps(meta.get(key)))
        out.heading("Tile times and helper log")
        try:
            out.row("Tiles", (run.dir / "stain" / "tiles.json").read_text()[-60000:])
        except OSError:
            pass
        try:
            out.row("Helper", (run.dir / "stain.log").read_text(errors="replace")[-60000:])
        except OSError:
            pass
        try:                            # the training patches' helper: what it printed, a failure's traceback with it
            out.row("Patch helper", (run.dir / "stain-patches.log").read_text(errors="replace")[-20000:])
        except OSError:
            pass
        return
    with run.lock:
        frames = [dict(f) for f in run.frames]
    labels = option_labels()
    schedule = temperature_schedule(meta)
    moves: list[dict] = []
    changes = knob_changes(frames, moves, schedule)
    out.title(f"Run {run_id}", "-")
    out.row("Sample", f"{meta.get('sample_name')} ({meta.get('sample_path') or 'a Xenium bundle'}), "
                      f"{meta.get('sample_kind', 'xenium')} sample {meta.get('sample')}")
    out.row("Region", region_text(meta))
    out.row("Mode", "whole-sample (tiled snapshots)" if meta.get("mode") == "tiled" else "full detail")
    out.row("Prior", prior_text(meta.get("prior")))
    out.row("Schedule", schedule_text(meta, moves))
    options = meta.get("options") or {}
    out.row("Options", "; ".join(f"{knob_name(k, labels)} = {value(v)}" for k, v in sorted(options.items()))
            or "none changed (proseg's defaults)")
    out.row("proseg args", " ".join(str(a) for a in meta.get("proseg_args") or []) or "-")
    out.row("Threads", num(meta.get("threads")))
    out.row("Speed", meta.get("speed") or "-")
    if "start_paused" in meta:                           # runs made before every run started paused on its prior
        out.row("Started paused", "yes (on the prior)" if meta["start_paused"] else "no")
    ack = meta.get("ack") or {}
    state = f"{meta.get('state')} (phase {meta.get('phase')})" + (f": {meta['message']}" if meta.get("message") else "")
    if ack:
        state += f"\nproseg's last acknowledgement: {ack.get('state')} at iteration {ack.get('iter')}"
    out.row("State", state)
    out.row("Times", f"created {local(meta.get('created'))}, started {local(meta.get('started'))}, finished "
                     f"{local(meta.get('finished'))}")
    prepared = meta.get("prepared") or {}
    if prepared:
        out.row("Prepared", prepared_text(prepared))
    info = meta.get("info") or {}
    if info:
        model = [f"{num(info[k])} {name}" for k, name in (("transcripts", "transcripts"), ("cells", "cells"),
                                                          ("genes", "genes"), ("layers", "layers")) if k in info]
        if "burnin_voxel_size" in info and "voxel_size" in info:
            model.append(f"voxels {value(info['burnin_voxel_size'])} µm in burn-in, then {value(info['voxel_size'])} µm")
        if "prior_weight" in info:
            model.append(f"prior weight {value(info['prior_weight'])}")
        if "factorization" in info:
            model.append(f"factorisation {'on' if info['factorization'] else 'off'}")
        if info.get("proseg"):
            model.append(f"proseg {info['proseg']}")
        out.row("Modelled", ", ".join(model))
    if meta.get("retried_without_factorization"):
        out.row("Retried", "proseg could not start with factorisation on this region and was retried without it")
    out.row("Snapshots", f"{num(len(frames))}" + (f" (iterations {frames[0].get('iter')}-{frames[-1].get('iter')})"
                                                   if frames else "") + f", {size(meta.get('disk_bytes'))} on disk")
    out.part("proseg's count", counts_part, src, run_id, meta, frames)
    outputs = meta.get("outputs") or []
    if outputs:
        out.row("Results", "\n".join(f"{o.get('name')} ({size(o.get('bytes'))})" for o in outputs))
    if meta.get("exit_code") is not None:
        out.row("proseg exit", str(meta.get("exit_code")))
    save = meta.get("zarr_save")
    if save:
        out.row("Saved to sample", f"{save.get('state')}: {save.get('shapes_key')} + {save.get('table_key')}"
                + (f", {num(save.get('cells'))} cells" if save.get("cells") else "")
                + (f" ({save.get('message')})" if save.get("message") else ""))
    live_values = meta.get("live_values") or {}
    if live_values:
        out.row("Knobs asked", "; ".join(f"{knob_name(k, labels)} = {value(v)}" for k, v in sorted(live_values.items())))
    out.part("phases", phases_part, frames)
    out.part("knobs you changed", knobs_part, frames, labels, schedule, changes)
    out.part("Clusters & UMAP", analysis_part, src, run_id, frames)
    out.part("mixing check", qc_run_part, src, run, meta)
    out.part("Go deeper", deeper_run_part, src, run_id)
    out.part("events", run_events_part, src, run_id)
    out.part("proseg's log", proseg_part, run)


COUNT_REASONS = ("init", "upscale", "request", "every", "pause", "hold", "finish", "end")   # a counts line's, in order


def counts_part(out: Text, src: Sources, run_id: str, meta: dict, frames: list[dict]):
    """The run's row for proseg's own count (counts_text), with what its spool folder held at most."""
    spool = getattr(src.runs, "spool", None)
    peak = spool.peak_mb(run_id) if spool is not None else None
    out.row("Proseg's count", counts_text(meta, frames, peak, getattr(spool, "run_cap_mb", None)))


def counts_text(meta: dict, frames: list[dict], peak_mb: float | None = None, cap_mb: int | None = None) -> str:
    """Proseg's own count, which every number about the run's cells reads: its basis (analysis/counts_file.py
    LIVE_COUNT) and how proseg-live keeps it; the iterations it counted, by why, and those that clipped a value its file
    could not hold (their lines' `saturated`); the bytes they took in the run's folder
    and in the spool (the most the run's spool folder held, `peak_mb`, of its cap); how often one came (proseg's clock);
    proseg's time to count one and to write it. Or why the run has none (its proseg_counts reason)."""
    reason = (meta.get("proseg_counts") or {}).get("reason")
    if reason == "old_proseg":
        return ("not kept: this proseg-live is older than proseg's own count, so the run has no clusters, mixing "
                "check, Go deeper or test of a setting; update proseg-live")
    if reason == "no_counts":
        return "not kept: this run was made before proseg's own count was kept"
    from ..analysis.counts_file import LIVE_COUNT
    text = ("proseg's count, averaged over its random background draw; at an iteration counted for a pause, the hold, a "
            "finish or the end, the count proseg saves there (one random background draw)" if LIVE_COUNT == "averaged"
            else "proseg's count, one random background draw (the sampled count)")
    kept = (meta.get("info") or {}).get("counts")
    if isinstance(kept, dict):
        text += f" ({kept.get('mode')} mode{', with its start block' if kept.get('start') else ''})"
    counted = [f for f in frames if isinstance(f.get("counts"), dict)]
    if not counted:
        return text + "\nno iteration counted yet"
    lines = [text]
    why: dict[str, int] = {}
    for f in counted:
        why[str(f["counts"].get("reason"))] = why.get(str(f["counts"].get("reason")), 0) + 1
    order = [r for r in COUNT_REASONS if r in why] + sorted(r for r in why if r not in COUNT_REASONS)
    reasons = ", ".join(f"{r} {num(why[r])}" for r in order)
    lines.append(f"{num(len(counted))} iteration{'s' if len(counted) != 1 else ''} counted ({reasons}), the newest at "
                 f"iteration {counted[-1].get('iter')}")
    # A value past what the file's 16 bits hold is kept clipped (counts_file.py), and the line says how many: never
    # silent, though no number on the page says it (the saved table keeps the true count).
    clipped = [f for f in counted if is_number(f["counts"].get("saturated")) and f["counts"]["saturated"] > 0]
    if clipped:
        from ..analysis.counts_file import E_SCALE, U16_MAX
        most = max(int(f["counts"]["saturated"]) for f in clipped)
        lines.append(f"{num(len(clipped))} of them clipped a cell's count of a gene ({num(most)} values at most), the "
                     f"first at iteration {clipped[0].get('iter')}: proseg-live's file holds {U16_MAX / E_SCALE:,.2f} "
                     f"averaged and {num(U16_MAX)} transcripts at most, so those cells read low there; the saved table "
                     f"keeps the true count")
    stored = {True: 0, False: 0}                     # durable (the run's folder) or spooled
    for f in counted:
        line = f["counts"]
        if is_number(line.get("bytes")):
            stored[not os.path.isabs(str(line.get("file", "")))] += line["bytes"]
    spooled = f"{size(stored[False])} written to the spool"
    if peak_mb is not None:
        spooled += f" (it held {peak_mb:,.1f} MB at most" + (f", of {cap_mb:,} MB" if cap_mb else "") + ")"
    lines.append(f"{size(stored[True])} kept in the run's folder, {spooled}")
    times = [f["t"] for f in counted if is_number(f.get("t"))]
    gaps = [b - a for a, b in zip(times, times[1:]) if b >= a]
    if gaps:
        lines.append(f"one every {statistics.median(gaps):.2f} s (median, proseg's clock)")
    timing = [f["counts"]["ms"] for f in counted if isinstance(f["counts"].get("ms"), dict)]
    extract = [t["extract"] for t in timing if is_number(t.get("extract"))]
    write = [t["write"] for t in timing if is_number(t.get("write"))]
    if extract:
        lines.append(f"counting took {statistics.median(extract):.0f} ms (slowest {max(extract):.0f}) on proseg's main "
                     f"thread" + (f", writing {statistics.median(write):.0f} ms" if write else ""))
    return "\n".join(lines)


def phases_part(out: Text, frames: list[dict]):
    out.heading("Phases (from proseg's snapshots; times on proseg's clock)")
    if not frames:
        out.line("no snapshot yet")
        return
    out.line(f"{'phase':<10}{'snapshots':>10}  {'iterations':<14}{'seconds':>9}  {'per iteration':<22}{'voxel':>7}"
             f"{'changed':>12}")
    for r in phases(frames):
        each = "-" if r["median_ms"] is None else f"{r['median_ms'] / 1000:.2f} s (slowest {r['slowest_ms'] / 1000:.2f})"
        seconds = "-" if r["seconds"] is None else f"{r['seconds']:.1f}"
        changed = "-" if r["changed"] is None else num(int(r["changed"]))
        iterations = f"{r['first']}-{r['last']}"
        voxel = f"{value(r['voxel'])} µm"
        out.line(f"{str(r['phase']):<10}{r['snapshots']:>10}  {iterations:<14}{seconds:>9}  {each:<22}{voxel:>7}"
                 f"{changed:>12}")
    waits = gaps(frames)
    if waits:
        out.line("Long waits between snapshots (a pause, a slow step or proseg waiting):")
        for g in waits:
            out.line(f"{g['seconds']:.1f} s before iteration {g['before']}", 4)


def knobs_part(out: Text, frames: list[dict], labels: dict, schedule: dict | None = None, changes: list | None = None):
    """The knobs the user changed while the run sampled (proseg's own temperature schedule is under Schedule)."""
    out.heading("Knobs you changed (as the snapshots show them)")
    if changes is None:
        changes = knob_changes(frames, None, schedule)
    if not changes:
        out.line("none")
    for c in changes[:KNOB_LIST]:
        clock = f" ({c['t']:.1f} s)" if isinstance(c.get("t"), (int, float)) else ""
        out.line(f"iteration {c['iter']}{clock}: {knob_name(c['flag'], labels)} {value(c['before'])} -> "
                 f"{value(c['after'])}")
    if len(changes) > KNOB_LIST:
        out.line(f"... and {len(changes) - KNOB_LIST} more")


def analysis_part(out: Text, src: Sources, run_id: str, frames: list[dict] | None = None):
    frames = frames or []
    a = src.analysis.log_info(run_id)
    r = a["run"]
    out.heading("Clusters & UMAP")
    if not a["available"]:
        out.line(f"cannot run: {why_not(a)}")
        return
    if r.get("explicit") and not r["enabled"]:
        out.row("  Switch", "off: switched off through the API")
    status = r["status"] or {}
    state, seq = status.get("state", "idle"), status.get("seq")
    at = f"iteration {iteration_of(frames, seq)}"
    out.row("  Status", {"ready": f"ready ({at})", "running": f"analysing {at}",
                         "error": f"failed at {at}: {status.get('error')}"
                                  + (" (too few cells here to group into clusters: no more automatic passes until "
                                     "there are more)" if status.get("reason") == "too_few_cells" else "")
                         }.get(state, state))
    if r["interval_s"] is not None:
        out.row("  Cadence", "back to back" if r["interval_s"] == 0 else
                f"at most one automatic analysis every {r['interval_s']:.2g} s")
    settings, defaults = r.get("settings"), a.get("defaults") or {}
    if settings:                       # the results below are all of the run's current settings (its generation)
        changed = {k: v for k, v in defaults.items() if k in settings and settings[k] != v}
        out.row("  Settings", settings_text(settings) + (
            f" (changed from the defaults: {settings_text(changed)}); the results below are of these settings"
            if r.get("changed") and changed else " (the defaults)"))
    results = r["results"]
    if not results:
        out.row("  Results", "none yet")
        return
    iters = [s.get("iter") for _, s in results if isinstance(s, dict)]
    out.row("  Results", f"{num(len(results))} kept"
                         + (f" (iterations {iters[0]}-{iters[-1]})" if iters and None not in (iters[0], iters[-1]) else "")
                         + "; each one is listed in the appendix at the end of the log")
    stats = step_stats(results)
    if stats:
        out.row("  Step times", "median (slowest) over the results kept, ms: " + " · ".join(stats))


def appendix_part(out: Text, src: Sources, run_id: str):
    """Every Clusters & UMAP result kept, one line each (the newest RESULT_LINES): the page moves this to the end."""
    run = src.runs.get(run_id)
    with run.lock:
        frames = [dict(f) for f in run.frames]
    results = src.analysis.log_info(run_id)["run"]["results"]
    if not results:
        return
    shown = results[-RESULT_LINES:]
    out.title(APPENDIX + (f" (the newest {RESULT_LINES} of {num(len(results))})" if len(results) > RESULT_LINES else ""))
    out.line("Each clustering result: its iteration, clusters, cells (a fresh layout when not warm-started) and step "
             "times", 0)
    for seq, s in shown:
        if not isinstance(s, dict):
            continue
        what = [f"{num(s['nclusters'])} clusters" if s.get("nclusters") is not None else None,
                f"{num(s['ncells'])} cells" if s.get("ncells") is not None else None,
                "fresh layout" if s.get("warm") is False else None]
        it = s.get("iter", iteration_of(frames, seq))
        upscale = " (finer voxels)" if phase_of(frames, seq) == "upscale" else ""
        out.line(f"iteration {it}{upscale}: {', '.join(w for w in what if w)} · {ms_line(s.get('ms'))}", width=APPENDIX_WRAP)


def qc_run_part(out: Text, src: Sources, run, meta: dict):
    out.heading("Mixing check")
    qc, models = src.qc, src.qc_models
    if qc is None or models is None:
        out.line("not part of this engine")
        return
    state = getattr(qc, "states", {}).get(run.meta["id"])
    if state is None:                                    # never looked at in this session: read its switch, write nothing
        out.row("  Switch", stored_check(run, src))
        enabled = False
    else:
        with state.lock:
            enabled, changes, key = state.enabled, state.switch_n, state.key
            start = dict(state.start) if state.start else None
            points = [state.points[k] for k in sorted(state.points)]
            errors = list(state.errors)[-5:]
            trial = dict(state.trial) if state.trial else None
        out.row("  Switch", f"{'on' if enabled else 'off'} ({changes} change{'s' if changes != 1 else ''})")
        if not enabled:
            out.row("  Model", "none used (the check is off)")
        elif key:
            out.row("  Model used", key)
        if start:
            out.row("  Start line", f"{start.get('per100')} in 100 look mixed ({num(start.get('mixed'))} of "
                                    f"{num(start.get('judged'))} judged cells)")
        if points:
            last = points[-1]
            out.row("  Results", f"{num(len(points))} iterations checked ({points[0].get('iter')}-{last.get('iter')});"
                                 f" newest: {last.get('per100')} in 100 (shown {last.get('shown')}) at iteration "
                                 f"{last.get('iter')}")
        for e in errors:
            out.row("  Error", f"iteration {iteration_of(run.frames, e.get('seq'))}: {e.get('error')}")
        if trial:
            out.row("  Trial", f"{trial.get('flag')} {value(trial.get('from'))} -> {value(trial.get('to'))}, verdict "
                               f"{trial.get('verdict')}" + (f", {trial['outcome']}" if trial.get("outcome") else ""))
    sample = models.snapshot().get(meta.get("sample"))
    if sample is None:
        out.row("  Sample", "not learned or looked at in this session")
    else:
        progress = sample.get("progress") or {}
        text = SAMPLE_STATES.get(sample.get("state"), str(sample.get("state"))) + \
            (f" ({sample['reason']})" if sample.get("reason") else "")
        if progress.get("of"):
            text += f": step {num(progress.get('step'))} of {num(progress.get('of'))}"
        if sample.get("eta_s"):
            text += f", about {span(sample['eta_s'])} left"
        if sample.get("message"):
            text += f": {sample['message']}"
        out.row("  Sample", text)
        model = sample.get("model") or {}
        if model:
            out.row("  Sample model", f"{model.get('id')} from {model.get('table')}: {num(model.get('cells'))} cells, "
                                      f"{num(model.get('genes'))} genes, learned {local(model.get('created'))} in "
                                      f"{span(model.get('seconds'))}, {num(model.get('epochs'))} passes, "
                                      f"{num(model.get('steps'))} steps")
            out.row("  Provenance", provenance_text(model.get("provenance")))
            out.row("  Go deeper", details_text(model, sample.get("detail")))
    if not enabled:
        return
    info = qc.log_info(run.meta["id"]) if hasattr(qc, "log_info") else None
    if info and info.get("seq") is not None:
        for name, text in mixing_detail(info, run.frames):
            out.row(f"  {name}", text)
    try:
        scheduler = src.analysis.describe_qc(run.meta["id"])
    except Exception:                                    # noqa: BLE001
        scheduler = {}
    status = scheduler.get("status") or {}
    if status.get("state") not in (None, "idle"):
        out.row("  Scoring", f"{status.get('state')}"
                + (f" iteration {iteration_of(run.frames, status.get('seq'))}" if status.get("seq") is not None else "")
                + (f": {status['error']}" if status.get("error") else ""))
    if scheduler.get("backfill"):
        out.row("  Backfill", f"{scheduler['backfill']} earlier iterations still to check")
    if scheduler.get("unavailable"):
        out.row("  Unavailable", scheduler["unavailable"])


def provenance_text(p) -> str:
    """How the run's model was learned against the reference settings (nothing names files outside the app, the log
    included): table, priors, passes, cells and whether it was learned with the reference
    settings (provenance `notebook_equivalent`)."""
    if not isinstance(p, dict):
        return "not recorded"
    diffs = ", ".join(str(d) for d in p.get("differences") or []) or "-"
    return (f"learned from {p.get('table') or '-'}" + (f" with {p['other_table']}" if p.get("other_table") else "")
            + f", {p.get('priors') or '?'} starting estimates, {num(p.get('epochs_run'))} passes over the cells (the "
            f"reference: {num(p.get('epochs_notebook'))}), {num(p.get('cells'))} cells; "
            + ("the reference settings" if p.get("notebook_equivalent") is True else f"not the reference settings ({diffs})"))


def details_text(model: dict, detail) -> str:
    """Whether the sample's model holds Go deeper's details (deep.npz), or how adding them goes."""
    detail = detail if isinstance(detail, dict) else {}
    state = detail.get("state")
    if model.get("deep"):
        return "the model holds its details (deep.npz)"
    if state in ("queued", "learning"):
        progress = detail.get("progress") or {}
        return f"adding the details: {state}" + (f", step {num(progress.get('step'))} of {num(progress.get('of'))}"
                                                 if progress.get("of") else "")
    if state == "failed":
        return f"the details could not be added ({detail.get('reason') or 'failed'}): Learn the details tries again"
    return "no details kept (learned before Go deeper): the window offers Learn the details"


def mixing_detail(info: dict, frames) -> list[tuple[str, str]]:
    """The numbers behind the check's words for its newest result (QcManager.log_info): the count it reads and the
    thresholds' measurement (basis, count, constants), judged coverage, transcript retention, the like-for-like change
    (deterministic on proseg's count: both sides scaled to the shallower depth) and its thresholds; a run with no start
    line (start_kind other than "cells") says it has nothing to compare with, never a change with a reason, and a change
    not worked out yet is never said to have too few cells."""
    rows = [("Newest result", f"iteration {info.get('iter') if info.get('iter') is not None else iteration_of(frames, info.get('seq'))}")]
    if info.get("basis") == "proseg":
        k = info.get("constants") if isinstance(info.get("constants"), dict) else {}
        count = ("the count proseg saves at this iteration (one random background draw)" if info.get("count") == "sampled"
                 else "proseg's count, averaged over its random background draw")
        rows.append(("Count", count + (f"; thresholds {k.get('id')} ({k.get('mode')} runs)" if k.get("id") else "")
                     + ("; not yet re-measured on proseg's count" if k.get("measured") is False else "")))
    c = info.get("coverage")
    if isinstance(c, dict):
        share = c.get("judged_share")
        rows.append(("Coverage", f"{num(c.get('judged'))} of {num(c.get('cells'))} cells judged (100 or more "
                                 f"transcripts of the model's genes)" + (f", {share * 100:.1f} %" if is_number(share) else "")
                     + f"; {num(c.get('empty'))} empty, {num(c.get('under20'))} with 1-19, {num(c.get('under100'))} "
                       f"with 20-99"))
    r = info.get("retention")
    if isinstance(r, dict) and is_number(r.get("ratio")):
        rows.append(("Retention", f"the cells hold {r['ratio'] * 100:.1f} % of the model genes' transcripts they held at "
                                  f"the start ({num(r.get('tx'))} now, {num(r.get('start_tx'))} at the start)"))
    m = info.get("matched")
    if isinstance(m, dict) and m.get("error"):
        rows.append(("Like for like", f"not worked out: {m['error']}"))
    elif isinstance(m, dict) and is_number(m.get("per100_now")):
        how = ("both scaled to the shallower of their two depths (no draws)" if not m.get("seeds") else
               f"each thinned to its shallower depth (sd {num(m.get('sd'))} over {num(m.get('seeds'))} draws)")
        rows.append(("Like for like", f"{m['per100_now']:.2f} in 100 now against {m.get('per100_start', 0):.2f} at the "
                                      f"start over {num(m.get('paired'))} cells judged in both, {how}"))
    ch = info.get("change")
    if info.get("start_kind") not in (None, "cells"):    # no start line: nothing is compared with a start
        rows.append(("Change", "none: this run started from nuclei (alone or with their cells), so it has no start line "
                               "to compare with"))
    elif isinstance(ch, dict):
        def pts(v):
            return f"{v:+.2f}" if is_number(v) else "-"
        verdict = {"fewer": "fewer look mixed", "more": "more look mixed", "same": "about the same"}.get(ch.get("direction"),
                                                                                                    "no verdict")
        rows.append(("Change", f"raw {pts(ch.get('raw'))}, like for like {pts(ch.get('matched'))} points since the "
                               f"start; it counts from {num(ch.get('threshold'))} (sampler) and "
                               f"{num(ch.get('model_threshold'))} (another learning of the model): {verdict}"
                     + (" (cells still forming)" if ch.get("forming") else "")
                     # Too few only for a change worked out on too few cells: without one, nothing was compared yet.
                     + ("" if ch.get("enough") is not False else " (too few cells to judge)" if is_number(ch.get("matched"))
                        else " (nothing compared with the start yet)")))
    return rows


def deeper_run_part(out: Text, src: Sources, run_id: str):
    """Go deeper for this run: the cell the engine follows, the request waiting, failures, and the answers kept."""
    out.heading("Go deeper")
    try:
        sched = (src.analysis.log_info(run_id)["run"] or {}).get("deeper") or {}
    except Exception:                                    # noqa: BLE001
        sched = {}
    manager = src.deeper.log_info(run_id) if src.deeper is not None and hasattr(src.deeper, "log_info") else {}
    kept = manager.get("run") or {}
    follow, requested = sched.get("follow"), sched.get("requested")
    if follow is None and not requested and not kept.get("kept") and not kept.get("on_disk") and not kept.get("pending"):
        out.line("not used for this run in this session")
        return
    out.row("  Following", f"cell {follow} (label - 1) with every live pass" if follow is not None else "no cell")
    if requested:
        out.row("  Asked", f"frame {requested[0]}, cell {requested[1]} (waiting for the worker)")
    if kept.get("pending"):
        pending = kept["pending"]
        out.row("  Waiting", f"frame {pending.get('seq')}, cell {pending.get('cell')} until the check (or its details) "
                             "are ready")
    if sched.get("failures"):
        out.row("  Failures", f"{num(sched['failures'])} in a row for the followed cell (each listed under the events)")
    out.row("  Answers", f"{num(kept.get('kept') or 0)} kept in memory"
            + (f", {num(kept['on_disk'])} on disk" if kept.get("on_disk") is not None else "")
            + f"; {num(kept.get('watchers') or 0)} page(s) watching")


def run_events_part(out: Text, src: Sources, run_id: str):
    events = [e for e in src.analysis.log_info(run_id)["events"] if e[1] == run_id]
    out.heading("Engine events for this run")
    if not events:
        out.line("none")
    for t, _, text in events[-100:]:
        out.line(f"{when(t)}  {text}")
    log = Path(src.config["paths"]["state_dir"]) / "logs" / "server.log"
    rows = grep_tail(log, run_id, RUN_ENGINE_LINES)
    if rows:
        out.heading("Engine log lines naming this run (server.log)")
        for row in rows:
            out.line(row)


def proseg_part(out: Text, run):
    rows = tail(run.dir / "proseg.log", PROSEG_LINES)
    out.heading(f"proseg's own log (proseg.log, the last {PROSEG_LINES} lines)")
    if rows is None:
        out.line("none yet")
        return
    for row in rows:
        if row.startswith("$ "):                         # the command proseg ran: whole, and pastes back as one line
            out.command(row)
        else:
            out.line(row)


def telemetry_sentence(report: dict) -> str:
    """One telemetry report (telemetry.mjs) in words: what the page was showing, how often it drew, long tasks, input
    delay, memory and what it downloaded."""
    def n(v, unit=""):
        return f"{v:g}{unit}" if is_number(v) else "?"
    run, rates = report.get("run") or {}, report.get("rates") or {}
    where = ", ".join(str(run[k]) for k in ("state", "view", "level") if run.get(k))
    draws = [f"{name} {n(rates.get(key))}/s" for key, name in (("stage", "the tissue"), ("umap", "the UMAP"),
                                                                ("timeline", "the timeline"))
             if is_number(rates.get(key)) and rates[key] > 0]
    bits = ["drew " + ", ".join(draws) if draws else "drew nothing"]
    tasks = report.get("long_tasks") or {}
    bits.append(f"{tasks['count']} long task{'s' if tasks['count'] != 1 else ''} (the longest {n(tasks.get('max_ms'), ' ms')})"
                if is_number(tasks.get("count")) and tasks["count"] > 0 else "no long tasks")
    inputs = report.get("input") or {}
    if is_number(inputs.get("count")) and inputs["count"] > 0:
        bits.append(f"input waited {n(inputs.get('delay_p95_ms'), ' ms')} at most for 95 % of {num(inputs['count'])} inputs")
    if is_number(report.get("heap_mb")):
        bits.append(f"memory {n(report['heap_mb'], ' MB')}")
    fetches = [f for f in (report.get("fetch") or {}).values() if isinstance(f, dict)]
    count = sum(f.get("n", 0) for f in fetches if is_number(f.get("n")))
    if count:
        bits.append(f"downloaded {size(sum(f.get('bytes', 0) for f in fetches if is_number(f.get('bytes'))))} in "
                    f"{num(count)} requests")
    if report.get("stretched"):
        bits.append("this report came late (the page was busy or hidden)")
    return (f"{where}: " if where else "") + "; ".join(bits)


def telemetry_part(out: Text, src: Sources):
    path = Path(src.config["paths"]["state_dir"]) / "logs" / "telemetry.jsonl"
    rows = tail(path, TELEMETRY_REPORTS, 256 << 10)
    out.title("Page telemetry (the newest reports, every 10 s while a run is open)", "-")
    if not rows:
        out.line("none")
        return
    for row in rows:
        try:
            report = json.loads(row)
        except ValueError:
            continue
        out.line(f"{local(report.get('t'))}  {telemetry_sentence(report)}", 2)


def worker_lines(rows: list[str], run_id: str | None) -> tuple[list[str], int, int]:
    """The analysis worker's messages (analysis.log) about the run or about no run in particular (its own failures, its
    newest warm-up), without those naming another run (a run deleted while it was analysed, earlier sessions' runs) and
    without earlier warm-ups (every worker start logs one, WARM_KEPT lines): (the lines kept, the messages about other
    runs left out, the earlier warm-up lines left out). A message is a line starting "analysis:" with the lines after it
    (a traceback). Without a run no message is left out for naming one."""
    blocks: list[list[str]] = []
    for row in rows:
        if row.startswith("analysis:") or not blocks:
            blocks.append([row])
        else:
            blocks[-1].append(row)
    kept, other = [], 0
    for block in blocks:
        named = {a or b for line in block for a, b in RUN_NAMED.findall(line)}
        if run_id is not None and named and run_id not in named:
            other += 1
            continue
        kept.append(block)
    warm = [i for i, block in enumerate(kept) if block[0].startswith("analysis: warm-up")]
    early = set(warm[:-WARM_KEPT])
    return [line for i, block in enumerate(kept) if i not in early for line in block], other, len(early)


def trainer_lines(rows: list[str]) -> list[str]:
    """The mixing check's fits (qc-trainer.log): the engine's own "qc:" lines (their UTC times as this computer's) and
    errors; not the libraries' warnings and tips."""
    out = []
    for row in rows:
        if not TRAINER_KEEP.search(row):
            continue
        match = ISO_START.match(row)
        out.append(f"{local(match.group(1))} {row[match.end():]}" if match else row)
    return out


def fit_this_session(src: Sources) -> bool:
    """A fit of the mixing check ran (or waits) in this engine session: a sample waiting, learning or failed, or a model
    learned since the engine started."""
    if src.qc_models is None:
        return False
    for s in src.qc_models.snapshot().values():
        if s.get("state") in ("waiting", "learning", "failed"):
            return True
        created = (s.get("model") or {}).get("created")
        try:
            if created and datetime.fromisoformat(str(created).replace("Z", "+00:00")).timestamp() >= src.started:
                return True
        except ValueError:
            continue
    return False


def logs_part(out: Text, src: Sources, run_id: str | None = None):
    logs = Path(src.config["paths"]["state_dir"]) / "logs"
    rows = tail(logs / "server.log", ENGINE_LINES)
    out.title(f"server.log (the engine's own output, the last {ENGINE_LINES} lines)", "-")
    if rows is None:
        out.line("no such file (the engine's output may go elsewhere)")
    elif not rows:
        out.line("empty")
    for row in rows or ():
        out.line(row)
    rows = tail(logs / "analysis.log", 10 * WORKER_LINES)
    kept, other, early = worker_lines(rows or [], run_id)
    out.title("analysis.log (the analysis worker's messages"
              + (" about this run or no run in particular" if run_id else "")
              + f", the last {WORKER_LINES} lines)", "-")
    if rows is None:
        out.line("none")
    elif not kept:
        out.line("nothing about this run" if run_id else "empty")
    for row in kept[-WORKER_LINES:]:
        out.line(row)
    left = [f"{other} about other runs (e.g. runs deleted while they were analysed)" if other else "",
            f"{early} lines of earlier warm-ups" if early else ""]
    if any(left):
        out.line(f"(left out: {'; '.join(x for x in left if x)})")
    state = getattr(src.qc, "states", {}).get(run_id) if run_id is not None and src.qc is not None else None
    check_on = bool(state is not None and state.enabled)
    if run_id is not None and not check_on and not fit_this_session(src):
        return                                           # the check neither ran for this run nor learned anything now
    rows = tail(logs / "qc-trainer.log", 20 * WORKER_LINES)
    kept = trainer_lines(rows or [])
    out.title(f"qc-trainer.log (the mixing check's model fits: its own lines and errors, the last {WORKER_LINES})", "-")
    if rows is None:
        out.line("none")
    elif not kept:
        out.line("nothing but the libraries' own messages")
    for row in kept[-WORKER_LINES:]:
        out.line(row)


def build(src: Sources, run_id: str | None = None) -> str:
    """The whole log as text (see the module docstring); `run_id`: the run open on the page, or None. Blanked as a crash
    report is, by the same redactor and secrets: a line the one keeps secret, the other does too."""
    from .crashlog import redact
    out = Text()
    out.part("header", header, src, run_id)
    out.part("summary", summary_part, src, run_id)
    out.part("engine", engine_part, src)
    if run_id is not None:
        out.part("run", run_part, src, run_id)
    out.part("telemetry", telemetry_part, src)
    out.part("logs", logs_part, src, run_id)
    if run_id is not None:
        out.part("appendix", appendix_part, src, run_id)
    return redact(out.text(), src.secrets)
