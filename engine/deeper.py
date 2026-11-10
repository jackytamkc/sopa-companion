"""Go deeper: one cell's RNA explained by the run's mixing-check model (ResolVI), for the window the page opens from the
selected cell's row in the mixing check (the recipe itself runs in the GPU worker: analysis/worker.py op "deeper"; the
model's details, deep.npz, come from analysis/qc_trainer.py).

DeeperManager serves the API (server/app.py: POST/DELETE/GET /runs/{id}/qc/deeper..., the WebSocket's "deeper" events)
and is the analysis scheduler's deeper provider (engine/analysis.py `attach_deeper`):
  * what Go deeper can do for a run now, the answer states of 6.4 in the window's order of 9 (`situation_of`): from
    QcManager (the run's switch, its sample's state) and QcModels (model.deep, the details job). Go deeper works only
    while the run's check is on and its model holds the details; nothing here learns anything;
  * explicit requests go to the scheduler ahead of every other job (the newest per run wins); a followed cell (the
    window open on a live run) rides along each live pass of its run. A request made while the sample is being learned
    (or its details are) is kept and sent by itself once that is done, so the window fills in by itself;
  * answers (the worker's v1 result, on proseg's own count of the frame: its `count`, "averaged", or "sampled", the
    count proseg saves, for a frame proseg counted for a pause, the hold, a finish or the end; none: "averaged"): an LRU
    of CACHE_KEPT in memory (about 5 MB) keyed (run, seq, cell, model, v, count), which also holds the followed cell's
    answers; the answers asked for are also kept on disk beside the run's other results of that count,
    <run>/qc/<model8>.pc1/deeper/<seq:06d>-<cell>.json (<seq:06d>-<cell>.sampled.json on the count proseg saves;
    start-<cell>.json for the start line; engine/analysis.py qc_folder; written atomically by a thread of their own, at
    most DISK_KEPT per run, the oldest going first, older counts' answers too; deleted with the run). An answer is
    served only for the count its frame is read on now (counts_file.count_of of its counts line): a frame counted
    again (a request's count, then the hold's) is answered again. A cached answer never reaches the worker; one worked
    out on the outlines' count (an older folder) is never read;
  * the history: the cell's stored bytes over the run (at most `points` evenly spaced .qc.bin frames of the same
    folder, read once per cell and count: a frame scored again on another count is read again), for the window's small
    line;
  * the WebSocket events of the watched run: every answer asked for (at most EXPLICIT_PUSHED a tick) and the newest
    followed one; the details job of each sample at most once a second.
Cells are numbered as in the frames: `cell` = label - 1 (GET /runs/{id}/cells names them).
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

from ..analysis.counts_file import count_of
from .analysis import DEEPER_V, DEEP_FILE, qc_folder
from .qc import QcError, counted_again
from .qc_rules import NO_START_WORDS, Steady, start_of

RESULT_V = DEEPER_V            # the worker's result version; part of every cache key
CACHE_KEPT = 512               # answers held in memory (about 9 KB each)
DISK_KEPT = 2000               # answers asked for, kept on disk per run
HISTORY_KEPT = 64              # cells whose bytes over the run are held (each a few hundred numbers)
HISTORY_POINTS = 200           # the small line's points by default (GET .../history?points=)
EXPLICIT_KEPT = 8              # answers asked for, waiting in a run's outbox for the pages that watch it
EXPLICIT_PUSHED = 4            # ... of which at most this many go out per WebSocket tick (the newest)
ASKED_KEPT = 256               # when each explicit request was made (latency_ms of its answer)
SAME_KEPT = 64                 # cells whose neighbours' same-kind classes are remembered (the worker's hysteresis)
PENDING_TTL_S = 1800.0         # a request kept while its sample (or its details) is learned is dropped after this
PENDING_CHECK_S = 0.5          # kept requests are looked at again at most this often (when the check's state moved)
UNWATCHED_GRACE_S = 30.0       # a run no page has watched for this long stops being followed (a page that went away)
DETAIL_EVERY_S = 1.0           # the details job's progress: at most one event per sample per this many seconds
STATES = ("ready", "working", "off", "waiting", "learning", "failed", "unavailable", "needs_detail", "detail",
          "detail_failed")
REASON_KEYS = ("worker", "needs_detail", "no_start", "no_transcripts", "off", "run_deleted", "no_counts", "old_proseg")
DETAIL_STATES = {"queued": "detail", "learning": "detail", "ready": "ready", "failed": "detail_failed"}


class DeeperError(QcError):
    """A refused Go deeper request (`status`: 404, 422, 503); server/app.py answers it as the check's errors, with
    `extra` beside its words: where Clusters & UMAP cannot run, {"reason": "analysis"}, which the window puts into its
    own words (the engine answers 503 for other troubles too: a storage error, its shutdown)."""

    def __init__(self, message, status=422, **extra):
        super().__init__(message, status)
        self.extra = extra


def situation_of(view: dict, sample: dict | None, deep_on_disk: bool = False) -> dict:
    """What Go deeper can do for a run now (in the window's order of states): {state, model, deep, reason?,
    progress?, eta_s?, estimate_s?}; state "go" when a cell can be worked out now. `view`: GET /runs/{id}/qc
    (QcManager.describe: enabled, state, reason, sample_state, progress, eta_s, model {id, deep?, ...}, detail?);
    `sample`: GET /samples/{id}/qc (QcModels.describe: reason, estimate_s, model, detail {state, progress, eta_s,
    reason}), or None; `deep_on_disk`: the run's model folder holds deep.npz (what the worker reads)."""
    sample = sample or {}
    model = view.get("model") or sample.get("model") or {}
    deep = bool(model.get("deep")) or bool(deep_on_disk)
    out = {"model": model.get("id"), "deep": deep}
    if view.get("state") == "unavailable" or view.get("sample_state") == "unavailable":
        return {**out, "state": "unavailable", "reason": view.get("reason") or sample.get("reason") or "no_table"}
    if not view.get("enabled"):
        return {**out, "state": "off"}
    if view.get("state") in ("waiting", "learning", "failed"):
        return {**out, "state": view["state"], "reason": view.get("reason"), "progress": view.get("progress"),
                "eta_s": view.get("eta_s")}
    if not out["model"]:                              # ready a moment ago, being looked at again
        return {**out, "state": "waiting", "reason": "queue", "progress": None, "eta_s": None}
    if not deep:
        detail = view.get("detail") or sample.get("detail") or {}
        state = detail.get("state") if isinstance(detail, dict) else None
        if state in ("queued", "learning"):
            return {**out, "state": "detail", "progress": detail.get("progress"), "eta_s": detail.get("eta_s")}
        if state == "failed":
            return {**out, "state": "detail_failed", "reason": detail.get("reason")}
        return {**out, "state": "needs_detail", "estimate_s": sample.get("estimate_s")}
    return {**out, "state": "go"}


def error_key(record: dict) -> str:
    """A failure's plain reason key for the page (it words them itself): one of REASON_KEYS, else "failed"."""
    reason = record.get("reason")
    if reason in REASON_KEYS:
        return reason
    return "worker" if "GPU worker" in str(record.get("error") or "") else "failed"


def _finite(value):
    """A worker's result as JSON can carry it: a non-finite number becomes null (checked in C first: almost always
    there is none)."""
    try:
        json.dumps(value, allow_nan=False)
        return value
    except ValueError:
        pass

    def clean(item):
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        return item
    return clean(value)


def _count_named(result) -> str:
    """The count an answer was worked out on (the worker's result `count`: "averaged" | "sampled"; none: "averaged",
    as every answer before the count proseg saves was read)."""
    count = result.get("count") if isinstance(result, dict) else None
    return count if count in ("averaged", "sampled") else "averaged"


def _seq_of(value):
    """A frame number or "start" from a request (int, or digits as text); None when it is neither."""
    if value == "start":
        return "start"
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _atomic_write(path: Path, data: bytes):
    """Write a file whole or not at all (a temporary file beside it, then one rename)."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    try:
        with open(temporary, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _read_codes(path: Path) -> bytes:
    """A .qc.bin / start.bin: one byte per cell, a zstd frame (or raw)."""
    raw = Path(path).read_bytes()
    if raw[:4] == b"\x28\xb5\x2f\xfd":
        import zstandard
        return zstandard.ZstdDecompressor().decompressobj().decompress(raw)
    return raw


class DeeperManager:
    """See the module docstring. `qc`: QcManager (describe, qc_model, series, version); `models`: QcModels (describe,
    snapshot); `analysis`: the scheduler (available, attach_deeper, request_deeper, snap_deeper, follow_deeper,
    stop_deeper).
    Thread-safe; never calls any of them while holding its own lock, and does no file work under it."""

    def __init__(self, config: dict | None, runs, qc, models, analysis, *, clock=time.monotonic):
        self.runs, self.qc, self.models, self.analysis, self.clock = runs, qc, models, analysis, clock
        self.lock = threading.RLock()
        self.cache: OrderedDict = OrderedDict()       # (run, seq, cell, model, v, count) -> {iter, result, disk}
        self.asked: OrderedDict = OrderedDict()       # (run, seq, cell, model) -> clock when asked (explicit)
        self.pending: dict[str, dict] = {}            # run -> {seq, cell, follow, t}: asked while not ready yet
        self.pending_seen = (None, -math.inf)         # (the check's version, clock) of the last look at them
        self.outbox: dict[str, dict] = {}             # run -> {"explicit": [(n, event)], "follow": (n, event) | None}
        self.n = 0                                    # events numbered in the order they were made
        self.local_version = 0
        self.watchers: dict[str, int] = {}            # run -> page connections watching it
        self.unwatched: dict[str, float] = {}         # run -> clock when its last watcher left
        self.histories: OrderedDict = OrderedDict()   # (run, model, cell) -> {seq: (byte, count) | "start": byte}
        self.same_kind: OrderedDict = OrderedDict()   # (run, model, cell) -> {neighbour cell: same kind?} as last shown
        self.start_words: OrderedDict = OrderedDict()  # (run, model, cell) -> its word since the start as last shown
        self.start_steady = Steady(("now", "start"))  # (run, model, cell) -> that word held steady while followed
        self.disk: dict[str, OrderedDict] = {}        # run -> its answers on disk (path -> None), oldest first
        self.gone: set[str] = set()                   # deleted runs: nothing more is kept or written for them
        self.writes: queue.Queue = queue.Queue()
        self.writer: threading.Thread | None = None
        self.closed = False
        if analysis is not None and hasattr(analysis, "attach_deeper"):
            analysis.attach_deeper(self)

    @property
    def version(self) -> int:
        """Moves whenever a page may have something new: an answer, or the check's or a sample's state."""
        return self.local_version + int(getattr(self.qc, "version", 0) or 0)

    # ---- the API ---------------------------------------------------------------------------------------------------
    def request(self, run_id: str, body) -> dict:
        """POST /runs/{id}/qc/deeper {seq: n | "start", cell, follow?}: {state, seq, cell, model, deep, reason?,
        progress?, eta_s?, estimate_s?, iter?, result?} (states: STATES). `follow`: true explains the cell with every
        live pass from now on, false stops following, absent leaves it. 404 for an unknown run, frame or cell (and the
        start line of a run that has none: one whose cells started with nuclei or from nuclei only, the refusal saying
        reason "no_start" and its start_kind), 422 for a bad body, 503 where Clusters & UMAP cannot run (the
        scheduler's unavailable_why says why)."""
        run = self.runs.get(run_id)
        seq, cell, follow = self._parse(run, body)
        if not getattr(self.analysis, "available", False):
            why = self.analysis.unavailable_why() if hasattr(self.analysis, "unavailable_why") else \
                "no analysis worker is attached"
            raise DeeperError(f"Go deeper needs Clusters and the UMAP, which can't run here: {why}", 503,
                              reason="analysis")
        return self._answer(run, seq, cell, follow)

    def stop(self, run_id: str):
        """DELETE /runs/{id}/qc/deeper: the window closed: no more following, and nothing asked is kept waiting."""
        self.runs.get(run_id)
        with self.lock:
            self.pending.pop(run_id, None)
        if hasattr(self.analysis, "stop_deeper"):
            self.analysis.stop_deeper(run_id)

    def cached(self, run_id: str, seq, cell: int, model: str | None = None) -> dict:
        """The kept answer to frame `seq` of cell `cell` (kept_answer's), 404 until it has been worked out."""
        return self.kept_answer(run_id, seq, cell, model)[0]

    def kept_answer(self, run_id: str, seq, cell: int, model: str | None = None,
                    count: str | None = None) -> tuple[dict, bool]:
        """GET /runs/{id}/qc/deeper/{seq}/{cell}?model=&count=: (the answer, whether it is the frame's own), 404 until
        it has been worked out. Without `model` the run's current model's. A frame is answered as the counted frame a
        request for it is (`_snapped`): not its own then, and answered with its own once its count is in (a request
        frame's counts line, or a pause's, can merge after its record). The answer is the one on the count that frame is
        read on now; `count` names the count the caller means ("averaged" | "sampled"): 404 when the frame is read on
        another now (it was counted again: a pause's, the hold's, a finish's or the end's count)."""
        run = self.runs.get(run_id)
        seq = self._frame(run, _seq_of(seq))
        self._cell(run, cell)
        if model is None:
            current = self.qc.qc_model(run)
            if not current:
                raise DeeperError("No mixing check for this run yet", 404)
            model = str(current["id"])
        at = self._snapped(run, seq, cell, str(model))
        if count is not None and count != self._count_of(run, at):
            raise DeeperError("This iteration's count changed", 404)
        hit = self._lookup(run, at, cell, str(model))
        if hit is None:
            raise DeeperError("Not worked out yet", 404)
        return hit["result"], at == seq

    def history(self, run_id: str, cell: int, points: int = HISTORY_POINTS) -> dict:
        """GET /runs/{id}/qc/deeper/history?cell=&points=: {model, start: byte | null, points: [[iter, byte]]}: the
        cell's stored byte (floor(share x 250), 255 = not judged) in at most `points` evenly spaced frames the run's
        model scored (the newest kept), and at the start line (null without one). Each frame's file is read once per
        cell and count, a frame being scored again (engine/qc.py counted_again) once its new result is in; 404 for an
        unknown run or cell, or before the run's check has a model."""
        run = self.runs.get(run_id)
        self._cell(run, cell)
        try:
            points = max(2, min(int(points), 2000))
        except (TypeError, ValueError):
            raise DeeperError("points must be a whole number") from None
        series = self.qc.series(run_id, points)        # QcError 404 before the check has a model
        model = series.get("model")
        folder = qc_folder(run.dir, model)              # the worker's files of proseg's count (docs/ARCHITECTURE.md)
        key = (run_id, model, cell)
        with self.lock:
            known = dict(self.histories.get(key) or {})
        for point in series.get("points") or []:
            seq, count = point.get("seq"), point.get("count") or "averaged"
            if isinstance(seq, int) and (seq not in known or known[seq][1] != count):   # (scored again on another
                if counted_again(run, seq, count):     # count: its bytes changed). Being scored again now: its bytes
                    continue                           # may be either count's, so it is read once its result is in
                byte = self._byte(folder / f"{seq:06d}.qc.bin", cell)
                if byte is not None:                   # a failed read (a file being replaced, a disk hiccup) is
                    known[seq] = (byte, count)         # looked at again with the next request, never kept as a gap
        if known.get("start") is None:                 # not kept while missing: the start line may come later
            known.pop("start", None)
            start = self._byte(folder / "start.bin", cell) if series.get("start") is not None else None
            if start is not None:
                known["start"] = start
        with self.lock:
            if run_id not in self.gone:
                self.histories[key] = known
                self.histories.move_to_end(key)
                while len(self.histories) > HISTORY_KEPT:
                    self.histories.popitem(last=False)
        line: list[list] = []
        for point in series.get("points") or []:
            byte = known.get(point.get("seq")) if isinstance(point.get("seq"), int) else None
            if byte is None:
                continue
            byte = byte[0]
            if line and line[-1][0] == point.get("iter"):   # the upscale frame repeats its iteration (200): the
                line[-1] = [point.get("iter"), byte]         # newer frame stands, so the line's iterations rise
            else:
                line.append([point.get("iter"), byte])
        return {"model": model, "start": known.get("start"), "points": line}

    def log_info(self, run_id: str | None = None) -> dict:
        """What the Log button may say of Go deeper (engine/runlog.py): answers kept in memory (and of the run's), the
        run's request kept while its sample learns, its answers on disk counted this session, the pages watching it.
        A copy, read under the lock; nothing is computed or started."""
        with self.lock:
            info = {"kept": len(self.cache), "writes_queued": self.writes.qsize()}
            if run_id is not None:
                pending = self.pending.get(run_id)
                info["run"] = {"kept": sum(1 for key in self.cache if key[0] == run_id),
                               "pending": None if pending is None else {k: pending[k] for k in ("seq", "cell", "follow")},
                               "on_disk": len(self.disk[run_id]) if run_id in self.disk else None,
                               "watchers": self.watchers.get(run_id, 0)}
        return info

    def forget(self, run_id: str):
        """A deleted run: everything kept for it goes (its folder, with the answers on disk, went with it)."""
        with self.lock:
            self.gone.add(run_id)
            for key in [k for k in self.cache if k[0] == run_id]:
                del self.cache[key]
            for key in [k for k in self.asked if k[0] == run_id]:
                del self.asked[key]
            for key in [k for k in self.histories if k[0] == run_id]:
                del self.histories[key]
            for key in [k for k in self.same_kind if k[0] == run_id]:
                del self.same_kind[key]
            for key in [k for k in self.start_words if k[0] == run_id]:
                del self.start_words[key]
            for key in [k for k in self.start_steady.verdicts if k[0] == run_id]:
                del self.start_steady.verdicts[key]
            for table in (self.pending, self.outbox, self.disk, self.watchers, self.unwatched):
                table.pop(run_id, None)
            self.local_version += 1
        if hasattr(self.analysis, "stop_deeper"):
            try:
                self.analysis.stop_deeper(run_id)
            except Exception:
                pass

    def close(self):
        """Engine shutdown: the answers still queued for the disk are written (a few seconds at most)."""
        self.closed = True
        writer = self.writer
        if writer is not None and writer.is_alive():
            self.writes.put(None)
            writer.join(timeout=5)

    # ---- the WebSocket (server/app.py live_session) ----------------------------------------------------------------
    def watching(self, previous: str | None, current: str | None):
        """A page connection stopped watching `previous` and/or now watches `current` (None: none). A run no page has
        watched for UNWATCHED_GRACE_S stops being followed (`events` looks)."""
        with self.lock:
            if previous:
                left = self.watchers.get(previous, 1) - 1
                if left > 0:
                    self.watchers[previous] = left
                else:
                    self.watchers.pop(previous, None)
                    self.unwatched[previous] = self.clock()
            if current:
                self.watchers[current] = self.watchers.get(current, 0) + 1
                self.unwatched.pop(current, None)

    def events(self, cursor: dict, watched: str | None, now: float) -> list[dict]:
        """Go deeper events one page connection has not had yet (`cursor`: that connection's own dict; `now`: its
        clock). Each sample's details job: {ev: "deeper", sample, state: "detail" | "ready" | "detail_failed",
        progress, eta_s, reason}, at most once a second while it learns (cursor["pending"] is set when one was held
        back: ask again shortly) and "ready" only to a page that saw it being learned. The watched run: {ev: "deeper",
        run, seq, iter, cell, model, result, explicit, cached, latency_ms, worker_ms} for answers, {ev: "deeper", run,
        seq, cell, state: "error", error: <reason key>, explicit} for failures: every answer asked for (at most
        EXPLICIT_PUSHED a tick, the newest) and the newest followed one; the first look after a watch gives only the
        newest followed answer (a page asks again for what it shows)."""
        self._sweep()
        out: list[dict] = []
        cursor["pending"] = False
        sent = cursor.setdefault("samples", {})
        try:
            samples = self.models.snapshot()
        except Exception:
            samples = {}
        for sid, view in samples.items():
            detail = view.get("detail") if isinstance(view, dict) else None
            state = DETAIL_STATES.get(detail.get("state")) if isinstance(detail, dict) else None
            if state is None:
                continue
            progress = detail.get("progress")
            signature = (state, progress.get("step") if isinstance(progress, dict) else progress, detail.get("reason"))
            last = sent.get(sid)
            if last is not None and last[0] == signature:
                continue
            if state == "ready" and (last is None or last[0][0] == "ready"):
                sent[sid] = (signature, now)          # it never saw it being learned: nothing to fill in
                continue
            if last is not None and state == "detail" and last[0][0] == "detail" and now - last[1] < DETAIL_EVERY_S:
                cursor["pending"] = True
                continue
            sent[sid] = (signature, now)
            out.append({"ev": "deeper", "sample": sid, "state": state, "progress": progress,
                        "eta_s": detail.get("eta_s"), "reason": detail.get("reason")})
        if watched:
            runs = cursor.setdefault("runs", {})
            with self.lock:
                box = self.outbox.get(watched) or {"explicit": [], "follow": None}
                first, seen = watched not in runs, runs.get(watched, 0)
                follow = box["follow"]
                if first:
                    fresh = [follow] if follow else []
                else:
                    fresh = [item for item in box["explicit"] if item[0] > seen][-EXPLICIT_PUSHED:]
                    if follow and follow[0] > seen:
                        fresh.append(follow)
                numbers = [n for n, _ in box["explicit"]] + ([follow[0]] if follow else [])
                runs[watched] = max([seen, *numbers])
                out.extend(dict(event) for _, event in sorted(fresh, key=lambda item: item[0]))
        return out

    # ---- the analysis scheduler's deeper provider (engine/analysis.py) ------------------------------------------------
    def deeper_cached(self, run_id: str, seq, cell: int, model: str, explicit: bool) -> bool:
        """Whether the answer on the count its frame is read on now is kept (memory; asked-for ones also on disk): then
        it goes out from here (an event, and the disk for one asked for) and the worker is not asked."""
        try:
            run = self.runs.get(run_id)
        except Exception:
            return False
        count = self._count_of(run, seq)
        key = (run_id, seq, cell, model, RESULT_V, count)
        with self.lock:
            hit = self.cache.get(key)
            if hit is not None:
                self.cache.move_to_end(key)
        if hit is None and explicit:
            hit = self._lookup(run, seq, cell, model, count)
        if hit is None:
            return False
        self._push(run_id, self._event(run_id, seq, cell, model, hit, explicit, cached=True), explicit)
        if explicit:
            self._save(run_id, seq, cell, model, hit, run)
        return True

    def deeper_prev(self, run_id: str, cell: int, model: str | None) -> dict:
        """{neighbour cell (as text): same kind?} as the newest worked-out answer for this cell showed them (the
        request block's `same_prev`): the worker keeps a neighbour's class while its likeness sits between the two
        cut-offs around 0.9 (analysis/worker.py same_kind_classes), so tagged genes do not come and go with every
        pass."""
        with self.lock:
            known = self.same_kind.get((run_id, model, int(cell)))
            return {str(k): bool(v) for k, v in known.items()} if known else {}

    def deeper_start_prev(self, run_id: str, cell: int, model: str | None) -> str | None:
        """The cell's word since the start ("more", "fewer", "same") as the newest worked-out answer showed it (the
        request block's `start_prev`): the worker keeps a word while the cell's like-for-like change stays at half its
        band or more (analysis/worker.py start_word), so it does not flip from frame to frame."""
        with self.lock:
            return self.start_words.get((run_id, model, int(cell)))

    def _steady_start(self, run_id: str, model: str, cell: int, result: dict) -> dict:
        """A followed cell's word since the start held steady as the tissue's verdicts are (qc_rules.Steady: the word
        shown changes once the answers' own word has been the new one for 5 s over at least 2 answers, at the latest
        after 15 s): its like-for-like estimate moves by a few points from one frame to the next, and the word at the
        band flipped with it. A held word says so: `held`, and the answer's own word as `direction_now`; it keeps the
        two numbers it compared (`now`, `start`: those of the newest answer whose own word was the one shown), so the
        step line that puts them beside the word never contradicts it, while `points` stays the answer's own. Answers
        asked for (a click, a step, a scrubbed frame) are judged alone. An answer read on the count proseg saves (a
        frame proseg counted for a pause, the hold, a finish or the end: the run rests there) says its own word with
        its own two numbers, as the panel's verdict does there: the word held before it is let go
        (qc_rules.Steady.release), and the answers that follow are held from it."""
        start = result.get("start_change")
        if not isinstance(start, dict) or start.get("direction") not in ("more", "fewer", "same"):
            return result
        key = (run_id, model, int(cell))
        block = {"direction": start["direction"], "now": start.get("now"), "start": start.get("start")}
        with self.lock:
            if _count_named(result) == "sampled":       # at rest: the frame's own word and numbers
                self.start_steady.release(key, block)
                shown = block
            else:
                shown = self.start_steady.apply(key, block, self.clock())
            while len(self.start_steady.verdicts) > SAME_KEPT:
                self.start_steady.verdicts.pop(next(iter(self.start_steady.verdicts)))
        if not shown.get("held"):
            return result
        return {**result, "start_change": {**start, "direction": shown["direction"], "held": True,
                                           "direction_now": shown["direction_now"], "now": shown["now"],
                                           "start": shown["start"]}}

    def _remember_same(self, run_id: str, model: str, cell: int, result: dict):
        """Keep the neighbours' same-kind classes an answer used (its `neighbours[].same`), and its word since the
        start (`start_change.direction`)."""
        start = result.get("start_change") if isinstance(result, dict) else None
        word = start.get("direction") if isinstance(start, dict) else None
        if word in ("more", "fewer", "same"):
            key = (run_id, model, int(cell))
            with self.lock:
                if run_id not in self.gone:
                    self.start_words[key] = word
                    self.start_words.move_to_end(key)
                    while len(self.start_words) > SAME_KEPT:
                        self.start_words.popitem(last=False)
        rows = result.get("neighbours") if isinstance(result, dict) else None
        classes = {int(n["cell"]): bool(n["same"]) for n in rows or [] if isinstance(n, dict)
                   and isinstance(n.get("cell"), int) and not isinstance(n.get("cell"), bool)
                   and isinstance(n.get("same"), bool)}
        if not classes:
            return
        key = (run_id, model, int(cell))
        with self.lock:
            if run_id in self.gone:
                return
            self.same_kind[key] = classes
            self.same_kind.move_to_end(key)
            while len(self.same_kind) > SAME_KEPT:
                self.same_kind.popitem(last=False)

    def deeper_result(self, record: dict):
        """An answer or a failure from the scheduler (engine/analysis.py's record): kept, and pushed."""
        run_id, seq, cell, model = record["run"], record["seq"], int(record["cell"]), record.get("model")
        explicit = bool(record.get("explicit"))
        with self.lock:
            if run_id in self.gone:
                return
        result = record.get("result")
        if isinstance(result, dict) and model:
            if not explicit:
                result = self._steady_start(run_id, model, cell, result)
            self._remember_same(run_id, model, cell, result)
            count = _count_named(result)
            hit = {"iter": record.get("iter"), "result": _finite(result), "disk": False, "count": count}
            self._remember((run_id, seq, cell, model, RESULT_V, count), hit)
            self._push(run_id, self._event(run_id, seq, cell, model, hit, explicit, cached=False), explicit)
            if explicit:
                self._save(run_id, seq, cell, model, hit)
            return
        with self.lock:
            self.asked.pop((run_id, seq, cell, model), None)
        self._push(run_id, {"ev": "deeper", "run": run_id, "seq": seq, "cell": cell, "state": "error",
                            "error": error_key(record), "explicit": explicit}, explicit)

    # ---- what a request gets ------------------------------------------------------------------------------------------
    def _parse(self, run, body) -> tuple:
        if not isinstance(body, dict):
            raise DeeperError('Send {seq, cell}: the frame (a number, or "start") and the cell (label - 1)')
        seq, cell, follow = _seq_of(body.get("seq")), body.get("cell"), body.get("follow")
        if seq is None:
            raise DeeperError('seq must be a frame number or "start"')
        if isinstance(cell, bool) or not isinstance(cell, int):
            raise DeeperError("cell must be a cell number (label - 1)")
        if follow is not None and not isinstance(follow, bool):
            raise DeeperError("follow must be true or false")
        return self._frame(run, seq), self._cell(run, cell), follow

    @staticmethod
    def _frame(run, seq):
        if seq is None:
            raise DeeperError("Unknown frame", 404)
        if seq != "start":
            with run.lock:
                frames = len(run.frames)
            if not 0 <= seq < frames:
                raise DeeperError("Unknown frame", 404)
        return seq

    @staticmethod
    def _cell(run, cell) -> int:
        cells = (run.info or {}).get("cells") or 0
        if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < int(cells):
            raise DeeperError("Unknown cell", 404)
        return cell

    def _situation(self, run, seq) -> dict:
        """situation_of for this run, with the model's folder (None unless a model is ready). The start line only for
        a run whose cells started without nuclei (qc_rules.start_of "cells": proseg's count at iteration 0 holds the
        starting cells); any other start is refused (404, reason "no_start", its start_kind: "nuclei" or "paired",
        whose count at iteration 0 holds only the nuclei)."""
        run_id = run.meta["id"]
        if seq == "start":
            kind = start_of(run.meta)
            if kind != "cells":
                raise DeeperError(NO_START_WORDS.get(kind, NO_START_WORDS["paired"]), 404, reason="no_start",
                                  start_kind=kind)
        view = self.qc.describe(run_id)
        if run.info and not (run.live / "transcripts.bin.gz").exists():
            return {"state": "unavailable", "reason": "no_transcripts", "model": None, "deep": False}
        sample = None
        if run.meta.get("sample_kind") == "zarr":
            try:
                sample = self.models.describe(run.meta["sample"])
            except QcError:
                sample = None
        current, deep_file = None, False
        if view.get("enabled") and view.get("state") in ("starting", "ready"):
            current = self.qc.qc_model(run)
            if current:
                folder = Path(current["path"])
                folder = folder if folder.suffix != ".npz" else folder.parent
                deep_file = (folder / DEEP_FILE).is_file()
        found = situation_of(view, sample, deep_file)
        if current:
            found["model"] = str(current["id"])
        return found

    @staticmethod
    def _count_of(run, seq) -> str:
        """The count frame `seq` is read on now (counts_file.count_of of its counts line: "sampled", the count proseg
        saves, for a frame proseg counted for a pause, the hold, a finish or the end; else "averaged", the start line's
        too)."""
        if seq == "start":
            return "averaged"
        with run.lock:
            line = run.frames[seq].get("counts") if isinstance(seq, int) and 0 <= seq < len(run.frames) else None
        return count_of(line)

    def _snapped(self, run, seq, cell: int, model: str | None):
        """The frame the scheduler answers frame `seq` with (its snap_deeper: the nearest counted frame at or before it
        whose file is there, or whose answer for this cell and model is kept): a frame without proseg's count of its
        own (between counted frames, pruned from the spool, gone with an engine that closed) is answered, kept and named
        as that one, so asking again finds it. "start", or a scheduler that does not snap, leaves it as it is."""
        snap = getattr(self.analysis, "snap_deeper", None)
        if seq == "start" or snap is None:
            return seq

        def kept(s):
            return model is not None and self._lookup(run, s, cell, model) is not None
        try:
            return snap(run.meta["id"], seq, kept)
        except Exception:                                 # an unknown frame is refused by _frame before this
            return seq

    def _answer(self, run, seq, cell: int, follow, push: bool = False, since: float | None = None) -> dict:
        """The answer to a request (state, and the result when kept), sending it to the scheduler when it can be
        worked out now, else keeping it until it can (`since`: when it was first asked). `push`: also push a kept
        answer to the pages (a request sent by itself later: nobody waits on an HTTP answer)."""
        run_id = run.meta["id"]
        found = self._situation(run, seq)
        model = found.get("model")
        base = {"seq": seq, "cell": cell, "model": model, "deep": bool(found.get("deep"))}
        if found["state"] != "go":
            with self.lock:
                if run_id not in self.gone:
                    self.pending[run_id] = {"seq": seq, "cell": cell, "follow": follow,
                                            "t": self.clock() if since is None else since}
            return {**base, **{k: v for k, v in found.items() if k not in ("model", "deep")}}
        with self.lock:
            self.pending.pop(run_id, None)
        seq = self._snapped(run, seq, cell, model)            # the frame its answer is worked out, kept and named at
        base["seq"] = seq
        hit = self._lookup(run, seq, cell, model)
        if hit is not None:
            if follow is not None:
                self.analysis.follow_deeper(run_id, cell if follow else None)
            if push:
                self._push(run_id, self._event(run_id, seq, cell, model, hit, True, cached=True), True)
            self._save(run_id, seq, cell, model, hit, run)
            return {**base, "state": "ready", "iter": hit["iter"], "result": hit["result"]}
        with self.lock:
            self.asked[(run_id, seq, cell, model)] = self.clock()
            self.asked.move_to_end((run_id, seq, cell, model))
            while len(self.asked) > ASKED_KEPT:
                self.asked.popitem(last=False)
        self.analysis.request_deeper(run_id, seq, cell, follow)
        return {**base, "state": "working"}

    def _sweep(self):
        """Runs nobody has watched for a while stop being followed (their kept request and queued events go); requests
        kept while not ready are sent once the check's state moved to ready (looked at again at most every
        PENDING_CHECK_S)."""
        clock = self.clock()
        version = int(getattr(self.qc, "version", 0) or 0)
        with self.lock:
            stale = [run_id for run_id, t in self.unwatched.items() if clock - t >= UNWATCHED_GRACE_S]
            for run_id in stale:                      # its window went with its page: nothing waits for it any more
                del self.unwatched[run_id]
                self.pending.pop(run_id, None)
                self.outbox.pop(run_id, None)
            due = bool(self.pending) and version != self.pending_seen[0] and clock - self.pending_seen[1] >= PENDING_CHECK_S
            if due:
                self.pending_seen = (version, clock)
                pending = dict(self.pending)
        for run_id in stale:
            try:
                self.analysis.follow_deeper(run_id, None)
            except Exception:
                pass
        if not due:
            return
        for run_id, ask in pending.items():
            keep = clock - ask["t"] < PENDING_TTL_S
            run = None
            if keep:
                try:
                    run = self.runs.get(run_id)
                except Exception:
                    keep = False
            if keep:
                try:
                    self._answer(run, ask["seq"], ask["cell"], ask["follow"], push=True, since=ask["t"])
                    continue
                except Exception:                         # a frame or run that went, the worker's environment gone
                    pass
            with self.lock:
                if self.pending.get(run_id) is ask:
                    del self.pending[run_id]

    # ---- the caches ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _path(run_dir: Path, seq, cell: int, model: str, count: str = "averaged") -> Path:
        """An answer's file: <seq:06d>-<cell>.json (the averaged count's, as before), <seq:06d>-<cell>.sampled.json (the
        count proseg saves), start-<cell>.json."""
        name = f"start-{cell}.json" if seq == "start" else \
            f"{int(seq):06d}-{cell}{'.sampled' if count == 'sampled' else ''}.json"
        return qc_folder(run_dir, model) / "deeper" / name

    def _remember(self, key: tuple, hit: dict):
        with self.lock:
            if key[0] in self.gone:
                return
            self.cache[key] = hit
            self.cache.move_to_end(key)
            while len(self.cache) > CACHE_KEPT:
                self.cache.popitem(last=False)

    def _lookup(self, run, seq, cell: int, model: str, count: str | None = None) -> dict | None:
        """The kept answer on `count` (None: the count the frame is read on now, `_count_of`): memory first, then the
        disk (answers asked for), else None."""
        count = self._count_of(run, seq) if count is None else count
        key = (run.meta["id"], seq, cell, model, RESULT_V, count)
        with self.lock:
            hit = self.cache.get(key)
            if hit is not None:
                self.cache.move_to_end(key)
                return hit
        try:
            stored = json.loads(self._path(run.dir, seq, cell, model, count).read_text())
        except (OSError, ValueError):
            return None
        if not (isinstance(stored, dict) and stored.get("v") == RESULT_V and stored.get("model") == model
                and stored.get("seq") == seq and stored.get("cell") == cell and isinstance(stored.get("result"), dict)
                and _count_named(stored["result"]) == count):
            return None
        hit = {"iter": stored.get("iter"), "result": stored["result"], "disk": True, "count": count}
        self._remember(key, hit)
        return hit

    def _event(self, run_id: str, seq, cell: int, model: str, hit: dict, explicit: bool, cached: bool) -> dict:
        with self.lock:
            asked = self.asked.pop((run_id, seq, cell, model), None) if explicit else None
        result = hit["result"]
        ms = next((result[k] for k in ("worker_ms", "ms")              # the op's whole share of its pass, else the
                   if isinstance(result.get(k), (int, float)) and not isinstance(result.get(k), bool)), None)   # recipe's
        return {"ev": "deeper", "run": run_id, "seq": seq, "iter": hit.get("iter"), "cell": cell, "model": model,
                "result": result, "explicit": explicit, "cached": cached,
                "latency_ms": None if asked is None else round((self.clock() - asked) * 1000.0, 1), "worker_ms": ms}

    def _push(self, run_id: str, event: dict, explicit: bool):
        """Queue an event for the pages watching the run: answers asked for in order (the newest EXPLICIT_KEPT), the
        followed cell's newest only (the same answer again is not news)."""
        with self.lock:
            if run_id in self.gone:
                return
            box = self.outbox.setdefault(run_id, {"explicit": [], "follow": None})
            if not explicit and box["follow"] is not None:
                last = box["follow"][1]
                if all(last.get(k) == event.get(k) for k in ("seq", "cell", "model", "state")) and event.get("cached") \
                        and _count_named(last.get("result")) == _count_named(event.get("result")):   # (counted again:
                    return                                                                       # news again)
            self.n += 1
            if explicit:
                box["explicit"].append((self.n, event))
                del box["explicit"][:-EXPLICIT_KEPT]
            else:
                box["follow"] = (self.n, event)
            self.local_version += 1

    def _save(self, run_id: str, seq, cell: int, model: str, hit: dict, run=None):
        """Keep an answer asked for on disk (its writer thread), once."""
        with self.lock:
            if hit.get("disk") or run_id in self.gone:
                return
            hit["disk"] = True
        if run is None:
            try:
                run = self.runs.get(run_id)
            except Exception:
                return
        try:
            data = json.dumps({"v": RESULT_V, "model": model, "seq": seq, "cell": cell, "iter": hit.get("iter"),
                               "result": hit["result"]}, separators=(",", ":"), allow_nan=False).encode()
        except (TypeError, ValueError):              # not JSON (never from the worker): kept in memory only
            return
        path = self._path(run.dir, seq, cell, model, hit.get("count") or _count_named(hit["result"]))
        self.writes.put((run_id, Path(run.dir), path, data))
        with self.lock:
            if self.writer is None or not self.writer.is_alive():
                self.writer = threading.Thread(target=self._write_loop, name="deeper-writer", daemon=True)
                self.writer.start()

    def _write_loop(self):
        while True:
            item = self.writes.get()
            if item is None:
                return
            run_id, run_dir, path, data = item
            with self.lock:
                if run_id in self.gone:
                    continue
            try:                                     # never parents=True: a deleted run's folder is not made again
                path.parent.parent.mkdir(exist_ok=True)        # <run>/qc/<model8> (<run>/qc holds the run's switch)
                path.parent.mkdir(exist_ok=True)
                _atomic_write(path, data)
            except FileNotFoundError:                # the run went meanwhile
                continue
            except OSError as exc:
                print(f"SoPA companion: Go deeper could not keep an answer of run {run_id}: {exc}", flush=True)
                continue
            self._index(run_id, run_dir, path)

    def _index(self, run_id: str, run_dir: Path, path: Path):
        """Count the run's answers on disk (listed once per session) and remove the oldest past DISK_KEPT."""
        with self.lock:
            index = self.disk.get(run_id)
        if index is None:
            found = []
            for item in run_dir.glob("qc/*/deeper/*.json"):
                try:
                    found.append((item.stat().st_mtime, str(item)))
                except OSError:
                    continue
            with self.lock:
                if run_id in self.gone:
                    return
                index = self.disk.setdefault(run_id, OrderedDict((name, None) for _, name in sorted(found)))
        with self.lock:
            index.pop(str(path), None)
            index[str(path)] = None
            drop = []
            while len(index) > DISK_KEPT:
                drop.append(index.popitem(last=False)[0])
        for name in drop:
            try:
                os.unlink(name)
            except OSError:
                pass

    @staticmethod
    def _byte(path: Path, cell: int) -> int | None:
        """The cell's byte in one stored file (.qc.bin, start.bin); None for a missing, short or damaged one."""
        try:
            data = _read_codes(path)
        except Exception:                             # missing, unreadable, or a damaged zstd frame
            return None
        return int(data[cell]) if 0 <= cell < len(data) else None
