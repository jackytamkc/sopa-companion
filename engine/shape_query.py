"""Box queries over a sample's exported shapes (the picker's prior outlines), answered in worker processes.

The picker asks for the outlines in view on every camera move: both prior layers at once, up to 20,000 polygons each.
That JSON is GIL-bound to build (a dict per polygon, then the encoders), and FastAPI re-encoded the whole structure
on the event loop, so on whole samples every request, WebSocket push and health check of the engine stalled for up
to 1.5 s (a thread does not help with the GIL). Here two worker processes
(`python -m sopa_companion.engine.shape_query`, the engine's own interpreter and code) load an element's exported
outlines once (<key>.bin + <key>.json from zarr_bridge/helper.py `shapes`, layout in that .json), keep them while the
files stay the same, and send the response body back as bytes: the engine thread only waits on a pipe.

Protocol on the worker's stdin/stdout: a request is a u32 little-endian length and JSON {base, key, box, limit}; the
reply is a status byte (0 the body, 1 a JSON error {type, message}), a u64 length and the payload. A worker exits when
its stdin closes, so it never outlives the engine (no other process holds that pipe).

The body is byte-identical to the old one: coordinates are the float32 values rounded to 0.01 (in float32, as np.round
did per polygon) and written as the float64 numbers they are, e.g. 8341.48046875.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

WORKERS = 2               # the picker asks for the cells and the nuclei layer at the same time
KEPT = 6                  # elements a worker keeps loaded (a dense whole sample: ~24 MB per layer)
TIMEOUT_S = 120.0
PIPE_BYTES = 1 << 20      # a bigger reply pipe: fewer wake-ups for a 5-50 MB body


def load_outlines(base: str | Path) -> dict:
    """An exported shapes element as arrays: ids (list), bounds f32[parts, 4], offsets u32[parts + 1],
    owner u32[parts] (index into ids), coords f32[vertices, 2]."""
    import numpy as np
    meta = json.loads(Path(f"{base}.json").read_text())
    raw = np.fromfile(f"{base}.bin", dtype=np.uint8)     # a copy, not a map: a replaced export never stays held
    parts, vertices = meta["parts"], meta["vertices"]
    edges = np.cumsum([0, parts * 16, (parts + 1) * 4, parts * 4, vertices * 8])
    if raw.size < edges[-1]:
        raise ValueError(f"{base}.bin is shorter than its description")
    return {"ids": meta["ids"],
            "bounds": raw[edges[0]:edges[1]].view("<f4").reshape(parts, 4),
            "offsets": raw[edges[1]:edges[2]].view("<u4"),
            "owner": raw[edges[2]:edges[3]].view("<u4"),
            "coords": raw[edges[3]:edges[4]].view("<f4").reshape(vertices, 2)}


def encode(data: dict, key: str, box, limit: int) -> bytes:
    """The body of GET /samples/{id}/shapes/{key}: {key, total, truncated, polygons: [{id, poly}]}, the polygons whose
    bounds touch the box (all without one), the first `limit` of them: vectors.hits and vectors.gather, the selection
    of every prior source."""
    import numpy as np
    import orjson
    from .vectors import gather, hits
    found, total = hits(data["bounds"], box, limit)
    edges, xy = gather(data, found)
    xy, ids = xy.astype(np.float64), data["ids"]          # one float64 view per polygon for orjson
    polygons = [{"id": ids[owner], "poly": xy[a:b]}
                for owner, a, b in zip(data["owner"][found].tolist(), edges[:-1].tolist(), edges[1:].tolist())]
    return orjson.dumps({"key": key, "total": total, "truncated": total > len(polygons), "polygons": polygons},
                        option=orjson.OPT_SERIALIZE_NUMPY)


# ---- worker ------------------------------------------------------------------------------------------------------
_loaded: OrderedDict = OrderedDict()      # base -> (file signature, arrays), most recently used last


def query(base: str, key: str, box, limit: int) -> bytes:
    """The element's outlines (loaded again once the export was replaced), encoded for the box."""
    signature = tuple((s.st_ino, s.st_size, s.st_mtime_ns) for s in (os.stat(f"{base}.bin"), os.stat(f"{base}.json")))
    found = _loaded.get(base)
    if found is None or found[0] != signature:
        found = _loaded[base] = (signature, load_outlines(base))
    _loaded.move_to_end(base)
    while len(_loaded) > KEPT:
        _loaded.popitem(last=False)
    return encode(found[1], key, box, limit)


def _exactly(stream, size: int) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data += chunk
    return bytes(data)


def main() -> int:
    replies = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                          # a stray print must never corrupt a reply
    requests = sys.stdin.buffer
    while True:
        head = _exactly(requests, 4)
        if head is None:                   # the engine closed the pipe (or is gone)
            return 0
        request = json.loads(_exactly(requests, int.from_bytes(head, "little")) or b"{}")
        try:
            status, body = 0, query(request["base"], request["key"], request["box"], request["limit"])
        except Exception as exc:
            status, body = 1, json.dumps({"type": type(exc).__name__, "message": str(exc)}).encode()
        replies.write(bytes([status]) + len(body).to_bytes(8, "little"))
        replies.write(body)
        replies.flush()


# ---- engine side -------------------------------------------------------------------------------------------------
class WorkerError(Exception):
    pass


class Unavailable(WorkerError):
    """The pool is closed: the engine is shutting down (503, never a reason to retry here)."""


def _read(fd: int, size: int, deadline: float) -> memoryview:
    """`size` bytes of the pipe, as a view of the one buffer they were read into (a body can be hundreds of MB: no
    second copy). poll(), not select(): an engine with many descriptors gets pipes numbered past select's 1024. Windows
    has neither poll nor readv: the pipe is looked at as portable.readable does, and read into the view by FileIO."""
    data, got = bytearray(size), 0
    view = memoryview(data)
    if not hasattr(select, "poll") or not hasattr(os, "readv"):
        import io
        from .. import portable
        stream = io.FileIO(fd, "rb", closefd=False)
        while got < size:
            left = deadline - time.monotonic()
            if left <= 0 or not portable.readable(fd, left):
                raise TimeoutError("the shapes worker did not answer in time")
            got_now = stream.readinto(view[got:got + PIPE_BYTES])
            if not got_now:
                raise WorkerError("the shapes worker exited")
            got += got_now
        return view
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    while got < size:
        left = deadline - time.monotonic()
        if left <= 0 or not poller.poll(max(1, math.ceil(left * 1000))):
            raise TimeoutError("the shapes worker did not answer in time")
        got_now = os.readv(fd, [view[got:got + PIPE_BYTES]])
        if not got_now:
            raise WorkerError("the shapes worker exited")
        got += got_now
    return view


class ShapeQueries:
    """Up to `workers` worker processes, started on demand; one request at a time each."""

    def __init__(self, workers: int = WORKERS):
        self.workers = workers
        self.cv = threading.Condition()
        self.idle: list[subprocess.Popen] = []
        self.out = 0                       # workers handed out (or being started)
        self.closed = False

    def _spawn(self) -> subprocess.Popen:
        from .. import portable
        from . import children
        from .config import PACKAGE
        env = children.env("same")                   # this interpreter: the app as it found it, no desktop
        env.update(dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"), "1"))
        worker = subprocess.Popen([sys.executable, "-m", f"{PACKAGE}.engine.shape_query"], cwd=children.app_path(),
                                  env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True,
                                  **portable.no_window())
        portable.tie(worker)                         # macOS and Windows: it ends with the engine
        try:                                         # Linux's bigger pipe (elsewhere the system's own)
            import fcntl
            fcntl.fcntl(worker.stdout.fileno(), fcntl.F_SETPIPE_SZ, PIPE_BYTES)
        except (ImportError, AttributeError, OSError):
            pass
        return worker

    def _take(self, deadline: float) -> subprocess.Popen:
        """An idle worker, or a new one while fewer than `workers` are out; waits for one until `deadline`."""
        with self.cv:
            while True:
                if self.closed:
                    raise Unavailable("The engine is shutting down")
                while self.idle:
                    worker = self.idle.pop()
                    if worker.poll() is None:
                        self.out += 1
                        return worker
                    _stop(worker)                  # died while idle (killed): only its pipes are left
                if self.out < self.workers:
                    self.out += 1
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("every shapes worker is busy")
                self.cv.wait(left)
        try:
            return self._spawn()
        except BaseException:
            self._give(None)
            raise

    def _give(self, worker: subprocess.Popen | None):
        with self.cv:
            self.out -= 1
            keep = worker is not None and not self.closed
            if keep:
                self.idle.append(worker)
            self.cv.notify()
        if worker is not None and not keep:
            _stop(worker)

    def query(self, base: str | Path, key: str, box, limit: int, timeout: float = TIMEOUT_S) -> memoryview:
        """The response body for the box (bytes-like). Blocks the calling thread, without the GIL, until a worker
        answers, at most `timeout` s in all (waiting for a free worker included: TimeoutError); a worker that died is
        replaced once. Errors in the worker come back as their own type where it matters (FileNotFoundError: the
        export was replaced meanwhile), else as WorkerError; Unavailable once the pool is closed."""
        request = json.dumps({"base": str(base), "key": key, "box": None if box is None else [float(v) for v in box],
                              "limit": int(limit)}).encode()
        deadline = time.monotonic() + timeout
        for attempt in range(2):
            worker = self._take(deadline)
            try:
                os.write(worker.stdin.fileno(), len(request).to_bytes(4, "little") + request)
                head = bytes(_read(worker.stdout.fileno(), 9, deadline))
                body = _read(worker.stdout.fileno(), int.from_bytes(head[1:], "little"), deadline)
            except BaseException as exc:
                # Whatever went wrong (gone, hung, a reply too big to hold), this worker is never reused and its
                # slot is free again: a leaked slot would park every later request for good.
                self._give(None)
                _stop(worker)
                if isinstance(exc, TimeoutError) or not isinstance(exc, (OSError, WorkerError)):
                    raise
                if attempt:
                    if isinstance(exc, WorkerError):
                        raise
                    raise WorkerError(f"the shapes worker failed: {exc}") from exc     # not a storage error (503)
                continue
            self._give(worker)
            if head[0] == 0:
                return body
            error = json.loads(bytes(body))
            if error.get("type") == "FileNotFoundError":
                raise FileNotFoundError(error.get("message"))
            raise WorkerError(f"{error.get('type')}: {error.get('message')}")
        raise AssertionError("unreachable")

    def close(self):
        with self.cv:
            self.closed, idle, self.idle = True, self.idle, []
            self.cv.notify_all()
        for worker in idle:
            _stop(worker)


def _stop(worker: subprocess.Popen):
    try:
        worker.stdin.close()               # its read sees the end: it exits by itself
    except OSError:
        pass
    try:
        worker.wait(timeout=2)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()
    worker.stdout.close()


if __name__ == "__main__":
    sys.exit(main())
