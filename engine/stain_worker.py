"""Small CPU-only bridge to the patch and comparison code in the installed Shapely environment.

One helper process serves every call, started by whichever request thread calls first and kept for the next ones. It
is never started under setpriv --pdeathsig: that signal follows the THREAD that started it, and the server retires idle
request threads, so the helper died under the calls of other threads. It ends instead when
its stdin closes (zarr_bridge/stain_patches.py serve): close() here, or the engine's exit, whatever ends it.
"""
import json
import subprocess
import threading

from .. import portable
from . import children
from .runs import RunError


# How long a call waits for the helper's answer: an export writes every patch (up to 64 of up to 4096 x 4096 pixels)
# before it answers, so it waits as long as a model's export may run (engine/stain_patches.py, 900 s); a read, 60 s.
WAIT_S = {"export": 900}
READ_WAIT_S = 60


class StainWorker:
    def __init__(self, python, module="sopa_companion.zarr_bridge.stain_patches"):
        self.module = module
        self.python, self.process = python, None
        self.lock = threading.RLock()

    def close(self):
        with self.lock:
            process, self.process = self.process, None
            if process is None:
                return
            process.stdin.close()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()

    def call(self, run, action, **values):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.close()
                # The app as its environment needs it (the desktop: the app alone, as before), no desktop.
                env = children.env(children.kind(self.python), inherit=False)
                env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           NUMBA_CACHE_DIR=str(run.dir / "stain/cache/numba"),
                           MPLCONFIGDIR=str(run.dir / "stain/cache/matplotlib"))
                with (run.dir / "stain-patches.log").open("a") as log:
                    self.process = subprocess.Popen(
                        [self.python, "-B", "-m", self.module], env=env, cwd=run.dir, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=log, close_fds=True, **portable.no_window())
                    portable.tie(self.process)          # macOS and Windows: it ends with the engine
            try:
                payload = json.dumps({"root": str(run.dir), "meta": run.meta, "action": action, **values}, allow_nan=False)
                self.process.stdin.write(payload.encode()+b"\n")
                self.process.stdin.flush()
                if not portable.readable(self.process.stdout.fileno(), WAIT_S.get(action, READ_WAIT_S)):
                    raise TimeoutError
                answer = json.loads(self.process.stdout.readline())
            except (OSError, ValueError, TimeoutError) as exc:
                self.close()
                raise RunError("The patches could not be read. Try opening the run again", 503) from exc
            if not answer["ok"]:
                raise RunError(answer["detail"], answer["status"])
            return answer["result"]
