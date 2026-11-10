"""The notebook route end to end, as an alpha tester runs it (release.yml's install job, on Linux, macOS and Windows):

    python ci/launch_smoke.py WORK_DIR

1. sopa's toy sample written to WORK_DIR/toy.zarr; sopa_companion.launch(sdata, open="none") in a real Jupyter kernel
   of this environment.
2. Through the engine's API, as the page does: the session's sample ready; proseg on a box of it, a few dozen
   iterations, finished and saved into the sample; a Cellpose stain run (cpsam, on the CPU) on a smaller box, saved.
3. app.close() in the kernel: both results added to the notebook's object, the session closed.
4. A second session whose kernel is killed outright while proseg runs: the engine and proseg must end by themselves
   (the end of the engine's control pipe; on macOS and Windows the engine's reaper, portable.tie).
5. Nothing of either session left running.

Needs jupyter_client and ipykernel besides the app (the README's install). Exit 0 when every step passed; otherwise the
step that failed, with the engine's log and the run's log.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import sys
import textwrap
import time
import urllib.error
import urllib.request

import psutil
from jupyter_client.manager import KernelManager

BASE = "/api/v1"
ITERATIONS = 30
STEP_S = {"kernel": 120, "launch": 600, "ready": 600, "frame0": 900, "iterations": 900, "finish": 900, "save": 900,
          "stain": 2400, "close": 600, "gone": 120}
STARTED = time.time()


class Failed(Exception):
    pass


def say(*parts):
    print(f"[{time.time() - STARTED:7.1f} s]", *parts, flush=True)


def wait(what: str, predicate, timeout: float, interval: float = 0.5):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise Failed(f"{what}: not within {timeout:g} s")
        time.sleep(interval)


# ---- the kernel ------------------------------------------------------------------------------------------------------
class Kernel:
    def __init__(self, work: Path):
        self.manager = KernelManager(kernel_name="python3")
        self.manager.start_kernel(cwd=str(work), env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.client = self.manager.client()
        self.client.start_channels()
        self.client.wait_for_ready(timeout=STEP_S["kernel"])
        self.pid = self.manager.provisioner.pid if self.manager.provisioner else None

    def run(self, code: str, timeout: float) -> str:
        """Run a cell; its printed output. Failed with the traceback when it raised."""
        msg_id = self.client.execute(textwrap.dedent(code), store_history=False)
        out, deadline = [], time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise Failed(f"a cell did not finish within {timeout:g} s:\n{code}")
            try:
                message = self.client.get_iopub_msg(timeout=min(left, 5))
            except queue.Empty:
                continue
            if message["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "stream":
                out.append(content["text"])
            elif kind == "error":
                raise Failed("a cell raised:\n" + "\n".join(content["traceback"]))
            elif kind == "status" and content["execution_state"] == "idle":
                return "".join(out)

    def value(self, code: str, timeout: float) -> dict:
        """A cell whose last printed line is JSON."""
        text = self.run(code, timeout)
        lines = [line for line in text.splitlines() if line.strip()]
        try:
            return json.loads(lines[-1])
        except (IndexError, ValueError):
            raise Failed(f"a cell printed no JSON:\n{text}") from None

    def kill(self):
        self.client.stop_channels()
        self.manager.shutdown_kernel(now=True)          # the kernel's process killed, no clean-up of its own

    def close(self):
        self.client.stop_channels()
        self.manager.shutdown_kernel()


LAUNCH = """
import json, time
import spatialdata, sopa_companion
sdata = spatialdata.read_zarr({path!r})
app = sopa_companion.launch(sdata, open="none", timeout=600)
for _ in range(1200):
    if app._port is not None or app.state in ("failed", "closed"):
        break
    time.sleep(0.5)
print(json.dumps({{"port": app._port, "token": app._config["server"]["token"], "state": app.state,
                  "state_dir": str(app._folders.state), "engine": app._process.pid,
                  "failure": None if app._failure is None else str(app._failure)}}))
"""


# ---- the engine's API ------------------------------------------------------------------------------------------------
class Api:
    def __init__(self, port: int, token: str):
        self.base = f"http://127.0.0.1:{port}"
        self.headers = {"Authorization": "Bearer " + token, "Origin": self.base, "Content-Type": "application/json"}

    def call(self, method: str, path: str, body=None, *, expect=(200,)):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + BASE + path, data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                status, text = response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            status, text = error.code, error.read().decode("utf-8", "replace")
        if status not in expect:
            raise Failed(f"{method} {path} -> {status}: {text[:600]}")
        try:
            return json.loads(text)
        except ValueError:
            return text

    def run(self, run_id: str) -> dict:
        return self.call("GET", f"/runs/{run_id}")

    def log(self, run_id: str) -> str:
        try:
            text = self.call("GET", f"/runs/{run_id}/log")
            return text if isinstance(text, str) else json.dumps(text)
        except Failed as exc:
            return str(exc)


def ended(meta: dict) -> bool:
    return meta.get("state") in ("done", "failed", "stopped")


def box_in(extent, side_um: float) -> list[float]:
    x0, y0, x1, y1 = extent
    side = max(20.0, min(side_um, 0.4 * min(x1 - x0, y1 - y0)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [round(cx - side / 2, 2), round(cy - side / 2, 2), round(cx + side / 2, 2), round(cy + side / 2, 2)]


def prior_of(detail: dict) -> dict:
    suggested = detail.get("suggested_prior") or {}
    if suggested.get("cells") or suggested.get("nuclei"):
        return {key: value for key, value in suggested.items() if value}
    for part in ("cells", "nuclei"):
        for option in (detail.get("prior_options") or {}).get(part) or []:
            return {part: {key: value for key, value in option.items()
                           if key in ("kind", "key", "column", "unassigned", "nuclear")}}
    raise Failed("the toy sample offers no prior for proseg")


def save(api: Api, run_id: str, shapes_key: str, table_key: str | None) -> dict:
    api.call("POST", f"/runs/{run_id}/zarr", {"shapes_key": shapes_key, "table_key": table_key,
                                               "set_default": False, "overwrite": False})
    saved = wait(f"the save of {shapes_key}",
                 lambda: (lambda state: state if state.get("state") in ("saved", "failed") else None)(
                     api.run(run_id).get("zarr_save") or {}), STEP_S["save"])
    if saved["state"] != "saved":
        raise Failed(f"the save of {shapes_key} failed: {saved.get('message')}\n{api.log(run_id)[-4000:]}")
    # Left as the page leaves a saved run (one run is open at a time), once the notebook has it.
    api.call("POST", f"/runs/{run_id}/leave", {"answer": "add"})
    wait(f"the run of {shapes_key} closed", lambda: run_id not in {run["id"] for run in api.call("GET", "/runs")},
         STEP_S["save"])
    return saved


def proseg(api: Api, sample: str, detail: dict, *, finish: bool) -> str:
    body = {"sample": sample, "region": {"kind": "box", "box": box_in(detail["extent"], 150)},
            "prior": prior_of(detail), "options": {}}
    run_id = api.call("POST", "/runs", body)["id"]
    say("proseg run", run_id, "on", body["region"]["box"])
    meta = wait("proseg's first frame", lambda: (lambda m: m if ended(m) or (m.get("frames", 0) >= 1 and
                                                                               m.get("state") == "paused") else None)(
        api.run(run_id)), STEP_S["frame0"])
    if ended(meta):
        raise Failed(f"proseg ended before its first frame: {meta.get('message')}\n{api.log(run_id)[-4000:]}")
    api.call("POST", f"/runs/{run_id}/control", {"action": "resume"})
    meta = wait(f"{ITERATIONS} iterations", lambda: (lambda m: m if ended(m) or ((m.get("last") or {}).get("iter") or 0)
                                                    >= ITERATIONS else None)(api.run(run_id)), STEP_S["iterations"])
    if ended(meta):
        raise Failed(f"proseg ended while sampling: {meta.get('message')}\n{api.log(run_id)[-4000:]}")
    say("proseg at iteration", (meta.get("last") or {}).get("iter"), "with", meta.get("frames"), "frames")
    if not finish:
        return run_id
    api.call("POST", f"/runs/{run_id}/control", {"action": "finish"})
    meta = wait("Finish & save", lambda: (lambda m: m if ended(m) else None)(api.run(run_id)), STEP_S["finish"])
    if meta.get("state") != "done":
        raise Failed(f"Finish & save ended {meta.get('state')}: {meta.get('message')}\n{api.log(run_id)[-4000:]}")
    save(api, run_id, "smoke_proseg_boundaries", "smoke_proseg_table")
    say("proseg's cells saved into the sample")
    return run_id


def stain(api: Api, sample: str, detail: dict) -> str:
    images = detail.get("images") or []
    image = next((i for i in images if i.get("key") == detail.get("image_key")), images[0] if images else None)
    if image is None:
        raise Failed("the toy sample has no image for Cellpose")
    channels = [str(c) for c in image.get("channels") or []]
    nucleus = next((c for c in channels if "dapi" in c.lower()), channels[0])
    settings = {"method": "cellpose", "model": "cpsam", "image": image["key"], "nucleus": nucleus, "cytoplasm": None,
                "diameter": None, "flow_threshold": 0.4, "cellprob_threshold": 0, "tissue_only": False}
    region = {"kind": "box", "box": box_in(detail["extent"], 60)}
    run_id = api.call("POST", "/runs", {"sample": sample, "kind": "stain", "region": region, "stain": settings})["id"]
    say("Cellpose run", run_id, "on", region["box"])
    api.call("POST", f"/runs/{run_id}/control", {"action": "resume"})
    meta = wait("the Cellpose run", lambda: (lambda m: m if ended(m) else None)(api.run(run_id)), STEP_S["stain"], 2)
    if meta.get("state") != "done":
        raise Failed(f"the Cellpose run ended {meta.get('state')}: {meta.get('message')}\n{api.log(run_id)[-4000:]}")
    say("Cellpose found", (meta.get("summary") or {}).get("cells"), "cells on", (meta.get("stain_runtime") or {}).get("device"))
    save(api, run_id, "smoke_cellpose_boundaries", None)
    say("Cellpose's cells saved into the sample")
    return run_id


def session_ready(api: Api) -> dict:
    def ready():
        try:
            health = api.call("GET", "/health")
        except (Failed, OSError):
            return None
        sample = (health.get("session") or {}).get("sample") or {}
        if sample.get("state") == "failed":
            raise Failed(f"the session's sample failed to open: {sample.get('message')}")
        return health if sample.get("state") == "ready" and sample.get("id") else None
    health = wait("the session's sample", ready, STEP_S["ready"], 1)
    wrong = {key: health.get(key) for key in ("ok", "proseg", "zarr") if health.get(key) is not True}
    if wrong or not (health.get("stain") or {}).get("available"):
        raise Failed(f"the engine's health: {wrong}, stain {health.get('stain')}")
    return health


def leftovers(since: float, engines: list[int], state_dirs=()) -> list[str]:
    """This smoke's processes still running: the engines it started (and, behind a Windows venv's python.exe, the
    interpreter it started: any process naming a session's settings), and any proseg-live started since."""
    found = []
    named = [str(folder) for folder in state_dirs]
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            line = " ".join(process.info["cmdline"] or [])
            if process.info["pid"] in engines or any(folder in line for folder in named) or \
                    (process.info["create_time"] >= since - 1 and "proseg-live" in (process.info["name"] or "") + line):
                if process.status() != psutil.STATUS_ZOMBIE:
                    found.append(f"{process.info['pid']}: {line[:200]}")
        except (psutil.Error, OSError):
            continue
    return found


def gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def tail(path: Path, lines: int = 60) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError as exc:
        return f"({path}: {exc})"


def main(work: Path) -> int:
    for stream in (sys.stdout, sys.stderr):   # the engine's log in a failure's report, whatever the console's code page
        stream.reconfigure(errors="backslashreplace")
    work.mkdir(parents=True, exist_ok=True)
    import sopa
    sample = work / "toy.zarr"
    sopa.io.toy_dataset(length=1000).write(sample)
    say("toy sample written:", sample)
    engines, state_dirs, kernel = [], [], None
    try:
        # ---- 1-3: a whole session, closed from the notebook ----------------------------------------------------------
        kernel = Kernel(work)
        say("kernel", kernel.pid, "ready")
        opened = kernel.value(LAUNCH.format(path=str(sample)), STEP_S["launch"])
        state_dirs.append(Path(opened["state_dir"]))
        engines.append(opened["engine"])
        if opened["port"] is None:
            raise Failed(f"launch() gave no engine: {opened.get('failure') or opened}")
        say("engine", opened["engine"], "on port", opened["port"])
        api = Api(opened["port"], opened["token"])
        health = session_ready(api)
        sample_id = health["session"]["sample"]["id"]
        detail = api.call("GET", f"/samples/{sample_id}")
        proseg(api, sample_id, detail, finish=True)
        stain(api, sample_id, detail)
        closed = kernel.value("""
            app.close(timeout=600)
            print(json.dumps({"state": app.state, "added": list(app.added), "shapes": sorted(sdata.shapes),
                              "tables": sorted(sdata.tables)}))
        """, STEP_S["close"])
        say("closed:", closed)
        missing = [key for key in ("smoke_proseg_boundaries", "smoke_cellpose_boundaries") if key not in closed["shapes"]]
        if closed["state"] != "closed" or missing or "smoke_proseg_table" not in closed["tables"]:
            raise Failed(f"app.close() left {closed}: missing {missing}")
        wait("the first engine's end", lambda: gone(engines[0]), STEP_S["gone"])
        kernel.close()

        # ---- 4: a kernel killed while proseg runs ----------------------------------------------------------------------
        kernel = Kernel(work)
        opened = kernel.value(LAUNCH.format(path=str(sample)), STEP_S["launch"])
        state_dirs.append(Path(opened["state_dir"]))
        engines.append(opened["engine"])
        api = Api(opened["port"], opened["token"])
        health = session_ready(api)
        sample_id = health["session"]["sample"]["id"]
        proseg(api, sample_id, api.call("GET", f"/samples/{sample_id}"), finish=False)
        say("killing kernel", kernel.pid, "while proseg runs")
        kernel.kill()
        left = []
        deadline = time.monotonic() + STEP_S["gone"]
        while time.monotonic() < deadline:
            left = leftovers(STARTED, engines, state_dirs)
            if not left:
                break
            time.sleep(1)
        if left:
            raise Failed("still running after the kernel was killed:\n" + "\n".join(left))
        say("the engine and proseg ended with the kernel")
    except Failed as exc:
        say("FAILED:", exc)
        try:
            kernel.kill()                      # its engine closes with it: nothing of this smoke goes on running
        except Exception:
            pass
        time.sleep(5)
        for folder in state_dirs:
            say(f"engine log ({folder}):\n" + tail(folder / "logs" / "server.log"))
        for process in leftovers(STARTED, engines, state_dirs):
            say("left running:", process)
        return 1
    say("ok: launch, proseg, Cellpose, saves, close and a killed kernel on", sys.platform)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1]).resolve()))
