"""A real ipykernel, private IPC channels and a fake engine; no server, display or GPU.

The fake is enough to exercise the delivery without the engine: it writes unconsolidated staged shapes/counts,
optionally writes a backed sample via write_element, emits deliver, and removes staging at received.
"""
import json
import errno
import os
from pathlib import Path
from queue import Empty
import socket
import sys
import time

import pytest

pytest.importorskip("spatialdata", reason="kernel delivery tests need the sopa environment")
jupyter_client = pytest.importorskip("jupyter_client", reason="kernel delivery tests need jupyter_client")
pytest.importorskip("ipykernel", reason="kernel delivery tests need ipykernel")


FAKE_ENGINE = r'''
import json, os, select, shutil, sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from anndata import AnnData
from scipy.sparse import csr_matrix
from shapely.geometry import box
from spatialdata.models import ShapesModel, TableModel
from sopa_companion.api.protocol import ENV_EVENTS_FD, build_event, decode_control_line, encode_line

root = Path(sys.argv[1])
backed = sys.argv[2] == "True"
events = os.fdopen(int(os.environ[ENV_EVENTS_FD]), "w", buffering=1)
events.write(encode_line(build_event("session", state="ready", message=None)))
while not (root / "trigger").exists():
    if select.select([sys.stdin], [], [], .02)[0] and not sys.stdin.readline():
        sys.exit(0)

key = "proseg_boundaries"
ids = ["aaaaaaaa-1", "aaaaaaab-1"]
shapes = ShapesModel.parse(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(2, 2, 4, 4)], index=ids))
obs = pd.DataFrame({"region": pd.Categorical([key] * 2), "cell_id": ids}, index=ids)
table = TableModel.parse(AnnData(X=csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.int32)), obs=obs,
                                var=pd.DataFrame(index=["G1", "G2"]),
                                uns={"sopa_companion": {"session": "api"}}),
                         region=key, region_key="region", instance_key="cell_id")
stage = root / "staged.zarr"
sd.SpatialData(shapes={key: shapes}, tables={"proseg_table": table}).write(stage, consolidate_metadata=False)
written_to = None
if backed:
    target = sd.read_zarr(root / "sample.zarr")
    for name, element in ((key, shapes), ("proseg_table", table)):
        target[name] = element
        target.write_element(name, overwrite=False)
    # This mapping is freshly read from disk, as in the helper's transaction, never the caller's stale attrs.
    target.attrs = {**target.attrs, "boundaries_shapes": key}
    target.write_attrs()
    written_to = str(target.path)
event = build_event("deliver", id="fake-delivery", run="fake-run", kind="proseg", staged=str(stage),
                    shapes_key=key, table_key="proseg_table", set_default=True, overwrite=False, replaced=[],
                    written_to=written_to, cells=2, genes=2, provenance={"kind": "proseg", "session": "api"})
events.write(encode_line(event))
with (root / "controls.jsonl").open("w", buffering=1) as controls:
    for line in sys.stdin:
        message = decode_control_line(line)
        if message is None:
            continue
        controls.write(json.dumps(message) + "\n")
        if message["op"] == "received":
            shutil.rmtree(stage)
            (root / "received").write_text("read before apply")
        if message["op"] in ("applied", "receive_failed"):
            events.write(encode_line(build_event("ended", added=["fake-delivery"], not_added=[])))
            (root / "engine-ended").write_text("done")
            break
'''


def wait_file(path, *, timeout=30, log=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    pytest.fail(f"{path.name} was not written; engine log: {log.read_text() if log and log.exists() else ''}")


def finish_cell(client, message_id, *, timeout=60):
    deadline, messages = time.monotonic() + timeout, []
    while time.monotonic() < deadline:
        try:
            message = client.get_iopub_msg(timeout=min(10, max(0.01, deadline - time.monotonic())))
        except Empty:
            continue
        if message.get("parent_header", {}).get("msg_id") != message_id:
            continue
        messages.append(message)
        if message["msg_type"] == "error":
            pytest.fail("\n".join(message["content"]["traceback"]))
        if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
            return messages
    pytest.fail("The kernel did not finish its cell")


@pytest.fixture
def kernel(tmp_path):
    root = tmp_path / "kernel"
    root.mkdir()
    with socket.socket(socket.AF_UNIX) as probe:
        try:
            probe.bind(str(root / "probe"))
        except OSError as exc:
            if exc.errno in (errno.EPERM, errno.EACCES):
                pytest.skip(f"The execution sandbox blocks kernel IPC sockets: {exc}")
            raise
    (root / "probe").unlink()
    log = (root / "kernel.log").open("w")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
           "CUDA_VISIBLE_DEVICES": "", "IPYTHONDIR": str(root / "ipython"),
           "JUPYTER_CONFIG_DIR": str(root / "jupyter"), "JUPYTER_RUNTIME_DIR": str(root / "runtime"),
           "XDG_CACHE_HOME": str(root / "cache"), "MPLCONFIGDIR": str(root / "mpl"),
           "NUMBA_CACHE_DIR": str(root / "numba"), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    manager = jupyter_client.KernelManager(connection_file=str(root / "connection.json"),
                                           transport="ipc", ip=str(root / "ipc"))
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = None
    try:
        manager.start_kernel(cwd=str(root), env=env, stdout=log, stderr=log)
        client = manager.client()
        client.start_channels()
        try:
            client.wait_for_ready(timeout=60)
        except RuntimeError as exc:
            log.flush()
            pytest.fail(f"{exc}\n{(root / 'kernel.log').read_text()}")
        yield client, root
    finally:
        if client is not None:
            # Only the fake child made in this kernel is terminated. A closed control pipe also makes it exit.
            if manager.is_alive():
                cleanup = client.execute("""
if 'delivery_manager' in globals():
    delivery_manager.close()
    os.close(events_r)
if 'engine' in globals():
    engine.stdin.close()
    try:
        engine.wait(timeout=5)
    except __import__('subprocess').TimeoutExpired:
        engine.terminate()
        engine.wait(timeout=5)
    engine_log.close()
""")
                try:
                    finish_cell(client, cleanup, timeout=15)
                except (Empty, AssertionError):
                    pass
            client.stop_channels()
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        log.close()


def start_fake(client, root, *, backed=False):
    code = f'''
import json, os, subprocess, sys, threading, time
from pathlib import Path
import spatialdata as sd
from sopa_companion.api.deliver import DeliveryManager
from sopa_companion.api.protocol import ENV_EVENTS_FD, encode_line
root = Path({str(root)!r})
sdata = sd.SpatialData(attrs={{"caller_note": "keep"}})
if {backed!r}:
    sdata.write(root / "sample.zarr")
events_r, events_w = os.pipe()
engine_log = (root / "engine.log").open("w")
engine = subprocess.Popen([sys.executable, "-c", {FAKE_ENGINE!r}, str(root), {str(backed)!r}],
                          stdin=subprocess.PIPE, stdout=engine_log, stderr=engine_log,
                          pass_fds=(events_w,), start_new_session=True,
                          env={{**os.environ, ENV_EVENTS_FD: str(events_w)}})
os.close(events_w)
def send(message):
    engine.stdin.write(encode_line(message).encode())
    engine.stdin.flush()
def on_event(event):
    if event["ev"] == "session" and event["state"] == "ready":
        (root / "ready").write_text("ready")
def on_applied(result):
    (root / "applied.json").write_text(json.dumps({{"state": result.state, "keys": list(result.keys),
        "thread": threading.current_thread().name, "main_thread": threading.current_thread() is threading.main_thread(),
        "attrs": sdata.attrs, "at": time.monotonic(), "written_to": result.event["written_to"]}}))
delivery_manager = DeliveryManager(sdata, send, on_event=on_event, on_applied=on_applied)
assert delivery_manager.mainthread.automatic
delivery_manager.start(events_r)
'''
    finish_cell(client, client.execute(code))
    wait_file(root / "ready", timeout=60, log=root / "engine.log")


@pytest.mark.parametrize("backed", [False, True])
def test_idle_kernel_gets_results_without_any_followup_call(kernel, backed):
    client, root = kernel
    start_fake(client, root, backed=backed)
    (root / "trigger").write_text("save")
    # No cell, wait(), result(), or apply() is executed while waiting for the callback's evidence.
    wait_file(root / "applied.json", log=root / "engine.log")
    wait_file(root / "engine-ended", log=root / "engine.log")
    outcome = json.loads((root / "applied.json").read_text())
    assert outcome["state"] == "added" and outcome["keys"] == ["proseg_boundaries", "proseg_table"]
    assert outcome["main_thread"] and outcome["attrs"]["boundaries_shapes"] == "proseg_boundaries"
    assert outcome["attrs"]["caller_note"] == "keep" and not (root / "staged.zarr").exists()
    assert outcome["written_to"] == (str(root / "sample.zarr") if backed else None)
    controls = [json.loads(line) for line in (root / "controls.jsonl").read_text().splitlines()]
    assert [message["op"] for message in controls] == ["received", "applied"]
    if backed:
        import spatialdata as sd
        disk = sd.read_zarr(root / "sample.zarr")
        assert disk["proseg_table"].X.toarray().tolist() == [[1, 2], [3, 4]]
        assert disk.attrs["boundaries_shapes"] == "proseg_boundaries"
    # The engine's stdout/stderr belongs in its log, and automatic apply itself produces no cell output.
    unsolicited = []
    while True:
        try:
            unsolicited.append(client.get_iopub_msg(timeout=0.1))
        except Empty:
            break
    assert not [message for message in unsolicited if message["msg_type"] in ("stream", "error")]


def test_busy_kernel_defers_apply_to_cell_end_and_reports_waiting(kernel):
    client, root = kernel
    start_fake(client, root)
    busy = client.execute('''
(root / "busy-started").write_text("busy")
time.sleep(4)
(root / "busy-ended").write_text(str(time.monotonic()))
''')
    wait_file(root / "busy-started")
    (root / "trigger").write_text("save")
    wait_file(root / "received", log=root / "engine.log")
    assert not (root / "busy-ended").exists() and not (root / "applied.json").exists()
    finish_cell(client, busy)
    wait_file(root / "applied.json", log=root / "engine.log")
    wait_file(root / "engine-ended", log=root / "engine.log")
    outcome = json.loads((root / "applied.json").read_text())
    assert outcome["state"] == "added" and outcome["main_thread"]
    assert outcome["at"] >= float((root / "busy-ended").read_text())
    controls = [json.loads(line) for line in (root / "controls.jsonl").read_text().splitlines()]
    assert [message for message in controls if message["op"] == "notebook"] == [
        {"op": "notebook", "busy": True}, {"op": "notebook", "busy": False}]
