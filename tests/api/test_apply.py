"""Delivery uses real, small SpatialData elements, never real samples or a running engine."""
import asyncio
from copy import deepcopy
import errno
import hashlib
import os
from pathlib import Path
import shutil
import selectors
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sd = pytest.importorskip("spatialdata", reason="API delivery tests need the sopa environment")
import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from shapely.geometry import box
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import Identity, get_transformation

from sopa_companion.api.deliver import DeliveryManager
from sopa_companion.api.mainthread import MainThread
from sopa_companion.api.protocol import build_event, encode_line, validate_control


@pytest.fixture(scope="module", autouse=True)
def poll_when_socket_wakeup_is_blocked():
    # A sandbox that refuses even socketpair.send() leaves zarr's worker asleep forever.
    # Poll only under that restriction; the filesystem IO and all delivered SpatialData elements remain real.
    probe = asyncio.new_event_loop()
    try:
        try:
            probe._csock.send(b"\0")
        except OSError as exc:
            if exc.errno != errno.EPERM:
                raise
        else:
            yield
            return
    finally:
        probe.close()

    class PollingSelector(selectors.DefaultSelector):
        def select(self, timeout=None):
            return super().select(0.005 if timeout is None else min(timeout, 0.005))

    class PollingPolicy(asyncio.DefaultEventLoopPolicy):
        def new_event_loop(self):
            return asyncio.SelectorEventLoop(selector=PollingSelector())

    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(PollingPolicy())
    try:
        yield
    finally:
        from zarr.core.sync import cleanup_resources
        cleanup_resources()
        asyncio.set_event_loop_policy(previous)


def result_elements(shapes_key="proseg_boundaries"):
    ids = ["aaaaaaaa-1", "aaaaaaab-1"]
    shapes = ShapesModel.parse(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(2, 2, 4, 4)], index=ids),
                               transformations={"global": Identity()})
    obs = pd.DataFrame({"region": pd.Categorical([shapes_key] * 2), "cell_id": ids, "slide": ["sample"] * 2,
                        "area": [1.0, 4.0]}, index=ids)
    table = TableModel.parse(AnnData(X=csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.int32)), obs=obs,
                                    var=pd.DataFrame(index=["G1", "G2"]),
                                    obsm={"spatial": np.array([[0.5, 0.5], [3.0, 3.0]])},
                                    uns={"proseg_run": {"iterations": 3}, "sopa_companion": {"session": "api"}}),
                             region=shapes_key, region_key="region", instance_key="cell_id")
    return shapes, table


@pytest.fixture
def staged(tmp_path):
    shapes, table = result_elements()
    path = tmp_path / "staged.zarr"
    sd.SpatialData(shapes={"proseg_boundaries": shapes}, tables={"proseg_table": table}).write(
        path, consolidate_metadata=False)
    event = build_event("deliver", id="delivery-1", run="run-1", kind="proseg", staged=str(path),
                        shapes_key="proseg_boundaries", table_key="proseg_table", set_default=True, overwrite=False,
                        replaced=[], written_to=None, cells=2, genes=2, provenance={"kind": "proseg", "run": "run-1"})
    return event, shapes, table


def digest(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def receive_in_thread(manager, event):
    errors = []

    def receive():
        try:
            manager.receive(event)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=receive)
    thread.start()
    thread.join(timeout=15)
    assert not thread.is_alive()
    assert not errors, errors


def test_received_means_stage_can_be_deleted_before_main_thread_apply(staged, monkeypatch):
    event, shapes, table = staged
    sdata = sd.SpatialData(attrs={"note": "keep"})
    controls, callback_threads = [], []

    def send(message):
        controls.append(validate_control(message))
        if message["op"] == "received":
            shutil.rmtree(event["staged"])

    manager = DeliveryManager(sdata, send, on_applied=lambda result: callback_threads.append(threading.get_ident()))
    monkeypatch.setattr(sd.SpatialData, "write_element", lambda *a, **kw: pytest.fail("kernel wrote to disk"))
    monkeypatch.setattr(sd.SpatialData, "write_attrs", lambda *a, **kw: pytest.fail("kernel wrote attrs"))
    receive_in_thread(manager, event)
    assert not Path(event["staged"]).exists()
    assert manager.pending == ("run-1",) and not sdata.shapes and not sdata.tables
    assert [m["op"] for m in controls] == ["received"]
    result, = manager.apply_ready()
    assert result.state == "added" and result.keys == ("proseg_boundaries", "proseg_table")
    assert callback_threads == [threading.main_thread().ident]
    assert sdata["proseg_boundaries"].equals(shapes)
    assert get_transformation(sdata["proseg_boundaries"], get_all=True) == get_transformation(shapes, get_all=True)
    np.testing.assert_array_equal(sdata["proseg_table"].X.toarray(), table.X.toarray())
    assert sdata["proseg_table"].obs.equals(table.obs)
    assert sdata["proseg_table"].uns["sopa_companion"]["session"] == "api"
    assert sdata.attrs == {"note": "keep", "boundaries_shapes": "proseg_boundaries"}
    assert controls[-1] == {"op": "applied", "id": "delivery-1", "state": "added",
                            "keys": ["proseg_boundaries", "proseg_table"], "clash": [], "message": None}
    assert manager.pending == () and manager.apply_ready() == ()
    manager.close()


def test_backed_helper_commit_and_memory_have_the_same_elements(staged, tmp_path, monkeypatch):
    event, shapes, table = staged
    path = tmp_path / "sample.zarr"
    sd.SpatialData(attrs={"note": "on disk"}).write(path)
    sdata = sd.read_zarr(path)
    # The fake helper uses the same write_element calls as a save, in its own SpatialData, before its deliver event.
    helper = sd.read_zarr(path)
    for key, element in (("proseg_boundaries", shapes), ("proseg_table", table)):
        helper[key] = element
        helper.write_element(key, overwrite=False)
    event = {**event, "written_to": str(path), "set_default": False}
    before = digest(path)
    monkeypatch.setattr(sd.SpatialData, "write_element", lambda *a, **kw: pytest.fail("duplicate kernel write"))
    monkeypatch.setattr(sd.SpatialData, "write_attrs", lambda *a, **kw: pytest.fail("kernel wrote whole attrs"))
    manager = DeliveryManager(sdata, lambda message: None)
    receive_in_thread(manager, event)
    result, = manager.apply_ready()
    assert result.state == "added" and result.event["written_to"] == str(path)
    disk = sd.read_zarr(path)
    assert disk["proseg_boundaries"].equals(sdata["proseg_boundaries"])
    np.testing.assert_array_equal(disk["proseg_table"].X.toarray(), sdata["proseg_table"].X.toarray())
    assert disk["proseg_table"].uns == sdata["proseg_table"].uns
    assert digest(path) == before
    manager.close()


def test_backed_with_auto_save_off_only_changes_memory(staged, tmp_path, monkeypatch):
    from sopa import settings
    monkeypatch.setattr(settings, "auto_save_on_disk", False)
    path = tmp_path / "sample.zarr"
    sd.SpatialData(attrs={"note": "on disk"}).write(path)
    sdata = sd.read_zarr(path)
    before = digest(path)
    manager = DeliveryManager(sdata, lambda message: None)
    receive_in_thread(manager, staged[0])
    result, = manager.apply_ready()
    assert result.state == "added" and result.event["written_to"] is None
    assert "proseg_boundaries" in sdata and "proseg_table" in sdata
    assert digest(path) == before
    assert not sd.read_zarr(path).shapes
    manager.close()


@pytest.mark.parametrize("key", ["proseg_boundaries", "proseg_table", "PROSEG_boundaries"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_new_notebook_name_is_never_replaced(staged, key, overwrite):
    event, shapes, table = staged
    sdata = sd.SpatialData(attrs={"boundaries_shapes": "original"})
    attrs = sdata.attrs
    controls = []
    manager = DeliveryManager(sdata, controls.append)
    element = table if key == "proseg_table" else shapes
    sdata[key] = element
    receive_in_thread(manager, {**event, "overwrite": overwrite, "replaced": [key]})
    result, = manager.apply_ready()
    assert result.state == "failed" and result.reason == "clash" and result.clash == (key,)
    assert sdata[key] is element and sdata.attrs is attrs
    assert len(list(sdata.gen_elements())) == 1
    assert controls[-1]["state"] == "failed" and controls[-1]["keys"] == []
    manager.close()


def test_replace_is_allowed_for_the_original_elements(staged):
    event, shapes, table = staged
    old_shapes, old_table = shapes.copy(), table.copy()
    sdata = sd.SpatialData(shapes={"proseg_boundaries": old_shapes}, tables={"proseg_table": old_table})
    manager = DeliveryManager(sdata, lambda message: None)
    receive_in_thread(manager, {**event, "overwrite": True, "replaced": ["proseg_boundaries", "proseg_table"]})
    result, = manager.apply_ready()
    assert result.state == "added"
    assert sdata["proseg_boundaries"] is not old_shapes and sdata["proseg_table"] is not old_table
    manager.close()


def test_replace_refuses_a_different_element_under_an_original_name(staged):
    event, shapes, table = staged
    sdata = sd.SpatialData(shapes={"proseg_boundaries": shapes}, tables={"proseg_table": table})
    manager = DeliveryManager(sdata, lambda message: None)
    newer = shapes.copy()
    sdata["proseg_boundaries"] = newer
    receive_in_thread(manager, {**event, "overwrite": True, "replaced": ["proseg_boundaries", "proseg_table"]})
    result, = manager.apply_ready()
    assert result.clash == ("proseg_boundaries",)
    assert sdata["proseg_boundaries"] is newer and sdata["proseg_table"] is table
    manager.close()


def test_replace_does_not_break_another_tables_annotation(staged):
    event, shapes, table = staged
    sdata = sd.SpatialData(shapes={"proseg_boundaries": shapes}, tables={"other_table": table})
    manager = DeliveryManager(sdata, lambda message: None)
    receive_in_thread(manager, {**event, "overwrite": True, "replaced": ["proseg_boundaries"]})
    result, = manager.apply_ready()
    assert result.state == "failed" and "other_table" in result.message
    assert sdata["proseg_boundaries"] is shapes and "proseg_table" not in sdata
    manager.close()


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("mutate_first", [False, True])
def test_failure_during_second_assignment_restores_the_whole_object(staged, monkeypatch, exception, mutate_first):
    event = staged[0]
    sdata = sd.SpatialData(attrs={"boundaries_shapes": "old", "note": {"shared": True}})
    attrs = sdata.attrs
    original = sd.SpatialData.__setitem__

    def fail(self, key, element):
        if key == "proseg_table":
            if mutate_first:
                original(self, key, element)
            raise exception("injected second assignment failure")
        original(self, key, element)

    manager = DeliveryManager(sdata, lambda message: None)
    receive_in_thread(manager, event)
    monkeypatch.setattr(sd.SpatialData, "__setitem__", fail)
    result, = manager.apply_ready()
    assert result.state == "failed" and "injected second assignment" in result.message
    assert not sdata.shapes and not sdata.tables and not sdata._shared_keys
    assert sdata.attrs is attrs and attrs["boundaries_shapes"] == "old"
    # Removing partially registered shared keys matters: the same names remain usable afterwards.
    sdata["proseg_boundaries"] = staged[1]
    manager.close()


def test_failure_setting_attrs_restores_replacements_and_original_attrs(staged, monkeypatch):
    event, shapes, table = staged
    sdata = sd.SpatialData(shapes={"proseg_boundaries": shapes}, tables={"proseg_table": table}, attrs={"old": True})
    attrs = sdata.attrs
    manager = DeliveryManager(sdata, lambda message: None)
    receive_in_thread(manager, {**event, "overwrite": True, "replaced": ["proseg_boundaries", "proseg_table"]})
    prop = sd.SpatialData.attrs

    def fail(self, value):
        prop.fset(self, value)
        raise RuntimeError("injected attrs failure")

    monkeypatch.setattr(sd.SpatialData, "attrs", property(prop.fget, fail))
    result, = manager.apply_ready()
    assert result.state == "failed"
    assert sdata["proseg_boundaries"] is shapes and sdata["proseg_table"] is table
    assert sdata._shared_keys == {"proseg_boundaries", "proseg_table"} and sdata.attrs is attrs
    manager.close()


@pytest.mark.parametrize("method", ["cellpose", "instanseg"])
def test_stain_metadata_detaches_the_crops_shared_attrs(tmp_path, method):
    shapes, _ = result_elements()
    attrs = {"boundaries_shapes": "seed", "sopa_companion": {"other": True, "layers": {"seed": {"note": "old"}}}}
    parent = sd.SpatialData(shapes={"seed": shapes}, attrs=attrs)
    crop = parent.subset(["seed"])
    assert crop.attrs is parent.attrs
    before = deepcopy(parent.attrs)
    key = f"{method}_boundaries_1"
    shapes = shapes.copy()
    shapes.index = [f"{method}-1", f"{method}-2"]
    path = tmp_path / "stain.zarr"
    sd.SpatialData(shapes={key: shapes}).write(path, consolidate_metadata=False)
    provenance = {"method": method, "model": "cpsam", "resolution": {"level": 1},
                  "filters": {"size": {"min": 2}}, "covered": {"kind": "box", "bbox": [0, 0, 4, 4]}}
    event = build_event("deliver", id="stain-1", run="run-s", kind="stain", staged=str(path), shapes_key=key,
                        table_key=None, set_default=True, overwrite=False, replaced=[], written_to=None, cells=2,
                        genes=None, provenance=provenance)
    manager = DeliveryManager(crop, lambda message: None)
    receive_in_thread(manager, event)
    result, = manager.apply_ready()
    assert result.state == "added" and result.keys == (key,) and not crop.tables
    assert parent.attrs == before and crop.attrs is not parent.attrs
    assert crop.attrs["sopa_companion"] is not parent.attrs["sopa_companion"]
    assert crop.attrs["sopa_companion"]["layers"] is not parent.attrs["sopa_companion"]["layers"]
    assert crop.attrs["boundaries_shapes"] == key
    stored = crop.attrs["sopa_companion"]["layers"][key]
    assert stored == {**provenance, "kind": "stain", "run": "run-s", "session": "api"}
    assert get_transformation(crop[key], get_all=True) == get_transformation(shapes, get_all=True)
    assert crop[key].index.tolist() == [f"{method}-1", f"{method}-2"]
    manager.close()


@pytest.mark.parametrize("name", ["sopa_companion", "proseg_live"])
def test_a_stain_results_whole_provenance_comes_from_its_staged_shapes(tmp_path, name):
    """The deliver event carries a stain save's provenance without its outlines' points, so that one event stays small
    however many areas were removed (engine/runs.py _without_outlines); the object's record of the layer is the one the
    helper wrote on the staged shapes, as the sample's zarr holds it, points included: under sopa_companion, or the
    previous version's proseg_live."""
    import zarr

    shapes, _ = result_elements()
    key = "cellpose_boundaries_1"
    shapes = shapes.copy()
    shapes.index = ["cellpose-1", "cellpose-2"]
    path = tmp_path / "stain.zarr"
    sd.SpatialData(shapes={key: shapes}).write(path, consolidate_metadata=False)
    ring = [[float(i), float(i % 5)] for i in range(500)]
    whole = {"method": "cellpose", "model": "cpsam", "region": {"kind": "polygon", "points": ring, "bbox": [0, 0, 499, 4]},
             "filters": {"areas": [{"id": i, "points": ring} for i in range(200)], "size": None}, "removed_cells": 3}
    zarr.open_group(str(path / "shapes" / key), mode="r+").attrs[name] = whole
    slim = {"method": "cellpose", "model": "cpsam", "region": {"kind": "polygon", "bbox": [0, 0, 499, 4]},
            "filters": {"areas": [{"id": i} for i in range(200)], "size": None}}
    event = build_event("deliver", id="stain-2", run="run-s", kind="stain", staged=str(path), shapes_key=key,
                        table_key=None, set_default=False, overwrite=False, replaced=[], written_to=None, cells=2,
                        genes=None, provenance=slim)
    assert len(encode_line(event)) < 8192
    sdata = sd.SpatialData()
    manager = DeliveryManager(sdata, lambda message: None)
    receive_in_thread(manager, event)
    result, = manager.apply_ready()
    assert result.state == "added"
    assert sdata.attrs["sopa_companion"]["layers"][key] == {**whole, "kind": "stain", "run": "run-s", "session": "api"}
    manager.close()


def test_unticked_default_keeps_the_attrs_object(staged):
    sdata = sd.SpatialData(attrs={"boundaries_shapes": "existing"})
    attrs = sdata.attrs
    manager = DeliveryManager(sdata, lambda message: None)
    receive_in_thread(manager, {**staged[0], "set_default": False})
    assert manager.apply_ready()[0].state == "added"
    assert sdata.attrs is attrs
    manager.close()


@pytest.mark.parametrize("changes", [{"shapes_key": "missing"}, {"table_key": "missing"}, {"cells": 3},
                                     {"genes": 3}, {"shapes_key": "../escape"}, {"table_key": None},
                                     {"table_key": "PROSEG_boundaries"}])
def test_bad_staging_is_reported_before_any_result_is_added(staged, changes):
    sdata = sd.SpatialData()
    controls = []
    manager = DeliveryManager(sdata, controls.append)
    receive_in_thread(manager, {**staged[0], **changes})
    assert [message["op"] for message in controls] == ["receive_failed"]
    result, = manager.apply_ready()
    assert result.state == "failed" and result.reason == "failed" and result.message
    assert not sdata.shapes and not sdata.tables and manager.pending == ()
    assert [message["op"] for message in controls] == ["receive_failed"]
    manager.close()


def test_duplicate_delivery_does_not_load_deleted_staging_or_apply_twice(staged):
    sdata, controls = sd.SpatialData(), []
    manager = DeliveryManager(sdata, controls.append)
    receive_in_thread(manager, staged[0])
    shutil.rmtree(staged[0]["staged"])
    receive_in_thread(manager, staged[0])
    assert len(manager.pending) == 1
    manager.apply_ready()
    receive_in_thread(manager, staged[0])
    assert len(manager.outcomes) == 1 and manager.apply_ready() == ()
    assert [message["op"] for message in controls] == ["received", "received", "applied", "received", "applied"]
    manager.close()


def test_broken_control_pipe_does_not_undo_a_successful_apply(staged, caplog):
    def broken(message):
        raise BrokenPipeError("engine gone")

    sdata = sd.SpatialData()
    manager = DeliveryManager(sdata, broken)
    receive_in_thread(manager, staged[0])
    assert manager.apply_ready()[0].state == "added" and "proseg_table" in sdata
    assert "Could not send an API delivery acknowledgement" in caplog.text
    manager.close()


def test_plain_script_blocking_launch_loop_applies_without_result_or_wait(staged, caplog):
    sdata, controls, events = sd.SpatialData(), [], []
    manager = DeliveryManager(sdata, controls.append, on_event=events.append)
    assert not manager.mainthread.automatic
    read_fd, write_fd = os.pipe()
    try:
        manager.start(read_fd)
        payload = b'{"ev":"future_event"}\nnot json\n' + encode_line(staged[0]).encode()
        os.write(write_fd, payload[:25])
        os.write(write_fd, payload[25:])
        os.close(write_fd)
        write_fd = None
        assert manager.mainthread.run_until(manager.eof.is_set, timeout=15)
        assert "proseg_table" in sdata and len(manager.outcomes) == 1
        assert [event["ev"] for event in events] == ["deliver"]
        assert "Ignoring unknown API" in caplog.text and "Could not read an API event" in caplog.text
    finally:
        manager.close()
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
    assert not manager.thread.is_alive()


def test_close_stops_a_reader_with_an_unfinished_line():
    manager = DeliveryManager(sd.SpatialData(), lambda message: None)
    read_fd, write_fd = os.pipe()
    try:
        manager.start(read_fd)
        os.write(write_fd, b'{"ev":')
        manager.close()
        assert not manager.thread.is_alive()
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_running_asyncio_loop_applies_automatically_on_its_main_thread(staged):
    async def scenario():
        sdata = sd.SpatialData()
        applied = asyncio.Event()
        threads = []

        def on_applied(result):
            threads.append(threading.get_ident())
            applied.set()

        manager = DeliveryManager(sdata, lambda message: None, on_applied=on_applied)
        assert manager.mainthread.automatic
        await asyncio.to_thread(manager.receive, staged[0])
        await asyncio.wait_for(applied.wait(), timeout=15)
        assert "proseg_table" in sdata and threads == [threading.main_thread().ident]
        manager.close()

    asyncio.run(scenario())


def test_terminal_ipython_drains_and_unregisters_its_cell_hook():
    handlers, called = {}, []
    events = SimpleNamespace(register=lambda name, fn: handlers.update({name: fn}),
                             unregister=lambda name, fn: handlers.pop(name))
    queue = MainThread(shell=SimpleNamespace(events=events))
    thread = threading.Thread(target=lambda: queue.submit(lambda: called.append(threading.get_ident())))
    thread.start()
    thread.join()
    assert not called and queue.automatic
    handlers["post_run_cell"](object())
    assert called == [threading.main_thread().ident]
    queue.close()
    assert handlers == {}


def test_worker_cannot_apply_a_result(staged):
    sdata = sd.SpatialData()
    manager = DeliveryManager(sdata, lambda message: None)
    receive_in_thread(manager, staged[0])
    errors = []

    def apply():
        try:
            manager.apply_ready()
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=apply)
    thread.start()
    thread.join()
    assert errors == ["Results must be added on the main thread"] and not sdata.shapes
    assert manager.apply_ready()[0].state == "added"
    manager.close()


def test_scheduled_interrupt_is_raised_after_the_batch_finishes():
    queue, steps = MainThread(), []
    previous = signal.getsignal(signal.SIGINT)

    def callback():
        steps.append("first")
        os.kill(os.getpid(), signal.SIGINT)
        steps.append("last")

    queue.submit(callback)
    with pytest.raises(KeyboardInterrupt):
        queue._run_scheduled()
    assert steps == ["first", "last"] and signal.getsignal(signal.SIGINT) == previous
    queue.close()


@pytest.mark.parametrize("shapes_key,table_key", [("proseg_boundaries_2", "proseg_table_2"),
                                                ("proseg_boundaries", "table")])
def test_event_names_are_used_without_renaming_the_table_target(tmp_path, shapes_key, table_key):
    shapes, table = result_elements(shapes_key)
    path = tmp_path / "named.zarr"
    sd.SpatialData(shapes={shapes_key: shapes}, tables={table_key: table}).write(path, consolidate_metadata=False)
    event = build_event("deliver", id="named", run="run-1", kind="proseg", staged=str(path),
                        shapes_key=shapes_key, table_key=table_key, set_default=True, overwrite=False, replaced=[],
                        written_to=None, cells=2, genes=2, provenance={})
    sdata = sd.SpatialData()
    manager = DeliveryManager(sdata, lambda message: None)
    receive_in_thread(manager, event)
    result, = manager.apply_ready()
    assert result.keys == (shapes_key, table_key)
    assert sdata[table_key].obs["region"].tolist() == [shapes_key, shapes_key]
    assert sdata.attrs["boundaries_shapes"] == shapes_key
    manager.close()


def test_waiting_state_clears_when_the_main_thread_applies(staged, monkeypatch):
    import sopa_companion.api.deliver as module
    controls = []

    def send(message):
        # The handle may read status while writing a control; no state lock may be held across its callback.
        assert isinstance(manager.pending, tuple)
        controls.append(message)

    manager = DeliveryManager(sd.SpatialData(), send)
    receive_in_thread(manager, staged[0])
    queued_at = manager._queued_at[staged[0]["id"]]
    with monkeypatch.context() as patch:
        patch.setattr(module.time, "monotonic", lambda: queued_at + 1.99)
        manager.poll()
        assert [message["op"] for message in controls] == ["received"]
        patch.setattr(module.time, "monotonic", lambda: queued_at + 2)
        manager.poll()
        manager.poll()
        assert controls[-1] == {"op": "notebook", "busy": True}
        assert manager.apply_ready()[0].state == "added"
    assert [message for message in controls if message["op"] == "notebook"] == [
        {"op": "notebook", "busy": True}, {"op": "notebook", "busy": False}]
    manager.close()


class HandleStandIn:
    """The handle's side of the delivery boundary (api/handle.py): its control queue and its record of names added."""

    def __init__(self):
        self.controls, self.added, self.owner = [], [], threading.get_ident()

    def _send(self, message):
        self.controls.append(validate_control(message))

    def _record_added(self, keys):
        assert threading.get_ident() == self.owner, "names are recorded on the thread that opened the app"
        self.added.extend(keys)


def test_launch_attaches_the_handles_delivery_which_adds_results_and_records_their_names(staged, monkeypatch):
    """The boundary launch() crosses (launch._attach_delivery -> deliver.attach): a deliver event handed over on the io
    thread is acknowledged, applied on the main thread, and its names recorded on the handle; a result that waits for
    the main thread tells the engine the notebook is busy, and not busy once applied."""
    import sopa_companion.api.deliver as module
    import importlib
    L = importlib.import_module("sopa_companion.api.launch")    # the module (the package exports the function)
    monkeypatch.setattr(module, "POLL_S", 0.01)
    event, shapes, table = staged
    app, sdata = HandleStandIn(), sd.SpatialData()
    attached = L._attach_delivery(app, sdata)
    assert isinstance(attached, module.Attached)
    receive_in_thread(attached, event)
    assert [m["op"] for m in app.controls] == ["received"] and not sdata.shapes
    queued_at = attached.manager._queued_at[event["id"]]
    monkeypatch.setattr(module.time, "monotonic", lambda: queued_at + 2)
    deadline = time.monotonic_ns() + 5_000_000_000
    while {"op": "notebook", "busy": True} not in app.controls and time.monotonic_ns() < deadline:
        threading.Event().wait(0.01)
    assert {"op": "notebook", "busy": True} in app.controls
    result, = attached.apply_pending()
    assert result.state == "added" and app.added == ["proseg_boundaries", "proseg_table"]
    assert sdata["proseg_boundaries"].equals(shapes)
    assert app.controls[-2:] == [{"op": "notebook", "busy": False},
                                 {"op": "applied", "id": "delivery-1", "state": "added",
                                  "keys": ["proseg_boundaries", "proseg_table"], "clash": [], "message": None}]
    attached.close()
    assert not attached._poller.is_alive()


def test_a_status_callback_can_read_results_without_reentering_the_batch():
    queue, steps = MainThread(), []
    queue.submit(lambda: steps.append(queue.drain()))
    queue.submit(lambda: steps.append("second"))
    assert queue.drain() == 2 and steps == [0, "second"]
    assert not queue.run_until(lambda: False, timeout=0.01)
    queue.close()
    with pytest.raises(RuntimeError, match="queue is closed"):
        queue.submit(lambda: None)


def test_importing_delivery_modules_does_not_load_science_or_server_packages():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'spatialdata', 'zarr', 'fastapi', 'uvicorn', 'httpx', 'psutil'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import sopa_companion.api.deliver, sopa_companion.api.mainthread
'''
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15,
                               env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])})
    assert completed.returncode == 0, completed.stdout + completed.stderr
