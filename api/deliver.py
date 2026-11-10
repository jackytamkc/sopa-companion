"""Reading the engine's deliver events, and adding their results to the in-memory SpatialData all or nothing.

The helper has already committed a backed target using write_element before emitting deliver; written_to records
that commit. This module must not write it again, or write the caller's whole attrs mapping to disk.
The caller builds DeliveryManager on the main thread, passes its thread-safe control send(dict) function, and starts the
reader with the events pipe. on_event(dict) and on_eof() run on the reader; on_applied(ApplyResult) runs on the main
thread. With no event loop, launch() blocks in manager.mainthread.run_until(manager.eof.is_set). The handle
(api/handle.py), which reads the events pipe itself, holds attach(app, sdata): Attached, below.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import selectors
import threading
import time

from ..names import PROVENANCE_KEYS
from .mainthread import MainThread, require_main_thread
from .protocol import build_control, decode_event_line, validate_event


_LOG = logging.getLogger(__name__)
_KINDS = ("images", "labels", "points", "shapes", "tables")
_MISSING = object()


@dataclass(frozen=True)
class Delivery:
    event: dict
    shapes: object
    table: object | None
    provenance: dict | None = None      # a stain result's, as recorded on its staged shapes; None: the event's


@dataclass(frozen=True)
class ApplyResult:
    event: dict
    state: str
    keys: tuple[str, ...] = ()
    clash: tuple[str, ...] = ()
    message: str | None = None

    @property
    def reason(self):
        return None if self.state == "added" else "clash" if self.clash else "failed"

    def control(self):
        return build_control("applied", id=self.event["id"], state=self.state, keys=list(self.keys),
                             clash=list(self.clash), message=self.message)


def _delivery_event(event):
    event = validate_event(event)
    if event["ev"] != "deliver":
        raise ValueError("Expected a deliver event")
    if (event["kind"] == "proseg") != (event["table_key"] is not None):
        raise ValueError("Proseg results need a table; stain results contain shapes only")
    from spatialdata._core.validation import check_valid_name
    keys = [event["shapes_key"]]
    if event["table_key"] is not None:
        keys.append(event["table_key"])
    for key in keys:
        check_valid_name(key)
    if len({key.lower() for key in keys}) != len(keys):
        raise ValueError("The shapes and table need different names, ignoring case")
    return event


def read_delivery(event):
    """Called by the io thread. Only eager shapes/tables survive the received acknowledgement."""
    event = _delivery_event(event)
    import spatialdata as sd
    staged = sd.read_zarr(event["staged"], selection=("shapes", "tables"))
    shapes = staged.shapes[event["shapes_key"]]
    table = staged.tables[event["table_key"]] if event["table_key"] is not None else None
    if len(shapes) != event["cells"]:
        raise ValueError("The staged shapes do not contain the expected number of cells")
    if table is not None:
        if table.isbacked:
            table = table.to_memory()
        if table.n_obs != event["cells"] or (event["genes"] is not None and table.n_vars != event["genes"]):
            raise ValueError("The staged table does not contain the expected cells and genes")
        region = table.uns.get("spatialdata_attrs", {}).get("region")
        regions = [region] if isinstance(region, str) else list(region) if region is not None else []
        if regions != [event["shapes_key"]]:
            raise ValueError("The staged table does not refer to the delivered shapes")
        # The helper owns table provenance too. Preserve it verbatim, matching the already committed target.
    return Delivery(event, shapes, table, _staged_provenance(event) if event["kind"] == "stain" else None)


def _staged_provenance(event):
    """A stain result's provenance as the helper recorded it on the staged shapes (zarr_bridge/helper.py _stage), as
    the sample's zarr holds it: whole, its outlines' points included, which the deliver event leaves out to stay small
    (engine/runs.py _without_outlines). None when the staged shapes hold none."""
    import zarr
    try:
        group = zarr.open_group(str(Path(event["staged"]) / "shapes" / event["shapes_key"]), mode="r")
        attrs = dict(group.attrs)
    except Exception:          # the record is not readable: the event's own, without the points, is kept instead
        _LOG.warning("Could not read the staged result's provenance", exc_info=True)
        return None
    # Under the first provenance key there is, as every reader of the app's records takes it (names.PROVENANCE_KEYS).
    value = next((attrs[key] for key in PROVENANCE_KEYS if isinstance(attrs.get(key), Mapping)), None)
    return deepcopy(dict(value)) if value is not None else None


def snapshot_elements(sdata):
    """Remember identities on the main thread, so Replace cannot clobber a newer notebook element."""
    require_main_thread()
    return {name: (kind, element) for kind in _KINDS
            for name, element in getattr(sdata, kind).data.copy().items()}


def _clashes(sdata, event, expected):
    current = snapshot_elements(sdata)
    clashes = []
    for kind, key in (("shapes", event["shapes_key"]), ("tables", event["table_key"])):
        if key is None:
            continue
        for name, (existing_kind, element) in current.items():
            if name.lower() != key.lower():
                continue
            previous = expected.get(name)
            allowed = (event["overwrite"] and name == key and existing_kind == kind and previous is not None
                       and previous[0] == kind and previous[1] is element)
            if not allowed:
                clashes.append(name)
    return tuple(dict.fromkeys(clashes))


def _attrs(sdata, event, provenance=None):
    attrs = dict(sdata.attrs)
    if event["set_default"]:
        attrs["boundaries_shapes"] = event["shapes_key"]
    if event["kind"] == "stain":
        companion = dict(attrs.get("sopa_companion", {}))
        layers = dict(companion.get("layers", {}))
        recorded = event["provenance"] if provenance is None else provenance
        layers[event["shapes_key"]] = {**deepcopy(recorded), "kind": "stain", "run": event["run"], "session": "api"}
        companion["layers"] = layers
        attrs["sopa_companion"] = companion
    return attrs


def _check_annotations(sdata, event):
    key = event["shapes_key"]
    if key not in sdata.shapes:
        return
    for name, table in sdata.tables.items():
        if name == event["table_key"]:
            continue
        region = table.uns.get("spatialdata_attrs", {}).get("region")
        regions = [region] if isinstance(region, str) else list(region) if region is not None else []
        if key in regions:
            raise ValueError(f"Table '{name}' uses '{key}': choose another shapes name")


def apply_delivery(sdata, delivery, *, expected=None):
    """No awaits and no disk writes. A failure, including an interrupt, restores elements and attrs together.

    expected is a snapshot_elements() result taken before the user could make a newer element. Without a snapshot,
    all existing names are refused, even with overwrite=True. DeliveryManager maintains this snapshot itself.
    """
    require_main_thread()
    event = delivery.event
    try:
        event = _delivery_event(event)
        clashes = _clashes(sdata, event, expected or {})
        if clashes:
            return ApplyResult(event, "failed", clash=clashes,
                               message=f"Not added: your SpatialData already has {', '.join(clashes)}")
        _check_annotations(sdata, event)
        attrs = _attrs(sdata, event, delivery.provenance)
        elements = [("shapes", event["shapes_key"], delivery.shapes)]
        if event["table_key"] is not None:
            elements.append(("tables", event["table_key"], delivery.table))
        original = [(getattr(sdata, kind), key, getattr(sdata, kind).data.get(key, _MISSING))
                    for kind, key, _ in elements]
        old_attrs = sdata.attrs
        shared_keys = sdata._shared_keys.copy()
        try:
            for _, key, element in elements:
                sdata[key] = element
            if event["set_default"] or event["kind"] == "stain":
                sdata.attrs = attrs
            return ApplyResult(event, "added", keys=tuple(key for _, key, _ in elements))
        except BaseException:
            # Restore the underlying mappings: re-entering a failing validator/setter could fail a second time.
            for container, key, previous in original:
                if previous is _MISSING:
                    container.data.pop(key, None)
                else:
                    container.data[key] = previous
            sdata._shared_keys.clear()
            sdata._shared_keys.update(shared_keys)
            sdata._attrs = old_attrs
            raise
    except BaseException as exc:
        message = " ".join(str(exc).splitlines()) or type(exc).__name__
        return ApplyResult(event, "failed", message=f"Could not add the result: {message}")


class DeliveryManager:
    def __init__(self, sdata, send, *, mainthread=None, on_event=None, on_applied=None, on_eof=None):
        require_main_thread()
        self.sdata = sdata
        self.mainthread = mainthread if mainthread is not None else MainThread()
        self._send_control = send
        self._on_event = on_event
        self._on_applied = on_applied
        self._on_eof = on_eof
        self._expected = snapshot_elements(sdata)
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._busy_lock = threading.Lock()
        self._responses = {}
        self._pending = {}
        self._queued_at = {}
        self._notebook_busy = False
        self._outcomes = []
        self._stop = threading.Event()
        self.eof = threading.Event()
        self.thread = None

    @property
    def pending(self):
        with self._lock:
            return tuple(event["run"] for event in self._pending.values())

    @property
    def outcomes(self):
        with self._lock:
            return tuple(self._outcomes)

    def _send(self, message):
        try:
            with self._send_lock:
                self._send_control(message)
        except Exception:
            _LOG.warning("Could not send an API delivery acknowledgement", exc_info=True)

    def _respond(self, id, message):
        with self._lock:
            self._responses[id].append(message)
        self._send(message)

    def receive(self, event):
        """Read and queue a deliver event on the io thread; a duplicate never reads deleted staging again."""
        event = validate_event(event)
        if event["ev"] != "deliver":
            raise ValueError("Expected a deliver event")
        id = event["id"]
        with self._lock:
            duplicate = id in self._responses
            responses = tuple(self._responses.get(id, ()))
            if not duplicate:
                self._responses[id] = []
                self._pending[id] = event
        if duplicate:
            for response in responses:
                self._send(response)
            return
        try:
            delivery = read_delivery(event)
        except Exception as exc:
            message = " ".join(str(exc).splitlines()) or type(exc).__name__
            self._respond(id, build_control("receive_failed", id=id, message=message))
            outcome = ApplyResult(event, "failed", message=f"Could not read the result: {message}")
            self.mainthread.submit(lambda: self._record(outcome, acknowledge=False))
            return
        self._respond(id, build_control("received", id=id))
        with self._lock:
            self._queued_at[id] = time.monotonic()
        self.mainthread.submit(lambda: self._apply(delivery))

    def poll(self):
        """Tell the engine when a loaded result has waited two seconds for the main thread."""
        with self._busy_lock:
            with self._lock:
                busy = any(time.monotonic() - queued >= 2 for queued in self._queued_at.values())
                changed = busy != self._notebook_busy
                self._notebook_busy = busy
            if changed:
                self._send(build_control("notebook", busy=busy))

    def _apply(self, delivery):
        result = apply_delivery(self.sdata, delivery, expected=self._expected)
        if result.state == "added":
            for key in result.keys:
                kind = "shapes" if key == result.event["shapes_key"] else "tables"
                self._expected[key] = (kind, self.sdata[key])
        self._record(result)

    def _record(self, result, *, acknowledge=True):
        require_main_thread()
        with self._lock:
            self._pending.pop(result.event["id"], None)
            self._queued_at.pop(result.event["id"], None)
            self._outcomes.append(result)
        self.poll()
        if acknowledge:
            self._respond(result.event["id"], result.control())
        if self._on_applied is not None:
            try:
                self._on_applied(result)
            except Exception:
                _LOG.exception("Could not update the API result display")

    def apply_ready(self):
        require_main_thread()
        before = len(self.outcomes)
        self.mainthread.drain()
        return self.outcomes[before:]

    def _line(self, line):
        try:
            event = decode_event_line(line)
            if event is None:
                return
            if self._on_event is not None:
                try:
                    self._on_event(event)
                except Exception:
                    _LOG.exception("Could not update the API session display")
            if event["ev"] == "deliver":
                self.receive(event)
        except Exception:
            _LOG.exception("Could not read an API event")

    def read_events(self, events):
        """Read an fd (or a file with fileno); the caller owns closing it after the reader stops."""
        fd = events if isinstance(events, int) else events.fileno()
        pending = b""
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(fd, selectors.EVENT_READ)
                while not self._stop.is_set():
                    self.poll()
                    if not selector.select(0.1):
                        continue
                    data = os.read(fd, 65536)
                    if not data:
                        if pending:
                            self._line(pending)
                        break
                    lines = (pending + data).split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        self._line(line)
        except OSError:
            if not self._stop.is_set():
                _LOG.exception("The API events pipe closed unexpectedly")
        finally:
            self.eof.set()
            if self._on_eof is not None:
                self._on_eof()

    def start(self, events):
        require_main_thread()
        if self.thread is not None:
            raise RuntimeError("The API events reader has already started")
        self.thread = threading.Thread(target=self.read_events, args=(events,), name="companion-io", daemon=True)
        self.thread.start()
        return self.thread

    def close(self):
        require_main_thread()
        self._stop.set()
        if self.thread is not None:
            self.thread.join()
        self.mainthread.close()


# ---- the handle's side (launch._attach_delivery) ---------------------------------------------------------------------
POLL_S = 0.25          # how often a result waiting for the main thread is looked at (the engine hears "busy" at 2 s)


class Attached:
    """What the handle holds (launch._attach_delivery: receive, apply_pending, close). The handle reads the events pipe
    itself and hands each deliver event to receive() on its io thread: the DeliveryManager reads the staged result
    there, acknowledges it through the handle's control lines (app._send) and queues the apply on the main thread
    (MainThread: the running loop once the kernel is idle, after each cell in a terminal IPython); apply_pending()
    drains it from a plain script's pump and at the end. Each result added is recorded on the handle
    (app._record_added, on the main thread that opened it). A thread of its own tells the engine when a result waits
    for the main thread (DeliveryManager.poll: "notebook busy" after 2 s), as the manager's own reader would."""

    def __init__(self, app, sdata):
        self.app = app
        self.manager = DeliveryManager(sdata, app._send, on_applied=self._applied)
        self._closed = threading.Event()
        self._poller = threading.Thread(target=self._poll, name="companion-busy", daemon=True)
        self._poller.start()

    def _applied(self, result):
        if result.state == "added" and result.keys:
            self.app._record_added(list(result.keys))

    def _poll(self):
        while not self._closed.wait(POLL_S):
            self.manager.poll()

    def receive(self, event):
        self.manager.receive(event)

    def apply_pending(self):
        return self.manager.apply_ready()

    def close(self):
        self._closed.set()
        self._poller.join(timeout=5)
        self.manager.close()


def attach(app, sdata):
    """The handle's delivery (launch._attach_delivery), made on the main thread that opens the app."""
    return Attached(app, sdata)
