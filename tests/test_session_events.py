"""The API session's events channel (engine/session.py): every event line reaches the notebook whole, however large,
in order, while the notebook reads at its own pace; a channel the engine gives up reads as ended in the notebook. Real
pipes, a reader that drains them as the handle does (api/handle.py: every 50 ms), no engine."""
from __future__ import annotations

import os
import stat
import threading
import time

import pytest

from sopa_companion.api import protocol
from sopa_companion.engine import session as module
from sopa_companion.engine.session import Session


@pytest.fixture
def api_config(tmp_path):
    return protocol.build_session_config(id="1f3a9c2e", token="t" * 43, data_dir=str(tmp_path / "sopa-companion-1f3a9c2e"),
                                         state_dir=str(tmp_path / "1f3a9c2e"), python="/fake/python", kernel_pid=os.getpid())


def drawn(areas=200, points=40, id="d1"):
    """A stain deliver event as large as many removed areas made it before the event left their points out."""
    ring = [[round(1000 + i * 0.37, 3), round(2000 + (i * 7919) % 997 * 0.11, 3)] for i in range(points)]
    return protocol.build_event("deliver", id=id, run="r1", kind="stain", staged="/fake/staged.zarr",
                                shapes_key="cellpose_boundaries_1", table_key=None, set_default=False, overwrite=False,
                                replaced=[], written_to=None, cells=40, genes=None,
                                provenance={"filters": {"areas": [{"id": i, "points": ring} for i in range(areas)]}})


class Reader(threading.Thread):
    """The notebook's side: reads the events pipe every `every` seconds, as much as there is, until it ends or the test
    stops it (stop(), before the test closes the pipe)."""

    def __init__(self, fd, every=0.05):
        super().__init__(daemon=True)
        self.fd, self.every, self.data, self.done = fd, every, bytearray(), threading.Event()
        self.start()

    def run(self):
        while not self.done.wait(self.every):
            try:
                chunk = os.read(self.fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                return
            self.data.extend(chunk)

    def stop(self):
        self.done.set()
        self.join(5)

    def lines(self):
        """The whole lines read so far, as events."""
        return [protocol.decode_event_line(line) for line in bytes(self.data).split(b"\n")[:-1]]


def test_an_event_larger_than_the_pipe_arrives_whole_and_the_channel_goes_on(api_config):
    """A stain save with many removed areas made an event of about 80 KB, more than the pipe holds (64 KiB): it was cut
    off at the pipe's size and the channel closed, so that result and every later event (the end of the session
    included) never reached the notebook. The line now waits for the notebook to read, and arrives whole."""
    reader_fd, writer_fd = os.pipe()
    os.set_blocking(reader_fd, False)
    session = Session(api_config, events_fd=writer_fd)
    reader = Reader(reader_fd)
    try:
        big = drawn()
        assert len(protocol.encode_line(big)) > 80_000
        session.deliver(big)
        session.deliver(protocol.build_event("session", state="ready", message=None))
        assert session._events_fd is not None and "d1" not in session._send_failed
        deadline = time.monotonic() + 10
        while len(reader.lines()) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert reader.lines() == [big, {"ev": "session", "state": "ready", "message": None}]
    finally:
        reader.stop()
        session.stop()
        os.close(writer_fd)
        os.close(reader_fd)


def test_events_from_several_threads_are_never_interleaved(api_config):
    reader_fd, writer_fd = os.pipe()
    os.set_blocking(reader_fd, False)
    session = Session(api_config, events_fd=writer_fd)
    reader = Reader(reader_fd, every=0.01)
    try:
        threads = [threading.Thread(target=session.deliver, args=(drawn(areas=60, id=f"d{i}"),)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        deadline = time.monotonic() + 10
        while len(reader.lines()) < 6 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert sorted(event["id"] for event in reader.lines()) == [f"d{i}" for i in range(6)]
    finally:
        reader.stop()
        session.stop()
        os.close(writer_fd)
        os.close(reader_fd)


def test_an_event_nobody_makes_room_for_is_not_sent_and_the_channel_stays(api_config, monkeypatch, caplog):
    """A full pipe that the notebook does not read: the event waits EVENTS_WRITE_S at most, is not sent (its delivery
    counts as not received), and nothing of it is in the pipe, so the channel stays usable for the next one."""
    monkeypatch.setattr(module, "EVENTS_WRITE_S", 0.2)
    reader_fd, writer_fd = os.pipe()
    os.set_blocking(reader_fd, False)
    session = Session(api_config, events_fd=writer_fd)
    try:
        filler = 0
        while True:
            try:
                filler += os.write(session._events_fd, b"x" * 4096)
            except BlockingIOError:
                break
        start = time.monotonic()
        session.deliver(drawn(areas=1, id="late"))
        assert 0.15 <= time.monotonic() - start < 2
        assert not session.received("late", 0.1) and "made no room" in caplog.text
        assert session._events_fd is not None
        drained = b""
        while len(drained) < filler:
            drained += os.read(reader_fd, 65536)
        assert drained == b"x" * filler                     # nothing of the event was written into the pipe
        event = protocol.build_event("session", state="ready", message=None)
        session.deliver(event)
        assert protocol.decode_event_line(os.read(reader_fd, 65536)) == event
    finally:
        session.stop()
        os.close(writer_fd)
        os.close(reader_fd)


def test_a_line_cut_off_part_way_ends_the_channel_and_the_notebook_reads_its_end(api_config, monkeypatch, caplog):
    """The notebook stops reading part-way through a line: after EVENTS_WRITE_S the channel ends rather than append
    another message to the unfinished one, and the notebook reads its end (EOF): the descriptor the handle passed
    (SOPA_COMPANION_EVENTS_FD) is let go of as the Session starts, so no write end of the pipe is left open. It used to
    stay open: the notebook then waited on a channel nobody wrote to any more."""
    monkeypatch.setattr(module, "EVENTS_WRITE_S", 0.3)
    reader_fd, writer_fd = os.pipe()
    os.set_blocking(reader_fd, False)
    monkeypatch.setenv(protocol.ENV_EVENTS_FD, str(writer_fd))
    session = Session(api_config)
    try:
        assert protocol.ENV_EVENTS_FD not in os.environ
        assert stat.S_ISCHR(os.fstat(writer_fd).st_mode)       # the passed number now leads to /dev/null
        session.deliver(drawn(areas=400))                     # ~170 KB: the pipe takes 64 KiB, nobody reads more
        assert session._events_fd is None and not session.received("d1", 0.1)
        assert "stopped reading part-way" in caplog.text
        data = b""
        while chunk := os.read(reader_fd, 65536):              # ...then the end of the channel, never a wait
            data += chunk
        assert data and not data.endswith(b"\n")
        session.deliver(protocol.build_event("session", state="ready", message=None))
        assert "events pipe is closed" in caplog.text
    finally:
        session.stop()
        os.close(writer_fd)
        os.close(reader_fd)


def test_the_passed_descriptor_is_let_go_so_the_notebook_reads_the_end_when_the_session_stops(api_config, monkeypatch):
    reader_fd, writer_fd = os.pipe()
    os.set_blocking(reader_fd, False)
    monkeypatch.setenv(protocol.ENV_EVENTS_FD, str(writer_fd))
    session = Session(api_config)
    event = protocol.build_event("ready", port=49157, pid=os.getpid())
    try:
        session.deliver(event)
        assert protocol.decode_event_line(os.read(reader_fd, 4096)) == event
        assert not os.get_inheritable(writer_fd)              # nor does a helper get the /dev/null left in its place
    finally:
        session.stop()
    try:
        assert os.read(reader_fd, 4096) == b""                 # the end: no copy of the pipe's write end is left
    finally:
        os.close(writer_fd)
        os.close(reader_fd)


def test_an_event_longer_than_the_notebook_reads_is_not_sent_and_the_channel_stays(api_config, caplog):
    """The handle refuses a line over 1 MiB and ends the session (protocol.MAX_EVENT_BYTES): such an event is never
    sent, so the session goes on."""
    reader_fd, writer_fd = os.pipe()
    os.set_blocking(reader_fd, False)
    session = Session(api_config, events_fd=writer_fd)
    reader = Reader(reader_fd, every=0.01)
    try:
        huge = protocol.validate_event({**drawn(areas=1, id="huge"), "provenance": {"large": "x" * protocol.MAX_EVENT_BYTES}})
        session.deliver(huge)
        assert not session.received("huge", 0.1) and "more than the notebook reads" in caplog.text
        event = protocol.build_event("session", state="ready", message=None)
        session.deliver(event)
        deadline = time.monotonic() + 5
        while not reader.lines() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert reader.lines() == [event]
    finally:
        reader.stop()
        session.stop()
        os.close(writer_fd)
        os.close(reader_fd)
