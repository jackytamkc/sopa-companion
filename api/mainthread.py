"""Main-thread hand-off for API deliveries; importing this module needs only the standard library.

Construct MainThread in launch(). An asyncio kernel receives callbacks automatically, without a result()/wait()
call. With no running loop, launch() must use run_until() as its blocking loop. Terminal IPython also drains after
each cell. No worker thread ever runs a callback itself.
"""
from __future__ import annotations

import asyncio
import builtins
from collections import deque
from contextlib import contextmanager
import signal
import threading
import time


def require_main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Results must be added on the main thread")


@contextmanager
def _defer_interrupt():
    """Finish a scheduled apply before delivering a user's interrupt to the kernel."""
    previous = signal.getsignal(signal.SIGINT)
    pending = []

    def remember(signum, frame):
        pending.append((signum, frame))

    signal.signal(signal.SIGINT, remember)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if pending:
            if callable(previous):
                previous(*pending[0])
            elif previous == signal.SIG_DFL:
                signal.default_int_handler(*pending[0])


class MainThread:
    def __init__(self, *, loop=None, shell=None):
        require_main_thread()
        if shell is None:
            get_ipython = getattr(builtins, "get_ipython", None)
            shell = get_ipython() if get_ipython is not None else None
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                kernel = getattr(shell, "kernel", None)
                loop = getattr(getattr(kernel, "io_loop", None), "asyncio_loop", None)
        self._loop = loop if loop is not None and loop.is_running() and not loop.is_closed() else None
        self._callbacks = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._scheduled = False
        self._draining = False
        self._closed = False
        self._events = getattr(shell, "events", None) if self._loop is None else None
        if self._events is not None:
            self._events.register("post_run_cell", self._after_cell)

    @property
    def automatic(self):
        """False means launch() must keep the main thread in run_until()."""
        return self._loop is not None or self._events is not None

    @property
    def pending(self):
        with self._lock:
            return len(self._callbacks)

    def submit(self, callback):
        """Queue from any thread, scheduling once on the loop captured on the main thread."""
        with self._lock:
            if self._closed:
                raise RuntimeError("The main-thread queue is closed")
            self._callbacks.append(callback)
            self._wake.set()
            if self._loop is not None and not self._scheduled:
                self._scheduled = True
                try:
                    self._loop.call_soon_threadsafe(self._run_scheduled)
                except RuntimeError:       # a loop closing during teardown: blocking calls can still drain
                    self._loop = None
                    self._scheduled = False

    def _after_cell(self, *args, **kwargs):
        self._run_scheduled()

    def _run_scheduled(self):
        require_main_thread()
        with _defer_interrupt():
            self.drain()

    def drain(self):
        """Apply a batch synchronously, with no awaits. Also used by the handle's explicit calls."""
        require_main_thread()
        if self._draining:
            return 0
        self._draining = True
        count = 0
        try:
            with self._lock:
                self._scheduled = False
                # A finite batch lets a blocking launch check for shutdown even if more deliveries keep arriving.
                count_ready = len(self._callbacks)
            for _ in range(count_ready):
                with self._lock:
                    callback = self._callbacks.popleft()
                callback()
                count += 1
        finally:
            self._draining = False
            with self._lock:
                if not self._callbacks:
                    self._wake.clear()
        return count

    def run_until(self, done, timeout=None):
        """Plain-script launch loop. Return False at timeout, keeping pending callbacks available."""
        require_main_thread()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self.drain()
            if done():
                # EOF can be set just after the first drain queued its last delivery.
                self.drain()
                return True
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                return False
            self._wake.wait(0.05 if left is None else min(0.05, left))

    def close(self):
        """After the io thread has stopped, apply its last deliveries and detach the IPython hook."""
        require_main_thread()
        self.drain()
        with self._lock:
            self._closed = True
        if self._events is not None:
            self._events.unregister("post_run_cell", self._after_cell)
            self._events = None
