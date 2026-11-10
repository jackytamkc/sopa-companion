"""Who is here: the pages signed in to this engine, how the last one went, and the watcher that acts on it. The engine
never asks which browser shows a page, so the rule is the same in every browser, the app's own Chrome window included.

A page counts while one of its sockets is open (server/app.py: a socket that signed in as a page, under the id its
address names). A page that goes for good says goodbye on its way: its socket's {op: "bye"}, its log's last upload
(POST /page-log with pagehide) or its answer to closing (POST /closing with bye); any one of them is enough. Each names
the page's showing (1, then one more each time the browser brings the page back from its cache): a goodbye counts for
the showing that said it only, so one that reaches the engine after the page came back changes nothing (one that names
no showing is taken as the page's newest). The last page that leaves decides:
  * it said goodbye: a real close. Once no page came back within close_grace_s (close_grace_pending_s while a run
    waits for an answer, so a slow reload never turns into a discard), the engine shuts down, and a run nobody
    answered for is discarded with no dialog (server/app.py decide_closing).
  * it said nothing (a crash, a tab the browser put away, a lost connection, a computer gone to sleep): it is away. The
    runs still sampling pause, marked so that the next page can say why (engine/runs.py Runs.pause_for_away), and the
    engine waits for the app to be opened again: never stopping by itself while a run waits for an answer; with
    nothing waiting it stops after away_stop_s (0: never).
"""
from __future__ import annotations

import asyncio
import re
import threading
import time

from ..cli import command_words
from .security import page_id   # the id a page gives its socket's address and its goodbyes

CLOSED_LINE = "SoPA companion: the last window was closed; shutting down (its page said goodbye)"
AWAY_LINE = ("SoPA companion: the last window went without closing (a crash, a sleeping computer, a tab the browser "
             "put away or a lost connection): {n} run(s) paused; the engine waits for the app to be opened again "
             "({command} stop ends it)")
BACK_LINE = "SoPA companion: a page is back after {s:g} s away"
IDLE_LINE = "SoPA companion: no window for {m:g} minutes and nothing waiting for an answer: shutting down"
PAGES_KEPT = 64          # pages remembered at most: the oldest that is not open goes first
GRACE_S = 6.0            # how long a page that left may take to come back, for an engine that watches no closing
GOODBYES = ("socket", "page-log", "closing")
VISITS = 10 ** 9         # a page's showings are counted below this


def visit_number(value) -> int | None:
    """Which showing of its page a socket or a goodbye comes from (its socket's ?visit=, a goodbye's visit), given as a
    whole number or its digits; None for anything else (one that names no showing: taken as the page's newest)."""
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,9}", value):
        return int(value)
    return value if type(value) is int and 0 <= value < VISITS else None


def _say(message: str) -> None:
    print(message, flush=True)


class Presence:
    """The pages signed in to this engine and how the last one went. A page counts while one of its sockets is open; a
    page that goes says goodbye (its socket's bye, its log's or its answer's last request as it is hidden for good). A
    page that went without one (a crash, a tab the browser put away, a lost connection) is away. Thread-safe (a lock):
    the routes run in threads, the socket and the watcher on the loop."""

    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        # page id -> {"sockets": its open sockets, "bye": how it said goodbye, "visit": its newest showing heard of}
        self.pages: dict[str, dict] = {}
        self.connects = 0                    # sockets of pages signed in so far
        self.windows = 0                     # launch codes of a browser the app opened, traded so far (signed_in)
        self.last = None                     # the page that left last (its last socket closed) ...
        self.left = 0.0                      # ... and when
        self.changed = clock()               # the last change, for public()
        self.marked = None                   # what the watcher made of the last page's going: "closed" | "away"
        self.watching = False                # a watcher acts on it (watch): public() leaves its verdict to it

    def _page(self, page: str) -> dict:
        entry = self.pages.pop(page, None) or {"sockets": 0, "bye": None, "visit": 0}
        self.pages[page] = entry                                   # the newest at the end
        while len(self.pages) > PAGES_KEPT:
            gone = next((p for p, e in self.pages.items() if not e["sockets"] and p != page), None)
            if gone is None:
                break
            del self.pages[gone]
        return entry

    def _open(self) -> int:
        return sum(entry["sockets"] for entry in self.pages.values())

    def connected(self, page: str, visit: int | None = None) -> None:
        """A socket of `page`'s showing `visit` (None: its newest) opened (signed in): it is here, and a goodbye it said
        before (a page the browser kept and showed again) no longer counts. A socket of an earlier showing than the
        newest heard of (one on its way as the page went) leaves the newest showing's goodbye as it is."""
        with self.lock:
            entry = self._page(page)
            entry["sockets"] += 1
            visit = entry["visit"] if visit is None else visit
            if visit >= entry["visit"]:
                entry["visit"], entry["bye"] = visit, None
            self.connects += 1
            self.marked, self.changed = None, self.clock()

    def disconnected(self, page: str) -> None:
        """A socket of `page` closed; its last one: the page left, now."""
        with self.lock:
            entry = self._page(page)
            entry["sockets"] = max(0, entry["sockets"] - 1)
            if not entry["sockets"]:
                self.last, self.left = page, self.clock()
            self.changed = self.clock()

    def goodbye(self, page, via: str, visit: int | None = None) -> None:
        """`page`'s showing `visit` (None: its newest) said it goes for good (`via` "socket" | "page-log" | "closing"),
        before or after its socket closed. One said before the page was shown again (an earlier showing than one that
        connected since: it reached the engine late) changes nothing, nor does one that names no page, or not one a page
        can have."""
        if (page := page_id(page)) is None or via not in GOODBYES:
            return
        with self.lock:
            entry = self._page(page)
            visit = entry["visit"] if visit is None else visit
            if visit < entry["visit"]:
                return
            entry["visit"], entry["bye"] = visit, via
            self.changed = self.clock()

    def stale(self, page, visit: int | None) -> bool:
        """Whether `page`'s showing `visit` is earlier than one that connected since: what it says now (an answer to
        closing that reached the engine late) was said before the page came back, and a page that connects again takes
        its answer back. One that names no showing (None) is the page's newest."""
        if visit is None or (page := page_id(page)) is None:
            return False
        with self.lock:
            entry = self.pages.get(page)
            return entry is not None and visit < entry["visit"]

    def signed_in(self, kind: str) -> None:
        """A launch code was traded for the page's key (server/security.py on_sign_in), of `kind` "window" (the page of
        a browser the app opened: the launcher's, a notebook's) or "link" (one a person opened by hand)."""
        if kind == "window":
            with self.lock:
                self.windows += 1

    def open(self) -> int:
        """The sockets of pages open now."""
        with self.lock:
            return self._open()

    def _verdict(self, grace: float) -> str | None:
        if self._open() or self.last is None or self.clock() - self.left < grace:
            return None
        entry = self.pages.get(self.last)
        return "closed" if entry is not None and entry["bye"] else "away"

    def verdict(self, grace: float) -> str | None:
        """None while a page is here, before any came, or less than `grace` s after the last one left; "closed" when
        that last page said goodbye (before or after its socket closed); else "away"."""
        with self.lock:
            return self._verdict(grace)

    def absent(self) -> float | None:
        """Seconds since the last page left; None while a page is here, or before any came."""
        with self.lock:
            return None if self._open() or self.last is None else self.clock() - self.left

    def mark(self, verdict: str) -> None:
        """What the watcher made of the last page's going ("closed" | "away"), until a page connects."""
        with self.lock:
            self.marked, self.changed = verdict, self.clock()

    def public(self) -> dict:
        """{"pages": sockets of pages open now, "state": "none" (no page yet) | "here" | "leaving" (the last one left
        a moment ago) | "closed" | "away", "since_s": seconds since the last change, "connects": signed-in sockets so
        far, "windows": pages of browsers the app opened signed in so far (whatever other tab connects meanwhile: the
        launcher and a notebook wait for this one)}: nothing secret, and no page's id."""
        with self.lock:
            pages = self._open()
            if pages:
                state = "here"
            elif self.last is None:
                state = "none"
            else:
                state = self.marked or (None if self.watching else self._verdict(GRACE_S)) or "leaving"
            return {"pages": pages, "state": state, "since_s": round(max(0.0, self.clock() - self.changed), 1),
                    "connects": self.connects, "windows": self.windows}


async def _call(action, log, what: str):
    """`action()` in a thread; what it raises is said, never raised (the watcher goes on, or ends as it meant to)."""
    try:
        return await asyncio.to_thread(action)
    except Exception as exc:
        log(f"SoPA companion: {what} failed: {type(exc).__name__}: {exc}")
        return None


async def watch(presence: Presence, *, grace: float, pending_grace: float, away_stop_s: float, pending, on_closed,
                on_away, on_idle=None, tick: float = 1.0, clock=time.monotonic, log=None) -> None:
    """Shut down (on_closed()) once the last page said goodbye and none came back within `grace` (`pending_grace` while
    `pending()` says a run waits for an answer); once per absence without a goodbye, pause (on_away() -> the run ids
    paused, logged); and shut an engine left away with nothing waiting down (on_idle(), else on_closed()) after
    `away_stop_s` (0: never) of nothing waiting. `pending`, `on_closed`, `on_away` and `on_idle` are plain functions,
    run in a thread; `pending` is asked only while no page is open."""
    log = log or _say
    on_idle = on_idle or on_closed
    presence.watching = True
    away_since = idle_since = None       # the absence taken for away: when it began; since when nothing has waited
    while True:
        await asyncio.sleep(tick)
        if presence.open():
            if away_since is not None:
                log(BACK_LINE.format(s=round(max(0.0, clock() - away_since), 1)))
                away_since = idle_since = None
            continue
        if presence.verdict(min(grace, pending_grace)) is None:
            continue                     # a page is here, none came yet, or the last one left a moment ago
        try:
            waiting = bool(await asyncio.to_thread(pending))
        except Exception:                # unknown: as if a run waits (the longer grace, never the idle stop)
            waiting = True
        verdict = presence.verdict(pending_grace if waiting else grace)
        if verdict == "closed":
            presence.mark("closed")
            log(CLOSED_LINE)
            await _call(on_closed, log, "closing the engine")
            return
        if verdict != "away":
            continue
        if away_since is None:
            away_since = clock() - (presence.absent() or 0.0)
            presence.mark("away")
            paused = await _call(on_away, log, "pausing the runs")
            log(AWAY_LINE.format(n=len(paused or ()), command=command_words()))
        if waiting or not away_stop_s:
            idle_since = None
            continue
        idle_since = clock() if idle_since is None else idle_since
        if clock() - idle_since >= away_stop_s:
            log(IDLE_LINE.format(m=round(away_stop_s / 60, 1)))
            await _call(on_idle, log, "closing the engine")
            return
