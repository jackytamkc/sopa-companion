"""A stand-in for engine/qc_pool.py QcPools in the tests of a test's runs (tests/test_qc_test_engine.py) and of its API
(tests/test_qc_test.py): the pool and its matching come from the test's own `make(meta, cell) -> (Pool, Matched)`, and
the run's own reasons are the engine's (qc_pool.run_reasons), split by kind as the engine splits them: a refusal refuses
(OutsideDomain), a wait makes `pool` raise NotYet, a caveat goes with the pool. The check's and F's reasons are the
test's own (`extra`). Set it as QcTests.pools (application.state.qc_tests.pools), which QcTests reads when it uses it.
The real QcPools is tested in tests/test_qc_pool.py; this module holds no test of its own."""
from __future__ import annotations

import collections
import dataclasses
from pathlib import Path
import time
from typing import Callable

from sopa_companion.engine import qc_pool


class FakePools:
    """`runs`: the engine's Runs (or a stand-in with `get(run_id)` -> a run with `meta` and `info`, and `max_region`,
    which the run's own reasons say); `make(meta, cell)` -> (Pool, Matched) for that run and cell; `reasons`: the run's
    own reasons, as the engine has them; `extra`: more reasons (dicts: the check's and F's, as the engine would find
    them), a list the test may change as the check moves on (one given as a list is kept, not copied);
    `seq_from_meta`: the pool's frame is the run's end
    (schedule_done's seq and iter, once it has one) whatever `make` gave (False keeps make's, as a pool taken at an
    older frame); `delay_s`: how long working out a pool takes, from the first ask of its run and cell (`ready_at`).
    `calls` counts each method's calls by name."""

    def __init__(self, runs, make: Callable[[dict, int], tuple[qc_pool.Pool, qc_pool.Matched]], *,
                 reasons: bool = True, extra=(), seq_from_meta: bool = True, delay_s: float = 0.0):
        self.runs, self.make = runs, make
        self.reasons, self.seq_from_meta, self.delay_s = reasons, seq_from_meta, delay_s
        self.extra = extra if isinstance(extra, list) else list(extra)
        self.ready_at: dict = {}            # (run, cell) -> when its pool is worked out (time.monotonic)
        self.calls: collections.Counter = collections.Counter()

    def _found(self, run) -> tuple[list, list, list]:
        """(refusals, waits, caveats) of the run: its own (with `reasons`) and `extra`."""
        found = qc_pool.run_reasons(run.meta, run.info, max_um=self.runs.max_region) if self.reasons else []
        return qc_pool.split([*found, *(dict(r) for r in self.extra)])

    def pool(self, run_id, cell, *, timeout=120.0, check=None) -> qc_pool.Pool:
        """The run's RunError passes through; the refusals raise OutsideDomain and the waits NotYet before anything
        else; then the pool is worked out delay_s from the first ask of its run and cell (asked again, the same work
        goes on, as in the engine's run worker), waited for at most `timeout` (StillWorking past it), check() called
        every 0.05 s meanwhile (it may raise); else make's Pool, at the run's end (with `seq_from_meta`), its caveats
        the run's then make's own."""
        self.calls["pool"] += 1
        run = self.runs.get(run_id)
        refusals, waits, caveats = self._found(run)
        if refusals:
            raise qc_pool.OutsideDomain(refusals)
        if waits:
            raise qc_pool.NotYet(waits)
        ready = self.ready_at.setdefault((run_id, cell), time.monotonic() + self.delay_s)
        deadline = min(ready, time.monotonic() + timeout)
        while True:
            if check is not None:
                check()
            left = deadline - time.monotonic()
            if left <= 0:
                break
            time.sleep(min(0.05, left))
        if time.monotonic() < ready:
            raise qc_pool.StillWorking()
        pool = self.make(run.meta, cell)[0]
        done = run.meta.get("schedule_done")
        if self.seq_from_meta and isinstance(done, dict):
            pool = dataclasses.replace(pool, seq=done.get("seq"), iter=done.get("iter"))
        return dataclasses.replace(pool, caveats=tuple(qc_pool.ordered([*caveats, *pool.caveats])))

    def prerequisites(self, run_id) -> tuple[list, list]:
        """(refusals, waits) of the run (its own and `extra`); its RunError passes through."""
        self.calls["prerequisites"] += 1
        refusals, waits, _ = self._found(self.runs.get(run_id))
        return refusals, waits

    def match_input(self, pool, input_path, options, *, check=None) -> qc_pool.Matched:
        """make's Matched for the pool's run and seed, holding the pool given and the input's path."""
        self.calls["match_input"] += 1
        if check is not None:
            check()
        matched = self.make(self.runs.get(pool.run).meta, pool.seed)[1]
        return dataclasses.replace(matched, pool=pool, input=str(Path(input_path)))

    def preview(self, run_id, cell, *, wait=0.0) -> dict:
        """{v, state: "ready", run, cell, offer, reasons, waits, caveats} only: the real reply's other fields are
        qc_pool's own."""
        self.calls["preview"] += 1
        refusals, waits, caveats = self._found(self.runs.get(run_id))
        return {"v": qc_pool.POOL_V, "state": "ready", "run": run_id, "cell": cell, "offer": not refusals,
                "reasons": refusals, "waits": waits, "caveats": caveats}

    def answers(self, run_id, cells) -> dict:
        self.calls["answers"] += 1
        return {}

    def forget(self, run_id) -> None:
        self.calls["forget"] += 1

    def close(self) -> None:
        self.calls["close"] += 1
