"""The mixing check's suggestions and trials through QcManager and the API (engine/qc_rules.py fed by engine/qc.py with
every live result): one card at a time from the confirmed areas and never while cells form or a trial runs; a trial
that follows its change (also when the page set the knob first), its band, rows and verdict; another change mixing it
up, and its own Undo not; the Undo card for a change that hurt and its "Keep"; R7 after a fair test that did not help;
the trial pushed to the page; a restart keeping the readout; drift judged like any other change; which runs the
scheduler should score first (`watching`); and a whole sample's summary with its cards staying under 8 KB. Every
verdict here is on the depth-matched change (the worker's "matched" and its per-bin sums), which these results carry;
the runs start from cells alone (a start line) unless a test says otherwise."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sopa_companion.engine import qc_constants as K
from sopa_companion.engine import qc_rules as R
from test_qc_api import (BASE, NCELLS, NX_B, NY_B, P1, P2, PAIRED, auth, bins_table, deliver_start_bins, make,
                         ready_run, write_run)
from test_qc_api import world  # noqa: F401  (the fixture)

KNOBS = {"cell-compactness": 0.03999999910593033, "prior-weight": 1.0, "no-diffusion": False}
TIGHT = 0.019999999552965164                              # 0.02 as a frame carries it (float32)


def knob_run(world, run_id: str, frames: int, knobs_at=None, created="2026-10-18T09:00:00+00:00", mode="detail",
             burnin=10, step=0.5, prior=None, phase_at=None):
    """A run on disk (test_qc_api.write_run: frame s is iteration s, `step` s apart, burn-in 10 by default: areas and
    bands count from iteration 30; `prior`: how its cells started, cells alone by default) whose frame s carries the
    knob values KNOBS updated by knobs_at(s), and the phase phase_at(s) when that gives one."""
    cfg, sample = world
    folder = write_run(Path(cfg["paths"]["data_dir"]), run_id, sample, created, frames=frames,
                       times=[step * s for s in range(frames)], burnin=burnin, mode=mode, prior=prior)
    path = folder / "live" / "frames.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    for frame in lines:
        frame["live"] = {**KNOBS, **(knobs_at(frame["seq"]) if knobs_at else {})}
        frame["phase"] = (phase_at(frame["seq"]) if phase_at else None) or frame["phase"]
    path.write_text("".join(json.dumps(frame) + "\n" for frame in lines))
    return cfg


class Live:
    """A run that grows as its results come (run.frames: the frames up to the newest result), sampling."""

    def __init__(self, application, run_id: str):
        self.qc, self.id = application.state.qc, run_id
        self.run = application.state.runs.get(run_id)
        self.frames = list(self.run.frames)
        self.run.frames = self.frames[:1]
        self.run.meta["state"] = "running"

    def done(self):
        """The run ended (closing the engine would discard a run still sampling)."""
        self.run.meta["state"] = "done"

    def deliver(self, seq: int, judged=400, mixed=40, bins=None, bands=None, tx=300.0, matched="auto",
                start=10.0) -> dict:
        """The worker's live result for frame seq (whole-tissue numbers, per-bin sums, its matched change) as the scheduler
        hands it over; returns the summary as the page gets it. `matched`: the worker's depth-matched change (by default
        the share's own change from `start` in 100 over the judged cells: nothing depth-related happened; None: an
        older worker without it); per-bin sums with paired cells get their matched sums the same way."""
        self.run.frames = self.frames[:seq + 1]
        state = self.qc._state(self.id)
        frame = self.frames[seq]
        summary = {"kind": "frame", "seq": seq, "judged": judged, "mixed": mixed, "ncells": NCELLS, "median_tx": tx}
        if matched == "auto":
            matched = {"paired": judged, "used": judged, "per100_now": 100.0 * mixed / judged, "per100_start": start,
                       "change": 100.0 * mixed / judged - start, "sd": None, "seeds": 0}
        if matched is not None:
            summary["matched"] = matched
            bins = with_matched(bins)
        if bands is not None:
            summary["bands"] = {"edges": [100, 250, 500, 1000], "judged": bands[0], "mixed": bands[1]}
        self.qc.qc_result({"kind": "frame", "run": self.id, "seq": seq, "model": state.key, "dir": str(state.dir),
                           "source": "live", "single": False,
                           "frame": {k: frame.get(k) for k in ("seq", "iter", "phase", "t", "live", "voxel")},
                           "summary": summary, "bins": bins, "cells": None, "start": None})
        return self.qc.summary(self.id, seq)


def with_matched(table):
    """Per-bin sums with the worker's depth-matched sums (its paired cells, as mixed now and at the start: nothing
    depth-related happened), unless they carry their own."""
    if table is None or "paired" not in table or "matched_paired" in table:
        return table
    return {**table, "matched_paired": np.asarray(table["paired"], np.float32),
            "matched_mixed_now": np.asarray(table["paired_mixed_now"], np.float32),
            "matched_mixed_start": np.asarray(table["paired_mixed_start"], np.float32)}


def around(cells, pad=2):
    """The bins within `pad` of `cells` (an area reaches past its flagged windows' centres)."""
    return sorted({(r + dr, c + dc) for r, c in cells for dr in range(-pad, pad + 1) for dc in range(-pad, pad + 1)
                   if 0 <= r + dr < NY_B and 0 <= c + dc < NX_B})


def growth_table(hot, start_hot, grown=(), squeezed=()):
    """bins_table with each bin's 25 cells about as big as at the start (growth histogram bucket 0.95-1.05), those in
    `grown` about 37 % bigger (1.25-1.5), and 2 tiny cells (< 10 µm²) in each of `squeezed`."""
    table = bins_table(hot, start_hot=start_hot)
    n = NY_B * NX_B
    hist = np.zeros((n, 13), np.int32)
    hist[:, 6] = 25
    for r, c in grown:
        hist[r * NX_B + c, 6], hist[r * NX_B + c, 9] = 0, 25
    table["growth_hist"] = hist
    table["tiny"], table["tiny_start"] = np.zeros(n, np.int32), np.zeros(n, np.int32)
    for r, c in squeezed:
        table["tiny"][r * NX_B + c] = 2
    return table


def trials_file(qc, run_id):
    return [json.loads(line) for line in (qc._state(run_id).dir / "trials.jsonl").read_text().splitlines()]


# ---- cards ------------------------------------------------------------------------------------------------------------
def test_cards_come_from_the_confirmed_areas_one_at_a_time(world):
    """No card while cells form (the start line's areas are not the run's); from burn-in + 20 the run's areas are
    confirmed, and the first card is shown once it has led for 2 s: one card (the top area's), the others under More
    ideas. Area A was already mixed at the start and did not grow (R1); area B rose with cells 37 % bigger (R4: a new
    run first, its timing names 1,000 iterations of this run)."""
    cfg = knob_run(world, "run-cards", 45)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-cards")
        live = Live(application, "run-cards")
        deliver_start_bins(live.qc, "run-cards")                      # P1 already mixed at the start
        table = growth_table(P1 + P2, P1, grown=around(P2))
        got = {s: live.deliver(s, bins=table) for s in range(1, 45)}
        view = client.get("/api/v1/runs/run-cards/qc", headers=auth()).json()
        stored = json.loads((live.qc._state("run-cards").dir / "000040.qc.json").read_text())
        live.done()
    for s in range(1, 34):                                            # forming (to 29), then entering (2 s from 30)
        assert got[s]["suggestion"] is None and "more_ideas" not in got[s], s
    assert got[29]["area_view"] == "start" and got[30]["area_view"] == "smoothed"
    late = got[40]
    assert [a["L"] for a in late["areas"]] == ["A", "B"] and [a["tag"] for a in late["areas"]] == ["already", "rose"]
    card = late["suggestion"]
    # Already high at the start, the cause not known (never "no setting is likely to fix this"): show the
    # starting outlines, look at its cells.
    assert card == {"id": "R1::None", "rule": "R1", "areas": ["A"], "confidence": "unknownCause", "timing": "later",
                    "actions": ["show_start", "inspect"], "main": "show_start"}
    local = {a["L"]: a["local"] for a in late["areas"]}
    assert local["B"]["enough"] and local["B"]["direction"] == "more" and local["B"]["matched_change"] > 10
    assert local["A"]["direction"] == "same" and local["A"]["matched_change"] == 0.0
    (idea,) = late["more_ideas"]
    assert (idea["rule"], idea["areas"], idea["flag"], idea["from"], idea["to"]) == ("R4", ["B"], "cell-compactness",
                                                                                      0.04, 0.02)
    assert idea["confidence"] == "rough" and idea["main"] == "new_run" and idea["actions"] == ["new_run", "try", "why"]
    assert idea["later_s"] == 500 and idea["growth"] == pytest.approx(0.369, abs=0.01)   # 0.5 s an iteration here
    assert got[34]["suggestion"]["rule"] == "R1" and late["trial"] is None and "previous" not in late
    assert not {"can_new_run", "active_runs", "max_active_runs"} & view.keys()
    assert stored["suggestion"] == card and stored["more_ideas"] == late["more_ideas"]     # each result keeps its own


def test_r5_hears_how_the_run_started(world, monkeypatch):
    """Through QcManager: the rules hear how the run started (ctx "start", qc_rules.start_of of the run's record). A
    run started from cells only gets its areas compared with its start (A already mixed: R1), but never R5's firmer
    hold for B, which rose while its small cells were squeezed (there the hold keeps any neighbouring RNA inside the
    starting outlines). A run started from cells with their nuclei has no start line (proseg's first iteration holds
    only the nuclei): its areas are compared with nothing, so none of the cards that need the start (R1, R4, R5,
    R6) comes, whatever the per-bin sums carry."""
    heard, evaluate = [], R.evaluate
    monkeypatch.setattr(R, "evaluate", lambda ctx: heard.append(ctx.get("start")) or evaluate(ctx))
    cards, tags, starts = {}, {}, {}
    for run_id, nuclei in (("run-r5-paired", {"kind": "shapes", "key": "xenium_nucleus_boundaries"}),
                           ("run-r5-cells", None)):
        cfg = knob_run(world, run_id, 45)
        application = make(cfg)
        with TestClient(application, base_url=BASE) as client:
            ready_run(client, run_id)
            live = Live(application, run_id)
            live.run.meta["prior"] = {"cells": {"kind": "shapes", "key": "xenium_cell_boundaries"}, "nuclei": nuclei}
            deliver_start_bins(live.qc, run_id)                       # P1 already mixed at the start
            table = growth_table(P1 + P2, P1, squeezed=around(P2))
            late = [live.deliver(s, bins=table) for s in range(1, 45)][-5]
            live.done()
        cards[run_id] = {(c["rule"], tuple(c["areas"])) for c in [late["suggestion"], *late.get("more_ideas", [])] if c}
        tags[run_id], starts[run_id] = {a["L"]: a["tag"] for a in late["areas"]}, late["start"]
        assert set(heard) == {"paired" if nuclei else "cells"}, run_id
        heard.clear()
    assert ("R1", ("A",)) in cards["run-r5-cells"] and not any(rule == "R5" for rule, _ in cards["run-r5-cells"])
    assert tags["run-r5-cells"]["A"] == "already" and tags["run-r5-cells"]["B"] == "rose"
    assert tags["run-r5-paired"] and set(tags["run-r5-paired"].values()) == {"similar"}
    assert starts["run-r5-paired"] is None and starts["run-r5-cells"] is not None
    assert not any(rule in ("R1", "R4", "R5", "R6") for rule, _ in cards["run-r5-paired"])


def test_r7_only_after_a_trial_that_made_its_area_more_mixed(world):
    """R7 ("Settings didn't help here") needs evidence that the setting did not help the area: a trial on area B whose
    own band said "more" (B has 300 or more judged cells, so it is judged on its own band), kept. A trial
    that ran its 300 iterations with "No clear change" is no evidence (a shape change after cells formed shows only
    after ~1,000 iterations, never within the 300 a trial is judged over, as measured): B keeps its
    card. The Undo card does not ask again about a change whose trial was kept."""
    cfg = knob_run(world, "run-r7", 460, lambda s: {"cell-compactness": TIGHT} if 101 <= s < 420 else {})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-r7")
        live = Live(application, "run-r7")
        deliver_start_bins(live.qc, "run-r7")
        table = growth_table(P1 + P2, P1, grown=around(P2))
        for s in range(1, 101):
            live.deliver(s, bins=table)
        assert client.post("/api/v1/runs/run-r7/qc/trial", headers=auth(), json={
            "rule": "R4", "area": "B", "flag": "cell-compactness", "from": 0.04, "to": 0.02}).status_code == 200
        got = {s: live.deliver(s, bins=table) for s in range(101, 412)}
        assert got[150]["trial"]["rows"][0]["kind"] == "area" and got[150]["trial"]["start_iter"] == 101
        assert got[150]["trial"]["on"] == "area"                         # B is big enough to be judged on its own
        assert got[401]["trial"]["verdict"] is None and got[402]["trial"]["verdict"] == "no_change"
        assert got[402]["trial"]["final"] and got[402]["trial"]["band"][-1]["iter"] == 401
        ended = client.post("/api/v1/runs/run-r7/qc/trial/end", json={"outcome": "undone"}, headers=auth()).json()
        assert ended["verdict"] == "no_change" and ended["outcome"] == "undone"
        after = {s: live.deliver(s, bins=table) for s in range(412, 430)}
        state = live.qc._state("run-r7")
        assert state.tried == []                                          # no clear change: no evidence against it
        live.done()
    cards = [after[429]["suggestion"], *after[429].get("more_ideas", [])]
    assert [(c["rule"], c["areas"]) for c in cards] == [("R1", ["A"]), ("R4", ["B"])]

    cfg = knob_run(world, "run-r7more", 200, lambda s: {"cell-compactness": TIGHT} if s >= 101 else {})
    application = make(cfg)
    rising = growth_table(P1 + P2, P1, grown=around(P2))
    for r, c in P2:
        rising["mixed"][r * NX_B + c] = 16                                # B's hot bins: 60 % -> 80 % mixed
    rising["paired_mixed_now"] = rising["mixed"].copy()
    rising["n10"] = np.maximum(rising["mixed"] - rising["paired_mixed_start"], 0).astype(np.int32)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-r7more")
        live = Live(application, "run-r7more")
        deliver_start_bins(live.qc, "run-r7more")
        for s in range(1, 101):
            live.deliver(s, bins=table)
        client.post("/api/v1/runs/run-r7more/qc/trial", headers=auth(), json={
            "rule": "R4", "area": "B", "flag": "cell-compactness", "from": 0.04, "to": 0.02})
        got = {s: live.deliver(s, bins=rising) for s in range(101, 160)}
        trial = got[159]["trial"]
        row = next(r for r in trial["rows"] if r["kind"] == "area")
        assert trial["verdict"] == "more" and trial["on"] == "area" and row["now"] > row["expected"]
        client.post("/api/v1/runs/run-r7more/qc/trial/end", json={"outcome": "kept"}, headers=auth())
        later = {s: live.deliver(s, bins=rising) for s in range(160, 175)}
        state = live.qc._state("run-r7more")
        assert state.tried == [{"area": "B", "flag": "cell-compactness", "from": 0.04, "to": 0.02, "verdict": "more"}]
        live.done()
    cards = [later[174]["suggestion"], *later[174].get("more_ideas", [])]
    assert ("R7", ["B"]) in [(c["rule"], c["areas"]) for c in cards]
    assert not any(c["rule"] == "U" for s in later for c in [later[s]["suggestion"]] if c)   # kept: not asked again
    r7 = next(c for c in cards if c["rule"] == "R7")
    assert r7["tried"] == {"flag": "cell-compactness", "from": 0.04, "to": 0.02} and r7["actions"] == ["inspect"]
    assert r7["confidence"] == "seen"                                     # what the trial showed, nothing more


def test_an_area_trial_is_judged_on_the_areas_own_band(world):
    """Area B (300 or more judged cells) gets less mixed after the change while the whole tissue stays inside its band:
    the trial says "less" for B (its own band, its row 28 -> 19 in 100), not "No clear change", and no R7 follows."""
    cfg = knob_run(world, "run-own", 460, lambda s: {"cell-compactness": TIGHT} if 101 <= s < 420 else {})
    application = make(cfg)
    before = growth_table(P1 + P2, P1, grown=around(P2))
    after = growth_table(P1 + P2, P1, grown=around(P2))
    for r, c in P2:
        after["mixed"][r * NX_B + c] = 7                                  # B's hot bins: 60 % -> 35 % mixed
    after["paired_mixed_now"] = after["mixed"].copy()
    after["n10"] = np.maximum(after["mixed"] - after["paired_mixed_start"], 0).astype(np.int32)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-own")
        live = Live(application, "run-own")
        deliver_start_bins(live.qc, "run-own")
        for s in range(1, 101):
            live.deliver(s, bins=before)
        assert client.post("/api/v1/runs/run-own/qc/trial", headers=auth(), json={
            "rule": "R4", "area": "B", "flag": "cell-compactness", "from": 0.04, "to": 0.02}).status_code == 200
        got = {s: live.deliver(s, bins=after) for s in range(101, 412)}
        client.post("/api/v1/runs/run-own/qc/trial/end", json={"outcome": "kept"}, headers=auth())
        later = {s: live.deliver(s, bins=after) for s in range(412, 430)}
        state = live.qc._state("run-own")
        live.done()
    trial = got[300]["trial"]
    area = next(r for r in trial["rows"] if r["kind"] == "area")
    assert area["now"] < area["expected"] - 1.0 and trial["on"] == "area" and trial["verdict"] == "less"
    assert got[402]["trial"]["verdict"] == "less" and state.tried == []
    cards = [later[429]["suggestion"], *later[429].get("more_ideas", [])]
    assert not any(c and c["rule"] == "R7" for c in cards)


# ---- trials --------------------------------------------------------------------------------------------------------------
def test_a_trial_follows_its_change_and_says_less_mixed_once_the_band_allows(world):
    """Try it at iteration 110; proseg applies the value from frame 120. The band is the 50 iterations before it (10.0
    +- 1.0 in 100); mixing falls at once, but the verdict waits for 5 results and 20 iterations, then 2 results in a row
    below the band. No card while it runs; the page hears the trial when it registers and when it meets its frame;
    Keep ends it."""
    cfg = knob_run(world, "run-try", 200, lambda s: {"cell-compactness": TIGHT} if s >= 120 else {})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-try")
        live = Live(application, "run-try")
        qc = live.qc
        for s in range(1, 111):
            live.deliver(s)
        cursor = {}
        qc.events(cursor, "run-try", 0.0)
        trial = client.post("/api/v1/runs/run-try/qc/trial", headers=auth(), json={
            "rule": "R4", "area": None, "flag": "cell-compactness", "from": 0.04, "to": 0.02}).json()
        assert trial["asked_seq"] == 110 and trial["start_seq"] is None and trial["rows"] == [] and trial["verdict"] is None
        assert trial["timing"] == "later" and trial["notes"] == [] and trial["started"]
        pushed = [e for e in qc.events(cursor, "run-try", 1.0) if e.get("state") == "trial"]
        assert len(pushed) == 1 and pushed[0]["trial"]["id"] == 1 and pushed[0]["trial"]["start_seq"] is None
        got = {s: live.deliver(s, mixed=40 if s < 120 else 30) for s in range(111, 160)}
        pushed = [e for e in qc.events(cursor, "run-try", 2.0) if e.get("state") == "trial"]
        assert pushed and pushed[-1]["trial"]["start_seq"] == 120                # it met its first frame
        assert all(got[s]["suggestion"] is None and "more_ideas" not in got[s] for s in got)
        first = got[120]["trial"]
        assert (first["start_seq"], first["start_iter"], first["verdict"]) == (120, 120, None)
        # The matched change since is smoothed as the headline is: a region's weight, 0.2 a second (measured on
        # proseg's count), so results 0.5 s apart move it 1 - 0.8 ** 0.5 = 0.106 of the way: -2.5 reads -0.26 at once.
        step = 1 - (1 - K.value("region", "headline_weight")) ** 0.5
        assert first["rows"][0] == {"kind": "whole", "expected": 10.0, "now": 9.7, "change": -0.26} == \
            {"kind": "whole", "expected": 10.0, "now": round(10.0 - 2.5 * step, 1), "change": round(-2.5 * step, 2)}
        assert first["rows"][1]["kind"] == "depth"
        assert "started" not in first and "asked_iter" not in first              # the summary carries what moves
        assert [got[s]["trial"]["verdict"] for s in (125, 139, 140)] == [None, None, None]
        assert got[141]["trial"]["verdict"] == "less" and got[141]["trial"]["results"] == 18
        # The band: the history's share +- the run's own sampler threshold (400 paired cells at 10 in 100: 1.76).
        half = R.change_threshold("region", 0.1, 400)
        low, high = round(10.0 - half, 2), round(10.0 + half, 2)
        assert half > 1.0 and got[141]["trial"]["band"] == [{"iter": 120, "expected": 10.0, "low": low, "high": high},
                                                            {"iter": 141, "expected": 10.0, "low": low, "high": high}]
        # The matched change since, 22 results on: most of the way to -2.5.
        assert got[141]["trial"]["rows"][0]["change"] == pytest.approx(-2.5 * (1 - (1 - step) ** 22), abs=0.01)
        assert got[141]["trial"]["horizon_iter"] == 420
        view = client.get("/api/v1/runs/run-try/qc", headers=auth()).json()["trial"]
        assert view["verdict"] == "less" and view["start_seq"] == 120 and view["started"] == trial["started"]
        ended = client.post("/api/v1/runs/run-try/qc/trial/end", json={"outcome": "kept"}, headers=auth()).json()
        assert ended["outcome"] == "kept" and ended["verdict"] == "less" and ended["ended"]
        pushed = [e for e in qc.events(cursor, "run-try", 3.0) if e.get("state") == "trial"]
        assert pushed[-1]["trial"] is None
        assert live.deliver(160, mixed=30)["trial"] is None
        lines = trials_file(qc, "run-try")
        assert [l["event"] for l in lines] == ["start", "end"] and lines[1]["verdict"] == "less"
        assert lines[1]["start_seq"] == 120 and qc._state("run-try").tried == []    # it helped: no R7
        live.done()


def test_a_trial_registered_after_its_change_showed_starts_where_it_showed(world):
    """The page may set the knob first and register the trial a moment later: the trial starts at the frame the change
    showed in (up to 20 frames before it was asked for), not at the frame it was asked at."""
    cfg = knob_run(world, "run-late", 160, lambda s: {"cell-compactness": TIGHT} if s >= 120 else {})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-late")
        live = Live(application, "run-late")
        for s in range(1, 126):
            live.deliver(s, mixed=40 if s < 120 else 30)
        trial = client.post("/api/v1/runs/run-late/qc/trial", headers=auth(), json={
            "rule": "R4", "area": None, "flag": "cell-compactness", "to": 0.02}).json()
        assert trial["asked_seq"] == 125 and trial["start_seq"] == 120 and trial["from"] == 0.04
        assert trial["rows"][0]["expected"] == 10.0
        assert live.deliver(126, mixed=30)["trial"]["start_seq"] == 120
        live.done()


def test_another_change_mixes_the_trial_up_but_its_own_undo_does_not(world):
    """Prior weight changed during a trial of Cell compactness: the comparison is mixed up (no verdict, no band, the
    rows as they were). The trial's own value put back (its Undo, before the page ended it) only freezes it."""
    for run_id, later in (("run-mixed", {"prior-weight": 2.0}), ("run-undone", {"cell-compactness": KNOBS["cell-compactness"]})):
        cfg = knob_run(world, run_id, 170,
                       lambda s, later=later: {**({"cell-compactness": TIGHT} if s >= 120 else {}), **(later if s >= 150 else {})})
        application = make(cfg)
        with TestClient(application, base_url=BASE) as client:
            ready_run(client, run_id)
            live = Live(application, run_id)
            for s in range(1, 111):
                live.deliver(s)
            client.post(f"/api/v1/runs/{run_id}/qc/trial", headers=auth(), json={
                "rule": "R4", "area": None, "flag": "cell-compactness", "from": 0.04, "to": 0.02})
            got = {s: live.deliver(s, mixed=40 if s < 120 else 30) for s in range(111, 160)}
            live.done()
        before, after, later_on = got[149]["trial"], got[150]["trial"], got[159]["trial"]
        assert before["verdict"] == "less" and before["notes"] == []
        if run_id == "run-mixed":
            assert after["mixed_up"] and "mixed_up" in after["notes"] and after["verdict"] is None and after["band"] is None
        else:
            assert not after["mixed_up"] and after["verdict"] == "less" and after["band"] is not None
        assert later_on["rows"] == before["rows"] and got[159]["suggestion"] is None     # frozen; still running


def test_a_manual_change_that_hurts_brings_the_undo_card_and_keep_sends_it_away(world):
    """Prior weight lowered in Live knobs (no trial): mixing rises above the band of the 50 iterations before; after 5
    results in a row above it (from 20 iterations on) the Undo card comes at once, first. "Keep" answers it: it goes, and
    the answer is kept with the run's trials."""
    cfg = knob_run(world, "run-harm", 160, lambda s: {"prior-weight": 0.5} if s >= 120 else {})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-harm")
        live = Live(application, "run-harm")
        got = {s: live.deliver(s, mixed=40 if s < 120 else 52) for s in range(1, 145)}
        assert got[143]["suggestion"] is None
        card = got[144]["suggestion"]
        assert {k: v for k, v in card.items() if k != "after_per100"} == {
            "id": "U:prior-weight:1.0:1", "rule": "U", "flag": "prior-weight", "from": 0.5, "to": 1.0, "areas": [],
            "before_per100": 10.0,
            "changed": {"flag": "prior-weight", "from": 1.0, "to": 0.5}, "confidence": "seen", "timing": "later",
            "actions": ["undo", "keep"], "main": "undo"}
        assert 12.5 < card["after_per100"] <= 13.0
        end = "/api/v1/runs/run-harm/qc/trial/end"
        answered = client.post(end, json={"outcome": "kept", "rule": "U"}, headers=auth()).json()
        assert answered["rule"] == "U" and answered["outcome"] == "kept" and answered["changed"]["to"] == 0.5
        assert live.deliver(145, mixed=52)["suggestion"] is None
        assert client.post(end, json={"outcome": "kept"}, headers=auth()).status_code == 404
        (harm,) = [l for l in trials_file(live.qc, "run-harm") if l["event"] == "harm"]
        assert harm["outcome"] == "kept" and harm["changed"] == {"flag": "prior-weight", "from": 1.0, "to": 0.5}
        live.done()


def test_a_trials_readout_survives_an_engine_restart_and_ends_with_its_run(world):
    """The run does not go on after a restart (it is stopped, here finished), so its trial ends with it: its readout as
    it was shown, with the outcome "ended", and the end in trials.jsonl."""
    cfg = knob_run(world, "run-restart", 150, lambda s: {"cell-compactness": TIGHT} if s >= 120 else {})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-restart")
        live = Live(application, "run-restart")
        for s in range(1, 111):
            live.deliver(s)
        client.post("/api/v1/runs/run-restart/qc/trial", headers=auth(), json={
            "rule": "R4", "area": None, "flag": "cell-compactness", "from": 0.04, "to": 0.02})
        for s in range(111, 145):
            last = live.deliver(s, mixed=40 if s < 120 else 30)
        shown = client.get("/api/v1/runs/run-restart/qc", headers=auth()).json()["trial"]
        live.done()
    assert shown["verdict"] == "less" == last["trial"]["verdict"]
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        again = client.get("/api/v1/runs/run-restart/qc", headers=auth()).json()["trial"]
        lines = trials_file(application.state.qc, "run-restart")
    assert again == {**shown, "outcome": "ended", "ended": again["ended"]}
    assert [l["event"] for l in lines] == ["start", "end"] and lines[1]["outcome"] == "ended"
    assert lines[1]["verdict"] == "less"


def test_a_trial_ends_when_its_run_ends(world):
    """A run finished (or stopped) while its trial runs: the trial ends with the outcome "ended" (trials.jsonl, the
    page's push), so the card slot is not held by a trial that can never end; a new trial is refused (409)."""
    cfg = knob_run(world, "run-ends", 150, lambda s: {"cell-compactness": TIGHT} if s >= 120 else {})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-ends")
        live = Live(application, "run-ends")
        qc = live.qc
        for s in range(1, 111):
            live.deliver(s)
        client.post("/api/v1/runs/run-ends/qc/trial", headers=auth(), json={
            "rule": "R4", "area": None, "flag": "cell-compactness", "from": 0.04, "to": 0.02})
        for s in range(111, 130):
            live.deliver(s, mixed=40 if s < 120 else 30)
        cursor = {}
        qc.events(cursor, "run-ends", 0.0)
        live.done()                                                     # Finish & save
        pushed = [e for e in qc.events(cursor, "run-ends", 1.0) if e.get("state") == "trial"]
        view = client.get("/api/v1/runs/run-ends/qc", headers=auth()).json()
        lines = trials_file(qc, "run-ends")
        refused = client.post("/api/v1/runs/run-ends/qc/trial", headers=auth(), json={
            "rule": "R4", "area": None, "flag": "cell-compactness", "from": 0.02, "to": 0.04}).status_code
    assert pushed and pushed[-1]["trial"]["outcome"] == "ended" and pushed[-1]["trial"]["id"] == 1
    assert view["trial"]["outcome"] == "ended" and qc._state("run-ends").trial is None
    assert [l["event"] for l in lines] == ["start", "end"] and lines[1]["outcome"] == "ended" and refused == 409


def test_trials_need_a_sampling_run_and_one_at_a_time(world):
    cfg = knob_run(world, "run-one", 40)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-one")
        url = "/api/v1/runs/run-one/qc/trial"
        body = {"rule": "R3", "area": None, "flag": "no-diffusion", "from": True, "to": False}
        assert client.post(url, json=body, headers=auth()).status_code == 409         # finished
        live = Live(application, "run-one")
        live.run.meta["state"] = "paused"                                             # paused: applies on Resume
        trial = client.post(url, json=body, headers=auth()).json()
        assert trial["start_seq"] is None and "no_verdict" not in trial["notes"] and trial["verdict"] is None
        assert client.post(url, json=body, headers=auth()).status_code == 409         # one at a time
        others = application.state.runs
        for run_id in ("run-a", "run-b"):
            others.get(run_id).meta["state"] = "running"
        view = client.get("/api/v1/runs/run-one/qc", headers=auth()).json()
        assert not {"active_runs", "can_new_run"} & view.keys()
        for run_id in ("run-a", "run-b"):
            others.get(run_id).meta["state"] = "done"
        live.done()


# ---- the previous run ----------------------------------------------------------------------------------------------------






# ---- size ---------------------------------------------------------------------------------------------------------------
def test_a_whole_samples_summary_with_eight_areas_and_its_cards_stays_under_8_kb(world, monkeypatch):
    """8 areas on 100 µm bins (as test_qc_api's 8 KB test) with their tags and growth: already mixed (R1), grown (R4),
    squeezed and flat (R6: a run started from cells alone, the only start with a start line, never gets R5's firmer
    hold), so a card and two more ideas. The summary as the page gets it (now with its coverage, retention, matched
    change, comparisons, count, constants, provenance and each area's own comparison) stays within SUMMARY_BYTES: the
    last more ideas give way, never the card."""
    from sopa_companion.engine import qc as Q
    nx, ny, frames = 64, 40, 40
    cfg = knob_run(world, "run-tiled", frames, created="2026-10-18T11:00:00+00:00", mode="tiled", burnin=0, step=1.0)
    rng = np.random.default_rng(11)
    n = nx * ny
    judged, mixed, start_mixed = np.full((ny, nx), 30), np.full((ny, nx), 3), np.full((ny, nx), 3)
    hist, tiny = np.zeros((ny, nx, 13), np.int32), np.zeros((ny, nx), np.int32)
    hist[..., 6] = 35
    for k in range(8):                                                # 8 blobs, far apart
        r0, c0 = 2 + 20 * (k // 4), 2 + 16 * (k % 4)
        blob = rng.random((10, 10)) < 0.6
        mixed[r0:r0 + 10, c0:c0 + 10][blob] = 16
        if k < 2:                                                     # already mixed at the start
            start_mixed[r0:r0 + 10, c0:c0 + 10][blob] = 16
        elif k < 4:                                                   # cells grew about 37 %
            hist[r0 - 2:r0 + 12, c0 - 2:c0 + 12, 6], hist[r0 - 2:r0 + 12, c0 - 2:c0 + 12, 9] = 0, 35
        elif k < 6:                                                   # small cells squeezed
            tiny[r0:r0 + 10, c0:c0 + 10] = 1
    now, start = mixed.ravel(), start_mixed.ravel()
    table = {"grid": np.array([10050.0, 9030.0, 100.0, nx, ny]), "bin": np.arange(n, dtype=np.int32),
             "cells": np.full(n, 35, np.int32), "judged": judged.ravel().astype(np.int32), "mixed": now.astype(np.int32),
             "tx": np.full(n, 9000.0, np.float32), "assigned": np.full(n, 10000.0, np.float32),
             "foreground": np.full(n, 8500.0, np.float32), "paired": judged.ravel().astype(np.int32),
             "paired_mixed_now": now.astype(np.int32), "paired_mixed_start": start.astype(np.int32),
             "n10": np.maximum(now - start, 0).astype(np.int32), "n01": np.maximum(start - now, 0).astype(np.int32),
             "growth_hist": hist.reshape(n, 13), "tiny": tiny.ravel(), "tiny_start": np.zeros(n, np.int32),
             "judged_start": judged.ravel().astype(np.int32), "mixed_start": start.astype(np.int32)}
    start_table = {**{k: v for k, v in table.items() if k not in ("paired", "paired_mixed_now", "paired_mixed_start",
                                                                  "n10", "n01", "growth_hist", "tiny", "tiny_start",
                                                                  "judged_start", "mixed_start")},
                   "mixed": start.astype(np.int32)}
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-tiled")
        state = qc._state("run-tiled")
        state.dir.mkdir(parents=True, exist_ok=True)
        np.savez(state.dir / "start.bins.npz", **start_table)
        qc.qc_result({"kind": "start", "run": "run-tiled", "seq": None, "model": state.key, "dir": str(state.dir),
                      "source": "start", "single": True, "bins": start_table, "cells": None, "start": None,
                      "summary": {"kind": "start", "judged": int(judged.sum()), "mixed": int(start.sum()),
                                  "median_tx": 300.0, "ncells": n}})
        live = Live(application, "run-tiled")
        budget = Q.SUMMARY_BYTES
        monkeypatch.setattr(Q, "SUMMARY_BYTES", 10 ** 6)               # every card, to see them all
        for seq in range(20, frames - 2):                             # areas are shown from iteration 0 + 20
            live.deliver(seq, judged=int(judged.sum()), mixed=int(now.sum()), bins=table)
        full = qc.summary("run-tiled", frames - 3)
        monkeypatch.setattr(Q, "SUMMARY_BYTES", budget)
        served = live.deliver(frames - 2, judged=int(judged.sum()), mixed=int(now.sum()), bins=table)
        cut = len(json.dumps(served, separators=(",", ":"))) - 50     # a budget the last more idea does not fit in
        monkeypatch.setattr(Q, "SUMMARY_BYTES", cut)
        tight = live.deliver(frames - 1, judged=int(judged.sum()), mixed=int(now.sum()), bins=table)
        live.done()
    assert len(served["areas"]) == 8 and sorted({a["tag"] for a in served["areas"]}) == ["already", "rose"]
    cards = [full["suggestion"], *full.get("more_ideas", [])]
    assert {c["rule"] for c in cards} == {"R1", "R4", "R6"} and len(cards) == 3
    assert served["constants"]["mode"] == "tiled" and "basis" in served
    assert served["suggestion"] == full["suggestion"] and served["more_ideas"] == full["more_ideas"][:len(served["more_ideas"])]
    assert len(json.dumps(served, separators=(",", ":"))) <= budget <= 9600
    assert all(a["local"]["enough"] and isinstance(a["edge"], bool) for a in served["areas"])
    assert served["coverage"]["judged"] and served["change"]["matched"] is not None and served["provenance"]
    assert tight["suggestion"] == served["suggestion"]
    assert len(tight.get("more_ideas", [])) == len(served["more_ideas"]) - 1
    assert len(json.dumps(tight, separators=(",", ":"))) <= cut




def test_the_overview_grid_is_each_bins_exact_window_share_and_its_judged_cells(world):
    """GET /runs/{id}/qc/{seq}/bins (the overview layer): plane 1 is, for each bin, the share of judged cells that
    look mixed over its 3 x 3 window, sum M / sum J (floor(share x 250); 255 where the window judges nothing), plane 2
    the bin's own judged cells (at most 254), so a bin is weighed by what it holds and the page averages nothing."""
    cfg = knob_run(world, "run-bins", 20)
    rng = np.random.default_rng(5)
    judged = rng.integers(0, 30, (NY_B, NX_B))
    judged[:, :3] = 0                                                  # empty background on the left
    judged[0, 0] = 300                                                 # one over-full bin
    mixed = rng.binomial(judged, 0.2)
    n = NY_B * NX_B
    table = {"grid": np.array([0.0, 0.0, 50.0, NX_B, NY_B]), "bin": np.arange(n, dtype=np.int32),
             "cells": (judged + 5).ravel().astype(np.int32), "judged": judged.ravel().astype(np.int32),
             "mixed": mixed.ravel().astype(np.int32)}
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-bins")
        live = Live(application, "run-bins")
        live.deliver(1, bins=table)                                    # the first result: its own sums
        answer = client.get("/api/v1/runs/run-bins/qc/1/bins", headers=auth({"Accept-Encoding": "identity"}))
        live.done()
    assert json.loads(answer.headers["x-bins"]) == {"x0": 0.0, "y0": 0.0, "bin": 50.0, "nx": NX_B, "ny": NY_B, "planes": 2}
    share, own = np.frombuffer(answer.content, np.uint8).reshape(2, NY_B, NX_B)
    for r in range(NY_B):
        for c in range(NX_B):
            rows, cols = slice(max(0, r - 1), r + 2), slice(max(0, c - 1), c + 2)
            J, M = judged[rows, cols].sum(), mixed[rows, cols].sum()
            assert share[r, c] == (255 if J == 0 else int(np.floor(M / J * 250))), (r, c)
            assert own[r, c] == min(int(judged[r, c]), 254), (r, c)


def test_drift_switched_off_can_be_tried_while_cells_form(world):
    """With transcript drift off, "Turn transcript drift back on" (R3) is offered while cells still form, with the start
    line's areas on screen: Try it first (drift acts only after burn-in, so this is the time); nothing else is offered
    before the run's own areas, since nothing is compared with the start yet."""
    cfg = knob_run(world, "run-drift", 30, lambda s: {"no-diffusion": True}, burnin=40)     # burn-in to iteration 40
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-drift")
        live = Live(application, "run-drift")
        deliver_start_bins(live.qc, "run-drift")
        got = {s: live.deliver(s, bins=growth_table(P1, P1)) for s in range(1, 30)}
        live.done()
    assert all(got[s]["suggestion"] is None for s in range(1, 14))          # iteration < 10, then 2 s of leading
    card = got[20]["suggestion"]
    assert (card["rule"], card["areas"], card["timing"], card["main"]) == ("R3", [], "early", "try")
    assert got[20]["area_view"] == "start" and got[20]["phase"] == "burnin" and (card["from"], card["to"]) == (True, False)


# ---- honest readouts -----------------------------------------------------------------------------------------




def test_two_changes_between_two_results_are_both_counted_and_mix_the_trial_up(world):
    """Prior weight lands at frame 101 and the trial's Cell compactness at 102, results every other frame (a whole
    sample: about one a second): the watch holds both changes with their own new values, so the trial attaches but is
    mixed up (no verdict), and no Undo card is offered (which change hurt cannot be told)."""
    def knobs(s):
        return {**({"prior-weight": 0.5} if s >= 101 else {}), **({"cell-compactness": TIGHT} if s >= 102 else {})}
    cfg = knob_run(world, "run-gap", 200, knobs)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-gap")
        live = Live(application, "run-gap")
        for s in range(2, 101, 2):
            live.deliver(s)
        live.run.frames = live.frames[:101]
        client.post("/api/v1/runs/run-gap/qc/trial", headers=auth(), json={
            "rule": "R4", "area": None, "flag": "cell-compactness", "from": 0.04, "to": 0.02})
        got = {s: live.deliver(s, mixed=40 if s < 101 else 52) for s in range(102, 200, 2)}
        watch = live.qc._state("run-gap").trial_watch
        live.done()
    assert watch.changes == [{"flag": "cell-compactness", "from": 0.04, "to": 0.02},
                             {"flag": "prior-weight", "from": 1.0, "to": 0.5}]
    trial = got[198]["trial"]
    assert trial["start_seq"] == 101 and trial["mixed_up"] and "mixed_up" in trial["notes"] and trial["verdict"] is None
    assert all(got[s]["suggestion"] is None for s in got)


def test_a_drift_change_is_judged_like_any_other_and_two_knobs_at_once_get_no_undo_card(world):
    """Drift back on (R3), tried after cells formed while the line rises: proseg's count shows what drift moves, so the
    trial has its band and says "more" (no "no_verdict" note); kept, nothing asks about it again. Drift switched back on
    in Live knobs (no trial) while mixing rises: the Undo card offers to put back what the user changed. Two knobs
    changed in the same frame: no Undo card (it could put back only one of them)."""
    cfg = knob_run(world, "run-drift", 200, lambda s: {"no-diffusion": True} if s < 120 else {"no-diffusion": False})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-drift")
        live = Live(application, "run-drift")
        for s in range(1, 111):
            live.deliver(s)
        client.post("/api/v1/runs/run-drift/qc/trial", headers=auth(), json={
            "rule": "R3", "area": None, "flag": "no-diffusion", "from": True, "to": False})
        got = {s: live.deliver(s, mixed=40 if s < 120 else 52) for s in range(111, 150)}
        client.post("/api/v1/runs/run-drift/qc/trial/end", json={"outcome": "kept"}, headers=auth())
        after = {s: live.deliver(s, mixed=52) for s in range(150, 175)}
        live.done()
    trial = got[149]["trial"]
    assert trial["verdict"] == "more" and trial["band"] and trial["start_seq"] == 120
    assert "no_verdict" not in trial["notes"] and "no_history" not in trial["notes"]
    assert trial["rows"][0]["kind"] == "whole" and trial["rows"][0]["now"] > trial["rows"][0]["expected"]
    assert all(after[s]["suggestion"] is None or after[s]["suggestion"]["rule"] != "U" for s in after)   # kept
    cfg = knob_run(world, "run-drift-on", 160, lambda s: {"no-diffusion": True} if s < 120 else {"no-diffusion": False})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-drift-on")
        live = Live(application, "run-drift-on")
        got = {s: live.deliver(s, mixed=40 if s < 120 else 52) for s in range(1, 160)}
        live.done()
    card = next(got[s]["suggestion"] for s in sorted(got) if (got[s]["suggestion"] or {}).get("rule") == "U")
    assert (card["flag"], card["from"], card["to"], card["actions"]) == ("no-diffusion", False, True, ["undo", "keep"])
    assert card["changed"] == {"flag": "no-diffusion", "from": True, "to": False}
    cfg = knob_run(world, "run-two", 160, lambda s: {"prior-weight": 0.5, "cell-compactness": 0.05} if s >= 120 else {})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-two")
        live = Live(application, "run-two")
        got = {s: live.deliver(s, mixed=40 if s < 120 else 52) for s in range(1, 160)}
        live.done()
    assert not any(got[s]["suggestion"] and got[s]["suggestion"]["rule"] == "U" for s in got)


def test_no_undo_card_right_after_undo_and_none_for_an_automatic_value(world):
    """A trial that made mixing rise, undone: the next result still carries the trial's value (the old one applies at
    the next iteration boundary), and no "Undo your last change?" asks about the change just undone. A temperature set
    by hand after cells formed changed it from "automatic" (the schedule's), which Live knobs cannot put back: no Undo
    card for it."""
    cfg = knob_run(world, "run-flash", 200, lambda s: {"cell-compactness": TIGHT} if 120 <= s < 152 else {})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-flash")
        live = Live(application, "run-flash")
        for s in range(1, 111):
            live.deliver(s)
        client.post("/api/v1/runs/run-flash/qc/trial", headers=auth(), json={
            "rule": "R4", "area": None, "flag": "cell-compactness", "from": 0.04, "to": 0.02})
        got = {s: live.deliver(s, mixed=40 if s < 120 else 52) for s in range(111, 150)}
        client.post("/api/v1/runs/run-flash/qc/trial/end", json={"outcome": "undone"}, headers=auth())
        flash = {s: live.deliver(s, mixed=52) for s in range(150, 153)}
        live.done()
    assert got[149]["trial"]["verdict"] == "more"
    assert all(flash[s]["suggestion"] is None or flash[s]["suggestion"]["rule"] != "U" for s in flash)
    cfg = knob_run(world, "run-temp", 160, lambda s: {"temperature": 1.0 if s < 120 else 0.5})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-temp")
        live = Live(application, "run-temp")
        got = {}
        for s in range(1, 160):
            if s == 120:
                live.run.meta["live_values"] = {"temperature": 0.5}
            got[s] = live.deliver(s, mixed=40 if s < 120 else 52)
        watch = live.qc._state("run-temp").watch
        live.done()
    assert watch.changes == [{"flag": "temperature", "from": None, "to": 0.5}] and watch.harm
    assert not any(got[s]["suggestion"] and got[s]["suggestion"]["rule"] == "U" for s in got)


def test_the_undo_card_puts_back_the_exact_value_asked_for(world):
    """Frames carry proseg's 32-bit floats (123.4567 comes back as 123.45670318603516): the Undo card puts back the
    number the run was asked for (its options, or the live value set), not a rounded one."""
    single = float(np.float32(123.4567))
    cfg = knob_run(world, "run-exact", 160, lambda s: {"prior-weight": single if s < 120 else 50.0})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-exact")
        live = Live(application, "run-exact")
        live.run.meta["options"] = {"prior-weight": 123.4567}
        got = {}
        for s in range(1, 160):
            if s == 120:
                live.run.meta["live_values"] = {"prior-weight": 50.0}
            got[s] = live.deliver(s, mixed=40 if s < 120 else 52)
        live.done()
    card = next(got[s]["suggestion"] for s in sorted(got) if got[s]["suggestion"])
    assert card["rule"] == "U" and card["to"] == 123.4567 and card["from"] == 50.0
    assert card["changed"] == {"flag": "prior-weight", "from": 123.4567, "to": 50.0}




def test_the_scheduler_is_told_while_a_knob_change_is_judged(world):
    """QcManager.watching (the analysis scheduler scores a watched run's newest counted frames first): no change yet,
    no; a trial registered, yes, until it ends; a change made in Live knobs, yes, until its Undo card is answered (or it
    reaches its horizon, or another change freezes it); an unknown run, no."""
    cfg = knob_run(world, "run-watched", 200, lambda s: {"prior-weight": 0.5} if s >= 120 else {})
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-watched")
        live = Live(application, "run-watched")
        qc = live.qc
        assert qc.watching("run-watched") is False and qc.watching("no-such-run") is False
        for s in range(1, 111):
            live.deliver(s)
        assert qc.watching("run-watched") is False                      # nothing changed yet
        client.post("/api/v1/runs/run-watched/qc/trial", headers=auth(), json={
            "rule": "R5", "area": None, "flag": "prior-weight", "from": 1.0, "to": 2.0})
        assert qc.watching("run-watched") is True                       # registered, before its change shows
        client.post("/api/v1/runs/run-watched/qc/trial/end", json={"outcome": "undone"}, headers=auth())
        assert qc.watching("run-watched") is False
        got = {s: live.deliver(s, mixed=40 if s < 120 else 52) for s in range(111, 150)}
        assert qc.watching("run-watched") is True                       # Prior weight lowered in Live knobs at 120
        assert any((got[s]["suggestion"] or {}).get("rule") == "U" for s in got)
        client.post("/api/v1/runs/run-watched/qc/trial/end", json={"outcome": "kept", "rule": "U"}, headers=auth())
        assert qc.watching("run-watched") is False                      # answered: nothing left to judge
        live.done()


@pytest.mark.parametrize("case", ["forming", "trial", "last_stage", "too_soon", "no_start_line"])
def test_no_priority_for_a_change_that_can_never_be_judged(world, case):
    """A change made while cells form (in Live knobs, or a trial's once its change showed), one made in the last
    settling stage, one made too soon after cells formed (5 results before it) and one on a run whose start has nuclei
    (no like-for-like change) never get a band: they are watched (their readout) until their horizon, but the scheduler
    is not asked to score their runs first."""
    at = {"forming": 5, "trial": 5, "last_stage": 130, "too_soon": 35, "no_start_line": 120}[case]
    cfg = knob_run(world, "run-never", at + 30, lambda s: {"prior-weight": 0.5} if s >= at else {},
                   prior=PAIRED if case == "no_start_line" else None,
                   phase_at=(lambda s: "hillclimb" if s >= 125 else None) if case == "last_stage" else None)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-never")
        live = Live(application, "run-never")
        for s in range(1, at + 20):
            if case == "trial" and s == 3:
                client.post("/api/v1/runs/run-never/qc/trial", headers=auth(), json={
                    "rule": "R5", "area": None, "flag": "prior-weight", "from": 1.0, "to": 0.5})
                assert live.qc.watching("run-never") is True            # registered, before its change shows
            live.deliver(s)
        state = live.qc._state("run-never")
        watch, trial_watch, watching = state.watch, state.trial_watch, live.qc.watching("run-never")
        live.done()
    assert watch is not None and watch.start_seq == at and watch.band is None and not watch.area_band
    assert not (watch.final or watch.frozen or watch.dismissed) and watch.iter_now < watch.horizon_iter
    assert watching is False and watch.judging is False
    assert trial_watch is (watch if case == "trial" else None)          # the trial runs on, with no verdict to come


@pytest.mark.parametrize("mode", ["tiled", "whole"])
def test_a_knob_change_on_a_tiled_region_or_a_whole_sample_is_watched_with_its_modes_constants(world, monkeypatch,
                                                                                              mode):
    """Every mode holds the same values today, so here the run's own mode is given values of its own: a knob change on
    a tiled region (or a whole sample) is judged on its mode's trial constants (here its horizon), its areas' own
    comparisons on its mode's (the tracker's), and only areas with its mode's minimum support are watched (here none),
    never on a dense region's."""
    monkeypatch.setitem(K.TABLE[mode], "trial", {**K.value(mode, "trial"), "horizon_iterations": 77})
    monkeypatch.setitem(K.TABLE[mode], "local_min_judged", 10 ** 6)
    cfg = knob_run(world, "run-mode", 140, lambda s: {"prior-weight": 0.5} if s >= 120 else {}, mode="tiled")
    if mode == "whole":
        info = Path(cfg["paths"]["data_dir"]) / "runs" / "run-mode" / "live" / "info.json"
        info.write_text(json.dumps({**json.loads(info.read_text()), "cells": 41190}))
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-mode")
        live = Live(application, "run-mode")
        deliver_start_bins(live.qc, "run-mode")
        got = {s: live.deliver(s, bins=bins_table(P1, start_hot=P1)) for s in range(1, 130)}
        state = live.qc._state("run-mode")
        live.done()
    watch = state.watch
    assert got[129]["constants"]["mode"] == mode and [a["L"] for a in got[119]["areas"]] == ["A"]
    assert watch.mode == mode and watch.start_seq == 120 and watch.horizon_iter == 120 + 77
    assert state.tracker.mode == mode and watch.area_bins == {}    # area A: about 500 judged cells, under 10**6


def test_the_checks_run_states_are_the_runs_own():
    """engine/qc.py names active and ended states as engine/runs.py does: a trial ends with its run."""
    from sopa_companion.engine import qc as Q
    from sopa_companion.engine import runs as RUNS
    assert Q.ACTIVE_STATES == RUNS.ACTIVE and Q.ENDED_STATES == RUNS.TERMINAL


def test_a_live_verdict_changes_only_once_the_new_one_has_held_for_5_s(world):
    """A live result's verdict on the like-for-like change (and an area's) is held steady
    on the run clock: a new direction shows only once it has held for 5 s over 2 results; meanwhile the summary says it
    is held and what the change's own verdict is (direction_now). The threshold itself never moves."""
    cfg = knob_run(world, "run-steady", 80, step=0.5)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-steady")
        live = Live(application, "run-steady")
        got = {s: live.deliver(s, judged=4000, mixed=400) for s in range(1, 41)}      # 10 in 100, as at the start
        got.update({s: live.deliver(s, judged=4000, mixed=200) for s in range(41, 70)})   # 5 in 100: fewer
        live.done()
    assert got[40]["change"]["direction"] == "same" and "held" not in got[40]["change"]
    turned = next(s for s in range(41, 70) if got[s]["change"].get("direction_now") == "fewer"
                  or got[s]["change"]["direction"] == "fewer")
    first = got[turned]["change"]
    assert first["held"] is True and first["direction"] == "same" and first["direction_now"] == "fewer"
    assert first["meaningful"] is False and first["meaningful_now"] is True and first["held_s"] == 0.0
    shown = next(s for s in range(turned, 70) if got[s]["change"]["direction"] == "fewer")
    assert 0.5 * (shown - turned) >= R.STEADY_HOLD_S - 1e-9, (turned, shown)          # 5 s of the run's clock
    assert "held" not in got[shown]["change"] and got[shown]["change"]["meaningful"] is True
    for s in range(turned, shown):                                                    # held all along, never flipping
        assert got[s]["change"]["direction"] == "same" and got[s]["change"]["held"]


def test_live_summaries_have_no_previous_run_link(world):
    cfg = knob_run(world, "run-current", 60)
    knob_run(world, "run-before", 60)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client, "run-current")
        ready_run(client, "run-before")
        before, current = Live(application, "run-before"), Live(application, "run-current")
        for seq in range(1, 60):
            before.deliver(seq, mixed=48)
            summary = current.deliver(seq)
        assert "previous" not in summary and "prev_change" not in summary
        assert client.get("/api/v1/runs/run-current/qc/previous", headers=auth()).status_code == 404
        state = current.qc._state("run-current")
        old = {**summary, "previous": {"run": "run-before"}, "prev_change": {"direction": "same"}}
        state.results[59] = json.dumps(old)
        assert not {"previous", "prev_change"} & current.qc.summary("run-current", 59).keys()
        before.done()
        current.done()
