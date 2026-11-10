"""Proseg's own count on the engine's side (engine/runs.py): whether a proseg-live keeps it (its --help), the flags a
run starts with, an older proseg-live's runs (they start and segment, without counts: "old_proseg"), counts.jsonl merged
into the frames' records as proseg commits them, where exports, the counts and where tokens the scheduler, Go deeper and
a test ask for (served by the next frame, or at once by a paused state), the spool's stop, and what the engine says of
it all (counts_status, a run's proseg_counts)."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import pytest

from conftest import wait_for
from sopa_companion.engine import held, runs as R
from sopa_companion.engine.counts_spool import CountsSpool
from sopa_companion.tests import counts_fixtures as CF
from test_engine import SHORT, TEST, executed, make_runs, registered, shared_file, start

GRID = {"x0": 0.0, "y0": 0.0, "voxel": 1.0, "nx": 4, "ny": 4, "nz": 2}     # the fake's tiny sample's
ENTRIES = {key: np.zeros(1, np.int64) for key in ("i", "j", "k", "gene", "dest")} | {"n": np.ones(1, np.int64)}
STATUS_KEYS = {"mode", "reason", "issued", "served", "outstanding", "newest", "newest_at", "extract_ms", "bytes",
               "spool_mb", "can_issue", "why"}


def created(runs, speed: str = "max") -> R.Run:
    """A run of the fake's tiny sample, paused on its prior (frame 0, counted: init) as every run starts."""
    run = runs.get(runs.create(runs.samples.list()[0]["id"], [50, 50, 150, 150])["id"])
    runs.control(run.meta["id"], "speed", speed)
    wait_for(lambda: (run.ack or {}).get("state") == "paused" and len(run.frames) == 1 and "counts" in run.frames[0])
    return run


def counted(run: R.Run) -> list[int]:
    with run.lock:
        return [f["seq"] for f in run.frames if "counts" in f]


def sampling(runs, run_id: str = "r") -> R.Run:
    """A run that samples and keeps proseg's own count, known to the engine but with no proseg behind it: nothing
    serves its tokens but what the test writes as proseg would (counts_fixtures), read with run.read_new()."""
    run = R.Run(runs.root / run_id, {"id": run_id, "kind": "proseg", "state": "running", "paused": False,
                                     "speed": "max", "live_counts": "request"})
    (run.live / "frames").mkdir(parents=True)
    run.spool = runs.spool
    run.info = {"cells": 3, "counts": {"mode": "request"}}
    with runs.lock:
        runs.runs[run_id] = run
    return run


def gone(runs, run: R.Run):
    """The run stops (a request_where still waiting for it returns) and leaves the engine, which closes."""
    with run.lock:
        run.meta["state"] = "stopped"
    with runs.lock:
        runs.runs.pop(run.meta["id"], None)
    runs.close("discard")


class Waiting:
    """call() in a daemon thread of its own: a wait that never ends fails its test, never hangs the others."""

    def __init__(self, call):
        self.out: dict = {}
        self.thread = threading.Thread(target=self._call, args=(call,), daemon=True)
        self.thread.start()

    def _call(self, call):
        try:
            self.out["answer"] = call()
        except BaseException as exc:            # raised again where the answer is asked for
            self.out["error"] = exc

    def answer(self, limit: float = 8.0):
        self.thread.join(limit)
        assert not self.thread.is_alive(), f"no answer within {limit} s"
        if "error" in self.out:
            raise self.out["error"]
        return self.out["answer"]


# ---- what a proseg-live keeps, and the flags a run starts with -----------------------------------------------------
def test_whether_a_proseg_live_keeps_proseg_s_count_is_asked_of_its_help_once(config, fake_binary, old_fake_binary,
                                                                              tmp_path, monkeypatch):
    """counts_capable: --help lists --live-counts (proseg-live 3.2.0.1) or not (3.2.0), asked once per file (its real
    path, modification time and size: a new build is asked again); a program that does not answer is asked once more,
    then taken for an older one, said in the engine's log (the run's when one is named), and asked again next time; one
    that answers the second time is kept; a missing one keeps nothing."""
    runs = make_runs(config)
    asked, real = [], subprocess.run

    def run(command, **kwargs):
        asked.append(list(command))
        return real(command, **kwargs)
    monkeypatch.setattr(R.subprocess, "run", run)
    assert runs.counts_capable(str(fake_binary)) is True and runs.counts_capable(str(old_fake_binary)) is False
    assert runs.counts_capable(str(fake_binary)) is True and runs.counts_capable(str(old_fake_binary)) is False
    assert asked == [[str(fake_binary), "--help"], [str(old_fake_binary), "--help"]]
    link = tmp_path / "linked"
    link.symlink_to(fake_binary)
    assert runs.counts_capable(str(link)) is True and len(asked) == 2           # the same file, by its real path
    stamp = fake_binary.stat().st_mtime_ns + 10_000_000_000
    os.utime(fake_binary, ns=(stamp, stamp))                                    # rebuilt
    assert runs.counts_capable(str(fake_binary)) is True and len(asked) == 3
    assert runs.counts_capable(str(tmp_path / "nowhere")) is False and len(asked) == 3

    def silent(command, **kwargs):
        asked.append(list(command))
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))
    noted = []
    monkeypatch.setattr(runs, "note", lambda text, run_id=None: noted.append((text, run_id)))
    monkeypatch.setattr(R, "HELP_RETRY_S", 0.0)
    monkeypatch.setattr(R.subprocess, "run", silent)
    stamp += 10_000_000_000
    os.utime(fake_binary, ns=(stamp, stamp))
    assert runs.counts_capable(str(fake_binary), "r") is False and runs.counts_capable(str(fake_binary)) is False
    assert asked[3:] == [[str(fake_binary), "--help"]] * 4                     # twice each time: not kept
    [(text, run_id), (_, engine)] = noted
    assert text.startswith("run r: proseg-live did not answer --help (") and "timed out" in text
    assert (run_id, engine) == ("r", None)
    busy = [OSError(errno.ETXTBSY, "Text file busy")]                           # a build replacing it, for a moment

    def replaced(command, **kwargs):
        asked.append(list(command))
        if busy:
            raise busy.pop()
        return real(command, **kwargs)
    monkeypatch.setattr(R.subprocess, "run", replaced)
    assert runs.counts_capable(str(fake_binary), "r") is True and len(asked) == 9 and len(noted) == 2
    assert runs.counts_capable(str(fake_binary)) is True and len(asked) == 9   # kept


def test_a_proseg_live_that_does_not_answer_starts_its_run_without_counts_and_says_why(config, monkeypatch):
    """--help unanswered twice: the run starts without the flags (old_proseg, as an older proseg-live's: a run never
    fails to start for it), and the engine's log says why, as the run's own (else its Log would only say to update)."""
    runs = make_runs(config)
    noted = []
    monkeypatch.setattr(runs, "note", lambda text, run_id=None: noted.append((text, run_id)))
    monkeypatch.setattr(R, "HELP_RETRY_S", 0.0)
    monkeypatch.setattr(R.subprocess, "run",
                        lambda command, **kwargs: (_ for _ in ()).throw(OSError(errno.ETXTBSY, "Text file busy")))
    run = R.Run(runs.root / "r", {"id": "r", "mode": "detail", "options": {}, "proseg_args": ["--nthreads", "2"]})
    command = runs._command(run, [])
    assert "--live-counts" not in command and command[-1] == "--live-hold" and run.meta["live_counts"] == "old_proseg"
    assert noted == [(f"run r: proseg-live did not answer --help ([Errno {errno.ETXTBSY}] Text file busy): started "
                      "without proseg's own count, as an older proseg-live is", "r")]


def test_a_run_asks_for_proseg_s_count_and_a_cells_only_start_for_its_start_block(config, old_fake_binary):
    """A proseg-live that keeps it: --live-counts request into the run's own spool folder (meta live_counts
    "request"), and --live-counts-start for a user's run whose cells start without nuclei (the arrow's like-for-like
    start), never for a test's run or a start with nuclei. An older one: no flag, meta live_counts "old_proseg"."""
    runs = make_runs(config)
    folder = runs.root / "r"
    cells = {"id": "r", "mode": "detail", "options": {}, "proseg_args": ["--nthreads", "2"], "sample_kind": "zarr",
             "prior": {"cells": {"kind": "shapes", "key": "cellpose"}, "nuclei": None},
             "prepared": {"seeded_from": "cells"}}
    spool = ["--live-counts", "request", "--live-counts-dir", str(runs.spool.folder / "r")]
    run = R.Run(folder, dict(cells))
    command = runs._command(run, [])
    assert command[command.index("--live-counts"):] == [*spool, "--live-counts-start", "--live-hold"]
    assert run.meta["live_counts"] == "request" and (runs.spool.folder / "r").is_dir()
    paired = R.Run(folder, {**cells, "prepared": {"seeded_from": "nuclei"},
                            "prior": {"cells": {"kind": "shapes", "key": "cell_boundaries"},
                                      "nuclei": {"kind": "shapes", "key": "nucleus_boundaries"}}})
    assert runs._command(paired, [])[-5:] == [*spool, "--live-hold"]
    tiled = runs._command(R.Run(folder, {**cells, "mode": "tiled"}), [])
    assert tiled[-7:] == [*spool, "--live-counts-start", "--live-hold", "--live-tiled"]
    test = R.Run(folder, {**cells, "qc_test": dict(TEST)})
    assert runs._command(test, [])[-4:] == spool and test.meta["live_counts"] == "request"
    runs.binary = str(old_fake_binary)
    old = R.Run(folder, dict(cells))
    assert "--live-counts" not in runs._command(old, []) and old.meta["live_counts"] == "old_proseg"
    assert R.counts_reason(old) == "old_proseg"


def test_a_run_whose_spool_folder_cannot_be_made_counts_only_what_proseg_counts_by_itself(config, monkeypatch):
    """The spool's disk refuses the run's folder: proseg still keeps its count (into the run's folder, no
    --live-counts-dir), the engine says so in its log, and no frame is asked of proseg for it (low_space)."""
    runs = make_runs(config)
    noted = []
    monkeypatch.setattr(runs, "note", lambda text, run_id=None: noted.append((text, run_id)))
    monkeypatch.setattr(runs.spool, "run_dir", lambda run_id: (_ for _ in ()).throw(OSError("No space left")))
    run = R.Run(runs.root / "r", {"id": "r", "mode": "detail", "options": {}, "proseg_args": ["--nthreads", "2"]})
    command = runs._command(run, [])
    assert command[command.index("--live-counts"):] == ["--live-counts", "request", "--live-hold"]
    assert run.meta["live_counts"] == "request" and run.spool_lost and noted and noted[0][1] == "r"
    assert "No space left" in noted[0][0] and runs._spool_refusal(run) == "spool_lost"


def test_an_older_proseg_live_still_segments_without_proseg_s_count(config, old_fake_binary):
    """An installed proseg-live 3.2.0 (the pin replaces it, but it may still be there): its runs start and segment as
    before, and say why they have no proseg's count ("old_proseg"); no token is ever asked of it."""
    runs = make_runs({**config, "proseg": {**config["proseg"], "binary": str(old_fake_binary)}})
    try:
        run = runs.get(runs.create(runs.samples.list()[0]["id"], [50, 50, 150, 150])["id"])
        runs.control(run.meta["id"], "speed", "max")
        runs.control(run.meta["id"], "resume")
        wait_for(lambda: len(run.frames) >= 5)
        assert run.meta["live_counts"] == "old_proseg" and "counts" not in run.info
        assert run.public()["proseg_counts"] == {"reason": "old_proseg", "newest": None, "recent": [], "sampled": []}
        assert counted(run) == [] and not (run.live / "counts.jsonl").exists()
        assert runs.request_counts(run.meta["id"]) is None and runs.request_where(run.meta["id"], [0, 0, 2, 2]) is None
        status = runs.counts_status(run.meta["id"])
        assert (status["reason"], status["why"], status["can_issue"], status["mode"]) == \
            ("old_proseg", "old_proseg", False, None)
        control = (run.live / "control").read_text().splitlines()
        assert not any(line.startswith(("counts ", "where ")) for line in control)
        [line] = [line for line in (run.dir / "proseg.log").read_text().splitlines() if line.startswith("$ ")]
        assert "--live-counts" not in line
    finally:
        runs.close("discard")


def test_a_run_made_before_proseg_s_count_was_kept_says_so(tmp_path):
    """No `counts` in info.json: made before (no_counts); a stain run or training patches: no reason; nothing read
    yet: none either."""
    run = R.Run(tmp_path / "r", {"id": "r", "kind": "proseg"})
    assert R.counts_reason(run) is None
    run.info = {"cells": 2}
    assert R.counts_reason(run) == "no_counts"
    run.info = {"cells": 2, "counts": {"mode": "request"}}
    assert R.counts_reason(run) is None
    run.meta["live_counts"] = "old_proseg"
    assert R.counts_reason(run) == "old_proseg"
    for kind in ("stain", "patches"):
        assert R.counts_reason(R.Run(tmp_path / kind, {"id": kind, "kind": kind})) is None


# ---- counts.jsonl and where exports, read ----------------------------------------------------------------------------
def frame_record(seq: int) -> dict:
    return {"seq": seq, "iter": seq * 10, "phase": "burnin", "file": f"frames/{seq:06d}.bin.zst", "t": seq * 0.5}


def counted_frame(live: Path, seq: int, keep: float = 1.0, **line):
    """One counted frame of a 3-cell, 2-gene count, its line appended to counts.jsonl (counts_fixtures)."""
    owner = np.array([1, 1, 2, 3, 3, 3, 0])
    gene = np.array([0, 1, 1, 0, 0, 1, 0])
    c = CF.counts_of(owner, gene, 3, 2, keep)
    return CF.write_frame_counts(live, seq, seq * 10, ncells=3, ngenes=2, indptr=c["indptr"], genes=c["genes"],
                                 n=c["n"], e=c["e"], f=c["f"] if line.pop("has_f", False) else None, **line)


def test_counts_lines_are_merged_into_their_frames_as_proseg_commits_them(tmp_path):
    """read_new reads counts.jsonl as it grows (whole lines only), merges each line into its frame's record once that
    record is read (counts_file, counted_file, the line itself), the record replaced, never changed; a later line of a
    frame replaces the earlier (the hold's durable count of a paused state, which makes the frame one read on the count
    proseg saves: `sampled`, its page record's `count`); unreadable lines are passed over. The run's disk bytes take
    the durable files and the sidecars; spooled files are noted for pruning."""
    live = tmp_path / "r" / "live"
    (live / "frames").mkdir(parents=True)
    spool = CountsSpool(tmp_path / "spool")
    run = R.Run(tmp_path / "r", {"id": "r", "kind": "proseg"})
    run.spool = spool
    for seq in range(4):
        (live / f"frames/{seq:06d}.bin.zst").write_bytes(b"f" * 100)
    (live / "frames.jsonl").write_text("".join(json.dumps(frame_record(s)) + "\n" for s in (0, 1)))
    init = counted_frame(live, 0, reason="init", has_f=True)
    asked = counted_frame(live, 1, reason="request", token=1, spool=spool.run_dir("r"))
    ahead = counted_frame(live, 3, reason="request", token=2, spool=spool.run_dir("r"))
    with open(live / "counts.jsonl", "a") as stream:          # one being written: not read until it is whole
        stream.write('{"seq": 2, "iter": 20')
    assert run.read_new()
    assert counted(run) == [0, 1] and run.counted == [0, 1] and run.counts_served == 1
    assert run.frames[0] == {**frame_record(0), "counts_file": "frames/000000.counts.zst",
                             "counted_file": "frames/000000.counted.zst", "counts": init}
    assert run.frames[1]["counts_file"] == asked["file"] == str(spool.run_dir("r") / "000001.counts.zst")
    assert run.held_counts.keys() == {3}                      # its frame is not read yet
    sidecar = (live / "frames/000000.counted.zst").stat().st_size
    assert run.frame_bytes == 200 + init["bytes"] + sidecar + (live / "frames/000001.counted.zst").stat().st_size
    assert run.public()["proseg_counts"] == {"reason": None, "newest": 1, "recent": [0, 1], "sampled": []}
    assert [R.page_frame(run.frames[seq])["count"] for seq in (0, 1)] == ["averaged", "averaged"]   # init: not saved
    first = run.newest_at
    assert first is not None and spool.known["r"] == {os.path.realpath(asked["file"]): asked["file"]}
    with open(live / "counts.jsonl", "a") as stream:          # the line written whole, then some that cannot be read
        stream.write(', "file": "frames/000002.counts.zst", "counted_file": "frames/000002.counted.zst"}\n'
                     + "not json\n" + json.dumps({"seq": -1, "file": "a", "counted_file": "b"}) + "\n"
                     + json.dumps({"seq": True, "file": "a", "counted_file": "b"}) + "\n"
                     + json.dumps({"seq": 2, "counted_file": "b"}) + "\n")
    with open(live / "frames.jsonl", "a") as stream:
        stream.write("".join(json.dumps(frame_record(s)) + "\n" for s in (2, 3)))
    time.sleep(0.01)
    assert run.read_new() and counted(run) == [0, 1, 2, 3] and run.counts_served == 2 and run.newest_at > first
    assert run.frames[2]["counts"] == {"seq": 2, "iter": 20, "file": "frames/000002.counts.zst",
                                       "counted_file": "frames/000002.counted.zst"} and run.held_counts == {}
    # The hold counts the paused frame 1 again, durably and with f: its record is replaced (a reader holding the old one
    # keeps it as it was), its bytes now in the run's folder.
    old, before = run.frames[1], run.frame_bytes
    durable = counted_frame(live, 1, reason="hold", has_f=True)
    assert run.read_new()
    assert old["counts"] == asked and run.frames[1]["counts"] == durable and run.frames[1] is not old
    assert run.frames[1]["counts_file"] == "frames/000001.counts.zst" and run.counted == [0, 1, 2, 3]
    assert run.frame_bytes == before + durable["bytes"]
    assert run.sampled == [1] and run.public()["proseg_counts"]["sampled"] == [1]
    assert R.page_frame(run.frames[1])["count"] == "sampled" and R.page_frame(old)["count"] == "averaged"
    assert not run.read_new()                                 # nothing new
    removed = spool.prune("r", keep=set())                    # what the scheduler would no longer need
    assert removed == {asked["file"], ahead["file"]}
    again = counted_frame(live, 1, reason="request", token=3, spool=spool.run_dir("r"), has_f=True)
    assert run.read_new() and run.frames[1]["counts"] == again and run.sampled == []      # its newest line rules
    spool.close()


def test_where_exports_are_kept_as_proseg_keeps_them(tmp_path):
    """where.json (read when it changes) and each counts line's "where": the 4 newest exports by token, each with its
    file's absolute path; a record naming a file that is not proseg's, or missing a field, is passed over."""
    live = tmp_path / "r" / "live"
    (live / "frames").mkdir(parents=True)
    run = R.Run(tmp_path / "r", {"id": "r", "kind": "proseg"})
    entries = {key: np.zeros(1, np.int64) for key in ("i", "j", "k", "gene", "dest")} | {"n": np.ones(1, np.int64)}
    for token in range(1, 4):
        CF.write_where_export(live, token, 0, 0, [0.5, 0.5, 2.5, 2.5], GRID, entries)
        assert run.read_new() and run.where_served == token
    assert [w["token"] for w in run.wheres] == [1, 2, 3] and run.wheres[-1]["path"] == str(live / "where-3.bin.zst")
    assert held.read_where(run.wheres[-1]).n.tolist() == [1]
    assert not run.read_new()                                  # unchanged: not read again
    (live / "frames.jsonl").write_text(json.dumps(frame_record(0)) + "\n")
    export = CF.write_where_export(live, 5, 0, 0, [0.5, 0.5, 3.5, 3.5], GRID, entries)
    (live / "where.json").unlink()                             # (its line carries it too)
    counted_frame(live, 0, reason="request", token=1, where=export)
    assert run.read_new() and [w["token"] for w in run.wheres] == [1, 2, 3, 5] and run.where_served == 5
    CF.write_where_export(live, 6, 0, 0, [0.5, 0.5, 3.5, 3.5], GRID, entries)
    assert run.read_new() and [w["token"] for w in run.wheres] == [2, 3, 5, 6]
    for bad in ({**export, "token": 7, "file": "../run.json"}, {**export, "token": 7, "i0": "0"},
                {key: value for key, value in export.items() if key != "x0"} | {"token": 7}):
        (live / "where.json").write_text(json.dumps(bad))
        run.read_new()
        assert [w["token"] for w in run.wheres] == [2, 3, 5, 6], bad


def test_the_control_file_carries_the_newest_tokens(config, tmp_path):
    """`counts N` and `where N x0 y0 x1 y1` (2 decimals) after the other lines, each only once one was issued."""
    runs = make_runs(config)
    (tmp_path / "r").mkdir()
    run = R.Run(tmp_path / "r", {"id": "r", "kind": "proseg", "paused": True, "speed": "max"})
    runs._write_control(run)
    assert not any(line.startswith(("counts", "where")) for line in (run.live / "control").read_text().splitlines())
    run.meta.update(counts_token=7, where_token=8, where_box=[10, 20, 110.5, 120.25])
    runs._write_control(run)
    lines = (run.live / "control").read_text().splitlines()
    assert lines[:2] == ["paused 1", "step 0"] and lines[-2:] == ["counts 7", "where 8 10.00 20.00 110.50 120.25"]
    assert R.where_box([10.004, "20", 110.4999, 120.25]) == [10.0, 20.0, 110.5, 120.25]
    for bad, words in (([0, 0, 0, 1], "The box is empty"), ([0, 0, 0.004, 1], "The box is empty"),
                       ([0, 0, float("nan"), 1], "The box is empty"), ([0, 0, 1e31, 1], "The box is empty"),
                       ([0, 0, 1], "box must be"), ([0, 0, "x", 1], "box must be"), (None, "box must be")):
        with pytest.raises(R.RunError, match=words) as refused:
            R.where_box(bad)
        assert refused.value.status == 422


# ---- tokens and where requests against the fake proseg-live ---------------------------------------------------------
def test_a_counts_token_is_served_by_the_paused_state_or_the_next_frame(config, monkeypatch):
    """request_counts: while paused, proseg counts nothing new (its paused state's line is the count), so a token is
    served at once; while sampling, the next frame is counted for it (reason request, its line carrying the token),
    and until then the same token is answered (one at a time), unless it is TOKEN_STALE_S old."""
    runs = make_runs(config)
    try:
        run = created(runs, "slow")
        run_id = run.meta["id"]
        status = runs.counts_status(run_id)
        assert status.keys() == STATUS_KEYS and status["mode"] == "request" and status["newest"] == 0
        assert (status["issued"], status["served"], status["outstanding"], status["can_issue"], status["why"]) == \
            (0, 0, False, True, None)
        assert status["extract_ms"] == 0.0 and status["bytes"] == run.frames[0]["counts"]["bytes"] > 0
        assert runs.request_counts(run_id) == 1
        assert runs.counts_status(run_id)["served"] == 1 and runs.request_counts(run_id) == 2
        assert "counts 2" in (run.live / "control").read_text().splitlines()
        runs.control(run_id, "resume")
        token = runs.request_counts(run_id)
        assert token == 3 and runs.request_counts(run_id) == 3 and runs.counts_status(run_id)["outstanding"]
        wait_for(lambda: runs.counts_status(run_id)["served"] >= 3, timeout=30)
        [seq] = [s for s in counted(run) if run.frames[s]["counts"].get("token") == 3]
        assert run.frames[seq]["counts"]["reason"] == "request" and seq > 0
        assert run.frames[seq]["counts_file"] == str(runs.spool.folder / run_id / f"{seq:06d}.counts.zst")
        status = runs.counts_status(run_id)
        assert status["newest"] == max(counted(run)) >= seq and not status["outstanding"]
        assert status["newest_at"] <= time.monotonic() and status["spool_mb"] > 0
        assert run.public()["proseg_counts"]["newest"] == status["newest"]
        monkeypatch.setattr(R, "TOKEN_STALE_S", 0.0)
        first = runs.request_counts(run_id)
        assert runs.request_counts(run_id) == first + 1          # one never served no longer holds a new one back
        runs.control(run_id, "abort")
        wait_for(lambda: run.meta["state"] == "stopped")
        assert runs.request_counts(run_id) is None and runs.counts_status(run_id)["why"] == "ended"
        assert runs.request_counts("nobody") is None
        with pytest.raises(R.RunError) as unknown:
            runs.counts_status("nobody")
        assert unknown.value.status == 404
    finally:
        runs.close("discard")


def test_a_where_request_is_served_at_once_while_paused_and_with_the_next_frame_while_sampling(config):
    """request_where: paused, proseg exports the box at once from its paused state (seq = the ack's), and a box that
    export covers is answered from it with no new token; sampling, the export comes with the next counted frame (its
    line carries it), and wait_s=0 only asks: where_export finds it later. Each answer is where.json's record, "path"
    added, and its file reads as proseg's pools."""
    runs = make_runs(config)
    try:
        run = created(runs, "fast")
        run_id = run.meta["id"]
        found = runs.request_where(run_id, [0.5, 0.5, 3.5, 3.5], wait_s=10)
        assert found["token"] == 1 and found["seq"] == 0 == run.ack["seq"] and found["box"] == [0.5, 0.5, 3.5, 3.5]
        assert (found["i0"], found["j0"], found["i1"], found["j1"]) == (0, 0, 3, 3)
        assert found["path"] == str(run.live / "where-1.bin.zst") and held.read_where(found).token == 1
        assert run.meta["counts_token"] == 1                     # a counts token with it: none was on its way
        lines = (run.live / "control").read_text().splitlines()
        assert lines[-2:] == ["counts 1", "where 1 0.50 0.50 3.50 3.50"]
        assert runs.request_where(run_id, [1, 1, 2, 2]) == found and run.meta["where_token"] == 1
        assert runs.where_export(run_id, 0, [1, 1, 2, 2]) == found
        assert runs.where_export(run_id, 1, [1, 1, 2, 2]) is None
        assert runs.where_export(run_id, 0, "nonsense") is None
        runs.control(run_id, "resume")
        sampling = runs.request_where(run_id, [0.5, 0.5, 2.5, 2.5], wait_s=20)
        assert sampling["token"] == 2 and sampling["seq"] > 0
        line = run.frames[sampling["seq"]]["counts"]
        assert line["reason"] == "request" and line["where"]["token"] == 2 and line["token"] == run.meta["counts_token"]
        assert runs.request_where(run_id, [1.5, 1.5, 2.5, 2.5], wait_s=0) is None and run.meta["where_token"] == 3
        wait_for(lambda: any((run.frames[s]["counts"].get("where") or {}).get("token") == 3 for s in counted(run)))
        [seq] = [s for s in counted(run) if (run.frames[s]["counts"].get("where") or {}).get("token") == 3]
        later = runs.where_export(run_id, seq, [1.5, 1.5, 2.5, 2.5])
        assert later["token"] == 3 and later["seq"] == seq and Path(later["path"]).is_file()
        with pytest.raises(R.RunError) as empty:
            runs.request_where(run_id, [2, 2, 2, 3])
        assert empty.value.status == 422
        runs.control(run_id, "abort")
        wait_for(lambda: run.meta["state"] == "stopped")
        assert runs.request_where(run_id, [0.5, 0.5, 3.5, 3.5]) is None
        assert runs.request_where("nobody", [0, 0, 1, 1]) is None
    finally:
        runs.close("discard")


def test_an_export_answers_only_a_box_it_covers_paused_or_not(config):
    """An export holds proseg's pools of its own span only (held.weights gives a transcript outside it nothing, so an
    export that misses part of the box would drop pink dots and still say exact): while paused, a box the paused
    frame's export does not cover gets an export of its own, at once; where_export hands out, of a frame's exports, the
    newest that covers the box."""
    runs = make_runs(config)
    try:
        run = created(runs, "fast")
        run_id = run.meta["id"]
        first = runs.request_where(run_id, [0.5, 0.5, 1.5, 1.5], wait_s=10)
        assert first["token"] == 1 and (first["i0"], first["j0"], first["i1"], first["j1"]) == (0, 0, 1, 1)
        assert runs.where_export(run_id, 0, [1, 1, 1.5, 1.5]) == first
        assert runs.where_export(run_id, 0, [2.5, 2.5, 3.5, 3.5]) is None                # outside its span
        other = runs.request_where(run_id, [2.5, 2.5, 3.5, 3.5], wait_s=10)
        assert other["token"] == 2 and other["seq"] == 0 and (other["i0"], other["j0"], other["i1"], other["j1"]) == \
            (2, 2, 3, 3) and run.meta["where_token"] == 2
        assert runs.where_export(run_id, 0, [2.5, 2.5, 3.5, 3.5]) == other
        assert runs.where_export(run_id, 0, [1, 1, 1.5, 1.5]) == first                   # the newest covering it
        assert runs.where_export(run_id, 0, [1, 1, 3, 3]) is None                        # neither covers it all
    finally:
        runs.close("discard")


def test_a_where_request_while_held_is_answered_at_once_from_the_held_state(config):
    """Held at the end of its schedule (paused, for the page): proseg exports the box at once from the held state,
    whose count is the hold's; a box that export covers is answered from it, with no new token."""
    runs = make_runs(config)
    try:
        run = start(runs, SHORT)
        wait_for(lambda: run.meta.get("schedule_done") and run.meta.get("state") == "paused"
                 and (run.ack or {}).get("state") == "paused" and "counts" in run.frames[run.ack["seq"]], timeout=60)
        seq = run.ack["seq"]
        started = time.monotonic()
        found = runs.request_where(run.meta["id"], [0.5, 0.5, 3.5, 3.5], wait_s=10)
        assert found["seq"] == seq and found["token"] == 1 and time.monotonic() - started < 5
        assert run.frames[seq]["counts"]["reason"] == "hold"
        assert runs.request_where(run.meta["id"], [1, 1, 2, 2], wait_s=10) == found and run.meta["where_token"] == 1
    finally:
        runs.close("discard")


def test_a_full_spool_stops_new_counts_but_never_a_paused_state(config, monkeypatch):
    """With the run's spool folder at its cap (or the engine's, or its disk low), no counted frame is asked of proseg
    while it samples (the scheduler waits for pruning), and counts_status says why; while paused, tokens are still
    served (proseg writes nothing new for them). A low disk is said once in the engine's log, until there is room."""
    runs = make_runs(config)
    noted = []
    monkeypatch.setattr(runs, "note", lambda text, run_id=None: noted.append((text, run_id)))
    try:
        run = created(runs, "slow")
        run_id = run.meta["id"]
        refusal = {"why": "run_cap"}
        monkeypatch.setattr(runs.spool, "can_issue", lambda rid: (refusal["why"] is None, refusal["why"]))
        assert runs.request_counts(run_id) == 1                  # paused: served by the paused state
        assert runs.counts_status(run_id)["can_issue"] is True
        runs.control(run_id, "resume")
        assert runs.request_counts(run_id) is None and runs.request_where(run_id, [0, 0, 2, 2], wait_s=0) is None
        status = runs.counts_status(run_id)
        assert (status["can_issue"], status["why"], status["issued"]) == (False, "run_cap", 1)
        refusal["why"] = "low_space"
        assert runs.request_counts(run_id) is None and runs.counts_status(run_id)["why"] == "low_space"
        assert len(noted) == 1 and "proseg's own count" in noted[0][0] and noted[0][1] == run_id
        refusal["why"] = None
        assert runs.request_counts(run_id) == 2
        refusal["why"] = "low_space"
        assert runs.counts_status(run_id)["why"] == "low_space" and len(noted) == 2     # low again: said again
    finally:
        runs.close("discard")


def test_a_low_disk_is_said_once_while_it_stays_low(config, monkeypatch):
    """The spool's disk low: said once in the engine's log however often runs are refused meanwhile, for it or for a
    cap (the scheduler asks at every live pass), and again only after it has had room."""
    runs = make_runs({**config, "analysis": {**config["analysis"], "counts_spool_mb": 64}})
    noted = []
    monkeypatch.setattr(runs, "note", lambda text, run_id=None: noted.append(text))
    free = {"mb": 100.0}                                                       # under twice 64 MB
    monkeypatch.setattr(runs.spool, "free_mb", lambda: free["mb"])
    full, other = (R.Run(runs.root / name, {"id": name, "kind": "proseg"}) for name in ("full", "other"))
    with open(runs.spool.run_dir("full") / "000001.counts.zst", "wb") as stream:
        stream.truncate(70_000_000)                                            # past its 64 MB cap
    try:
        assert [runs._spool_refusal(run) for _ in range(5) for run in (full, other)] == ["run_cap", "low_space"] * 5
        assert len(noted) == 1 and "has under 128 MB free" in noted[0]
        free["mb"] = 500.0
        assert runs._spool_refusal(full) == "run_cap"                          # room again: looked at
        free["mb"] = 100.0
        assert runs._spool_refusal(other) == "low_space" and len(noted) == 2  # low again: said again
    finally:
        runs.close("discard")


def test_a_start_or_a_restart_serves_tokens_from_none(config, tmp_path):
    """Proseg serves its tokens from 0 at each start: tokens the run's record held from an earlier start are dropped
    before proseg reads its control file, so nothing is counted for them."""
    runs = make_runs(config)
    path, count = shared_file(runs, tmp_path)
    run = registered(runs, "square", path, count, counts_token=5, where_token=2, where_box=[0, 0, 2, 2])
    run.wheres, run.counts_served = [{"token": 9}], 9
    executed(runs, run)
    assert run.meta["state"] == "done" and "counts_token" not in run.meta and "where_token" not in run.meta
    reasons = [run.frames[s]["counts"]["reason"] for s in counted(run)]
    assert reasons == ["init", "upscale", "end"] and run.wheres == [] and run.counts_served == 0
    assert not (run.live / "where.json").exists()
    runs.close("discard")


# ---- where requests and exports, proseg's side written by the test --------------------------------------------------
def test_a_where_request_nobody_serves_is_answered_none_once_its_wait_is_over(config):
    """request_where while proseg samples, with no export coming (proseg busy, or slow): its tokens are written, and
    None comes back once `wait_s` is over, never later (Go deeper waits 2 s for it, a test's view loop not at all)."""
    runs = make_runs(config)
    run = sampling(runs)
    try:
        started = time.monotonic()
        assert Waiting(lambda: runs.request_where("r", [0, 0, 1, 1], wait_s=0.3)).answer() is None
        assert time.monotonic() - started >= 0.3
        assert (run.meta["where_token"], run.meta["counts_token"]) == (1, 1)
        assert (run.live / "control").read_text().splitlines()[-2:] == ["counts 1", "where 1 0.00 0.00 1.00 1.00"]
    finally:
        gone(runs, run)


def test_a_where_token_on_its_way_holds_a_new_one_back_until_it_is_served(config, monkeypatch):
    """Proseg serves only the newest where token: while one is on its way, a request for another box waits for it
    (within its wait_s) and asks for nothing, so the first caller's export is not lost to the second; once an export
    carries it, the next box gets a token of its own (with no counts token while one is on its way); a token
    TOKEN_STALE_S old holds nothing back."""
    runs = make_runs(config)
    run = sampling(runs)
    try:
        assert runs.request_where("r", [0, 0, 1, 1], wait_s=0) is None
        assert (run.meta["where_token"], run.meta["where_box"], run.meta["counts_token"]) == (1, [0, 0, 1, 1], 1)
        started = time.monotonic()
        assert Waiting(lambda: runs.request_where("r", [2, 2, 3, 3], wait_s=0.2)).answer() is None
        assert time.monotonic() - started >= 0.2
        assert (run.meta["where_token"], run.meta["where_box"]) == (1, [0, 0, 1, 1])
        assert (run.live / "control").read_text().splitlines()[-1] == "where 1 0.00 0.00 1.00 1.00"
        CF.write_where_export(run.live, 1, 0, 0, [0, 0, 1, 1], GRID, ENTRIES)         # proseg serves it
        assert run.read_new() and run.where_served == 1
        assert runs.request_where("r", [2, 2, 3, 3], wait_s=0) is None
        assert (run.meta["where_token"], run.meta["where_box"], run.meta["counts_token"]) == (2, [2, 2, 3, 3], 1)
        assert (run.live / "control").read_text().splitlines()[-2:] == ["counts 1", "where 2 2.00 2.00 3.00 3.00"]
        monkeypatch.setattr(R, "TOKEN_STALE_S", 0.0)
        assert runs.request_where("r", [1, 1, 2, 2], wait_s=0) is None
        assert (run.meta["where_token"], run.meta["counts_token"]) == (3, 2)       # stale: both asked again
    finally:
        gone(runs, run)


def test_an_export_is_handed_out_only_with_its_file_and_its_frames_count(config):
    """An export is handed out (where_export, and request_where for its token) only once its frame's counts line is
    merged (a cell's keep comes from that count) and while its file is there (proseg keeps its 4 newest): of a frame's
    exports covering the box, the newest that has both, else None."""
    runs = make_runs(config)
    run = sampling(runs)
    wide, box = [0.5, 0.5, 2.5, 2.5], [1, 1, 2, 2]

    def served(token: int, seq: int, asked=wide) -> dict:
        """The engine asks (wait_s=0: never waits) and proseg exports the box with frame `seq`: where.json first."""
        assert runs.request_where("r", asked, wait_s=0) is None and run.meta["where_token"] == token
        export = CF.write_where_export(run.live, token, seq, seq * 10, asked, GRID, ENTRIES)
        assert run.read_new() and run.where_served == token
        return export
    try:
        (run.live / "frames.jsonl").write_text("".join(json.dumps(frame_record(s)) + "\n" for s in (0, 1, 2)))
        counted_frame(run.live, 0, reason="init")
        served(1, 0)
        served(2, 0)
        assert runs.where_export("r", 0, box)["token"] == 2
        (run.live / "where-2.bin.zst").unlink()                                       # proseg deleted it
        assert runs.where_export("r", 0, box)["token"] == 1
        (run.live / "where-1.bin.zst").unlink()
        assert runs.where_export("r", 0, box) is None
        export = served(3, 1)                                                         # frame 1's: not counted yet
        assert runs.where_export("r", 1, box) is None
        counted_frame(run.live, 1, reason="request", token=run.meta["counts_token"], where=export)
        assert run.read_new() and runs.where_export("r", 1, box)["token"] == 3
        # request_where waits for its token's export and for that export's frame's count, both.
        waiting = Waiting(lambda: runs.request_where("r", box, wait_s=10))
        wait_for(lambda: run.meta.get("where_token") == 4)
        export = CF.write_where_export(run.live, 4, 2, 20, box, GRID, ENTRIES)
        assert run.read_new() and run.where_served == 4
        time.sleep(0.3)
        assert waiting.thread.is_alive()                                              # frame 2 has no count yet
        counted_frame(run.live, 2, reason="request", token=run.meta["counts_token"], where=export)
        assert run.read_new()
        found = waiting.answer()
        assert (found["token"], found["seq"], found["path"]) == (4, 2, str(run.live / "where-4.bin.zst"))
    finally:
        gone(runs, run)


def test_a_runs_public_view_names_its_64_newest_counted_frames(tmp_path):
    """... and the 64 newest of them read on the count proseg saves (`sampled`), each in order."""
    run = R.Run(tmp_path / "r", {"id": "r", "kind": "proseg"})
    run.counted = list(range(200))
    run.sampled = list(range(0, 200, 2))
    assert run.public()["proseg_counts"] == {"reason": None, "newest": 199, "recent": list(range(136, 200)),
                                             "sampled": list(range(72, 200, 2))}


def test_a_runs_public_view_names_the_constants_of_its_mode(tmp_path):
    """Run.public()["constants"]: the mixing check's constants block of the run's mode (qc_constants.run_mode: tiled by
    its meta or by its prior's tiles, a whole sample from 20,000 cells; a region until info.json is read), with the
    constants kept from the outlines' measurement there named (`kept`: the page's caveats on Go deeper's words)."""
    from sopa_companion.engine import qc_constants as K
    run = R.Run(tmp_path / "r", {"id": "r", "kind": "proseg"})
    region = {"mode": "region", "id": K.CONSTANTS_ID, "measured": True, "kept": []}
    tiled = {"mode": "tiled", "id": K.CONSTANTS_ID, "measured": True, "kept": ["glow", "where_words"]}
    whole = {"mode": "whole", "id": K.CONSTANTS_ID, "measured": False,
             "kept": ["glow", "where_words", "start_band", "min_counts", "like_depth"]}
    assert run.public()["constants"] == region
    run.info = {"cells": 4823}
    assert run.public()["constants"] == region
    run.info = {"cells": 4823, "prior": {"format": "tiles"}}
    assert run.public()["constants"] == tiled
    run.info, run.meta["mode"] = {"cells": 4823}, "tiled"
    assert run.public()["constants"] == tiled
    run.info = {"cells": 20000}
    assert run.public()["constants"] == whole
