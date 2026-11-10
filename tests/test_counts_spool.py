"""The spool of proseg's own counts (engine/counts_spool.py): one folder per engine and one per run in it, on the RAM
disk when it has room for it, unless told otherwise; capped per run and per engine, and stopped while its disk is low;
pruned only of frames whose counts lines the engine has read; gone with the run (a test's own runs too, and when proseg
starts the run again) and with the engine; and what engines that are gone left there is swept at the next start (with
no root given, in the temporary folder too, whichever root it takes), this user's only."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import types

import pytest

from conftest import wait_for
from sopa_companion.engine import counts_spool as S
from test_engine import SHORT, TEST, executed, make_runs, registered, shared_file, start


def spooled(spool, run_id: str, seq: int, size: int = 1000) -> str:
    """A counted frame as proseg-live leaves it in the run's spool folder (sparse: `size` bytes long, none written)."""
    path = spool.run_dir(run_id) / f"{seq:06d}.counts.zst"
    with open(path, "wb") as stream:
        stream.truncate(size)
    return str(path)


def dead_pid() -> int:
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    return done.pid


@pytest.fixture
def temporary(tmp_path):
    """The system's temporary folder as the spool sees it (tempfile.gettempdir()), the test's own: an engine given no
    root sweeps it, so a test never sweeps the computer's, and what is there is only what the test puts there. It
    outlasts a test's monkeypatch.undo()."""
    folder = tmp_path / "tmp"
    folder.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tempfile, "tempdir", str(folder))
        yield folder


def test_one_folder_per_engine_and_one_per_run(tmp_path):
    spool = S.CountsSpool(tmp_path, run_cap_mb=64)
    assert spool.folder.parent == tmp_path and spool.folder.name.startswith(f"{S.SPOOL_PREFIX}{os.getpid()}-")
    assert spool.folder.stat().st_mode & 0o777 == 0o700                     # the user's own: counts of their samples
    assert (spool.run_cap_mb, spool.engine_cap_mb) == (64, 128)
    run = spool.run_dir("r1")
    assert run == spool.folder / "r1" and run.is_dir() and spool.run_dir("r1") == run
    a, b = spooled(spool, "r1", 3), spooled(spool, "r1", 12, 2000)
    (run / ".000013.counts.zst.tmp").write_bytes(b"y" * 500)                # proseg still writing one
    (run / "000014.counted.zst").write_bytes(b"z")                          # not a counted frame
    assert spool.files("r1") == [a, b] and spool.files("nobody") == []
    assert spool.usage_mb("r1") == pytest.approx(3501 / 1e6) and spool.usage_mb() == pytest.approx(3501 / 1e6)
    spooled(spool, "r2", 0, 4000)
    assert spool.usage_mb() == pytest.approx(7501 / 1e6) and spool.usage_mb("nobody") == 0
    late = spooled(spool, "r1", 1_000_000, 0)                               # past 999,999: seven digits
    assert spool.files("r1") == [a, b, late]
    for bad in ("", ".", "..", "a/b", "a\x00b", None):
        with pytest.raises(ValueError):
            spool.run_dir(bad)
    spool.remove_run("r1")
    assert not run.exists() and spool.files("r1") == [] and (spool.folder / "r2").is_dir()
    spool.remove_run("r1")                                                  # gone already: nothing to do
    spool.close()
    assert not spool.folder.exists()
    with pytest.raises(OSError):                                            # a closed engine's spool takes no run
        spool.run_dir("r3")


def test_the_spool_is_on_the_ram_disk_when_this_user_may_write_there(tmp_path, temporary, monkeypatch):
    ram = tmp_path / "shm"
    ram.mkdir()
    monkeypatch.setattr(S, "RAM_DISK", str(ram))
    assert S.default_root() == ram
    spool = S.CountsSpool()
    assert spool.root == ram and spool.folder.parent == ram
    spool.close()
    monkeypatch.setattr(S, "RAM_DISK", str(tmp_path / "none"))
    assert S.default_root() == temporary
    monkeypatch.setattr(S, "RAM_DISK", str(ram))
    monkeypatch.setattr(os, "access", lambda path, mode: False)             # there, but not this user's to write
    assert S.default_root() == temporary
    monkeypatch.undo()
    # A root that takes no folder (here a file): the system's temporary folder instead, never no engine; what engines
    # that are gone left there is swept then, as this one spools there.
    (tmp_path / "file").write_text("")
    left = temporary / f"{S.SPOOL_PREFIX}{dead_pid()}-abcd"
    left.mkdir()
    spool = S.CountsSpool(tmp_path / "file")
    assert spool.root == temporary and spool.folder.parent == spool.root and spool.swept == [left]
    spool.close()


def test_a_ram_disk_without_room_for_the_spool_is_passed_over(tmp_path, temporary, monkeypatch, capsys):
    """A RAM disk with less than twice a run's cap free (a container's 64 MB one, say) would never take a frame
    (low_space): the system's temporary folder is the root instead (ram_free_mb says what it had; so does the engine's
    log), its room measured once what gone engines left there is swept (that may be what fills it). An engine given no
    root sweeps the temporary folder too whichever root it takes: one that is gone may have passed the RAM disk over.
    A root given is used whatever its room, and swept alone."""
    ram = tmp_path / "shm"
    ram.mkdir()
    monkeypatch.setattr(S, "RAM_DISK", str(ram))
    left = ram / f"{S.SPOOL_PREFIX}{dead_pid()}-abcd"
    left.mkdir()
    free = {"mb": 127.0}
    usage = shutil.disk_usage

    def disk_usage(path):            # the RAM disk's room: what is free, less what a gone engine's folder takes
        if not str(path).startswith(str(ram)):
            return usage(path)
        return types.SimpleNamespace(free=(free["mb"] - (500 if left.exists() else 0)) * 1e6)
    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    capsys.readouterr()
    spool = S.CountsSpool(run_cap_mb=64)
    assert spool.swept == [left] and spool.root == temporary and spool.ram_free_mb == 127.0
    assert spool.folder.parent == spool.root
    assert capsys.readouterr().out == (f"SoPA companion: proseg's counts are spooled in {spool.root}: {ram} has 127 MB "
                                       "free, under twice a run's cap (128 MB)\n")
    spool.close()
    free["mb"] = 128.0
    left.mkdir()                                                            # gone engines' leftovers fill it: swept
    there = temporary / f"{S.SPOOL_PREFIX}{dead_pid()}-efgh"                # one that passed the RAM disk over
    there.mkdir()
    spool = S.CountsSpool(run_cap_mb=64)
    assert spool.swept == [left, there] and not there.exists()
    assert spool.root == ram and spool.ram_free_mb is None and spool.folder.parent == ram
    spool.close()
    free["mb"] = 10.0
    there.mkdir()
    spool = S.CountsSpool(ram, run_cap_mb=64)                               # given: kept, and no frame asked for
    assert spool.root == ram and spool.ram_free_mb is None and spool.can_issue("r") == (False, "low_space")
    assert spool.swept == [] and there.is_dir()                             # given: its root alone is swept
    spool.close()
    assert capsys.readouterr().out == ""


# An engine of its own (a process) whose RAM disk has 100 MB free, under twice a 64 MB cap: it spools in the temporary
# folder, says so and where, and waits to be killed.
PASSED_OVER = """
import shutil, sys, tempfile, time, types
sys.path.insert(0, sys.argv[3])
from sopa_companion.engine import counts_spool as S
S.RAM_DISK, tempfile.tempdir = sys.argv[1], sys.argv[2]
usage = shutil.disk_usage
shutil.disk_usage = lambda path: types.SimpleNamespace(free=100e6) if str(path).startswith(sys.argv[1]) else usage(path)
spool = S.CountsSpool(run_cap_mb=64)
(spool.run_dir("r") / "000001.counts.zst").write_bytes(b"x" * 1000)
print("FOLDER", spool.folder, flush=True)
time.sleep(600)
"""


def test_a_crashed_engines_spool_is_swept_wherever_it_spooled(tmp_path, temporary, monkeypatch):
    """An engine that passed a RAM disk without room over spooled in the temporary folder, then crashed: the next one
    removes what it left there, although the RAM disk has room again and its own spool goes there."""
    ram = tmp_path / "shm"
    ram.mkdir()
    parent = Path(S.__file__).resolve().parents[2]                          # the package these tests import
    with subprocess.Popen([sys.executable, "-c", PASSED_OVER, str(ram), str(temporary), str(parent)],
                          stdout=subprocess.PIPE, text=True) as engine:
        silent = threading.Timer(60, engine.kill)                           # one that never says fails, never hangs
        silent.start()
        try:
            said = [engine.stdout.readline() for _ in range(2)]
        finally:
            silent.cancel()
            engine.kill()                                                   # it crashes: its folder is left behind
            engine.wait()
    assert said[0].startswith("SoPA companion: proseg's counts are spooled in ") and said[1].startswith("FOLDER "), said
    folder = Path(said[1].split(" ", 1)[1].strip())
    assert folder.parent == temporary and (folder / "r" / "000001.counts.zst").is_file()
    monkeypatch.setattr(S, "RAM_DISK", str(ram))
    usage = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda path: types.SimpleNamespace(free=10e9)
                        if str(path).startswith(str(ram)) else usage(path))
    spool = S.CountsSpool(run_cap_mb=64)                                    # the next engine: the RAM disk has room
    assert spool.root == ram and spool.ram_free_mb is None and spool.folder.parent == ram
    assert spool.swept == [folder] and not folder.exists()
    spool.close()


def test_the_start_sweeps_what_gone_engines_left_this_users_only(tmp_path, monkeypatch):
    """At an engine's start: a folder of this prefix whose engine is gone (its pid is no process) and that is this
    user's is removed; a live engine's, another user's, a link and a name that holds no pid are left as they are."""
    gone = dead_pid()
    left = tmp_path / f"{S.SPOOL_PREFIX}{gone}-abcd"
    (left / "run-1").mkdir(parents=True)
    (left / "run-1" / "000001.counts.zst").write_bytes(b"x")
    alive = tmp_path / f"{S.SPOOL_PREFIX}{os.getpid()}-efgh"
    alive.mkdir()
    theirs = tmp_path / f"{S.SPOOL_PREFIX}{gone}-theirs"
    theirs.mkdir()
    named = tmp_path / f"{S.SPOOL_PREFIX}nopid-ijkl"
    named.mkdir()
    other = tmp_path / "sopa-companion-bins-1-x"
    other.mkdir()
    link = tmp_path / f"{S.SPOOL_PREFIX}{gone}-link"
    link.symlink_to(other)
    stat = Path.stat

    def owner(path, *args, **kwargs):          # another user's folder, as its owner shows it
        found = stat(path, *args, **kwargs)
        return types.SimpleNamespace(st_uid=os.getuid() + 1, st_mode=found.st_mode) \
            if path.name.endswith("-theirs") else found
    monkeypatch.setattr(Path, "stat", owner)
    spool = S.CountsSpool(tmp_path)
    monkeypatch.undo()
    assert spool.swept == [left] and not left.exists()
    assert alive.is_dir() and theirs.is_dir() and named.is_dir() and other.is_dir() and link.is_symlink()
    assert spool.folder.is_dir()
    spool.close()
    assert alive.is_dir()                       # an engine closes its own folder only


def test_no_frame_is_asked_for_past_a_cap_or_while_the_disk_is_low(tmp_path, monkeypatch):
    """can_issue: a run's folder may not pass its cap with one more frame the size of its newest, the engine's twice
    that, and none is asked for while the spool's disk has less than twice a run's cap free. An empty folder takes a
    frame whatever its size (a whole sample's may be larger than a small cap)."""
    spool = S.CountsSpool(tmp_path, run_cap_mb=64)
    free = {"mb": 10_000.0}
    monkeypatch.setattr(spool, "free_mb", lambda: free["mb"])
    assert spool.can_issue("r1") == (True, None)
    spooled(spool, "r1", 1, 70_000_000)
    assert spool.can_issue("r1") == (False, "run_cap")
    spool.remove_run("r1")
    spooled(spool, "r1", 1, 30_000_000)
    assert spool.can_issue("r1") == (True, None)                            # 30 + 30 <= 64 MB
    spooled(spool, "r1", 2, 30_000_000)
    assert spool.can_issue("r1") == (False, "run_cap")                      # 60 + 30 > 64 MB
    spooled(spool, "r2", 1, 60_000_000)
    assert spool.can_issue("r3") == (True, None)                            # 120 MB in all, nothing of r3's yet
    spooled(spool, "r3", 1, 5_000_000)
    assert spool.can_issue("r3") == (False, "engine_cap")                   # 125 + 5 > 128 MB
    spool.remove_run("r1")
    spool.remove_run("r2")
    free["mb"] = 127.0                                                      # under twice a run's cap
    assert spool.can_issue("r3") == (False, "low_space")
    free["mb"] = 128.0
    assert spool.can_issue("r3") == (True, None)
    monkeypatch.undo()
    assert spool.free_mb() > 0
    monkeypatch.setattr(shutil, "disk_usage", lambda path: (_ for _ in ()).throw(OSError("gone")))
    assert spool.free_mb() == float("inf")                                  # not known: no stop for it
    spool.close()


def test_pruning_removes_only_frames_whose_lines_were_read(tmp_path):
    """prune(run, keep): of the run's frames whose counts lines the engine has read (note), those not kept; a frame
    proseg has just written, its line not read yet, stays, and so does anything outside the run's folder. A path given
    through a link to the spool names the same file, and the paths removed are the ones the lines gave."""
    (tmp_path / "root").mkdir()
    spool = S.CountsSpool(tmp_path / "root")
    a, b, c, d = (spooled(spool, "r", seq, 1_000_000 * (seq + 1)) for seq in (1, 2, 3, 4))
    for path in (a, b, c):
        spool.note("r", path)
    spool.note("r", str(tmp_path / "elsewhere" / "000005.counts.zst"))     # not in the spool
    spool.note("other", a)                                                  # another run's folder's: not its file
    assert spool.peak_mb("r") == pytest.approx(14.0) and spool.peak_mb("other") == 0.0
    assert spool.prune("r", keep={b}) == {a, c}
    assert spool.files("r") == [b, d]
    assert spool.prune("r", set()) == {b} and spool.files("r") == [d]
    assert spool.prune("r", set()) == set() and spool.files("r") == [d]     # d's line is not read yet
    alias = tmp_path / "alias"
    alias.symlink_to(spool.folder)
    through = str(alias / "r" / Path(d).name)
    spool.note("r", through)
    assert spool.prune("r", {d}) == set()                                   # kept, named either way
    assert spool.prune("r", set()) == {through} and spool.files("r") == []
    assert spool.peak_mb("r") == pytest.approx(14.0)                        # the most it held, not what it holds
    spool.remove_run("r")
    assert spool.peak_mb("r") == 0.0
    spool.close()


# ---- the engine's spool --------------------------------------------------------------------------------------------
def test_the_engine_makes_its_spool_from_the_settings_and_closes_it(config, tmp_path, temporary, monkeypatch):
    """Runs makes the spool as it starts ([analysis] counts_spool_mb: the run's cap; the tests' own root, from
    SOPA_COMPANION_COUNTS_DIR), and its folder goes when the engine closes, a test engine's too."""
    root = Path(os.environ[S.ROOT_ENV])
    runs = make_runs({**config, "analysis": {**config["analysis"], "counts_spool_mb": 100}})
    assert runs.spool.root == root and runs.spool.folder.parent == root and runs.spool.folder.is_dir()
    assert (runs.spool.run_cap_mb, runs.spool.engine_cap_mb) == (100, 200)
    runs.close("discard")
    assert not runs.spool.folder.exists()
    tester = make_runs({**config, "proseg": {**config["proseg"], "keep_unsaved": True}})
    assert make_runs(config).spool.run_cap_mb == 1024                       # none given: 1,024 MB
    tester.close()
    assert not tester.spool.folder.exists()
    monkeypatch.delenv(S.ROOT_ENV)
    monkeypatch.setattr(S, "RAM_DISK", str(tmp_path / "ram"))
    (tmp_path / "ram").mkdir()
    assert make_runs(config).spool.root == tmp_path / "ram"                 # an engine's own: the RAM disk's rule


def test_a_runs_spool_folder_comes_with_its_proseg_and_goes_with_the_run(config):
    """proseg-live gets the run's own spool folder (--live-counts-dir), and the state a pause stops on is counted
    there (frame 0's count, init, is durable, in the run's folder); the folder goes when the run is deleted."""
    runs = make_runs(config)
    try:
        run = runs.get(runs.create(runs.samples.list()[0]["id"], [50, 50, 150, 150])["id"])
        runs.control(run.meta["id"], "speed", "max")
        folder = runs.spool.folder / run.meta["id"]
        wait_for(lambda: (run.ack or {}).get("seq") == 0 and len(run.frames) == 1 and "counts" in run.frames[0])
        assert folder.is_dir() and run.meta["live_counts"] == "request"
        assert run.frames[0]["counts_file"] == "frames/000000.counts.zst" and runs.spool.files(run.meta["id"]) == []
        runs.control(run.meta["id"], "step")
        wait_for(lambda: (run.ack or {}).get("seq") == 1 and len(run.frames) == 2 and "counts" in run.frames[1])
        assert run.frames[1]["counts"]["reason"] == "pause"
        assert run.frames[1]["counts_file"] == str(folder / "000001.counts.zst")
        assert runs.spool.files(run.meta["id"]) == [run.frames[1]["counts_file"]]
        runs.control(run.meta["id"], "abort")
        wait_for(lambda: run.meta["state"] == "stopped")
        runs.delete(run.meta["id"])
        assert not folder.exists()
    finally:
        runs.close("discard")
    assert not runs.spool.folder.exists()


def test_a_tests_run_takes_its_spool_folder_with_it(config, tmp_path, monkeypatch):
    """A test's own run (delete(later=True), its folder emptied in the background) leaves nothing in the spool: its
    spool folder goes at once. And a run's spool folder is emptied whenever proseg starts it (or starts it again: the
    retry without factorisation), before proseg starts."""
    from sopa_companion.engine import leftovers
    runs = make_runs(config)
    runs.held = lambda folders: set()
    monkeypatch.setattr(leftovers, "remove_later", lambda folder, **kwargs: None)
    path, count = shared_file(runs, tmp_path)
    run = registered(runs, "square", path, count)
    executed(runs, run)
    folder = runs.spool.folder / "square"
    assert run.meta["state"] == "done" and folder.is_dir() and run.meta["qc_test"] == TEST
    (folder / "000002.counts.zst").write_bytes(b"x")                        # a request frame's, say
    runs.delete("square", later=True)
    assert not folder.exists() and "square" not in runs.runs
    # Started again: whatever the first proseg left in its spool folder is gone before the second starts.
    seen = []
    real = runs._command

    def command(run, extra):
        seen.append(sorted(os.listdir(runs.spool.folder / run.meta["id"]))
                    if (runs.spool.folder / run.meta["id"]).exists() else None)
        return real(run, extra)
    monkeypatch.setattr(runs, "_command", command)
    again = registered(runs, "again", path, count)
    (runs.spool.run_dir("again") / "000007.counts.zst").write_bytes(b"x")
    executed(runs, again)
    assert seen == [None] and again.meta["state"] == "done"
    runs.close("discard")


def test_short_runs_keep_the_frames_proseg_counts_by_itself(config):
    """A run left alone (no token ever asked) still has proseg's own count of its first frame, its upscale and the
    hold, durable in its folder: none in the spool."""
    runs = make_runs(config)
    try:
        run = start(runs, SHORT)
        wait_for(lambda: run.meta.get("schedule_done") and (run.ack or {}).get("seq") == 8
                 and "counts" in run.frames[8], timeout=60)
        reasons = {f["seq"]: f["counts"]["reason"] for f in run.frames if "counts" in f}
        assert reasons == {0: "init", 3: "upscale", 8: "hold"}
        assert all(run.frames[seq]["counts_file"] == f"frames/{seq:06d}.counts.zst" for seq in reasons)
        assert runs.spool.files(run.meta["id"]) == []
    finally:
        runs.close("discard")
