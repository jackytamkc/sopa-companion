"""Saves, reads and fits of this version and of the previous one (the package proseg_live) keep each other out, and
this version still recognises what the previous one leaves in a sample's folder.

A save takes the sample's lock file in both lock folders, the previous version's first (names.ZARR_LOCK_FOLDERS); the
previous version takes the first alone. Its code runs here as it was (tests/data/legacy_locks.py), in a process of its
own. HOME is a test's own folder throughout, so no test opens, creates or prunes the user's real lock files."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

from sopa_companion import names
from sopa_companion.analysis import qc_trainer as T
from sopa_companion.engine import qc as Q
from sopa_companion.engine.runs import sample_locked
from sopa_companion.engine.zarr_samples import unfinished_writes
from sopa_companion.zarr_bridge import helper

LEGACY = Path(__file__).resolve().with_name("data") / "legacy_locks.py"
HOLD_S = 0.8


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home folder of the test's own, and none of the private lock variables (the shared folders are the subject)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in (names.ZARR_LOCK_DIR_ENV, names.FIT_LOCK_ENV, *names.RETIRED_ENV):
        monkeypatch.delenv(name, raising=False)
    return home


@pytest.fixture
def zarr(tmp_path):
    path = tmp_path / "sample.zarr"
    path.mkdir()
    return path.resolve()


def legacy(home: Path, *args) -> subprocess.Popen:
    """The previous version's lock code in a process of its own, with only HOME (and PATH) in its environment."""
    return subprocess.Popen([sys.executable, str(LEGACY), *map(str, args)], env={"PATH": os.environ["PATH"],
                            "HOME": str(home)}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def said(process: subprocess.Popen, timeout: float = 20) -> str:
    out, err = process.communicate(timeout=timeout)
    assert process.returncode == 0, err
    return out.strip()


def folders(home: Path) -> tuple[Path, Path]:
    return tuple(names.home_path(folder, home) for folder in names.ZARR_LOCK_FOLDERS)


def test_a_save_of_the_previous_version_keeps_a_new_save_out(home, zarr):
    old = legacy(home, "hold", zarr, HOLD_S)
    assert old.stdout.readline().strip() == "held"
    started = time.monotonic()
    with helper._store_lock(zarr):                         # waits for the old save's lock, then takes both folders'
        waited = time.monotonic() - started
        assert all(path.exists() for path in helper._lock_paths(zarr, create=False))
    assert waited > HOLD_S / 2 and said(old) == "released"
    assert [path.parent for path in helper._lock_paths(zarr, create=False)] == list(folders(home))


def test_a_new_save_keeps_a_save_of_the_previous_version_out(home, zarr):
    with helper._store_lock(zarr):
        assert said(legacy(home, "try", zarr)) == "busy"
        old = legacy(home, "save", zarr)
        time.sleep(HOLD_S)
        assert old.poll() is None                          # still waiting for the new save
    assert said(old) == "saving"
    assert said(legacy(home, "try", zarr)) == "free"


def test_reads_wait_for_saves_of_either_version_and_create_nothing(home, zarr):
    # Nothing ever saved into the sample: one read, no lock file and no folder made.
    assert T.read_consistently(zarr, lambda: "table") == "table"
    assert not (home / ".cache").exists()
    assert T.store_lock_paths(zarr) == helper._lock_paths(zarr, create=False)
    for hold in (lambda: legacy(home, "hold", zarr, HOLD_S), None):
        events = []
        if hold:                                           # a save of the previous version: its folder's file only
            old = hold()
            assert old.stdout.readline().strip() == "held"
            T.read_consistently(zarr, lambda: events.append("read"))
            assert said(old) == "released" and events == ["read"]
        else:                                              # a save of this version
            saving, done = threading.Event(), threading.Event()

            def save():
                with helper._store_lock(zarr):
                    saving.set()
                    time.sleep(HOLD_S)
                    events.append("saved")
                done.set()
            threading.Thread(target=save).start()
            saving.wait(10)
            T.read_consistently(zarr, lambda: events.append("read"))
            done.wait(10)
            assert events == ["saved", "read"]
    # The previous version's reader (a shared lock on its folder's file) waits for a new save.
    with helper._store_lock(zarr):
        reader = legacy(home, "read", zarr)
        time.sleep(HOLD_S / 2)
        assert reader.poll() is None
    assert said(reader) == "reading"


def test_saves_and_reads_of_this_version_never_deadlock(home, zarr):
    events, lock = [], threading.Lock()

    def save(name):
        with helper._store_lock(zarr):
            with lock:
                events.append(f"{name}+")
            time.sleep(0.2)
            with lock:
                events.append(f"{name}-")

    threads = [threading.Thread(target=save, args=(name,)) for name in "abc"]
    threads.append(threading.Thread(target=lambda: T.read_consistently(zarr, lambda: events.append("read"))))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
    assert not any(thread.is_alive() for thread in threads)
    saves = [event for event in events if event != "read"]
    assert all(saves[i][0] == saves[i + 1][0] and saves[i].endswith("+") for i in range(0, len(saves), 2))


def test_sample_locked_sees_a_save_of_either_version(home, zarr):
    assert sample_locked(zarr) is False and not (home / ".cache").exists()
    old = legacy(home, "hold", zarr, HOLD_S)
    assert old.stdout.readline().strip() == "held"
    assert sample_locked(zarr) is True                     # the previous version holds its folder's file only
    assert said(old) == "released"
    assert sample_locked(zarr) is False                    # its lock file stays, held by nobody
    with helper._store_lock(zarr):
        assert sample_locked(zarr) is True
    assert sample_locked(zarr) is False


def test_stale_locks_are_pruned_in_both_folders_and_held_or_recent_ones_kept(home, tmp_path):
    assert helper.prune_locks() == [] and not (home / ".cache").exists()       # pruning makes no folder
    old_folder, new_folder = folders(home)
    stale = []
    for folder in (old_folder, new_folder):
        folder.mkdir(parents=True)
        for name in ("stale", "recent"):
            path = folder / f"{name}-{folder.parent.name}.lock"
            path.write_text("")
            if name == "stale":
                os.utime(path, (time.time() - 7200, time.time() - 7200))
                stale.append(path.name)
    held_zarr = tmp_path / "held.zarr"
    held_zarr.mkdir()
    old = legacy(home, "hold", held_zarr.resolve(), HOLD_S)
    assert old.stdout.readline().strip() == "held"
    held = helper._lock_paths(held_zarr.resolve(), create=False)[0]
    os.utime(held, (time.time() - 7200, time.time() - 7200))
    assert sorted(helper.prune_locks()) == sorted(stale)
    assert held.exists() and said(old) == "released"
    assert sorted(p.name for p in old_folder.iterdir()) == sorted([held.name, f"recent-{old_folder.parent.name}.lock"])


def test_a_private_lock_folder_is_the_only_one_used(home, tmp_path, zarr, monkeypatch):
    private = tmp_path / "private"
    monkeypatch.setenv(names.ZARR_LOCK_DIR_ENV, str(private))
    assert helper._lock_dirs(create=False) == [private] and not private.exists()
    with helper._store_lock(zarr):
        assert sample_locked(zarr) is True
        assert [p.parent for p in T.store_lock_paths(zarr)] == [private]
    stale = helper._lock_paths(zarr, create=False)[0]
    os.utime(stale, (time.time() - 7200, time.time() - 7200))
    assert helper.prune_locks() == [stale.name]
    assert not (home / ".cache").exists()                  # nothing outside it, in the shared folders' place


def test_the_previous_versions_private_variables_are_refused(home, tmp_path, zarr, monkeypatch):
    """Set alone (a caller that means a private folder, but names it the old way), each old variable is refused rather
    than ignored: ignored, it would put a test's or the harness's engine on the user's real lock files."""
    monkeypatch.setenv("PROSEG_LIVE_ZARR_LOCK_DIR", str(tmp_path / "old-locks"))
    with pytest.raises(RuntimeError, match="PROSEG_LIVE_ZARR_LOCK_DIR is no longer read: .*SOPA_COMPANION_ZARR_LOCK"):
        helper._lock_dirs(create=False)
    with pytest.raises(RuntimeError, match="no longer read"):
        with helper._store_lock(zarr):
            pass
    with pytest.raises(RuntimeError, match="no longer read"):
        helper.prune_locks()
    assert sample_locked(zarr) is False
    from sopa_companion import cli
    with pytest.raises(RuntimeError, match="no longer read"):
        cli.serve({"server": {"port": 0}, "paths": {"state_dir": str(tmp_path / "state"), "data_dir": str(tmp_path)}},
                  "127.0.0.1", 0)
    assert not (tmp_path / "old-locks").exists() and not (tmp_path / "state").exists()
    assert not (home / ".cache").exists()
    monkeypatch.setenv(names.ZARR_LOCK_DIR_ENV, str(tmp_path / "new-locks"))          # both set: the new one is used
    assert helper._lock_dirs(create=False) == [tmp_path / "new-locks"]
    monkeypatch.setenv("PROSEG_LIVE_QC_FIT_LOCK", str(tmp_path / "old-fit.lock"))
    with pytest.raises(RuntimeError, match="PROSEG_LIVE_QC_FIT_LOCK is no longer read"):
        Q.fit_lock_paths()
    monkeypatch.setenv(names.FIT_LOCK_ENV, str(tmp_path / "fit.lock"))
    assert Q.fit_lock_paths() == [tmp_path / "fit.lock"]


def test_fits_of_either_version_keep_each_other_off_gpu0(home, tmp_path):
    from test_qc_models import models
    assert Q.fit_lock_paths() == [names.home_path(lock, home) for lock in names.FIT_LOCKS]
    old = legacy(home, "fit", HOLD_S * 2)                  # the previous version's fit holds its lock
    assert old.stdout.readline().strip() == "free"
    m = models(tmp_path)
    assert m._fit_lock() is None                           # no fit now
    assert said(old) == ""
    held = m._fit_lock()                                   # the data dir's lock, then both computer-wide ones
    assert [Path(handle.name) for handle in held] == [m.root / ".fit.lock", *Q.fit_lock_paths()]
    assert said(legacy(home, "fit")) == "busy"             # and the previous version's fit waits for this one
    for handle in held:
        handle.close()
    assert said(legacy(home, "fit")) == "free"
    m.close()


def test_a_save_interrupted_under_either_backups_name_is_recognised(home, zarr, tmp_path):
    """Saves set aside what they replace in the previous version's backups folder (.proseg-live-backups), the only one
    that version rolls back from; a later version's (.sopa-companion-backups) is read too."""
    assert helper.BACKUPS == ".proseg-live-backups" and names.BACKUP_FOLDERS[1] == ".sopa-companion-backups"
    later = zarr.parent / ".sopa-companion-backups" / f"{zarr.name}-20261024-120000-{os.getpid()}"
    later.mkdir(parents=True)
    journal = {"state": "writing", "path": str(zarr), "writer": helper._writer(), "moved": [],
               "written": [["shapes", "cells_2"]]}
    (later / "journal.json").write_text(json.dumps(journal))
    with pytest.raises(helper.HelperError, match="Another save into this sample is under way") as refused:
        helper._recover(zarr)                              # a live writer: refused, nothing touched
    assert refused.value.details.get("unfinished") == str(later) if hasattr(refused.value, "details") else True
    assert later.exists()
    earlier = zarr.parent / ".proseg-live-backups" / f"{zarr.name}-20261024-110000-{os.getpid()}"
    earlier.mkdir(parents=True)
    (earlier / "journal.json").write_text(json.dumps({**journal, "written": [["tables", "table_2"]]}))
    assert unfinished_writes(zarr) == {("shapes", "cells_2"), ("tables", "table_2")}
    # A committed save of a process that has ended: only its clean-up was cut short, and it goes.
    gone = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead = {"pid": int(gone.stdout), "host": socket.gethostname(), "boot": journal["writer"]["boot"], "started": None}
    (later / "journal.json").write_text(json.dumps({**journal, "state": "committed", "writer": dead}))
    (earlier / "journal.json").write_text(json.dumps({**journal, "state": "committed", "writer": dead}))
    assert helper._recover(zarr) == []
    assert not later.exists() and not earlier.exists()
    assert not any((zarr.parent / name).exists() for name in names.BACKUP_FOLDERS)
