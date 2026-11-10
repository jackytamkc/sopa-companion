"""The previous version's locks (the package proseg_live), kept as it took them, for tests/test_lock_compat.py: an
engine of that version may run beside this one on a computer (a desktop install not updated yet, a notebook package not
reinstalled yet), and the two must keep each other out of a sample's save and of GPU 0's fit.

_lock_dir, _lock_path and _store_lock are zarr_bridge/helper.py's at commit 1c5d968, word for word but for `progress`;
the fit lock is engine/qc.py's FIT_LOCK and how QcModels._fit_lock took it; a read is analysis/qc_trainer.py's shared
lock on the sample's lock file. Run in a process of its own, with HOME (and only HOME) naming a test's folder:

    python legacy_locks.py hold <zarr> <seconds>    a save's lock, held (prints "held", then "released")
    python legacy_locks.py save <zarr>              a save's lock, waited for (prints "saving" once it has it)
    python legacy_locks.py try <zarr>               the save lock tried without waiting ("busy" or "free")
    python legacy_locks.py read <zarr>              a read's shared lock, waited for ("reading")
    python legacy_locks.py fit [<seconds>]          the computer-wide fit lock tried ("busy", or "free" and held)
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import time

LOCK_DIR_ENV = "PROSEG_LIVE_ZARR_LOCK_DIR"    # tests point it at a private folder
FIT_LOCK = Path.home() / ".cache" / "proseg_live" / "qc-fit.lock"


def progress(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _lock_dir(create: bool = True) -> Path:
    """The save locks shared by every Proseg Live engine and helper of this user."""
    folder = Path(os.environ.get(LOCK_DIR_ENV) or Path.home() / ".cache" / "proseg_live" / "zarr-locks")
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def _lock_path(path: Path) -> Path:
    """One lock per zarr store."""
    return _lock_dir() / (hashlib.sha1(str(path).encode()).hexdigest() + ".lock")


@contextmanager
def _store_lock(path: Path):
    target = _lock_path(path)
    while True:
        handle = open(target, "w")      # creates it, or truncates it: the new mtime tells prune_locks it is in use
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                progress("Waiting for another save into this sample to finish…")
                fcntl.flock(handle, fcntl.LOCK_EX)
            # prune_locks may have removed the file between our open and our lock: a lock on that orphan excludes
            # nobody who opens the path now. Only the file still at the path counts.
            mine, current = os.fstat(handle.fileno()), os.stat(target)
            if (mine.st_dev, mine.st_ino) == (current.st_dev, current.st_ino):
                break
        except FileNotFoundError:
            pass
        except BaseException:
            handle.close()
            raise
        handle.close()
    try:
        yield
    finally:
        handle.close()


def say(word: str) -> None:
    print(word, flush=True)


def main(argv: list[str]) -> int:
    command = argv[1]
    if command == "fit":
        FIT_LOCK.parent.mkdir(parents=True, exist_ok=True)
        with open(FIT_LOCK, "a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                say("busy")
                return 0
            say("free")
            time.sleep(float(argv[2]) if len(argv) > 2 else 0)
        return 0
    zarr = Path(argv[2])
    if command == "hold":
        with _store_lock(zarr):
            say("held")
            time.sleep(float(argv[3]))
        say("released")
    elif command == "save":
        with _store_lock(zarr):
            say("saving")
    elif command == "try":
        with open(_lock_path(zarr), "w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                say("free")
            except BlockingIOError:
                say("busy")
    elif command == "read":                 # as the trainer read: the lock file opened, never created
        fd = os.open(_lock_dir(create=False) / (hashlib.sha1(str(zarr).encode()).hexdigest() + ".lock"), os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            say("reading")
        finally:
            os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
