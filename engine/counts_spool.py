"""The spool of proseg's own counts: where proseg-live writes the counted frames the engine asks for (request frames)
and the states a pause stops on (`--live-counts-dir`), apart from the run's folder, which keeps only the durable ones
(init, upscale, hold, finish, end: native/proseg/src/live.rs).

  <root>/sopa-companion-counts-<engine pid>-<random>/     this engine's, made at its start and removed at its close
      <run id>/NNNNNN.counts.zst                          one run's: made with the run, removed with it

The root is the RAM disk (/dev/shm) when it is a folder this user may write, else the system's temporary folder: the
rule engine/analysis.py transient_dir keeps for the per-bin sums. A RAM disk with less than twice a run's cap free once
what gone engines left there is swept (a container's 64 MB one, say) would never take a frame, so the temporary
folder is the root then too (ram_free_mb; said in the engine's log). A run's folder is capped (`[analysis]
counts_spool_mb`, 1,024 MB) and the engine's at twice that. While either would pass its cap with one more frame, or the
root's disk has less than twice a run's cap free, no new counted frame is asked for (can_issue: Runs.request_counts,
request_where); proseg's own forced frames still come. The scheduler (engine/analysis.py) frees room by pruning what it
has analysed (prune), and only files whose counts lines the engine has read (note), never one proseg has just written.
Folders that engines which are gone left (a crash) are removed at the next start, this user's only: in the root, and
with no root given in the temporary folder too, where such an engine spooled if it passed the RAM disk over, whichever
root this one takes.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile
import threading

from .. import portable

SPOOL_PREFIX = "sopa-companion-counts-"
RAM_DISK = "/dev/shm"
# A private root for the spool instead of the RAM disk (the tests' own folder): read by engine/runs.py Runs.
ROOT_ENV = "SOPA_COMPANION_COUNTS_DIR"
COUNTS_FILE = re.compile(r"[0-9]{6,9}\.counts\.zst")      # a counted frame's file as proseg-live names it
MB = 1e6


def default_root() -> Path:
    """The RAM disk when it is a folder this user may write, else the system's temporary folder."""
    return Path(RAM_DISK if os.path.isdir(RAM_DISK) and os.access(RAM_DISK, os.W_OK) else tempfile.gettempdir())


def _real(path) -> str:
    """A path as one file has it whatever names lead there (a root reached through a link)."""
    return os.path.realpath(str(path))


def _free_mb(path) -> float:
    """MB free on the disk of `path` (unbounded when it cannot be told)."""
    try:
        return shutil.disk_usage(path).free / MB
    except OSError:
        return float("inf")


def sweep(root) -> list[Path]:
    """Remove the spool folders in `root` that engines which are gone left (their pid is no process), this user's only
    (another user's, a live engine's and anything not named as ours are left as they are): the folders removed."""
    removed = []
    try:
        found = list(Path(root).glob(SPOOL_PREFIX + "*"))
    except OSError:
        return removed
    for folder in found:
        try:
            if folder.is_symlink() or not folder.is_dir() or (not portable.WINDOWS and
                                                              folder.stat().st_uid != os.getuid()):
                continue                                    # (Windows: the user's own temporary folder)
            portable.probe(int(folder.name[len(SPOOL_PREFIX):].split("-")[0]))
        except ProcessLookupError:                          # its engine is gone
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(folder)
        except (ValueError, OSError):                       # alive (or not ours to judge): kept
            continue
    return removed


class CountsSpool:
    """One engine's spool (the module docstring). The runs' threads, the scheduler and the routes all use it."""

    def __init__(self, root=None, run_cap_mb: int = 1024):
        self.root = Path(root) if root is not None else default_root()
        self.run_cap_mb = int(run_cap_mb)
        self.engine_cap_mb = 2 * self.run_cap_mb
        self.lock = threading.Lock()
        self.known: dict[str, dict] = {}           # run id -> {real path: the path its counts line gave} (note)
        self.peaks: dict[str, float] = {}          # run id -> the most its folder held (MB), looked at with each line
        self.ram_free_mb: float | None = None      # the RAM disk's free MB when it was passed over for want of room
        self.swept: list[Path] = []                # the folders of engines that are gone this start removed (sweep)
        self._swept_roots: set[str] = set()        # the folders it swept them from (real paths), each once
        self._sweep(self.root)
        if root is None:
            # An engine given no root may have spooled in the system's temporary folder instead (it passed a RAM disk
            # without room over, or the RAM disk took no folder): what one that is gone left there goes too, whichever
            # root this one takes.
            self._sweep(tempfile.gettempdir())
        if root is None and str(self.root) == RAM_DISK and _free_mb(self.root) < 2 * self.run_cap_mb:
            # A RAM disk that would never take a frame (low_space), measured once the sweep has removed what engines
            # that are gone left there: the system's temporary folder instead, said in the engine's log.
            self.ram_free_mb = _free_mb(self.root)
            self.root = Path(tempfile.gettempdir())
            print(f"SoPA companion: proseg's counts are spooled in {self.root}: {RAM_DISK} has "
                  f"{self.ram_free_mb:,.0f} MB free, under twice a run's cap ({2 * self.run_cap_mb:,} MB)", flush=True)
        try:
            self.folder = Path(tempfile.mkdtemp(prefix=f"{SPOOL_PREFIX}{os.getpid()}-", dir=self.root))
        except OSError:                             # a root that takes nothing: the system's temporary folder
            self.root = Path(tempfile.gettempdir())
            self._sweep(self.root)
            self.folder = Path(tempfile.mkdtemp(prefix=f"{SPOOL_PREFIX}{os.getpid()}-", dir=self.root))

    def _sweep(self, root) -> None:
        """What engines that are gone left in `root` goes (sweep), once a start whatever names lead there."""
        if _real(root) not in self._swept_roots:
            self._swept_roots.add(_real(root))
            self.swept += sweep(root)

    def _path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or run_id in ("", ".", "..") or "/" in run_id or os.sep in run_id \
                or "\x00" in run_id:
            raise ValueError(f"not a run's id: {run_id!r}")
        return self.folder / run_id

    def run_dir(self, run_id: str) -> Path:
        """The run's folder (proseg-live's --live-counts-dir), made when it is not there: never the engine's folder once
        it is closed (OSError)."""
        path = self._path(run_id)
        path.mkdir(exist_ok=True)
        return path

    def files(self, run_id: str) -> list[str]:
        """The absolute paths of the run's counted frames in the spool (NNNNNN.counts.zst), in their order."""
        folder = self._path(run_id)
        try:
            names = os.listdir(folder)
        except OSError:
            return []
        return [str(folder / name) for name in sorted((name for name in names if COUNTS_FILE.fullmatch(name)),
                                                      key=lambda name: int(name.split(".", 1)[0]))]

    def _bytes(self, run_id: str | None = None) -> int:
        """What the run's folder holds (every file in it, proseg's half-written ones too), or the engine's."""
        folders = [self._path(run_id)] if run_id is not None else self._folders()
        total = 0
        for folder in folders:
            try:
                with os.scandir(folder) as entries:
                    for entry in entries:
                        try:
                            if entry.is_file(follow_symlinks=False):
                                total += entry.stat(follow_symlinks=False).st_size
                        except OSError:             # gone meanwhile (pruned, or proseg's rename)
                            continue
            except OSError:
                continue
        return total

    def _folders(self) -> list[Path]:
        try:
            return [path for path in self.folder.iterdir() if path.is_dir() and not path.is_symlink()]
        except OSError:
            return []

    def usage_mb(self, run_id: str | None = None) -> float:
        """MB the run's folder holds, or with no run the engine's (every run's)."""
        return self._bytes(run_id) / MB

    def peak_mb(self, run_id: str) -> float:
        """The most the run's folder held, as looked at with each counts line read (note)."""
        with self.lock:
            return self.peaks.get(run_id, 0.0)

    def free_mb(self) -> float:
        """MB free on the spool's disk (unbounded when it cannot be told)."""
        return _free_mb(self.folder)

    def low_space(self) -> bool:
        """The spool's disk has less than twice a run's cap free: no frame is asked for (can_issue)."""
        return self.free_mb() < 2 * self.run_cap_mb

    def can_issue(self, run_id: str) -> tuple[bool, str | None]:
        """Whether one more counted frame may be asked for the run: (False, "run_cap") when the run's folder would pass
        its cap with a frame the size of its newest, "engine_cap" when the engine's would pass twice that, "low_space"
        while the spool's disk has less than twice a run's cap free; else (True, None). An empty folder takes one frame
        whatever its size."""
        run, engine = self._bytes(run_id), self._bytes(None)
        newest = self.files(run_id)
        try:
            frame = os.stat(newest[-1]).st_size if newest else 0
        except OSError:
            frame = 0
        if run and run + frame > self.run_cap_mb * MB:
            return False, "run_cap"
        if engine and engine + frame > self.engine_cap_mb * MB:
            return False, "engine_cap"
        if self.low_space():
            return False, "low_space"
        return True, None

    def note(self, run_id: str, path) -> None:
        """A counted frame of the run whose counts line the engine has read (Run.read_new): prune may remove it from now
        on. A file outside the run's folder (another engine's spool, a run folder) is not the spool's."""
        key = _real(path)
        if os.path.dirname(key) != _real(self._path(run_id)):
            return
        with self.lock:
            self.known.setdefault(run_id, {})[key] = str(path)
        held = self.usage_mb(run_id)
        with self.lock:
            self.peaks[run_id] = max(self.peaks.get(run_id, 0.0), held)

    def prune(self, run_id: str, keep: set[str]) -> set[str]:
        """Remove the run's counted frames that are not in `keep` (absolute paths), of those whose counts lines the
        engine has read (note): a frame proseg has just written, its line not read yet, stays. -> the paths removed,
        as their counts lines name them."""
        keep = {_real(path) for path in keep}
        with self.lock:
            known = dict(self.known.get(run_id, {}))
        removed = {}
        for path in self.files(run_id):
            key = _real(path)
            if key in keep or key not in known:
                continue
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                continue
            removed[key] = known[key]
        with self.lock:
            left = self.known.get(run_id, {})
            for key in removed:
                left.pop(key, None)
        return set(removed.values())

    def remove_run(self, run_id: str) -> None:
        """The run's folder and everything in it go (the run is deleted, or proseg starts it again)."""
        with self.lock:
            self.known.pop(run_id, None)
            self.peaks.pop(run_id, None)
        shutil.rmtree(self._path(run_id), ignore_errors=True)

    def close(self) -> None:
        """The engine's folder goes, with every run's in it."""
        shutil.rmtree(self.folder, ignore_errors=True)
