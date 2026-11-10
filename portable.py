"""What differs between Linux, macOS and Windows in how the app's processes start, end, lock and share files.

Linux keeps exactly what it did before this module: setpriv's death signal, flock, process groups, os.kill and
os.replace. macOS and Windows get their own ways here, and nowhere else:
- a child that must end with the process that started it (setpriv --pdeathsig TERM on Linux) is watched by a reaper
  process of the starter's (tie), which ends it once the starter is gone, however it went;
- os.kill(pid, 0) is no probe on Windows (signal 0 is CTRL_C_EVENT there, a Ctrl+C to that process group, and any
  other signal ends the process): probe() is; os.killpg is signal_group();
- flock is lock()/unlock() (LockFileEx on Windows, on one byte far past any content, so the file stays readable);
- a file another process may have open is replaced with replace() (Windows refuses to replace an open file for a
  moment) and read with read_bytes()/read_text() (on Windows open so that its writer can still replace it).
Standard library only at import: the notebook's kernel imports it before anything of the engine; psutil, which the
app depends on, is imported only in the macOS and Windows code that needs it.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

LINUX = sys.platform.startswith("linux")
MACOS = sys.platform == "darwin"
WINDOWS = os.name == "nt"

# Windows' creation flags for every child the app starts: no console window of its own (none of the app's processes
# has one to share; without the flag each child would open one on the screen), and a process group of its own, so a
# console's Ctrl+C never reaches it (as a new session keeps a terminal's away on POSIX).
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
# SIGKILL where there is one; Windows ends a process the same way whatever it is sent (TerminateProcess).
SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
# os.open's flags that Windows lacks, as no flag there: a path is followed (the app's folders there are the user's
# own: private.py), and a folder is never opened with os.open (fsync_folder).
O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)          # Windows' os.open never hands a descriptor to a child anyway
O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
# Windows' os.open makes a text-mode descriptor without it (line ends rewritten, Ctrl-Z ending a read): every os.open
# of data takes it. 0 on POSIX.
O_BINARY = getattr(os, "O_BINARY", 0)
# How long replace() keeps trying on Windows while another process has the file open (they read it in microseconds).
REPLACE_WAIT_S = 2.0
# lock()'s byte on Windows, far past the content of any file the app locks (lock files and records of a few hundred
# bytes), as SQLite places its own: LockFileEx's locks keep other handles from reading what they cover, flock's never
# do. Below 4 GB, which every file system takes.
_LOCK_OFFSET = (0, 0x7FFFFFF0)                 # (high, low) 32 bits


# ---- starting a child ------------------------------------------------------------------------------------------------
def new_session() -> dict:
    """subprocess.Popen's arguments for a child in a session (POSIX) or process group (Windows) of its own, with no
    console window on Windows: start_new_session=True as before on POSIX."""
    if WINDOWS:
        return {"creationflags": CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW}
    return {"start_new_session": True}


def no_window() -> dict:
    """subprocess.Popen/run's arguments for a child that shares its starter's session: nothing on POSIX; no console
    window of its own on Windows."""
    return {"creationflags": CREATE_NO_WINDOW} if WINDOWS else {}


def child_environment(environment=None) -> dict:
    """A child's environment (default: this process's), on Windows with Python's UTF-8 mode on, so the app's helpers
    read and write their text files and pipes as UTF-8 there as on Linux and macOS (whose locales already are)."""
    environment = dict(os.environ if environment is None else environment)
    if WINDOWS:
        environment.setdefault("PYTHONUTF8", "1")
    return environment


# ---- a child that ends with its starter ------------------------------------------------------------------------------
# The reaper: a small Python process each process that ties children starts once. It reads "<pid> <start time>" lines
# on its standard input, the only copy of whose other end its starter holds; when that end closes (the starter ended,
# however: a crash, a kill, its normal exit), it ends every child it was told about that still runs (the same pid AND
# start time: never a later process given the pid), and exits. Linux never starts one: setpriv's death signal does it.
_REAPER = r"""
import os, signal, sys
import psutil
watched = []
for line in sys.stdin.buffer:
    parts = line.split()
    if len(parts) == 2:
        watched.append((int(parts[0]), float(parts[1])))
for pid, started in watched:
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - started) > 0.01:
            continue
        if os.name == "nt":
            for child in process.children(recursive=True):
                child.kill()
            process.kill()
        else:
            try:
                os.killpg(pid, signal.SIGTERM)       # a child in a session of its own leads its process group
            except OSError:
                process.terminate()
    except (psutil.Error, OSError, ValueError):
        pass
"""
_reaper = None
_reaper_lock = threading.Lock()


def tie(process) -> None:
    """End `process` (a subprocess.Popen) when this process ends, however it ends: Linux starts its children under
    setpriv --pdeathsig TERM for that and does nothing here; macOS and Windows tell this process's reaper. A child
    started to outlive its starter (the helper's save) is never tied. Never raises: a child that could not be tied is
    ended by its starter's own shutdown as before."""
    if LINUX:
        return
    global _reaper
    try:
        import psutil
        started = psutil.Process(process.pid).create_time()
        with _reaper_lock:
            if _reaper is None or _reaper.poll() is not None:
                # In a folder of nobody's (Windows removes no folder a process works in: the session's must go).
                _reaper = subprocess.Popen([sys.executable, "-I", "-c", _REAPER], stdin=subprocess.PIPE,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
                                           cwd=tempfile.gettempdir(), env=child_environment(), **new_session())
            _reaper.stdin.write(f"{process.pid} {started!r}\n".encode())
            _reaper.stdin.flush()
    except Exception:
        pass


# ---- signalling and probing ------------------------------------------------------------------------------------------
def probe(pid: int) -> None:
    """os.kill(pid, 0) as POSIX has it: returns when a process with this pid runs, ProcessLookupError when none does,
    PermissionError when one runs that this user may not signal, another OSError when it cannot be told. On Windows
    os.kill(pid, 0) sends a Ctrl+C (signal 0 is CTRL_C_EVENT there) and any other signal ends the process: the process
    is opened for its state instead."""
    if not WINDOWS:
        os.kill(pid, 0)
        return
    _win_probe(pid)


def alive(pid) -> bool | None:
    """Whether a process with this pid runs on this computer: True (also another user's), False, or None when it cannot
    be told (or `pid` is not a positive whole number)."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        probe(pid)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def signal_group(pid: int, sig=signal.SIGTERM) -> None:
    """os.killpg(pid, sig) on POSIX: `pid` leads a process group of its own (new_session). Windows has no process
    groups to signal: the process and every process it started are ended (TerminateProcess, whatever `sig`).
    ProcessLookupError when there is no such process, as os.killpg."""
    if not WINDOWS:
        os.killpg(pid, sig)
        return
    import psutil
    try:
        process = psutil.Process(pid)
        family = process.children(recursive=True)
    except psutil.NoSuchProcess:
        raise ProcessLookupError(errno.ESRCH, "No such process") from None
    for member in (*family, process):
        try:
            member.kill()
        except psutil.NoSuchProcess:
            pass


def end(process, sig=signal.SIGTERM) -> None:
    """process.send_signal(sig) (a subprocess.Popen); on Windows TerminateProcess whatever `sig`, as Popen does it. A
    process that has already ended is left alone."""
    if process.poll() is not None:
        return
    if WINDOWS:
        process.kill()
    else:
        process.send_signal(sig)


def shut_down_this_process() -> None:
    """What sending this process SIGTERM does on Linux and macOS (os.kill(os.getpid(), SIGTERM): its handler, uvicorn's
    or the engine's own, shuts it down in order). On Windows os.kill would end it at once: the signal is raised in this
    process instead, which runs the same handler on the main thread."""
    if WINDOWS:
        signal.raise_signal(signal.SIGTERM)
    else:
        os.kill(os.getpid(), signal.SIGTERM)


def boot_id() -> str | None:
    """This computer's current boot, as text that changes at every restart: Linux's boot_id (as before); macOS's boot
    time (kern.boottime, through psutil); None on Windows, whose boot time drifts with the clock: a restart there is
    told by the processes' start times alone."""
    if LINUX:
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
        except OSError:
            return None
    if MACOS:
        try:
            import psutil
            return f"boot-{int(psutil.boot_time())}"
        except Exception:
            return None
    return None


def started(pid: int) -> float | None:
    """When the process with this pid started (seconds since the epoch, psutil's create time), None when there is none
    or it cannot be told. For macOS and Windows, which have no /proc: Linux reads /proc/<pid>/stat where it did."""
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def niceness() -> int | None:
    """This process's niceness (os.getpriority) on POSIX; None on Windows, which has priority classes instead."""
    if WINDOWS:
        return None
    return os.getpriority(os.PRIO_PROCESS, 0)


def nicest() -> None:
    """The lowest priority there is: niceness 19 on POSIX (os.nice by what is missing, as before); the idle priority
    class on Windows. Never raises on Windows."""
    if not WINDOWS:
        os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
        return
    lower_priority(19)


def more_open_files(wanted: int = 10240) -> None:
    """macOS starts a process with room for 256 open files, which a sample's reads, the helpers' pipes and the page's
    connections can use up: raised there to `wanted` (within the hard limit; 10240 is macOS's own most), and the
    engine's helpers inherit it. Linux keeps its own (1024 or more); Windows has no such limit. Never raises."""
    if not MACOS:
        return
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ImportError, ValueError, OSError):
        pass


def lower_priority(level: int) -> None:
    """os.nice(level) on POSIX; on Windows the below-normal priority class (any level above 0) or the idle class (10 or
    more). Never raises."""
    try:
        if not WINDOWS:
            os.nice(level)
            return
        if level > 0:
            import psutil
            psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if level >= 10 else psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


# ---- locks -----------------------------------------------------------------------------------------------------------
def _fd(file) -> int:
    return file if isinstance(file, int) else file.fileno()


def lock(file, *, shared: bool = False, block: bool = True) -> None:
    """flock() of `file` (an open file or its descriptor): exclusive, or `shared`; without `block`, BlockingIOError
    while another holds it. Released by unlock() or by closing the file. Windows: LockFileEx of one byte far past any
    content, by this handle (a shared lock is not made exclusive in place there: unlock first)."""
    if not WINDOWS:
        import fcntl
        fcntl.flock(_fd(file), (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if block else fcntl.LOCK_NB))
        return
    _win_lock(_fd(file), shared, block)


def unlock(file) -> None:
    """Let go of lock()'s lock on `file`. On Windows, closing the file lets go of it only "when the system can", so the
    app unlocks first wherever another process may wait for it."""
    if not WINDOWS:
        import fcntl
        fcntl.flock(_fd(file), fcntl.LOCK_UN)
        return
    _win_unlock(_fd(file))


# ---- files that another process may have open ------------------------------------------------------------------------
def replace(source, target) -> None:
    """os.replace(source, target). On Windows a file that another process has open cannot be replaced for that moment:
    tried again until REPLACE_WAIT_S, then the last error."""
    if not WINDOWS:
        os.replace(source, target)
        return
    deadline, delay = time.monotonic() + REPLACE_WAIT_S, 0.001
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(2 * delay, 0.05)


def unlink(path, *, missing_ok: bool = False) -> None:
    """Path(path).unlink(missing_ok); on Windows tried again while another process has the file open, as replace()."""
    if not WINDOWS:
        Path(path).unlink(missing_ok=missing_ok)
        return
    deadline, delay = time.monotonic() + REPLACE_WAIT_S, 0.001
    while True:
        try:
            Path(path).unlink(missing_ok=missing_ok)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(2 * delay, 0.05)


def fsync_folder(path) -> None:
    """Make a folder's entries (a file just put in place with replace) durable: fsync of the folder on POSIX. Windows
    opens no folder with os.open, and its file system keeps a rename durable by itself."""
    if WINDOWS:
        return
    folder = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(folder)
    finally:
        os.close(folder)


def fchmod(fd: int, mode: int) -> None:
    """os.fchmod on POSIX; nothing on Windows, where a file's access is its folder's (private.py)."""
    if not WINDOWS:
        os.fchmod(fd, mode)


def open_shared(path, mode: str = "rb", **kwargs):
    """open(path, mode) for reading. On Windows the file is opened so that the process writing it can still replace or
    remove it while it is read (FILE_SHARE_DELETE: proseg-live replaces where.json, window.json and the like in place,
    and ends when it cannot); elsewhere a plain open."""
    if "r" not in mode or any(flag in mode for flag in "wax+"):
        raise ValueError("open_shared only reads")
    if not WINDOWS:
        return open(path, mode, **kwargs)
    return open(path, mode, opener=_win_shared_opener, **kwargs)


def read_bytes(path) -> bytes:
    """Path(path).read_bytes(), through open_shared."""
    with open_shared(path, "rb") as stream:
        return stream.read()


def read_text(path, encoding: str | None = None, errors: str | None = None) -> str:
    """Path(path).read_text(encoding, errors), through open_shared."""
    with open_shared(path, "r", encoding=encoding, errors=errors) as stream:
        return stream.read()


# ---- Windows ---------------------------------------------------------------------------------------------------------
_kernel32 = None


def _win():
    """kernel32 with the few functions used here declared, made once."""
    global _kernel32
    if _kernel32 is None:
        import ctypes
        from ctypes import wintypes

        class OVERLAPPED(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p), ("Offset", wintypes.DWORD),
                        ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k.GetExitCodeProcess.restype = wintypes.BOOL
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.WaitForSingleObject.restype = wintypes.DWORD
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        k.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                 ctypes.POINTER(OVERLAPPED)]
        k.LockFileEx.restype = wintypes.BOOL
        k.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                   ctypes.POINTER(OVERLAPPED)]
        k.UnlockFileEx.restype = wintypes.BOOL
        k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                  wintypes.DWORD, wintypes.HANDLE]
        k.CreateFileW.restype = wintypes.HANDLE
        k.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                                    ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        k.PeekNamedPipe.restype = wintypes.BOOL
        k.OVERLAPPED = OVERLAPPED
        _kernel32 = k
    return _kernel32


def _win_error(code: int, filename=None) -> OSError:
    import ctypes
    return OSError(None, ctypes.FormatError(code).strip(), filename, code)


def _win_probe(pid: int) -> None:
    import ctypes
    from ctypes import wintypes
    k = _win()
    PROCESS_QUERY_LIMITED_INFORMATION, SYNCHRONIZE = 0x1000, 0x00100000
    ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER = 5, 87
    WAIT_OBJECT_0, WAIT_TIMEOUT, STILL_ACTIVE = 0, 0x102, 259
    handle = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
    if not handle:
        code = ctypes.get_last_error()
        if code == ERROR_ACCESS_DENIED:
            handle = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                raise PermissionError(errno.EPERM, "Operation not permitted")
            try:                               # no right to wait on it: its exit code says whether it still runs
                status = wintypes.DWORD()
                if not k.GetExitCodeProcess(handle, ctypes.byref(status)):
                    raise _win_error(ctypes.get_last_error())
                if status.value != STILL_ACTIVE:
                    raise ProcessLookupError(errno.ESRCH, "No such process")
                return
            finally:
                k.CloseHandle(handle)
        if code == ERROR_INVALID_PARAMETER:
            raise ProcessLookupError(errno.ESRCH, "No such process")
        raise _win_error(code)
    try:
        state = k.WaitForSingleObject(handle, 0)
        if state == WAIT_OBJECT_0:
            raise ProcessLookupError(errno.ESRCH, "No such process")    # ended, its handle still held by someone
        if state != WAIT_TIMEOUT:
            raise _win_error(ctypes.get_last_error())
    finally:
        k.CloseHandle(handle)


def _overlapped():
    k = _win()
    overlapped = k.OVERLAPPED()
    overlapped.OffsetHigh, overlapped.Offset = _LOCK_OFFSET
    return overlapped


def _win_lock(fd: int, shared: bool, block: bool) -> None:
    import ctypes
    import msvcrt
    k = _win()
    LOCKFILE_FAIL_IMMEDIATELY, LOCKFILE_EXCLUSIVE_LOCK, ERROR_LOCK_VIOLATION = 0x1, 0x2, 33
    flags = (0 if shared else LOCKFILE_EXCLUSIVE_LOCK) | (0 if block else LOCKFILE_FAIL_IMMEDIATELY)
    overlapped = _overlapped()
    if not k.LockFileEx(msvcrt.get_osfhandle(fd), flags, 0, 1, 0, ctypes.byref(overlapped)):
        code = ctypes.get_last_error()
        if code == ERROR_LOCK_VIOLATION:
            raise BlockingIOError(errno.EWOULDBLOCK, "The lock is held by another")
        raise _win_error(code)


def _win_unlock(fd: int) -> None:
    import ctypes
    import msvcrt
    k = _win()
    ERROR_NOT_LOCKED = 158
    overlapped = _overlapped()
    if not k.UnlockFileEx(msvcrt.get_osfhandle(fd), 0, 1, 0, ctypes.byref(overlapped)):
        code = ctypes.get_last_error()
        if code != ERROR_NOT_LOCKED:
            raise _win_error(code)


def _win_shared_opener(path, flags: int) -> int:
    import ctypes
    import msvcrt
    k = _win()
    GENERIC_READ, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL = 0x80000000, 3, 0x80
    FILE_SHARE_ALL = 0x1 | 0x2 | 0x4                     # read, write, delete
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    handle = k.CreateFileW(os.fspath(path), GENERIC_READ, FILE_SHARE_ALL, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                           None)
    if handle in (None, INVALID_HANDLE_VALUE):
        raise _win_error(ctypes.get_last_error(), os.fspath(path))
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY)           # binary: no O_TEXT
    except BaseException:
        k.CloseHandle(handle)
        raise


# ---- pipes -----------------------------------------------------------------------------------------------------------
def pass_fd(fd: int) -> tuple[str, dict]:
    """How a child gets the pipe end `fd`: the text that names it to the child (in an environment variable) and
    subprocess.Popen's arguments. POSIX: its number, in pass_fds. Windows, which passes no descriptors: the pipe's
    handle, which that child alone inherits (STARTUPINFO's handle list) and makes a descriptor of with take_fd()."""
    if not WINDOWS:
        return str(fd), {"pass_fds": (fd,)}
    import msvcrt
    handle = msvcrt.get_osfhandle(fd)
    os.set_handle_inheritable(handle, True)
    info = subprocess.STARTUPINFO()
    info.lpAttributeList = {"handle_list": [handle]}
    return str(handle), {"startupinfo": info}


def take_fd(value: str, *, write: bool) -> int:
    """The descriptor pass_fd's text names in this child: POSIX the number itself; Windows a descriptor made of the
    inherited handle (closing it closes the handle)."""
    if not WINDOWS:
        return int(value)
    import msvcrt
    return msvcrt.open_osfhandle(int(value), os.O_WRONLY if write else os.O_RDONLY)


def _peek(fd: int) -> bool:
    """Windows: whether a read of the pipe `fd` would return at once (data waits, or its other end has closed)."""
    import ctypes
    from ctypes import wintypes
    import msvcrt
    k = _win()
    waiting = wintypes.DWORD()
    try:
        handle = msvcrt.get_osfhandle(fd)
    except OSError:
        return True                                        # closed: the read says so
    if not k.PeekNamedPipe(handle, None, 0, None, ctypes.byref(waiting), None):
        return True                                        # the other end closed (or another error): the read says
    return waiting.value > 0


def readable(fd: int, timeout: float | None) -> bool:
    """select.select([fd], [], [], timeout) for a pipe: whether a read would return at once, waiting up to `timeout`
    seconds (None: as long as it takes). Windows' select takes sockets only: the pipe is looked at every 10 ms there."""
    if not WINDOWS:
        import select
        return bool(select.select([fd], [], [], timeout)[0])
    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
    while True:
        if _peek(fd):
            return True
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0:
            return False
        time.sleep(0.01 if left is None else min(0.01, left))


def writable(fd: int, timeout: float) -> bool:
    """select.select([], [fd], [], timeout) for a pipe. Windows cannot tell whether a pipe has room: after a short wait
    (at most 10 ms) the pipe is said writable, and the non-blocking write that follows says whether it was."""
    if not WINDOWS:
        import select
        return bool(select.select([], [fd], [], timeout)[1])
    time.sleep(min(0.01, max(0.0, timeout)))
    return True


class PipeSelector:
    """selectors.DefaultSelector for pipes: on POSIX it is one (pipe_selector()). On Windows, whose select takes
    sockets only, a pipe registered for reading is ready when a read would return at once (_peek), and one registered
    for writing is said ready after a few milliseconds (no busy loop while a pipe is full); its non-blocking write
    says whether there was room (BlockingIOError)."""

    def __init__(self):
        import selectors
        self._selectors = selectors
        self._keys = {}

    def register(self, fileobj, events, data=None):
        fd = fileobj if isinstance(fileobj, int) else fileobj.fileno()
        if fd in self._keys:
            raise KeyError(f"{fileobj!r} is already registered")
        key = self._selectors.SelectorKey(fileobj, fd, events, data)
        self._keys[fd] = key
        return key

    def unregister(self, fileobj):
        fd = fileobj if isinstance(fileobj, int) else fileobj.fileno()
        return self._keys.pop(fd)

    def select(self, timeout=None):
        read, write = self._selectors.EVENT_READ, self._selectors.EVENT_WRITE
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        writers = [key for key in self._keys.values() if key.events & write]
        while True:
            ready = [(key, read) for key in self._keys.values() if key.events & read and _peek(key.fd)]
            if ready:
                return ready + [(key, write) for key in writers]
            left = None if deadline is None else deadline - time.monotonic()
            if writers or (left is not None and left <= 0):
                if writers:
                    time.sleep(0.005 if left is None else min(0.005, max(0.0, left)))
                return [(key, write) for key in writers]
            time.sleep(0.01 if left is None else min(0.01, left))

    def close(self):
        self._keys.clear()


def pipe_selector():
    """A selector for pipes: selectors.DefaultSelector on POSIX, PipeSelector on Windows."""
    if not WINDOWS:
        import selectors
        return selectors.DefaultSelector()
    return PipeSelector()


# ---- owners ----------------------------------------------------------------------------------------------------------
def owned_by_me(process) -> bool:
    """Whether a psutil.Process runs as this user: its real and effective uids are this process's on POSIX (as
    before); its user name is this process's on Windows, which has no uids."""
    if not WINDOWS:
        owners = process.uids()
        return owners.real == os.getuid() and owners.effective == os.getuid()
    import psutil
    return process.username() == psutil.Process().username()


def linked(info) -> bool:
    """Whether an os.lstat() result is a link: a symbolic link, or on Windows any reparse point (a junction too)."""
    import stat
    if stat.S_ISLNK(info.st_mode):
        return True
    return bool(WINDOWS and getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def private_info(info, mode: int) -> bool:
    """Whether an os.lstat()/fstat() result is this user's alone: owned by this user with exactly `mode` (0o700 for a
    folder, 0o600 for a file) on POSIX, as before; on Windows not a link or junction, the user's own folders' access
    list doing the rest (private.py)."""
    import stat
    if WINDOWS:
        return not linked(info)
    return info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == mode


def mine(info) -> bool:
    """Whether an os.stat()/lstat() result belongs to this user: its uid on POSIX; on Windows, which keeps no uid in
    it, whatever is not a link or junction (the app looks only in the user's own folders there)."""
    if WINDOWS:
        return not linked(info)
    return info.st_uid == os.getuid()


# ---- folder links ----------------------------------------------------------------------------------------------------
def folder_link(path) -> bool:
    """Whether `path` is a link to a folder: a symbolic link, or on Windows a directory junction too (link_folder)."""
    path = Path(path)
    return path.is_symlink() or (WINDOWS and path.is_junction())


def _remove_link(path: Path) -> None:
    """Windows: remove a folder link or junction (rmdir removes the link, never what it leads to; a real folder that
    holds anything is refused)."""
    try:
        os.rmdir(path)
    except FileNotFoundError:
        pass


def link_folder(link, target) -> None:
    """Make `link` lead to the folder `target`. POSIX: a symbolic link renamed over whatever link was there, in one
    step (a helper may be starting from it). Windows: a symbolic link needs Developer Mode or an administrator there,
    so a directory junction (the same, for a folder on this computer) when one cannot be made; and no folder link can
    be renamed over another there, so an old link is removed first."""
    link, target = Path(link), Path(target)
    spare = link.with_name(f".{link.name}.{os.getpid()}.{threading.get_ident()}")
    if not WINDOWS:
        spare.unlink(missing_ok=True)
        spare.symlink_to(target, target_is_directory=True)
        os.replace(spare, link)
        return
    if folder_link(spare):
        _remove_link(spare)
    try:
        spare.symlink_to(target, target_is_directory=True)
    except OSError:
        import _winapi
        _winapi.CreateJunction(str(target), str(spare))
    if folder_link(link):
        _remove_link(link)
    os.replace(spare, link)
