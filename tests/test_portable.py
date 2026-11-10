"""portable.py on the system the tests run on: Linux here and in every push's checks, macOS and Windows on GitHub's
runners when started by hand and before each release (tests.yml other_systems). Each test starts real processes and
takes real locks: what differs between the systems is exactly what is tested."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from sopa_companion import portable

APP_PARENT = str(Path(portable.__file__).resolve().parent.parent)


def python(code: str, *args, **kwargs) -> subprocess.Popen:
    """A Python child running `code` that imports this copy of the app."""
    environment = portable.child_environment({**os.environ, "PYTHONPATH": APP_PARENT})
    return subprocess.Popen([sys.executable, "-c", textwrap.dedent(code), *map(str, args)], env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)


def wait_for(predicate, timeout=20.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def sleeper(**kwargs) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)


def test_probing_a_running_process_leaves_it_running():
    """On Windows os.kill(pid, 0) is a Ctrl+C and any other signal ends the process: probe and alive leave it alone."""
    child = sleeper(**portable.new_session())
    try:
        for _ in range(5):
            portable.probe(child.pid)
            assert portable.alive(child.pid) is True
        time.sleep(0.3)
        assert child.poll() is None, "probing the process ended it"
    finally:
        child.kill()
        child.wait()


def test_a_finished_process_is_not_alive():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert portable.alive(child.pid) is False
    with pytest.raises(ProcessLookupError):
        portable.probe(child.pid)
    assert portable.alive(os.getpid()) is True
    assert portable.alive(0) is None and portable.alive(-3) is None and portable.alive(True) is None
    assert portable.alive("12") is None


def test_signal_group_ends_the_child_and_what_it_started():
    starter = python("""
        import subprocess, sys, time
        grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        print(grandchild.pid, flush=True)
        time.sleep(120)
    """, **portable.new_session())
    try:
        grandchild = int(starter.stdout.readline())
        assert portable.alive(grandchild)
        portable.signal_group(starter.pid, signal.SIGTERM)
        assert wait_for(lambda: starter.poll() is not None)
        assert wait_for(lambda: portable.alive(grandchild) is False), "the group's other process still runs"
    finally:
        if starter.poll() is None:
            starter.kill()
        starter.wait()
    with pytest.raises(ProcessLookupError):
        portable.signal_group(starter.pid)


def test_end_stops_a_process_and_leaves_an_ended_one():
    child = sleeper()
    portable.end(child)
    assert child.wait(timeout=20) != 0
    portable.end(child)                                  # already ended: nothing


def test_a_tied_child_ends_when_its_starter_is_killed(tmp_path):
    """tie() on macOS and Windows; the same reaper is tried on Linux, where the app itself uses setpriv's death signal."""
    starter = python("""
        import subprocess, sys, time
        from sopa_companion import portable
        portable.LINUX = False                           # the reaper, whatever the system
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], **portable.new_session())
        portable.tie(child)
        print(child.pid, flush=True)
        time.sleep(120)
    """)
    child = int(starter.stdout.readline())
    try:
        assert portable.alive(child)
        time.sleep(0.5)                                  # the reaper has read the child's line
        starter.kill()                                   # SIGKILL / TerminateProcess: no clean-up of its own
        starter.wait()
        assert wait_for(lambda: portable.alive(child) is False), "the tied child outlived its starter"
    finally:
        if portable.alive(child):
            portable.signal_group(child, portable.SIGKILL)


def test_an_untied_child_outlives_its_starter():
    starter = python("""
        import subprocess, sys, time
        from sopa_companion import portable
        portable.LINUX = False
        tied = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], **portable.new_session())
        portable.tie(tied)
        free = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], **portable.new_session())
        print(tied.pid, free.pid, flush=True)
        time.sleep(120)
    """)
    tied, free = map(int, starter.stdout.readline().split())
    try:
        time.sleep(0.5)
        starter.kill()
        starter.wait()
        assert wait_for(lambda: portable.alive(tied) is False)
        time.sleep(0.5)
        assert portable.alive(free) is True, "a child that was not tied was ended"
    finally:
        for pid in (tied, free):
            if portable.alive(pid):
                portable.signal_group(pid, portable.SIGKILL)


def test_shutting_down_this_process_runs_its_sigterm_handler():
    child = python("""
        import signal, sys, threading, time
        from sopa_companion import portable
        signal.signal(signal.SIGTERM, lambda *_: (print("handled", flush=True), sys.exit(7)))
        threading.Thread(target=portable.shut_down_this_process).start()       # from another thread, as the server does
        time.sleep(30)
    """)
    out, err = child.communicate(timeout=60)
    assert child.returncode == 7 and out.strip() == "handled", (child.returncode, out, err)


def test_an_exclusive_lock_keeps_another_process_out_until_unlocked(tmp_path):
    target = tmp_path / "sample.lock"
    target.write_text('{"pid": 1}')
    other = """
        import sys
        from sopa_companion import portable
        with open(sys.argv[1], "r+") as stream:
            try:
                portable.lock(stream, shared=sys.argv[2] == "shared", block=False)
            except BlockingIOError:
                print("held", stream.read(), flush=True)
            else:
                print("taken", flush=True)
                portable.unlock(stream)
    """
    with open(target, "r+") as mine:
        portable.lock(mine)
        for kind in ("exclusive", "shared"):
            child = python(other, target, kind)
            out, err = child.communicate(timeout=60)
            # Held, and its content still readable (Windows' locks would block reading what they cover).
            assert out.split() == ["held", '{"pid":', "1}"], (out, err)
        portable.unlock(mine)
        child = python(other, target, "exclusive")
        assert child.communicate(timeout=60)[0].strip() == "taken"


def test_shared_locks_admit_each_other_but_not_an_exclusive_one(tmp_path):
    target = tmp_path / "fit.lock"
    target.touch()
    other = """
        import sys
        from sopa_companion import portable
        with open(sys.argv[1], "a") as stream:
            try:
                portable.lock(stream, shared=sys.argv[2] == "shared", block=False)
            except BlockingIOError:
                print("held", flush=True)
            else:
                print("taken", flush=True)
    """
    with open(target, "a") as mine:
        portable.lock(mine, shared=True)
        assert python(other, target, "shared").communicate(timeout=60)[0].strip() == "taken"
        assert python(other, target, "exclusive").communicate(timeout=60)[0].strip() == "held"


def test_a_blocking_lock_waits_for_the_holder(tmp_path):
    target = tmp_path / "save.lock"
    target.touch()
    holder = python("""
        import sys, time
        from sopa_companion import portable
        with open(sys.argv[1], "a") as stream:
            portable.lock(stream)
            print("locked", flush=True)
            time.sleep(1.0)
            portable.unlock(stream)
    """, target)
    assert holder.stdout.readline().strip() == "locked"
    began = time.monotonic()
    with open(target, "a") as mine:
        portable.lock(mine)                             # waits for the holder's unlock
        waited = time.monotonic() - began
        portable.unlock(mine)
    holder.wait(timeout=60)
    assert waited > 0.3


def test_a_file_held_open_by_a_reader_is_replaced(tmp_path):
    target, source = tmp_path / "control", tmp_path / ".control.tmp"
    target.write_text("paused 1\n")
    source.write_text("paused 0\n")
    reader = portable.open_shared(target, "rb")
    threading.Timer(0.3, reader.close).start()            # a reader that lets go a moment later
    portable.replace(source, target)
    assert target.read_text() == "paused 0\n" and not source.exists()


def test_a_shared_read_lets_the_writer_remove_the_file(tmp_path):
    target = tmp_path / "where.json"
    target.write_text('{"token": 3}')
    with portable.open_shared(target, "rb") as stream:
        assert stream.read() == b'{"token": 3}'
        target.unlink()                                   # proseg-live's own clean-up while the engine reads
    assert not target.exists()
    with pytest.raises(FileNotFoundError):
        portable.read_bytes(tmp_path / "missing")


def test_read_text_and_bytes_read_what_was_written(tmp_path):
    target = tmp_path / "genes.txt"
    target.write_bytes("CD3E\nKRT8\nµ-gene\n".encode("utf-8"))
    assert portable.read_bytes(target) == "CD3E\nKRT8\nµ-gene\n".encode("utf-8")
    assert portable.read_text(target, encoding="utf-8") == "CD3E\nKRT8\nµ-gene\n"
    with pytest.raises(ValueError):
        portable.open_shared(target, "r+")
    with pytest.raises(ValueError):
        portable.open_shared(target, "wb")


def test_the_boot_is_told_the_same_way_twice():
    first, second = portable.boot_id(), portable.boot_id()
    assert first == second
    if portable.LINUX:
        assert first == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    elif portable.MACOS:
        assert first.startswith("boot-")
    else:
        assert first is None


def test_children_get_no_window_and_their_own_group():
    session = portable.new_session()
    if portable.WINDOWS:
        assert session == {"creationflags": portable.CREATE_NEW_PROCESS_GROUP | portable.CREATE_NO_WINDOW}
        assert portable.no_window() == {"creationflags": portable.CREATE_NO_WINDOW}
        assert portable.child_environment({})["PYTHONUTF8"] == "1"
    else:
        assert session == {"start_new_session": True} and portable.no_window() == {}
        assert portable.child_environment({"A": "1"}) == {"A": "1"}
    assert portable.SIGKILL == getattr(signal, "SIGKILL", signal.SIGTERM)


def test_lowering_the_priority_never_raises():
    """lower_priority and nicest never raise; on POSIX they lower this process's priority (niceness 10, then 19), and
    what the system answered is said when they do not."""
    child = python("""
        import json, os
        from sopa_companion import portable
        before = portable.niceness()
        portable.lower_priority(10)
        middle = portable.niceness()
        portable.lower_priority(0)
        portable.nicest()
        after = portable.niceness()
        try:
            tried = os.nice(0) if hasattr(os, "nice") else None
        except Exception as exc:
            tried = repr(exc)
        print(json.dumps({"before": before, "middle": middle, "after": after, "nice(0)": tried}), flush=True)
    """)
    out, err = child.communicate(timeout=60)
    assert child.returncode == 0, err
    said = json.loads(out.strip().splitlines()[-1])
    if portable.WINDOWS:
        assert said["before"] is None and said["after"] is None
    else:
        assert said["middle"] >= min(19, said["before"] + 10) and said["after"] == 19, said

# What Windows lacks: these modules, and these names of os, signal and select (macOS lacks a few of them too).
WINDOWS_LACKS = {"modules": ("fcntl", "resource", "termios", "pwd", "grp"),
                 "os": ("getuid", "geteuid", "getgid", "killpg", "getpgid", "setsid", "nice", "getpriority",
                        "setpriority", "sched_getaffinity", "O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC", "O_NONBLOCK",
                        "readv", "fchmod", "getloadavg", "pidfd_open"),
                 "signal": ("SIGKILL", "SIGHUP", "SIGUSR1", "SIGUSR2", "SIGCHLD", "SIGPIPE", "SIGALRM",
                            "pidfd_send_signal"),
                 "select": ("poll", "epoll")}
# The desktop app's own modules, which stay Linux's (the notebook route never imports them).
DESKTOP_ONLY = ("desktop.launcher", "desktop.migrate", "desktop.processes", "desktop.install")
IMPORT_CHECK = r"""
import importlib, json, os, select, signal, sys
lacks = json.loads(sys.argv[1])
for library in ("numpy", "pandas", "zarr", "fastapi", "uvicorn", "psutil", "httpx", "pyarrow", "orjson", "zstandard",
                "tifffile", "PIL", "scipy", "anndata", "spatialdata", "dask", "torch", "cellpose", "sklearn", "umap",
                "igraph", "numba", "asyncio", "subprocess", "selectors", "tempfile", "multiprocessing"):
    try:
        importlib.import_module(library)      # what checks for those names at its own import, loaded first
    except Exception:
        pass
for name in lacks["modules"]:
    sys.modules[name] = None
for owner, names in ((os, lacks["os"]), (signal, lacks["signal"]), (select, lacks["select"])):
    for name in names:
        if hasattr(owner, name):
            delattr(owner, name)
failed = []
for module in sys.argv[2:]:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name in lacks["modules"] or (exc.name or "").startswith("sopa_companion"):
            failed.append(f"{module}: {exc!r}")
    except Exception as exc:
        failed.append(f"{module}: {type(exc).__name__}: {exc}")
print(json.dumps(failed))
"""


def test_every_module_imports_without_what_windows_lacks():
    """Windows has no fcntl, resource, pwd or grp, no getuid, killpg, nice, poll or readv, no O_NOFOLLOW or O_DIRECTORY,
    no SIGKILL or SIGHUP: every module the notebook route can load imports without them (they are used only on the
    systems that have them, or through portable.py). Run here with them taken away, so a Linux-only name at a module's
    top shows on every push, not only on a Windows runner."""
    root = Path(portable.__file__).resolve().parent
    modules = []
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        if parts[0] in ("tests", "harness", "native", "frontend", "docs", "ci", "packaging", "var") or \
                any(part.startswith(".") for part in parts) or ".".join(parts) in DESKTOP_ONLY:
            continue
        if parts[-1] == "__main__":
            continue
        modules.append(".".join(("sopa_companion", *[p for p in parts if p != "__init__"])))
    environment = {**os.environ, "PYTHONPATH": APP_PARENT, "MPLBACKEND": "Agg"}
    done = subprocess.run([sys.executable, "-c", IMPORT_CHECK, json.dumps(WINDOWS_LACKS), *modules],
                          capture_output=True, text=True, timeout=600, env=environment)
    assert done.returncode == 0, done.stderr[-3000:]
    failed = json.loads(done.stdout.strip().splitlines()[-1])
    assert failed == [], "\n".join(failed)


def test_the_engine_behind_a_venvs_launcher_is_the_engine(tmp_path):
    """On Windows a virtual environment's python.exe is a launcher that runs the interpreter as its child: the pid the
    engine reports (os.getpid) is then the child's, never Popen's. The notebook takes that child as its engine; any other
    process is not (elsewhere the two are one process, the venv's python a link)."""
    import types
    from sopa_companion.api.handle import CompanionApp
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=300)
    python = venv / ("Scripts/python.exe" if portable.WINDOWS else "bin/python")
    started = subprocess.Popen([str(python), "-c", "import os, sys, time; print(os.getpid(), flush=True); "
                                "sys.stdin.read()"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        reported = int(started.stdout.readline())
        handle = types.SimpleNamespace(_process=started)
        assert CompanionApp._is_engine(handle, reported)
        assert CompanionApp._is_engine(handle, started.pid)
        assert not CompanionApp._is_engine(handle, os.getpid()) and not CompanionApp._is_engine(handle, "12")
        if portable.WINDOWS and reported != started.pid:
            assert portable.started(reported) is not None     # the launcher's child, a process of its own
    finally:
        started.stdin.close()
        started.wait(timeout=60)
