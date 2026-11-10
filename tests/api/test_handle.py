"""The kernel API against private folders and an engine speaking the real control and event lines."""
from __future__ import annotations

import fcntl
import html
import importlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import threading
import time
import tomllib
from urllib.parse import parse_qs, urlsplit

import pytest

from sopa_companion import dirs, locate
from sopa_companion.analysis.backend import CPU_MODULES
from sopa_companion.api import folders, protocol, window
from sopa_companion.api.result import CompanionError
from sopa_companion.desktop import browser

L = importlib.import_module("sopa_companion.api.launch")
OPEN_APP = browser.open_app             # desktop/browser.py's own, whatever a test puts in its place


# Invoked as the configured engine Python, with the real command-line arguments,
# private inherited events fd, and stdin control pipe. No engine/server dependency.
FAKE_ENGINE = r'''
import json
import os
from pathlib import Path
import select
import signal
import sys
import threading
import time
import tomllib
from sopa_companion.api import protocol as P

assert sys.argv[1:5] == ["-m", "sopa_companion", "serve", "--config"], sys.argv
config = P.validate_session_config(tomllib.loads(Path(sys.argv[5]).read_text()))
capture = Path(os.environ["FAKE_ENGINE_CAPTURE"])
mode = os.environ.get("FAKE_ENGINE_MODE", "normal")

def record(value):
    with capture.open("a") as stream:
        stream.write(json.dumps(value) + "\n")

record({"config": config, "pid": os.getpid(), "sid": os.getsid(0), "cwd": os.getcwd(),
        "env": {k: os.environ.get(k) for k in ["MPLBACKEND", "CUDA_VISIBLE_DEVICES", "PYTHONPATH"]}})
lock = open(Path(config["paths"]["data_dir"]) / ".engine.lock", "w")
import fcntl
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
events = int(os.environ[P.ENV_EVENTS_FD])
os.set_blocking(events, False)

def emit(name, **fields):
    data = P.encode_line(P.build_event(name, **fields)).encode()
    if mode == "split":
        os.write(events, data[:5])
        time.sleep(.01)
        os.write(events, data[5:])
    else:
        os.write(events, data)

# A sandbox may refuse socket(), loopback included. HTTP is replaced in the
# parent fixture; the private control/events pipes and process are real.
port = 20000 + int(config["session"]["id"], 16) % 40000
if mode == "crash":
    print("secret=" + config["server"]["token"], flush=True)
    print("fake engine broke", flush=True)
    sys.exit(7)
if mode == "hang":
    signal.signal(signal.SIGTERM, lambda *args: None)
if mode != "no_ready":
    if mode == "unknown":
        os.write(events, b'{"ev":"future_event","extra":true}\n')
    if mode == "malformed":
        os.write(events, b'{"ev":"ready","port":"bad","pid":1}\n')
    else:
        emit("ready", port=port, pid=os.getpid() if mode != "bad_pid" else os.getpid() + 1)
buffer = b""
bound_at = None
delivered = False
while True:
    if mode == "dont_read":
        time.sleep(.02)
        continue
    if bound_at and mode == "auto_end" and time.monotonic() - bound_at >= .25:
        emit("closing", pending="discard", by="window")
        emit("ended", added=["unapplied-delivery-id"], not_added=[])
        break
    if not select.select([0], [], [], .02)[0]:
        continue
    chunk = os.read(0, 65536)
    if not chunk:
        record({"eof": True})
        if mode == "hang":
            while True: time.sleep(.1)
        break
    buffer += chunk
    while b"\n" in buffer:
        line, buffer = buffer.split(b"\n", 1)
        message = P.decode_control_line(line)
        if message is None: continue
        record(message)
        if message["op"] == "bind":
            bound_at = time.monotonic()
            emit("session", state="ready", message=None)
            if mode == "delivery":
                delivered = True
                emit("deliver", id="d1", run="r1", kind="stain", staged=str(Path(config["paths"]["data_dir"]) / "deliveries/d1/staged.zarr"),
                     shapes_key="new_cells", table_key=None, set_default=True, overwrite=False, replaced=[],
                     written_to=message["zarr"], cells=1, genes=None, provenance={})
        if message["op"] == "applied" and mode == "delivery":
            emit("ended", added=["d1"], not_added=[])
            sys.exit(0)
        if message["op"] == "close" and mode != "hang":
            emit("closing", pending=message["pending"], by="handle")
            emit("ended", added=[], not_added=[])
            sys.exit(0)
os.close(events)
'''


@pytest.fixture
def fake_engine(private_home, monkeypatch):
    executable = private_home / "fake-engine-python"
    executable.write_text("#!" + sys.executable + "\n" + FAKE_ENGINE)
    executable.chmod(0o700)
    capture = private_home / "engine-messages.jsonl"
    monkeypatch.setenv("FAKE_ENGINE_CAPTURE", str(capture))
    monkeypatch.setenv("FAKE_ENGINE_MODE", "normal")
    monkeypatch.setattr(L, "_settings", lambda: {"api": {"engine_python": str(executable)},
                                              "proseg": {"binary": sys.executable}})
    # The kernel here has none of the optional features' packages ("auto" finds nothing: off), and the fake engine's
    # interpreter does not import this copy by itself (the session's pylib, as an engine in another environment).
    kernel_has(monkeypatch, set())
    monkeypatch.setattr(locate, "imports_this_copy", lambda python, env, **kwargs: False)
    monkeypatch.setattr(L, "_attach_delivery", lambda app, sdata: None)
    def http(port, token, path, **kwargs):
        config = next(line["config"] for line in records(capture) if "config" in line and line["config"]["server"]["token"] == token)
        # A link a person opens by hand asks for a ten-minute code (?link=1): marked in the fake's codes.
        return {"ok": True} if path == "/api/v1/health" else \
            {"code": "only-once-" + config["session"]["id"] + ("-link" if path.endswith("?link=1") else "")}
    monkeypatch.setattr(window, "request", http)
    yield capture
    # A failed assertion must not leave a test engine or its thread alive.
    from sopa_companion.api.handle import _APPS
    for app in tuple(_APPS):
        app.close(pending="discard", timeout=0)


@pytest.fixture
def sample(private_home, monkeypatch):
    import numpy as np
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel
    from sopa import settings

    sdata = SpatialData(images={"image": Image2DModel.parse(np.zeros((1, 2, 3), dtype="uint8"), dims=("c", "y", "x"))})
    # No user store is created or written. The fake engine only records bind.
    path = private_home / "sample-store"
    path.mkdir()
    sdata.path = path
    monkeypatch.setattr(SpatialData, "_symmetric_difference_with_zarr_store", lambda self: ([], []))
    monkeypatch.setattr(settings, "auto_save_on_disk", True)
    return sdata


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


REAL_PYTHON_FOR = locate.python_for


def kernel_has(monkeypatch, modules):
    """The kernel as having these packages and no others (locate.python_for's lookup, with nothing imported)."""
    monkeypatch.setattr(locate, "python_for", lambda feature, value=None: REAL_PYTHON_FOR(
        feature, value, find=lambda name: object() if name in modules else None))


def until(predicate, *, callbacks=None, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if callbacks is not None:
            while callbacks:
                callbacks.pop(0)()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("condition not reached")


@pytest.fixture
def notebook(monkeypatch):
    callbacks = []
    monkeypatch.setattr(L, "_running_loop", lambda: callbacks.append)
    return callbacks


@pytest.fixture
def outputs(monkeypatch):
    """The notebook's outputs (IPython's display faked): each link shown is one, its text as it reads now (Shown's
    updates run on the kernel's loop: the `notebook` fixture's callbacks)."""
    import builtins
    import IPython.display
    shown = []

    class Output:
        def __init__(self, value):
            self.text = value["text/plain"]

        def update(self, value, raw=False):
            self.text = value["text/plain"]

    def display(value, raw=False, display_id=False):
        shown.append(Output(value))
        return shown[-1]
    monkeypatch.setattr(builtins, "get_ipython", lambda: object(), raising=False)
    monkeypatch.setattr(IPython.display, "display", display)
    return shown


@pytest.fixture
def private_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    # The default scratch folder too, which launch() sweeps for sessions left behind: never the shared /tmp.
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    return tmp_path


def dead_owner(session):
    path = session.state / "owner.json"
    owner = json.loads(path.read_text())
    owner["kernel"]["started"] = -1  # same pid, different birth: PID reuse is not a live owner
    folders.private_json(path, owner)


def test_folders_are_private_fresh_and_config_is_i1(private_home):
    root = private_home
    a = folders.SessionFolders.create(root / "scratch", "/not-opened/sample.zarr")
    b = folders.SessionFolders.create(root / "scratch", "/not-opened/sample.zarr")
    try:
        assert a.id != b.id and len(a.id) == 8
        assert int(a.id, 16) >= 0
        assert stat.S_IMODE(a.state.stat().st_mode) == 0o700
        assert stat.S_IMODE((a.state / "owner.json").stat().st_mode) == 0o600
        config = protocol.build_session_config(id=a.id, token="t" * 43, data_dir=str(a.data),
                                              state_dir=str(a.state), python=sys.executable, kernel_pid=os.getpid())
        folders.write_config(a.state / "session.toml", config)
        assert stat.S_IMODE((a.state / "session.toml").stat().st_mode) == 0o600
        assert protocol.validate_session_config(tomllib.loads((a.state / "session.toml").read_text())) == config
        # The engine's pylib is made only when an interpreter needs it (api/handle.py), then kept as it is.
        assert not (a.state / "pylib").exists()
        assert a.pylib() == a.state / "pylib" == a.pylib()
        assert (a.state / "pylib/sopa_companion").resolve() == Path(__file__).resolve().parents[2]
        assert stat.S_IMODE((a.state / "pylib").stat().st_mode) == 0o700
        assert folders.sweep() == ()  # a living kernel's folders are never removed
    finally:
        a.remove()
        b.remove()
    assert not a.state.exists() and not a.data.exists()
    a.remove()  # idempotent


def test_folders_retry_both_exclusive_mkdir_collisions(private_home, monkeypatch):
    root, scratch = folders.state_root(), private_home / "scratch"
    (root / "11111111").mkdir()
    scratch.mkdir()
    (scratch / "sopa-companion-22222222").mkdir()
    ids = iter(["11111111", "22222222", "33333333"])
    monkeypatch.setattr(folders.secrets, "token_hex", lambda n: next(ids))
    session = folders.SessionFolders.create(scratch, "/sample")
    assert session.id == "33333333"
    assert (root / "11111111").is_dir()
    assert (scratch / "sopa-companion-22222222").is_dir()
    assert not (root / "22222222").exists()
    session.remove()


def test_private_state_rejects_symlinks_and_falls_back(private_home, monkeypatch):
    target = private_home / "other"
    target.mkdir()
    link = private_home / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(CompanionError, match="private folder"):
        folders.private_directory(link)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(private_home / "unusable"))
    (private_home / "unusable").write_text("a file")
    assert folders.state_root() == private_home / "cache/sopa-companion/sessions"


def test_sweeper_checks_pid_birth_engine_lock_and_data_marker(private_home):
    session = folders.SessionFolders.create(private_home / "scratch", "/sample")
    dead_owner(session)
    with (session.data / ".engine.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert folders.sweep() == ()
        assert session.data.is_dir()
    session.engine_started(os.getpid())
    assert folders.sweep() == ()
    owner_path = session.state / "owner.json"
    owner = json.loads(owner_path.read_text())
    owner["engine"] = None
    folders.private_json(owner_path, owner)
    # A forged data location is never recursively removed.
    marker = session.data / ".sopa-companion-session.json"
    correct = marker.read_text()
    marker.write_text('{}')
    assert folders.sweep() == ()
    marker.write_text(correct)
    assert folders.sweep() == (session.id,)
    assert not session.state.exists() and not session.data.exists()
    assert folders.sweep() == ()


def test_sweeper_never_follows_links_or_touches_sample(private_home):
    sample = private_home / "sample"
    sample.mkdir()
    keep = sample / "keep"
    keep.write_text("unchanged")
    session = folders.SessionFolders.create(private_home / "scratch", str(sample))
    (session.data / "linked-sample").symlink_to(sample, target_is_directory=True)
    dead_owner(session)
    root_link = session.state.parent / "ffffffff"
    root_link.symlink_to(sample, target_is_directory=True)
    assert folders.sweep() == (session.id,)
    assert keep.read_text() == "unchanged" and root_link.is_symlink()


@pytest.mark.parametrize("value,display,ssh,expected", [
    ("auto", False, False, "link"), ("auto", True, False, "window"), ("auto", True, True, "link"),
    ("window", False, False, "link"), ("none", False, False, "none"), ("link", True, False, "link"),
])
def test_open_modes(value, display, ssh, expected, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    # Only test mode selection; no process is ever attached to a display.
    if display:
        monkeypatch.setenv("WAYLAND_DISPLAY", "test-only")
    if ssh:
        monkeypatch.setenv("SSH_CONNECTION", "client port server port")
    assert window.open_mode(value) == expected


class Started:
    """The browser starts open_window would make, faked (desktop/browser.py open_app with a fake `which`, start and
    desktop): what was started, never a real browser."""

    def __init__(self, monkeypatch, *, found=("chromium",), default=None):
        self.started = []
        real = OPEN_APP
        which = lambda name: f"/usr/bin/{name}" if name in found else None
        answer = f"Default application for “x-scheme-handler/https”: {default}\n" if default else ""
        run = lambda argv, **kw: types.SimpleNamespace(returncode=0, stdout=answer)
        environ = {"HOME": os.environ["HOME"]}

        def opened(url, **kwargs):
            return real(url, **kwargs, which=which, spawn=self.start, run=run, environ=environ, platform="linux")
        monkeypatch.setattr(window.browser, "open_app", opened)
        monkeypatch.setattr(browser, "wsl", lambda environ=None: False)

    def start(self, argv, **kwargs):
        self.started.append(argv)
        return types.SimpleNamespace(pid=4242, argv=argv, wait=lambda timeout=None: 0)


def desktop_entry(home: Path, name: str, exec_line: str) -> None:
    folder = home / ".local/share/applications"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(f"[Desktop Entry]\nType=Application\nName={name}\nExec={exec_line}\n")


def test_the_session_opens_where_the_browser_setting_says_never_with_its_code_on_a_command_line(private_home,
                                                                                                monkeypatch):
    """api/window.py open_window: the user's [desktop] settings decide (desktop/browser.py open_app): Chrome's app window
    with the session's own profile and its flags, or the default browser; no browser, or flags the window cannot take:
    a link (why said), and no page left behind."""
    folder = private_home / "state"
    folder.mkdir(mode=0o700)
    url = "http://127.0.0.1:40001/?code=once&api=1f3a9c2e"
    started = Started(monkeypatch, found=("chromium", "gio", "firefox"), default="firefox.desktop")
    opened = window.open_window(url, folder=folder, settings={"browser": "chrome",
                                                              "chrome_flags": ["--password-store=basic"]})
    [command] = started.started
    assert opened.how == "chrome" and command[:2] == ["/usr/bin/chromium", "--password-store=basic"]
    assert "--user-data-dir=" + str(folder / "chrome-profile") in command and "--no-first-run" in command
    assert not any("once" in arg for arg in command) and opened.page.parent == folder
    assert f'location.replace("{url}&window=chrome")' in opened.page.read_text()
    desktop_entry(Path(os.environ["HOME"]), "firefox.desktop", "firefox %u")
    opened = window.open_window(url, folder=folder, settings={"browser": "default"})
    assert opened.how == "default" and started.started[-1] == ["/usr/bin/firefox", opened.page.as_uri()]
    assert f'location.replace("{url}")' in opened.page.read_text()
    for page in folder.glob("open-*.html"):
        page.unlink()
    # No browser at all; Chrome window flags it cannot take: a link, saying why, and nothing written.
    Started(monkeypatch, found=())
    assert window.open_window(url, folder=folder, settings={}).why == browser.WORDS["none"]
    refused = window.open_window(url, folder=folder, settings={"browser": "chrome", "chrome_flags": ["--user-data-dir=/x"]})
    assert refused.how == "link" and "chosen by the companion" in refused.why
    assert list(folder.glob("open-*.html")) == []


def test_profile_scan_matches_whole_arguments_and_current_user(tmp_path):
    profile = Path("/private/profile with spaces")
    commands = {
        101: ["chromium", "--user-data-dir=" + str(profile)],
        102: ["chromium", "--user-data-dir", str(profile)],
        103: ["sh", "-c", "chromium --user-data-dir=" + str(profile)],
        104: ["chromium", "--user-data-dir=" + str(profile) + "-other"],
        105: ["/usr/bin/chromium --user-data-dir=" + str(profile) + " --app=http://local"],
        106: ["/usr/bin/chromium --user-data-dir=" + str(profile) + "-other --app=http://local"],
    }
    for pid, args in commands.items():
        path = tmp_path / str(pid)
        path.mkdir()
        (path / "cmdline").write_bytes(b"\0".join(a.encode() for a in args) + b"\0")
    assert sorted(window.profile_processes(profile, proc=tmp_path)) == [101, 102, 105]


def test_the_package_imports_only_the_standard_library_and_its_notebook_names_on_first_use():
    """`import sopa_companion` is the notebook's entry point and every helper's first import: the standard library alone,
    and the notebook's names (api/) only once one is used, then exactly api's objects."""
    script = '''
import sys
sys.path[:0] = sys.argv[1:]
import sopa_companion as sc
assert not any(name.startswith("sopa_companion.") for name in sys.modules if name != "sopa_companion.names"), \\
    sorted(sys.modules)
assert {"launch", "CompanionApp", "CompanionError", "CompanionStartError", "CompanionWarning", "SOPA_API"} <= set(dir(sc))
assert sc.SOPA_API == 1
import sopa_companion.api as api
assert sc.launch is api.launch and sc.CompanionApp is api.CompanionApp and sc.CompanionError is api.CompanionError
try:
    sc.not_a_name
except AttributeError as error:
    assert "not_a_name" in str(error)
else:
    raise AssertionError("an unknown name")
unexpected = {name.split(".")[0] for name in sys.modules} - sys.stdlib_module_names - {"__main__", "sopa_companion"}
assert not unexpected, sorted(unexpected)
assert not any(name.startswith(("sopa_companion.engine", "sopa_companion.server")) for name in sys.modules)
# Of the desktop only the browser opener and Chrome's command, which the launcher shares (standard library only).
desktop = {name for name in sys.modules if name.startswith("sopa_companion.desktop")}
assert desktop <= {"sopa_companion.desktop", "sopa_companion.desktop.chrome", "sopa_companion.desktop.browser"}, \
    sorted(desktop)
'''
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(root.parent)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_notebook_returns_with_only_readonly_handle_and_real_pipes(sample, fake_engine, notebook):
    before = {thread.ident for thread in threading.enumerate() if thread.name.startswith("companion-")}
    start = time.monotonic()
    app = L.launch(sample, open="none", timeout=5)
    assert time.monotonic() - start < 1
    assert app.state in protocol.SESSION_STATES
    # The link comes once the engine's port is known, its state with the session's own event: either may come first.
    until(lambda: app.url is not None and app.state == "ready", callbacks=notebook)
    for name in ("state", "url", "added"):
        with pytest.raises(AttributeError):
            setattr(app, name, "changed")
    assert [name for name in dir(app) if not name.startswith("_")] == ["added", "close", "open", "state", "url"]
    assert app.added == ()
    assert "ready" in repr(app) and "code=" not in repr(app)
    lines = records(fake_engine)
    launched = lines[0]
    assert launched["sid"] == launched["pid"] != os.getsid(0)
    assert launched["cwd"] == launched["config"]["paths"]["state_dir"]
    assert launched["env"]["MPLBACKEND"] == "Agg"
    assert launched["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert launched["env"]["PYTHONPATH"] == str(Path(launched["cwd"]) / "pylib")     # it does not import this copy
    assert lines[1] == {"op": "hello", "protocol": 1}
    bind = next(line for line in lines if line.get("op") == "bind")
    assert bind == {"op": "bind", "path": str(sample.path), "view": "in_place", "name": "sample-store", "backed": True,
                    "zarr": str(sample.path), "memory_only": [], "points_key": None, "linked": [], "written": []}
    assert app.close() is None
    assert app.close() is None
    assert app.state == "closed" and "closed" in repr(app)
    assert not Path(launched["config"]["paths"]["state_dir"]).exists()
    assert not Path(launched["config"]["paths"]["data_dir"]).exists()
    assert {t.ident for t in threading.enumerate() if t.name.startswith("companion-")} == before
    # close() without an answer discards unsaved work, as closing the window does
    assert next(line for line in records(fake_engine) if line.get("op") == "close")["pending"] == "discard"


def test_an_engine_that_imports_this_copy_gets_no_pythonpath(sample, fake_engine, monkeypatch):
    """The engine's interpreter is asked, without PYTHONPATH and in the session's state folder, whether it imports this
    very copy by itself (the app installed in its environment): then it gets no PYTHONPATH, the kernel's own never,
    and no pylib is made; otherwise the session's pylib, made then (the test above)."""
    from sopa_companion.api import handle
    started, asked, made = [], [], []

    def popen(command, **kwargs):
        started.append(kwargs)
        raise OSError("not started in this test")
    monkeypatch.setattr(handle.subprocess, "Popen", popen)
    real_pylib = folders.SessionFolders.pylib
    monkeypatch.setattr(folders.SessionFolders, "pylib", lambda self: made.append(self.state) or real_pylib(self))
    monkeypatch.setenv("PYTHONPATH", "/kernel/own")
    for imports in (True, False):
        monkeypatch.setattr(locate, "imports_this_copy", lambda python, env, *, cwd, imports=imports:
                            asked.append((python, env.get("PYTHONPATH"), cwd)) or imports)
        with pytest.raises(OSError, match="not started"):
            L.launch(sample, open="none", timeout=5)
        python, pythonpath, cwd = asked[-1]
        assert python == str(fake_engine.parent / "fake-engine-python") and pythonpath is None
        assert started[-1]["cwd"] == cwd and Path(cwd).name == Path(cwd).name.lower() and len(Path(cwd).name) == 8
        if imports:
            assert "PYTHONPATH" not in started[-1]["env"] and made == []
        else:
            assert started[-1]["env"]["PYTHONPATH"] == str(Path(cwd) / "pylib") and made == [Path(cwd)]


def test_plain_script_launch_blocks_until_the_window_ends(sample, fake_engine, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_MODE", "auto_end")
    monkeypatch.setattr(L, "_running_loop", lambda: None)
    start = time.monotonic()
    app = L.launch(sample, open="none", timeout=5)
    assert time.monotonic() - start >= .25
    assert app.state == "closed" and app.url is not None and app.added == ()
    assert not app._folders.state.exists() and not app._folders.data.exists()


@pytest.mark.parametrize("mode", ["split", "unknown"])
def test_event_reader_handles_split_lines_and_ignores_unknown_events(mode, sample, fake_engine, notebook, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_MODE", mode)
    app = L.launch(sample, open="none", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    app.close(pending="discard")
    assert app.state == "closed"
    assert next(line for line in records(fake_engine) if line.get("op") == "close")["pending"] == "discard"


def test_link_uses_one_use_code_and_prints_ssh_forwarding(sample, fake_engine, notebook, monkeypatch, capsys):
    monkeypatch.setenv("SSH_CONNECTION", "127.0.0.1 10000 127.0.0.1 22")
    app = L.launch(sample, open="auto", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    app.close(pending="discard")
    config = records(fake_engine)[0]["config"]
    output = capsys.readouterr().out
    assert config["server"]["token"] not in output and "?token=" not in output
    assert app.url in output and "ssh -N -L" in output
    # A ten-minute code, and the session's id: its tab keeps its storage for this session.
    assert parse_qs(urlsplit(app.url).query) == {"code": ["only-once-" + config["session"]["id"] + "-link"],
                                                 "api": [config["session"]["id"]]}
    assert config["session"]["open"] == "link" and config["desktop"]["exit_when_closed"] is False


def test_window_receives_the_users_settings_and_the_sessions_folder(sample, fake_engine, notebook, monkeypatch):
    opened = []

    class Browser:
        def wait(self, timeout): return 0

    monkeypatch.setattr(L, "open_mode", lambda value: "window")
    settings = L._settings()
    monkeypatch.setattr(L, "_settings", lambda: {**settings, "desktop": {"browser": "chrome", "chrome_flags": []}})
    monkeypatch.setattr(window, "open_window", lambda url, **kw: opened.append((url, kw)) or
                        browser.Opened("chrome", "Google Chrome", None, Browser()))
    app = L.launch(sample, open="window", timeout=5)
    until(lambda: bool(opened), callbacks=notebook)
    [(url, kwargs)] = opened
    assert url == app.url and kwargs["folder"] == app._folders.state
    assert kwargs["settings"] == {"browser": "chrome", "chrome_flags": []}
    assert parse_qs(urlsplit(url).query)["api"] == [app._config["session"]["id"]]
    assert records(fake_engine)[0]["config"]["desktop"]["exit_when_closed"] is True
    assert "browser" not in records(fake_engine)[0]["config"]["desktop"]   # the engine never needs it
    assert not app.url.endswith("-link")                     # a window opened now: its short-lived code
    app.close(pending="discard")


def test_a_window_that_cannot_open_prints_a_ten_minute_link_with_why(sample, fake_engine, notebook, monkeypatch, capsys):
    """No browser after all (none found, a sandbox that cannot read the page): the why, then a link printed for a
    person, a code the engine keeps ten minutes, never the window's code of seconds; app.url is that link."""
    monkeypatch.setattr(L, "open_mode", lambda value: "window")
    monkeypatch.setattr(window, "open_window", lambda url, **kw: browser.Opened("link", why=browser.WORDS["none"]))
    app = L.launch(sample, open="window", timeout=5)
    until(lambda: app.url is not None and "-link" in app.url, callbacks=notebook)
    output = capsys.readouterr().out
    assert f"{browser.WORDS['none']}\nSoPA companion: {app.url}\n" in output
    app.close(pending="discard")


def test_a_default_browser_that_signs_nothing_in_gets_a_link_after_the_wait(sample, fake_engine, notebook, monkeypatch,
                                                                            capsys):
    """A browser took the page but no page signed in (a URL forwarder as the desktop's https handler, a browser that
    failed without a word): after the wait, a link too; a page that signs in meanwhile needs none. The app's own Chrome
    window is not waited for."""
    from sopa_companion.api import handle
    monkeypatch.setattr(handle, "OPEN_WAIT_S", 0.3)
    monkeypatch.setattr(handle, "SIGN_IN_LOOK_S", 0.05)
    monkeypatch.setattr(L, "open_mode", lambda value: "window")
    real, looks, signs = window.request, [0], [False]

    def request(port, token, path, **kw):
        if path != "/api/v1/health":
            return real(port, token, path, **kw)
        looks[0] += 1                                    # a page that signs in: one more at every look
        return {"ok": True, "presence": {"windows": looks[0] if signs[0] else 0, "connects": looks[0]}}
    monkeypatch.setattr(window, "request", request)
    how = ["default"]
    monkeypatch.setattr(window, "open_window", lambda url, **kw: browser.Opened(how[0], "Firefox", None))
    app = L.launch(sample, open="window", timeout=5)
    until(lambda: app.url is not None and "-link" in app.url, callbacks=notebook)
    assert f"{window.NOT_YET}\nSoPA companion: {app.url}\n" in capsys.readouterr().out
    app.close(pending="discard")
    for how[0], signs[0] in (("default", True), ("chrome", False)):
        app = L.launch(sample, open="window", timeout=5)
        until(lambda: app.url is not None, callbacks=notebook)
        time.sleep(0.6)
        assert "-link" not in app.url and "SoPA companion: http" not in capsys.readouterr().out, how[0]
        app.close(pending="discard")


def test_open_opens_the_app_again_with_a_fresh_code(sample, fake_engine, notebook, monkeypatch, capsys):
    """app.open(): a window or tab that went away (a crash, a closed laptop) is opened again, the session going on
    meanwhile: a new window when the session opens one, else a new link; never once the app is closed, and only on the
    thread that opened it."""
    codes = iter(range(100))
    real = window.request
    monkeypatch.setattr(window, "request", lambda port, token, path, **kw: {"code": f"fresh-{next(codes)}"}
                        if "launch-code" in path else real(port, token, path, **kw))
    app = L.launch(sample, open="link", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    first = app.url
    assert first in capsys.readouterr().out
    app.open()
    assert app.url != first and app.url in capsys.readouterr().out and "api=" in app.url
    app.close(pending="discard")
    with pytest.raises(CompanionError, match="not open"):
        app.open()
    opened = []
    monkeypatch.setattr(L, "open_mode", lambda value: "window")
    monkeypatch.setattr(window, "open_window", lambda url, **kw: opened.append(url) or browser.Opened("default", "Firefox"))
    app = L.launch(sample, open="window", timeout=5)
    until(lambda: bool(opened), callbacks=notebook)
    app.open()
    assert len(opened) == 2 and opened[0] != opened[1] and app.url == opened[1]
    errors = []

    def elsewhere():
        try:
            app.open()
        except CompanionError as error:
            errors.append(str(error))
    thread = threading.Thread(target=elsewhere)
    thread.start()
    thread.join()
    assert errors == ["Open the app on the thread that opened it"]
    app.close(pending="discard")


def test_a_link_shown_is_replaced_once_used_so_no_code_stays_in_the_notebook(monkeypatch, capsys):
    """In a notebook the link is one output, which the engine's word that a link signed a browser in replaces (on the
    kernel's loop) with words that hold no code; in a plain script those words are printed."""
    import builtins
    import IPython.display
    updates, shown_raw = [], []

    class Handle:
        def update(self, value, raw=False):
            updates.append((value, raw))

    def display(value, raw=False, display_id=False):
        shown_raw.append((value, raw, display_id))
        return Handle()
    monkeypatch.setattr(builtins, "get_ipython", lambda: object(), raising=False)
    monkeypatch.setattr(IPython.display, "display", display)
    loop = []
    url = "http://127.0.0.1:41000/?code=secret&api=1f3a9c2e"
    shown = window.show_link(url, 41000, why="No default web browser was found.", schedule=loop.append)
    assert shown_raw == [({"text/plain": f"No default web browser was found.\nSoPA companion: {url}"}, True, True)]
    at = time.mktime((2026, 10, 25, 14, 5, 0, 0, 0, -1))
    shown.used(at)
    assert updates == [] and len(loop) == 1                  # on the kernel's loop
    loop.pop()()
    [(value, raw)] = updates
    assert raw is True and "secret" not in value["text/plain"]
    assert value["text/plain"] == ("SoPA companion: a one-use link was shown here and used at 14:05. If that was not you, "
                                   "close the app with its close().")
    shown.used(at)
    assert loop == []                                        # once
    monkeypatch.delattr(builtins, "get_ipython")
    plain = window.show_link("http://127.0.0.1:41000/?code=other", 41000)
    assert capsys.readouterr().out == "SoPA companion: http://127.0.0.1:41000/?code=other\n"
    plain.used(at)
    assert "used at 14:05" in capsys.readouterr().out


def test_the_engines_word_that_a_link_was_used_reaches_the_link_shown(sample, fake_engine, notebook, monkeypatch):
    used = []
    monkeypatch.setattr(window, "show_link", lambda url, port, **kw: types.SimpleNamespace(used=used.append,
                                                                                          state="shown"))
    app = L.launch(sample, open="link", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    app._event(protocol.build_event("link", state="used"))
    assert len(used) == 1 and abs(used[0] - time.time()) < 5
    app.close(pending="discard")


def test_a_link_shown_while_no_page_signed_in_is_taken_out_once_the_browsers_page_signs_in(sample, fake_engine, notebook,
                                                                                          monkeypatch, outputs):
    """A browser that took the page but showed nothing within OPEN_WAIT_S gets a link to open by hand beside it. Its
    page signing in after all (its own launch code traded: presence.windows; another tab connecting is no sign of it)
    takes that link out of the notebook, so no live code stays there."""
    from sopa_companion.api import handle
    monkeypatch.setattr(handle, "OPEN_WAIT_S", 0.3)
    monkeypatch.setattr(handle, "SIGN_IN_LOOK_S", 0.05)
    monkeypatch.setattr(L, "open_mode", lambda value: "window")
    real, presence = window.request, {"connects": 0, "windows": 0}
    monkeypatch.setattr(window, "request", lambda port, token, path, **kw: {"ok": True, "presence": dict(presence)}
                        if path == "/api/v1/health" else real(port, token, path, **kw))
    monkeypatch.setattr(window, "open_window", lambda url, **kw: browser.Opened("default", "Firefox", None))
    app = L.launch(sample, open="window", timeout=5)
    until(lambda: outputs and "-link" in outputs[0].text, callbacks=notebook)
    assert outputs[0].text == f"{window.NOT_YET}\nSoPA companion: {app.url}"
    presence["connects"] = 3                               # another tab of the session connecting: not this browser
    time.sleep(0.4)
    until(lambda: True, callbacks=notebook)
    assert "code=" in outputs[0].text
    presence["windows"] = 1                                # the browser's own page signed in after all
    until(lambda: "code=" not in outputs[0].text, callbacks=notebook)
    assert outputs[0].text == window.TAKEN_OUT["opened"]
    app._event(protocol.build_event("link", state="used"))  # someone used that link's code, live for ten minutes yet
    until(lambda: outputs[0].text.startswith("SoPA companion: a one-use link was shown here and used at"), callbacks=notebook)
    app.close(pending="discard")


def test_a_newer_link_takes_the_older_out_and_the_word_that_a_link_was_used_reaches_the_newest(sample, fake_engine,
                                                                                               notebook, monkeypatch,
                                                                                               outputs):
    """app.open() in a session that shows links: the new link's output comes, and the older one's code goes from the
    notebook at once (one live code there at most). The engine's word that a link was used (it cannot say which) goes
    to the newest link not said to be used yet."""
    codes = iter(range(100))
    real = window.request
    monkeypatch.setattr(window, "request", lambda port, token, path, **kw: {"code": f"fresh-{next(codes)}"}
                        if "launch-code" in path else real(port, token, path, **kw))
    app = L.launch(sample, open="link", timeout=5)
    until(lambda: len(outputs) == 1, callbacks=notebook)
    first = app.url
    assert outputs[0].text == f"SoPA companion: {first}"
    app.open()
    until(lambda: len(outputs) == 2 and "code=" not in outputs[0].text, callbacks=notebook)
    assert outputs[0].text == window.TAKEN_OUT["newer"] and outputs[1].text == f"SoPA companion: {app.url}" != first
    app._event(protocol.build_event("link", state="used"))
    until(lambda: "code=" not in outputs[1].text, callbacks=notebook)
    assert outputs[1].text.startswith("SoPA companion: a one-use link was shown here and used at")
    assert outputs[0].text == window.TAKEN_OUT["newer"]
    app._event(protocol.build_event("link", state="used"))  # another use: the older link's code, used by someone
    until(lambda: outputs[0].text.startswith("SoPA companion: a one-use link was shown here and used at"),
          callbacks=notebook)
    app.close(pending="discard")


def test_the_sessions_chrome_window_is_closed_with_it_even_when_a_later_open_was_another_browser(sample, fake_engine,
                                                                                                notebook, monkeypatch):
    """The session's own Chrome window, once opened, ends with the session (its profile's processes, the one it started
    waited for), whatever app.open() opened since (Chrome failing to start the second time: the default browser)."""
    closed, waited = [], []
    monkeypatch.setattr(window, "close_profile", closed.append)
    monkeypatch.setattr(L, "open_mode", lambda value: "window")
    process = types.SimpleNamespace(wait=lambda timeout=None: waited.append(timeout) or 0)
    opened = iter([browser.Opened("chrome", "Google Chrome", None, process), browser.Opened("default", "Firefox", None)])
    monkeypatch.setattr(window, "open_window", lambda url, **kw: next(opened))
    app = L.launch(sample, open="window", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    app.open()
    app.close(pending="discard")
    assert closed == [app._folders.state / "chrome-profile"] and waited == [1]


def test_the_sessions_chrome_window_is_closed_with_it_and_only_one_it_opened(sample, fake_engine, notebook, monkeypatch):
    closed = []
    monkeypatch.setattr(window, "close_profile", closed.append)
    monkeypatch.setattr(L, "open_mode", lambda value: "window")
    process = types.SimpleNamespace(wait=lambda timeout=None: 0)
    for how in ("default", "chrome"):
        monkeypatch.setattr(window, "open_window", lambda url, how=how, **kw: browser.Opened(how, "B", None, process))
        app = L.launch(sample, open="window", timeout=5)
        until(lambda: app.url is not None, callbacks=notebook)
        app.close(pending="discard")
    assert closed == [app._folders.state / "chrome-profile"]   # the Chrome one only


def test_the_kernel_reads_where_the_app_opens_and_names_a_bad_setting_before_anything_starts(sample, fake_engine,
                                                                                            monkeypatch, private_home,
                                                                                            notebook):
    """[desktop] browser is the kernel's to read, from the app folder's files and then the user's own (the user's
    wins): a bad value is refused in plain words before any folder or engine; the session's engine is never given it."""
    settings = L._settings()
    monkeypatch.setattr(L, "_settings", lambda: {**settings, "desktop": {"browser": "firefox"}})
    with pytest.raises(ValueError) as refused:
        L.launch(sample, open="none", timeout=5)
    assert str(refused.value) == browser.CHOICE_WORDS.format(value="firefox")
    assert records(fake_engine) == [] and not list((private_home / "runtime").glob("sopa-companion/*"))


def test_the_users_own_file_says_where_the_app_opens(private_home, monkeypatch):
    """The app folder's file says "default", the user's own "chrome": the kernel opens the session's Chrome window
    (the engine, which reads the app folder's file, never needs to know: tests/test_api_session.py)."""
    root = private_home / "code"
    root.mkdir()
    (root / "config.example.toml").write_text('[desktop]\nbrowser = "default"\n')
    user = private_home / "config/sopa-companion/config.toml"
    user.parent.mkdir(parents=True)
    user.write_text('[desktop]\nbrowser = "chrome"\n')
    monkeypatch.setattr(L, "ROOT", root)
    assert L._settings()["desktop"]["browser"] == "chrome"
    assert browser.choice(L._settings()) == ("chrome", None)


def test_start_failure_is_redacted_and_removes_folders(sample, fake_engine, monkeypatch):
    from sopa_companion.api import CompanionStartError

    monkeypatch.setenv("FAKE_ENGINE_MODE", "crash")
    monkeypatch.setattr(L, "_running_loop", lambda: None)
    with pytest.raises(CompanionStartError, match="fake engine broke") as caught:
        L.launch(sample, open="none", timeout=5)
    captured = records(fake_engine)[0]
    assert captured["config"]["server"]["token"] not in str(caught.value)
    assert "secret=[private]" in str(caught.value)
    assert not Path(captured["config"]["paths"]["state_dir"]).exists()
    assert not Path(captured["config"]["paths"]["data_dir"]).exists()


@pytest.mark.parametrize("mode", ["no_ready", "malformed", "bad_pid"])
def test_bad_start_is_bounded_and_cleans_up(mode, sample, fake_engine, monkeypatch):
    from sopa_companion.api import CompanionStartError

    monkeypatch.setenv("FAKE_ENGINE_MODE", mode)
    monkeypatch.setattr(L, "_running_loop", lambda: None)
    with pytest.raises(CompanionStartError):
        L.launch(sample, open="none", timeout=.3)
    captured = records(fake_engine)[0]
    assert not Path(captured["config"]["paths"]["state_dir"]).exists()
    assert not Path(captured["config"]["paths"]["data_dir"]).exists()


def test_close_timeout_kills_only_its_child_and_reaps_it(sample, fake_engine, notebook, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_MODE", "hang")
    app = L.launch(sample, open="none", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    start = time.monotonic()
    app.close(pending="discard", timeout=.05)
    assert time.monotonic() - start < 4
    assert app._process.poll() is not None and not app._thread.is_alive()
    assert not app._folders.state.exists() and not app._folders.data.exists()


def test_object_names_and_auto_save_target_are_refreshed(sample, fake_engine, notebook, monkeypatch):
    from sopa import settings

    app = L.launch(sample, open="none", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    sample.images["another_image"] = sample.images["image"]
    monkeypatch.setattr(settings, "auto_save_on_disk", False)
    until(lambda: any(line.get("op") == "object" and "another_image" in line["names"] for line in records(fake_engine)),
          callbacks=notebook)
    latest = [line for line in records(fake_engine) if line.get("op") == "object"][-1]
    assert latest == {"op": "object", "names": ["another_image", "image"], "zarr": None, "auto_save": False}
    app.close(pending="discard")


@pytest.mark.parametrize("in_notebook", [False, True])
def test_delivery_hook_applies_on_main_thread_without_a_public_wait(sample, fake_engine, monkeypatch, in_notebook):
    callbacks, threads = [], []
    monkeypatch.setattr(L, "_running_loop", lambda: callbacks.append if in_notebook else None)
    monkeypatch.setenv("FAKE_ENGINE_MODE", "delivery")

    class Delivery:
        def __init__(self, app): self.app, self.events = app, []
        def receive(self, event):
            threads.append(("receive", threading.get_ident()))
            self.events.append(event)
            self.app._send("received", id=event["id"])
            if self.app._schedule is not None:
                self.app._schedule(self.apply_pending)
        def apply_pending(self):
            while self.events:
                event = self.events.pop(0)
                threads.append(("apply", threading.get_ident()))
                sample.attrs = {**sample.attrs, "saved_in_test": True}
                self.app._record_added([event["shapes_key"]])
                self.app._send("applied", id=event["id"], state="added", keys=[event["shapes_key"]], clash=[], message=None)
        def close(self): threads.append(("close", threading.get_ident()))

    monkeypatch.setattr(L, "_attach_delivery", lambda app, sdata: Delivery(app))
    app = L.launch(sample, open="none", timeout=5)
    until(lambda: app._done.is_set(), callbacks=callbacks)
    app.close()
    assert app.state == "closed" and app.added == ("new_cells",)
    assert sample.attrs["saved_in_test"] is True
    assert all(ident == threading.get_ident() for name, ident in threads if name in ("apply", "close"))
    assert all(ident != threading.get_ident() for name, ident in threads if name == "receive")
    assert sum(name == "close" for name, ident in threads) == 1
    assert "d1" not in app.added  # ended.added contains delivery ids, not keys


def test_request_headers_disable_proxies_and_redirects(monkeypatch):
    seen = []

    class Opener:
        def open(self, req, *, timeout):
            seen.append((req, timeout))
            return io.BytesIO(b'{"code":"one use + slash/"}')

    def opener(*handlers):
        assert any(isinstance(h, window.ProxyHandler) and h.proxies == {} for h in handlers)
        assert any(isinstance(h, window._NoRedirect) for h in handlers)
        return Opener()

    monkeypatch.setattr(window, "build_opener", opener)
    url = window.launch_url(40123, "private-token", timeout=.2)
    req, timeout = seen[0]
    assert req.full_url == "http://127.0.0.1:40123/api/v1/session/launch-code"
    window.launch_url(40123, "private-token", link=True, timeout=.2)
    assert seen[1][0].full_url == "http://127.0.0.1:40123/api/v1/session/launch-code?link=1"   # a link opened by hand
    assert req.method == "POST" and req.data == b"{}" and timeout == .2
    assert dict(req.header_items()) == {"Authorization": "Bearer private-token", "Origin": "http://127.0.0.1:40123", "Accept": "application/json"}
    assert parse_qs(urlsplit(url).query) == {"code": ["one use + slash/"]}
    assert "private-token" not in url
    assert window._NoRedirect().redirect_request(req, None, 302, "moved", {}, "http://elsewhere") is None


def test_private_profile_cleanup_leaves_other_profile_running(tmp_path):
    profile = tmp_path / "profile with spaces"
    command = [sys.executable, "-c", "import time; time.sleep(20)"]
    mine = subprocess.Popen([*command, "--user-data-dir=" + str(profile)], start_new_session=True)
    other = subprocess.Popen([*command, "--user-data-dir=" + str(profile) + "-other"], start_new_session=True)
    try:
        until(lambda: mine.pid in window.profile_processes(profile))
        window.close_profile(profile)
        assert mine.wait(timeout=2) < 0
        assert other.poll() is None
    finally:
        for process in (mine, other):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=2)


def test_unbacked_and_wrong_type_are_refused_before_spawn(sample, fake_engine):
    with pytest.raises(TypeError, match="SpatialData"):
        L.launch(object())
    sample.path = None
    with pytest.raises(ValueError, match="write it to a zarr first"):
        L.launch(sample)
    assert not fake_engine.exists()


@pytest.mark.parametrize("kwargs", [{"tool": "other"}, {"points_key": "missing"}, {"copy": True}, {"copy": 1},
    {"open": "browser"}, {"timeout": 0}, {"timeout": float("inf")}, {"timeout": float("nan")},
    {"timeout": True}, {"gpu": True}, {"gpu": -1}, {"gpu": {"qc": -1}}, {"gpu": {"unknown": 0}}])
def test_bad_options_do_not_spawn(kwargs, sample, fake_engine):
    with pytest.raises(ValueError):
        L.launch(sample, **kwargs)
    assert not fake_engine.exists()


def test_main_thread_required_before_any_process_or_folder(sample, fake_engine):
    errors = []

    def work():
        try:
            L.launch(sample)
        except CompanionError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert errors == ["Open the companion on the main thread"]
    assert not fake_engine.exists()


def test_engine_python_and_binary_fast_checks(sample, fake_engine, monkeypatch):
    config = L._settings()
    config["api"]["engine_python"] = "/no-such-python"
    monkeypatch.setattr(L, "_settings", lambda: config)
    with pytest.raises(CompanionError, match="engine Python"):
        L.launch(sample)
    config["api"]["engine_python"] = sys.executable
    real_spec = L.importlib.util.find_spec
    monkeypatch.setattr(L.importlib.util, "find_spec", lambda name: None if name == "fastapi" else real_spec(name))
    with pytest.raises(CompanionError) as refused:
        L.launch(sample)
    said = str(refused.value)                                  # every engine package this kernel lacks, fastapi first
    assert said.startswith("The engine needs fastapi") and said.endswith(
        "; install the app into this environment as the README's Install section says, or set api.engine_python")
    config["api"]["engine_python"] = str(fake_engine.parent / "fake-engine-python")
    config["proseg"]["binary"] = "/no-such-program"
    with pytest.raises(CompanionError) as refused:
        L.launch(sample)
    assert str(refused.value) == "The companion's proseg-live program is missing (/no-such-program is not there)"
    assert not fake_engine.exists()


def test_sopa_points_rule_and_image_only_validation(sample, fake_engine, notebook):
    import pandas as pd
    from spatialdata.models import PointsModel

    with pytest.raises(ValueError, match="Proseg needs transcripts"):
        L.launch(sample, tool="proseg")
    points = PointsModel.parse(pd.DataFrame({"x": [0., 1.], "y": [0., 1.], "gene": ["a", "b"]}), feature_key="gene")
    sample.points["transcripts"] = points
    sample.points["other_points"] = points
    with pytest.raises(ValueError, match="points_key"):
        L.launch(sample, tool="proseg")
    sample.attrs = {**sample.attrs, "transcripts_dataframe": "transcripts"}
    app = L.launch(sample, open="none", tool="proseg", gpu={"analysis": 2, "stain": 1}, timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    app.close(pending="discard")
    lines = records(fake_engine)
    assert next(line for line in lines if line.get("op") == "bind")["points_key"] == "transcripts"
    assert lines[0]["config"]["analysis"]["gpu"] == 2 and lines[0]["config"]["stain"]["gpu"] == 1


def test_the_mixing_check_learns_in_the_configured_environment_else_the_kernels(sample, fake_engine, notebook,
                                                                              monkeypatch, private_home):
    """[qc] python, as [analysis] python and [stain] python: an environment with scvi-tools named in the companion's
    settings (the desktop's sopa environment, say); "auto" (or none), the kernel's own when it has torch, scvi and
    scviva; one that is missing: a warning, and the check is off."""
    env = private_home / "qc-env" / "bin" / "python"
    env.parent.mkdir(parents=True)
    env.write_text("#!/bin/sh\n")
    env.chmod(0o700)
    settings = L._settings()
    def launched(qc=None):
        monkeypatch.setattr(L, "_settings", lambda: {**settings, **({"qc": qc} if qc is not None else {})})
        app = L.launch(sample, open="none", timeout=5)
        until(lambda: app.url is not None, callbacks=notebook)
        python = app._config["qc"]["python"]
        app.close(pending="discard")
        return python
    assert launched({"python": str(env)}) == str(env)
    assert launched() == "off"                                          # the kernel here has no scvi (fake_engine)
    kernel_has(monkeypatch, {"torch", "scvi", "scviva"})
    assert launched() == launched({"python": "auto"}) == os.path.abspath(sys.executable)
    assert launched({"python": "off"}) == "off"
    with pytest.warns(L.CompanionWarning, match="optional Python environment"):
        assert launched({"python": "/no-such-python"}) == "off"
    assert launched({"python": str(env)}) == str(env), "the one named wins over the kernel's"


def test_launch_writes_off_or_a_full_path_never_auto(sample, fake_engine, notebook, monkeypatch, private_home):
    """session.toml's interpreters are what launch() found in the kernel, never "auto" or "": the session's engine may
    run in another interpreter ([api] engine_python), which must not look again. Clusters & UMAP's "auto": the kernel's
    own Python when it has what their CPU path imports (analysis/backend.py: what `pip install sopa` brings), else
    "off", with what the kernel lacks beside it ([locate.analysis]; the mixing check's too), which the session's engine
    keeps: it names those packages rather than an "off" nobody set."""
    from sopa_companion.engine.config import load_config
    settings = L._settings()

    def handed(analysis):
        monkeypatch.setattr(L, "_settings", lambda: {**settings, "analysis": analysis})
        app = L.launch(sample, open="none", timeout=5)
        until(lambda: app.url is not None, callbacks=notebook)
        config = [line for line in records(fake_engine) if "config" in line][-1]["config"]
        engine = load_config(app._folders.state / "session.toml")          # what the session's engine reads
        app.close(pending="discard")
        return config, engine

    kernel_has(monkeypatch, set(CPU_MODULES))
    for value in ("auto", ""):
        config, engine = handed({"python": value})
        assert config["analysis"]["python"] == os.path.abspath(sys.executable), value
        assert engine["analysis"]["python"] == os.path.abspath(sys.executable)
        assert engine["locate"]["analysis"]["how"] == "setting"            # the engine does not look again
        assert "analysis" not in config.get("locate", {})
    kernel_has(monkeypatch, set())
    config, engine = handed({"python": "auto"})
    assert config["analysis"]["python"] == "off" and engine["analysis"]["python"] == ""
    for feature, modules in (("analysis", list(CPU_MODULES)), ("qc", ["torch", "scvi", "scviva"])):
        found = {"path": "", "how": "auto", "missing": modules, "problem": "", "fix": "", "passed": []}
        assert config["locate"][feature] == engine["locate"][feature] == found, feature
    assert engine["qc"]["python"] == ""
    config, engine = handed({"python": "off"})                             # set off: off, nothing to report
    assert engine["locate"]["analysis"]["how"] == "off" and "analysis" not in config["locate"]
    for section in ("qc", "stain", "analysis"):
        assert config[section]["python"] not in ("", "auto")
    assert config["stain"]["instanseg_python"] == "off"                       # dormant: off


def test_launch_finds_the_installed_proseg_live(sample, fake_engine, notebook, monkeypatch, private_home):
    """[proseg] binary "auto" (or none): the proseg-live installed with the app in the kernel's environment (its
    scripts folder), handed to the engine as a full path; missing, launch() says what makes it."""
    import types
    scripts = private_home / "env" / "bin"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(locate, "sysconfig", types.SimpleNamespace(
        get_path=lambda name, scheme=None: str(private_home / "user" / "bin" if scheme else scripts),
        get_preferred_scheme=lambda kind: "posix_user"))
    import shutil
    tools = private_home / "tools"                                          # a PATH with setpriv and no proseg-live
    tools.mkdir()
    (tools / "setpriv").symlink_to(shutil.which("setpriv") or "/usr/bin/setpriv")
    monkeypatch.setenv("PATH", str(tools))
    installed = private_home / "site-packages" / "sopa_companion"            # an installed copy: no pyproject.toml
    installed.mkdir(parents=True)
    monkeypatch.setattr(L, "ROOT", installed)
    settings = L._settings()
    for binary in ("auto", None):
        monkeypatch.setattr(L, "_settings", lambda: {**settings, "proseg": {} if binary is None else {"binary": binary}})
        with pytest.raises(CompanionError) as refused:
            L.launch(sample, open="none", timeout=5)
        assert str(refused.value) == ("The companion's proseg-live program is missing (install it as the README's "
                                      "Install section says)")
    program = scripts / "proseg-live"
    program.write_text("#!/bin/sh\n")
    program.chmod(0o755)
    app = L.launch(sample, open="none", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    assert app._config["proseg"]["binary"] == str(program)
    app.close(pending="discard")


def test_two_sessions_have_separate_ids_tokens_and_folders(sample, fake_engine, notebook):
    a = L.launch(sample, open="none", timeout=5)
    until(lambda: a.url is not None, callbacks=notebook)
    b = L.launch(sample, open="none", timeout=5)
    until(lambda: b.url is not None, callbacks=notebook)
    configs = [a._config, b._config]
    assert configs[0]["server"]["token"] != configs[1]["server"]["token"]
    assert configs[0]["session"]["id"] != configs[1]["session"]["id"]
    assert a._folders.data != b._folders.data and a._folders.state != b._folders.state
    a.close(pending="discard")
    assert b._process.poll() is None and b._folders.data.is_dir()
    b.close(pending="discard")


def test_phase_one_warns_for_memory_only_elements_and_auto_save_off(sample, fake_engine, notebook, monkeypatch):
    from sopa import settings

    monkeypatch.setattr(type(sample), "_symmetric_difference_with_zarr_store", lambda s: (["shapes/unsaved_cells"], []))
    monkeypatch.setattr(settings, "auto_save_on_disk", False)
    with pytest.warns(L.CompanionWarning) as seen:
        app = L.launch(sample, open="none", timeout=5)
    assert len(seen) == 2
    until(lambda: app.url is not None, callbacks=notebook)
    app.close(pending="discard")
    bind = next(line for line in records(fake_engine) if line.get("op") == "bind")
    assert bind["memory_only"] == ["unsaved_cells"] and bind["zarr"] == str(sample.path)
    current = next(line for line in records(fake_engine) if line.get("op") == "object")
    assert current["auto_save"] is False and current["zarr"] is None


def test_settings_merge_user_override_and_optional_python_checks(private_home, monkeypatch):
    root = private_home / "code"
    root.mkdir()
    (root / "config.example.toml").write_text('[analysis]\ngpu=0\npython=""\n[api]\nscratch_dir="/tmp/example"\n')
    (root / "config.local.toml").write_text('[analysis]\ngpu=1\n')
    user = private_home / "config/sopa-companion/config.toml"
    assert user == dirs.user_config_file()                                   # your own settings file
    user.parent.mkdir(parents=True)
    user.write_text('[api]\nengine_python="/opt/python"\n')
    monkeypatch.setattr(L, "ROOT", root)
    assert L._settings() == {"analysis": {"gpu": 1, "python": ""},
                             "api": {"scratch_dir": "/tmp/example", "engine_python": "/opt/python"}}
    assert L._python("qc", {"qc": {"python": sys.executable}}) == os.path.abspath(sys.executable)
    # A venv's bin/python is a symlink to its base interpreter: kept as given, never resolved (the base interpreter,
    # started by its own path, finds none of the venv's packages: the engine could not import fastapi).
    link = private_home / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(sys.executable)
    assert L._python("analysis", {"analysis": {"python": str(link)}}) == str(link) and L._path(str(link)) == str(link)
    with pytest.warns(L.CompanionWarning, match="optional Python environment"):
        assert L._python("stain", {"stain": {"python": "/no-such-python"}}) == "off"
    assert L._python("qc", {"qc": {"python": "off"}}) == "off"
    # "auto" (or empty, or none): the kernel's own when it has the feature's packages, looked up without importing.
    kernel_has(monkeypatch, {"sopa", "spatialdata"})
    for settings in ({}, {"zarr": {}}, {"zarr": {"python": ""}}, {"zarr": {"python": "auto"}}):
        assert L._python("zarr", settings) == os.path.abspath(sys.executable), settings
    kernel_has(monkeypatch, set())
    assert L._python("zarr", {}) == "off"


def test_loop_detection_uses_running_main_thread_loop_and_not_a_stale_kernel(monkeypatch):
    callbacks = []
    loop = types.SimpleNamespace(asyncio_loop=types.SimpleNamespace(is_running=lambda: False), add_callback=callbacks.append)
    shell = types.SimpleNamespace(kernel=types.SimpleNamespace(io_loop=loop))
    monkeypatch.setattr(L.builtins, "get_ipython", lambda: shell, raising=False)
    assert L._running_loop() is None
    loop.asyncio_loop.is_running = lambda: True
    schedule = L._running_loop()
    schedule("test")
    assert callbacks == ["test"]

    # asyncio's default loop creates a socketpair, also forbidden here. Use an
    # already-running loop stand-in to check capture without opening sockets.
    running = types.SimpleNamespace(call_soon_threadsafe=callbacks.append)
    monkeypatch.setattr(L.asyncio, "get_running_loop", lambda: running)
    assert L._running_loop() == callbacks.append


def test_attach_delivery_is_the_delivery_modules_and_a_missing_dependency_is_said(sample, monkeypatch):
    """launch()'s delivery is api/deliver.py's attach(app, sdata), which ships with the handle: a dependency missing
    inside it is raised, never taken for a build without delivery."""
    import sopa_companion.api.deliver as module

    app, adapter = object(), object()
    monkeypatch.setattr(module, "attach", lambda actual_app, actual_sdata: adapter if actual_app is app and actual_sdata is sample else None)
    assert L._attach_delivery(app, sample) is adapter

    def broken(app, sdata):
        raise ModuleNotFoundError("dependency missing", name="required_by_delivery")

    monkeypatch.setattr(module, "attach", broken)
    with pytest.raises(ModuleNotFoundError, match="dependency missing"):
        L._attach_delivery(app, sample)


def test_control_backpressure_does_not_block_the_kernel_or_close(sample, fake_engine, notebook, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_MODE", "dont_read")
    app = L.launch(sample, open="none", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    start = time.monotonic()
    app._send("object", names=["a" * 100000], zarr=None, auto_save=False)
    assert time.monotonic() - start < .5
    app.close(pending="discard", timeout=.1)
    assert time.monotonic() - start < 4
    assert not app._thread.is_alive() and app._process.poll() is not None
    assert not app._folders.data.exists() and not app._folders.state.exists()


def test_missing_delivery_handler_reports_receive_failed(sample, fake_engine, notebook, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_MODE", "delivery")
    app = L.launch(sample, open="none", timeout=5)
    until(lambda: any(line.get("op") == "receive_failed" for line in records(fake_engine)), callbacks=notebook)
    app.close(pending="discard")
    assert app.added == ()
    assert not any(line.get("op") == "received" for line in records(fake_engine))


def test_spawn_failure_closes_pipes_and_removes_both_folders(sample, fake_engine, monkeypatch):
    handle = importlib.import_module("sopa_companion.api.handle")
    created = []
    create = folders.SessionFolders.create

    def track(*args, **kwargs):
        session = create(*args, **kwargs)
        created.append(session)
        return session

    def cannot_spawn(*args, **kwargs):
        raise OSError("test spawn failed")

    before = len(list(Path("/proc/self/fd").iterdir()))
    monkeypatch.setattr(folders.SessionFolders, "create", track)
    monkeypatch.setattr(handle.subprocess, "Popen", cannot_spawn)
    with pytest.raises(OSError, match="test spawn failed"):
        L.launch(sample, open="none")
    assert created and not created[0].data.exists() and not created[0].state.exists()
    assert len(list(Path("/proc/self/fd").iterdir())) == before
    assert not fake_engine.exists()


def test_launch_sweeps_a_crashed_session_once(sample, fake_engine, notebook, private_home):
    crashed = folders.SessionFolders.create(private_home / "old-scratch", "/not-touched")
    dead_owner(crashed)
    with pytest.warns(L.CompanionWarning, match="Removed 1 temporary"):
        app = L.launch(sample, open="none", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    assert not crashed.state.exists() and not crashed.data.exists()
    assert folders.sweep() == ()
    app.close(pending="discard")


def test_sweeper_ignores_an_invalid_owner_record(private_home):
    state = folders.state_root() / "12345678"
    state.mkdir(mode=0o700)
    (state / "owner.json").write_text("[]")
    assert folders.sweep() == ()
    assert state.is_dir()


# ---- the window's launch code, the save target, settings, GPUs and folders ------------------------------------------
def test_a_window_opens_a_private_page_and_no_command_line_carries_its_code(sample, fake_engine, notebook, monkeypatch):
    """Any account on this computer can read every process's command line (/proc/<pid>/cmdline); a window opened with
    --app=<address with its one-use code> let another account use the code first and sign in as the window. Chrome
    starts on a page in the session's private folder (0700), readable by this user only (0600), that sends it on to
    the code's address; the page goes once its code can no longer sign in, and with the session's folder at the end."""
    monkeypatch.setattr(L, "open_mode", lambda value: "window")
    settings = L._settings()
    monkeypatch.setattr(L, "_settings", lambda: {**settings, "desktop": {"browser": "chrome"}})
    monkeypatch.setattr(window, "PAGE_KEEP_S", 0.5)
    started = Started(monkeypatch, found=("google-chrome",))
    app = L.launch(sample, open="window", timeout=5)
    until(lambda: bool(started.started), callbacks=notebook)
    code = parse_qs(urlsplit(app.url).query)["code"][0]
    [argv] = started.started
    assert argv[0] == "/usr/bin/google-chrome" and not any(code in arg or "code=" in arg for arg in argv)
    assert "--user-data-dir=" + str(app._folders.state / "chrome-profile") in argv
    page = Path(next(arg for arg in argv if arg.startswith("--app=")).removeprefix("--app=file://"))
    assert page.parent == app._folders.state and page.name.startswith("open-") and page.suffix == ".html"
    assert stat.S_IMODE(page.stat().st_mode) == 0o600 and stat.S_IMODE(page.parent.stat().st_mode) == 0o700
    text, opened = page.read_text(), app.url + "&window=chrome"
    assert f'location.replace("{opened}")' in text and f'content="0;url={html.escape(opened, quote=True)}"' in text
    until(lambda: not page.exists(), callbacks=notebook, timeout=5)        # its code is used or expired by then
    app.close(pending="discard")
    assert not app._folders.state.exists()


def test_the_save_target_stays_the_zarr_opened_when_the_working_folder_changes(sample, fake_engine, notebook,
                                                                               monkeypatch, private_home):
    """spatialdata keeps a relative sdata.path as it was given. Resolved at every look, it followed the kernel's working
    folder: after a %cd to another sample's folder of the same layout, saves went into that sample's zarr. The path is
    resolved when the app opens, and again only when the object's path changes (a write to another folder)."""
    for name in ("a", "b"):
        (private_home / name / "data" / "sample.zarr").mkdir(parents=True)
    monkeypatch.chdir(private_home / "a")
    sample.path = Path("data/sample.zarr")
    app = L.launch(sample, open="none", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    opened = str(private_home / "a" / "data" / "sample.zarr")
    bind = next(line for line in records(fake_engine) if line.get("op") == "bind")
    assert bind["path"] == bind["zarr"] == opened and app._snapshot()["zarr"] == opened
    monkeypatch.chdir(private_home / "b")                         # the other sample's folder: the same names
    assert app._snapshot()["zarr"] == opened
    monkeypatch.chdir(private_home)                               # a folder without one: still the zarr opened
    assert app._snapshot()["zarr"] == opened
    sample.path = private_home / "b" / "data" / "sample.zarr"     # the object written elsewhere: its new place
    assert app._snapshot()["zarr"] == str(private_home / "b" / "data" / "sample.zarr")
    app.close(pending="discard")


def test_the_engine_runs_the_proseg_program_launch_checked_and_stain_takes_the_configured_python(
        sample, fake_engine, notebook, monkeypatch, private_home):
    """launch() checked [proseg] binary (which the user's own config file may set) while the engine, which never reads
    that file, ran the app folder's program; and [stain] python was never read, so a kernel without Cellpose had no
    stain segmentation however the config named its environment. Both now reach the engine in session.toml, which it
    reads last (engine/config.py load_config)."""
    from sopa_companion.engine.config import load_config

    env = private_home / "stain-env" / "bin" / "python"
    env.parent.mkdir(parents=True)
    env.write_text("#!/bin/sh\n")
    env.chmod(0o700)
    settings = L._settings()
    monkeypatch.setattr(L, "_settings", lambda: {**settings, "stain": {"python": str(env)}})
    app = L.launch(sample, open="none", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    config = records(fake_engine)[0]["config"]
    assert config["proseg"] == {"binary": sys.executable} and config["stain"]["python"] == str(env)
    engine = load_config(app._folders.state / "session.toml")
    assert engine["proseg"]["binary"] == sys.executable and engine["stain"]["python"] == str(env)
    app.close(pending="discard")
    monkeypatch.setattr(L, "_settings", lambda: settings)          # none named: the kernel's own, with Cellpose
    kernel_has(monkeypatch, {"cellpose"})
    app = L.launch(sample, open="none", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    assert app._config["stain"]["python"] == os.path.abspath(sys.executable)
    app.close(pending="discard")


def test_the_dormant_instanseg_reaches_the_engine_only_with_its_internal_switch_on(sample, fake_engine, notebook,
                                                                                    monkeypatch, private_home):
    """InstanSeg is dormant: while the internal switch [stain] instanseg_enabled is off (unset, or anything but true), a
    [stain] instanseg_python the settings name (the desktop's config.toml on a computer that had InstanSeg), even a
    missing one, is left out without a word, and the session's engine gets the switch off. Switched on, the environment
    is checked and handed over as any optional one, with the switch."""
    import warnings
    from sopa_companion.engine.config import load_config

    env = private_home / "instanseg-env" / "bin" / "python"
    env.parent.mkdir(parents=True)
    env.write_text("#!/bin/sh\n")
    env.chmod(0o700)
    settings = L._settings()

    def launched(stain):
        monkeypatch.setattr(L, "_settings", lambda: {**settings, "stain": stain})
        app = L.launch(sample, open="none", timeout=5)
        until(lambda: app.url is not None, callbacks=notebook)
        handed = [line for line in records(fake_engine) if "config" in line][-1]["config"]["stain"]
        engine = load_config(app._folders.state / "session.toml")["stain"]       # what the session's engine reads
        app.close(pending="discard")
        # "off" reaches the engine as it has always read an environment that is off: "".
        assert (engine["instanseg_python"], engine["instanseg_enabled"]) == (
            "" if handed["instanseg_python"] == "off" else handed["instanseg_python"], handed["instanseg_enabled"])
        return handed["instanseg_python"], handed["instanseg_enabled"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert launched({"instanseg_python": str(env)}) == ("off", False)
        assert launched({"instanseg_python": "/no-such-python"}) == ("off", False)
        assert launched({"instanseg_python": str(env), "instanseg_enabled": "true"}) == ("off", False)   # true, not text
    assert not [w for w in caught if issubclass(w.category, L.CompanionWarning)], "quietly"
    assert launched({"instanseg_python": str(env), "instanseg_enabled": True}) == (str(env), True)
    with pytest.warns(L.CompanionWarning, match="optional Python environment"):
        assert launched({"instanseg_python": "/no-such-python", "instanseg_enabled": True}) == ("off", True)


@pytest.mark.parametrize("visible,gpu,mixing_check,refused", [
    (None, {"qc": 1}, False, "qc GPU 1 is GPU 1 in this notebook"),
    ("1", None, True, "qc GPU 0 is GPU 1 in this notebook"),
    ("", None, True, "qc GPU 0 is no GPU in this notebook"),
    ("GPU-5f1c", {"qc": 0}, False, "qc GPU 0 is GPU GPU-5f1c"),
    ("1,0", {"qc": 1}, True, None),                  # the second GPU this notebook sees is GPU 0
    (None, 1, False, None),                          # every helper on GPU 1: the mixing check cannot learn there, the rest can
    ("1", None, False, None),                        # nothing asked of the mixing check
])
def test_launch_refuses_a_qc_gpu_the_mixing_check_cannot_learn_on(sample, fake_engine, notebook, monkeypatch,
                                                                  visible, gpu, mixing_check, refused):
    """The mixing check's learning process runs on GPU 0 only (engine/qc.py TRAINER_GPU): launch(gpu={"qc": 1}) was
    accepted and could never work. A qc GPU named, or the mixing check asked to start, that is not GPU 0 here is refused
    with the reason."""
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    if refused:
        with pytest.raises(ValueError, match="The mixing check learns on GPU 0 only") as caught:
            L.launch(sample, open="none", timeout=5, gpu=gpu, mixing_check=mixing_check)
        assert refused in str(caught.value) and not records(fake_engine)
        return
    app = L.launch(sample, open="none", timeout=5, gpu=gpu, mixing_check=mixing_check)
    until(lambda: app.url is not None, callbacks=notebook)
    app.close(pending="discard")


def test_patches_dir_is_made_kept_past_the_session_and_must_be_a_folder_of_yours(sample, fake_engine, notebook,
                                                                                 private_home):
    folder = private_home / "my patches"
    app = L.launch(sample, open="none", timeout=5, patches_dir=folder)
    until(lambda: app.url is not None, callbacks=notebook)
    assert records(fake_engine)[0]["config"]["session"]["patches_dir"] == str(folder) and folder.is_dir()
    app.close(pending="discard")
    assert folder.is_dir() and not app._folders.data.exists()
    link = private_home / "patches link"
    link.symlink_to(folder, target_is_directory=True)
    with pytest.raises(ValueError, match="not a link"):
        L.launch(sample, open="none", timeout=5, patches_dir=link)
    # Inside a zarr store (by its name, or a folder on the way holding a zarr marker): refused before it is made.
    store = private_home / "store"
    store.mkdir()
    (store / "zarr.json").write_text("{}")
    for inside in (store / "images" / "patches", private_home / "sample.zarr" / "patches"):
        with pytest.raises(ValueError, match="must not be inside a zarr store"):
            L.launch(sample, open="none", timeout=5, patches_dir=inside)
        assert not inside.parent.exists()
    assert len([record for record in records(fake_engine) if "config" in record]) == 1     # no engine started


def test_the_kernel_and_the_engine_name_the_sessions_folder_and_window_alike_through_a_link(
        sample, fake_engine, notebook, monkeypatch, private_home):
    """The runtime folder reached through a link: the kernel started Chrome with the link's path and the engine, which
    resolves its state folder, looked for the other, so it never saw its window (a crashed page then ended the session
    as a closed window). The state root is resolved too."""
    from sopa_companion.desktop.window import chrome_profile
    from sopa_companion.engine.config import load_config

    real = private_home / "real-runtime"
    real.mkdir(mode=0o700)
    (private_home / "runtime-link").symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(private_home / "runtime-link"))
    app = L.launch(sample, open="none", timeout=5)
    until(lambda: app.url is not None, callbacks=notebook)
    assert app._folders.state.parent == real / "sopa-companion"
    engine = load_config(app._folders.state / "session.toml")
    assert chrome_profile(engine) == app._folders.state / "chrome-profile"
    owner = json.loads((app._folders.state / "owner.json").read_text())
    assert owner["state_dir"] == str(app._folders.state)
    app.close(pending="discard")


def test_a_sessions_data_folder_left_when_its_state_folder_went_is_swept_from_its_scratch(private_home):
    """A reboot or a logout clears the state root (the user's runtime folder) while a session is open: its data folder,
    in a scratch folder that outlives that, was never removed. launch() sweeps the scratch folders sessions used: only a
    data folder of this user's whose state folder is gone and whose engine lock nobody holds."""
    import shutil

    scratch = private_home / "scratch"
    live, gone, held = (folders.SessionFolders.create(scratch, "/sample.zarr") for _ in range(3))
    folders.remember_scratch(scratch)
    shutil.rmtree(gone.state)
    shutil.rmtree(held.state)
    with (held.data / ".engine.lock").open("w") as lock:          # its engine still runs
        fcntl.flock(lock, fcntl.LOCK_EX)
        (scratch / "sopa-companion-0badc0de").mkdir()             # no marker: not a session of this app
        linked = scratch / "sopa-companion-feedbeef"
        linked.symlink_to(gone.data, target_is_directory=True)    # a link: never followed
        foreign = scratch / "sopa-companion-0ddba11a"             # its state folder in no state root of this user's
        foreign.mkdir()
        folders.private_json(foreign / ".sopa-companion-session.json",
                             {"id": "0ddba11a", "state_dir": str(private_home / "elsewhere" / "0ddba11a")})
        assert folders.sweep() == ()                              # the state root holds no dead session
        assert folders.sweep_scratch() == (gone.id,)              # the remembered scratch folder does
        assert not gone.data.exists() and live.data.is_dir() and held.data.is_dir()
        assert (scratch / "sopa-companion-0badc0de").is_dir() and linked.is_symlink() and foreign.is_dir()
    assert folders.sweep_scratch(scratch) == (held.id,)           # its engine gone too
    assert live.data.is_dir() and live.state.is_dir()
    live.remove()


def test_a_scratch_folder_shared_with_another_computer_keeps_that_computers_sessions(private_home, monkeypatch):
    """A scratch folder on a disk two computers share holds the other computer's live sessions too. Their state folders
    are in that computer's runtime folder, under the same path as this user's own here (/run/user/<uid>), so here they
    look gone, and a file lock may not reach across computers: such a session's data folder was removed under it. The
    marker names its computer, and only this computer's are swept; a marker that names none is left alone."""
    import shutil
    import socket

    scratch = private_home / "scratch"
    here, there, unnamed = (folders.SessionFolders.create(scratch, "/sample.zarr") for _ in range(3))
    for session in (here, there, unnamed):
        shutil.rmtree(session.state)
    marker = lambda session: session.data / ".sopa-companion-session.json"
    assert json.loads(marker(here).read_text()) == {"id": here.id, "state_dir": str(here.state),
                                                    "host": socket.gethostname()}
    folders.private_json(marker(there), {**json.loads(marker(there).read_text()), "host": "another-computer"})
    folders.private_json(marker(unnamed), {"id": unnamed.id, "state_dir": str(unnamed.state)})
    assert folders.sweep_scratch(scratch) == (here.id,)
    assert not here.data.exists() and there.data.is_dir() and unnamed.data.is_dir()
    monkeypatch.setattr(socket, "gethostname", lambda: "another-computer")      # on that computer: its own
    assert folders.sweep_scratch(scratch) == (there.id,)
    assert unnamed.data.is_dir()
