"""One state folder, one engine (server/runtime_file.py, desktop/launcher.py). Two copies of the app with data folders of
their own can share a state folder (an installed copy's default and a source tree whose settings name the same one):
whichever engine holds it first keeps it. The other's launch and its engine's start are refused (exit 3) and its stop
and status say whose engine holds the folder, while nothing of the first is touched. An engine started as
`sopa-companion serve` (the console script pip installs) is the app's serve to stop, status, launch and the move, but
the folder it was typed in never makes it another copy's; and an installed copy's desktop entry never puts its
site-packages on a PYTHONPATH."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest

from sopa_companion.cli import parser
from sopa_companion.desktop import launcher
from sopa_companion.desktop.processes import DesktopError, serve_command, verify_process
from sopa_companion.server import runtime_file as R

TOKEN = "t" * 43


def copies(tmp_path) -> tuple[dict, dict]:
    """Settings of two copies sharing one state folder, each with its own data folder."""
    state = tmp_path / "state"
    a = {"paths": {"data_dir": str(tmp_path / "a-var"), "state_dir": str(state)}, "server": {"token": "auto"}}
    b = {"paths": {"data_dir": str(tmp_path / "b-var"), "state_dir": str(state)}, "server": {"token": "auto"}}
    return a, b


def stand_in(folder: Path, seconds: int) -> subprocess.Popen:
    """A process whose command line reads as the app's serve: the console script's `<python> <...>/sopa-companion serve`,
    the script here one that only waits."""
    script = folder / "bin" / "sopa-companion"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"import time\ntime.sleep({seconds})\n")
    return subprocess.Popen([sys.executable, str(script), "serve"], stdin=subprocess.DEVNULL)


@pytest.fixture
def engine_a(tmp_path):
    """A stand-in for copy A's engine (stand_in)."""
    child = stand_in(tmp_path / "a-env", 120)
    try:
        yield child
    finally:
        child.kill()
        child.wait()


def quiet_services(said: list) -> launcher.Services:
    def never(*args, **kwargs):
        raise AssertionError("nothing of the other engine is started, signalled or asked")
    return launcher.Services(spawn=never, run=never, stopper=never, request=never, discover=lambda *a, **k: None,
                             output=said.append, error=said.append, environ={}, sleep=lambda s: None)


def test_a_second_engine_on_the_same_state_folder_is_refused_and_the_first_keeps_its_record(tmp_path, engine_a,
                                                                                             monkeypatch, capsys):
    a, b = copies(tmp_path)
    R.write_runtime(a, TOKEN, "127.0.0.1", 8801, pid=engine_a.pid, argv=["python", "-m", "sopa_companion", "serve"])
    record = R.runtime_path(a)
    before = record.read_bytes()
    refusal = (f"Another SoPA companion engine (pid {engine_a.pid}) uses this state folder, {tmp_path / 'state'}, for "
               f"another data folder, {(tmp_path / 'a-var').resolve()}. Stop that engine first (with the stop command "
               "of the copy it belongs to), or give one of the two a state folder of its own: [paths] state_dir.")
    assert R.other_engine(b) == {"pid": engine_a.pid, "data_dir": str((tmp_path / "a-var").resolve()), "port": 8801}
    assert R.other_engine(a) is None                                         # its own record is no other engine
    # B's launch: refused before anything starts.
    said = []
    with pytest.raises(DesktopError) as refused:
        launcher.ensure_engine(b, services=quiet_services(said))
    assert refused.value.exit_code == 3 and str(refused.value) == refusal
    # B's engine itself (a serve started by hand, a launchers' race): its record never replaces A's.
    with pytest.raises(RuntimeError) as refused:
        R.write_runtime(b, TOKEN, "127.0.0.1", 8802, pid=os.getpid(), argv=["python", "-m", "sopa_companion", "serve"])
    assert str(refused.value) == refusal and record.read_bytes() == before
    # `sopa-companion serve` on B's settings: refused before it takes B's data folder or a port, exit 3.
    from sopa_companion import cli
    monkeypatch.setattr("sopa_companion.engine.config.load_config", lambda path=None, *args, **kwargs: b)
    assert cli.main(["serve", "--port", "0"]) == 3
    assert capsys.readouterr().err == f"OtherEngine: {refusal}\n"
    assert not (tmp_path / "b-var").exists() and record.read_bytes() == before
    # ... and should A's engine take the folder as B's starts, B's start fails at its record (the server's lifespan),
    # which the command raises again once the server has stopped.
    import asyncio
    from fastapi import FastAPI
    app = FastAPI()
    app.state.token = TOKEN
    R.install_lifecycle(app, b, "127.0.0.1", 8802)

    async def start():
        async with app.router.lifespan_context(app):
            pass
    with pytest.raises(R.OtherEngine):
        asyncio.run(start())
    assert isinstance(app.state.refused, R.OtherEngine) and app.state.refused.exit_code == 3
    assert record.read_bytes() == before
    # B's status: not running, and whose engine holds the folder.
    launcher.status(b, parser().parse_args(["status", "--json"]), services=quiet_services(said))
    status = json.loads(said[-1])
    assert status["running"] is False and status["other_engine"] == R.other_engine(b)
    said.clear()
    launcher.status(b, parser().parse_args(["status"]), services=quiet_services(said))
    line = (f"SoPA companion: not running (its state folder's runtime file belongs to another engine: pid {engine_a.pid}, "
            f"data folder {(tmp_path / 'a-var').resolve()})")
    assert said[0] == line
    # B's stop: nothing of B runs, so nothing is signalled; it says so, and exits 0.
    said.clear()
    assert launcher.stop(b, parser().parse_args(["stop", "--force"]), services=quiet_services(said)) == 0
    assert said == [line] and record.read_bytes() == before and engine_a.poll() is None
    # A's own status still reads its record (its engine is a stand-in, so not healthy: only the record is checked).
    assert R.read_runtime(a)["pid"] == engine_a.pid
    assert R.remove_runtime(b, os.getpid()) is False and record.read_bytes() == before   # never raises over it
    # Once A's engine has gone, its record is stale: B starts as it always has, replacing it.
    engine_a.kill()
    engine_a.wait()
    assert R.other_engine(b) is None
    written = R.write_runtime(b, TOKEN, "127.0.0.1", 8802, pid=os.getpid(), argv=["python", "-m", "sopa_companion", "serve"])
    assert R.read_runtime(b) == written and written["data_dir"] == str((tmp_path / "b-var").resolve())


def test_a_record_naming_a_reused_pid_is_no_other_engine(tmp_path):
    a, b = copies(tmp_path)
    # This test's own pid: pytest's command line is no serve.
    R.write_runtime(a, TOKEN, "127.0.0.1", 8801, pid=os.getpid(), argv=["python", "-m", "sopa_companion", "serve"])
    assert R.other_engine(b) is None
    # A pid started after the record was written: the record is stale, whatever runs under that pid now.
    child = stand_in(tmp_path / "env", 60)
    try:
        value = json.loads(R.runtime_path(a).read_text())
        value.update(pid=child.pid, started_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())
        R.atomic_private_json(R.runtime_path(a), value)
        assert R.read_runtime(a)["pid"] == child.pid and R.other_engine(b) is None
        value["started_at"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        R.atomic_private_json(R.runtime_path(a), value)
        assert R.other_engine(b)["pid"] == child.pid                         # started before it: that engine runs
        # Another user's process, or one whose command is not the app's serve, is none either.
        fake = types.SimpleNamespace(is_running=lambda: True, status=lambda: "sleeping", create_time=lambda: 0,
                                     uids=lambda: types.SimpleNamespace(real=os.getuid() + 1, effective=os.getuid() + 1),
                                     cmdline=lambda: ["python", "-m", "sopa_companion", "serve"])
        assert R.other_engine(b, process_factory=lambda pid: fake) is None
        fake.uids = lambda: types.SimpleNamespace(real=os.getuid(), effective=os.getuid())
        assert R.other_engine(b, process_factory=lambda pid: fake)["pid"] == child.pid
        fake.cmdline = lambda: ["python", "-m", "sopa_companion", "launch"]
        assert R.other_engine(b, process_factory=lambda pid: fake) is None
    finally:
        child.kill()
        child.wait()
    assert R.other_engine(b) is None                                         # a dead pid
    R.runtime_path(a).write_text("{not json")                               # a broken file
    assert R.other_engine(b) is None
    R.runtime_path(a).unlink()
    assert R.other_engine(b) is None                                         # no file


def test_the_console_scripts_serve_is_the_apps_serve():
    for argv in (["/env/bin/python3.12", "/env/bin/sopa-companion", "serve"],
                 ["/env/bin/python3.12", "/env/bin/sopa-companion", "--config", "/x.toml", "serve"],
                 ["/env/bin/python", "/env/bin/sopa-companion", "--config=/x.toml", "serve"],
                 ["C:\\env\\python.exe", "C:\\env\\Scripts\\sopa-companion.exe", "serve"],
                 ["/env/bin/Python3", "/env/Scripts/sopa-companion-script.py", "serve"],
                 ["py", "-m", "sopa_companion", "serve"], ["py", "-m", "proseg_live", "--config", "/x", "serve"]):
        assert serve_command(argv), argv
    for argv in (["/env/bin/python3.12", "/env/bin/sopa-companion", "launch"],
                 ["timeout", "600", "sopa-companion", "serve"], ["bash", "-c", "sopa-companion serve"],
                 ["/env/bin/python3.12", "/env/bin/sopa-companion-tools", "serve"],
                 ["/env/bin/sopa-companion", "serve"], ["/env/bin/python3.12", "/env/bin/sopa-companion"],
                 ["/env/bin/python3.12", "/env/bin/sopa-companion", "--config", "serve", "launch"]):
        assert not serve_command(argv), argv
    # `-m` is the interpreter's own only before its code, its standard input or a script: after them it is that
    # program's argument (`python -c CODE -m sopa_companion serve` runs CODE); options before `-m` are the interpreter's.
    for argv in (["py", "-c", "import time; time.sleep(9)", "-m", "sopa_companion", "serve"],
                 ["py", "-cimport time", "-m", "sopa_companion", "serve"], ["py", "-", "-m", "sopa_companion", "serve"],
                 ["py", "tool.py", "-m", "sopa_companion", "serve"], ["py", "-B", "-c", "x", "-m", "sopa_companion", "serve"]):
        assert not serve_command(argv), argv
    for argv in (["py", "-u", "-B", "-m", "sopa_companion", "serve"], ["py", "-X", "dev", "-W", "error", "-m",
                                                                       "sopa_companion", "--config", "/x", "serve"]):
        assert serve_command(argv), argv
    process = types.SimpleNamespace(uids=lambda: types.SimpleNamespace(real=os.getuid(), effective=os.getuid()),
                                    cmdline=lambda: ["/env/bin/python3.12", "/env/bin/sopa-companion", "serve"],
                                    create_time=lambda: 0)
    assert verify_process(4321, process_factory=lambda pid: process) is process
    assert serve_command is R.serve_command                                  # one rule, where the runtime file is


def test_an_installed_copys_desktop_entry_names_no_pythonpath(tmp_path, monkeypatch):
    from sopa_companion.desktop import install
    from sopa_companion.engine.config import ROOT, load_config
    site = tmp_path / "env" / "site-packages"
    installed = site / "sopa_companion"
    installed.mkdir(parents=True)
    shutil.copy(ROOT / "config.example.toml", installed / "config.example.toml")     # what the wheel carries
    state = tmp_path / "state"
    python = str(tmp_path / "env" / "bin" / "python")
    entry = install.entry_text(python, tmp_path / "server.log", installed, workdir=state)
    execs = [line for line in entry.splitlines() if line.startswith("Exec=")]
    assert len(execs) == 3 and not any("PYTHONPATH" in line or "site-packages" in line for line in execs)
    assert f"Exec=\"{python}\" -m sopa_companion launch" in execs and f"Path={state}" in entry.splitlines()
    # install-desktop on such a copy: the same entry, its launcher starting in the engine's state folder.
    monkeypatch.setattr(install, "ROOT", installed)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config = load_config(root=installed, overrides={"paths": {"data_dir": str(tmp_path / "var")}})
    said = []
    args = types.SimpleNamespace(home=None, uninstall=False, no_desktop_icon=True, dry_run=True)
    assert install.install(config, args, output=said.append, which=lambda name: None) == 0
    shown = said[-1].splitlines()
    assert said[-1] == install.entry_text(os.path.abspath(sys.executable), install.log_path(config), installed,
                                          workdir=(home / ".local/state/sopa-companion").resolve()).rstrip()
    assert f"Path={(home / '.local/state/sopa-companion').resolve()}" in shown
    assert not any("PYTHONPATH" in line for line in shown) and not (home / ".local/state").exists()   # a dry run
    # With --home, that home's state folder (the default one's place there).
    other = tmp_path / "other-home"
    args.home = str(other)
    said.clear()
    install.install(config, args, output=said.append, which=lambda name: None)
    assert f"Path={other.resolve() / '.local/state/sopa-companion'}" in said[-1].splitlines()
    # A source tree called sopa_companion keeps today's entry: its parent on PYTHONPATH, its own folder as Path.
    tree = tmp_path / "srv" / "sopa_companion"
    tree.mkdir(parents=True)
    (tree / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    entry = install.entry_text(python, tmp_path / "server.log", tree)
    assert f'Exec=env "PYTHONPATH={tree.parent}" "{python}" -m sopa_companion launch' in entry.splitlines()
    assert f"Path={tree}" in entry.splitlines()


def test_the_folder_a_console_scripts_engine_was_typed_in_never_makes_it_that_copys(config, tmp_path):
    """`sopa-companion serve` runs the copy installed in the script's environment, wherever it is typed: typed in a
    source tree, its working folder says nothing of its install. That source tree's launcher, stop and status never take
    it for their own engine (the launcher would offer to restart it, and signal another copy's engine); the engine lock
    of the data folder it holds still finds it, and a --config naming a data folder still tells that one."""
    from sopa_companion.desktop import processes
    data_dir = config["paths"]["data_dir"]
    here = tmp_path / "clone" / "sopa_companion"                 # a source tree whose settings name this data folder
    (here / "engine").mkdir(parents=True)
    (here / "cli.py").write_text("")
    (here / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    (here / "config.toml").write_text(f'[paths]\ndata_dir = "{data_dir}"\n')

    class Typed:
        pid = 4301
        info = {"pid": 4301, "cmdline": ["/env/bin/python3.12", "/env/bin/sopa-companion", "serve"],
                "uids": types.SimpleNamespace(real=os.getuid())}

        def cmdline(self):
            return self.info["cmdline"]

        def cwd(self):
            return str(here)

        def environ(self):
            return {}
    typed = Typed()
    assert serve_command(typed.cmdline()) and processes._install_root(typed) is None
    assert processes.foreign_engine(config, None, processes=lambda attrs: [typed]) is None
    typed.info["cmdline"] = ["/env/bin/python3.12", "/env/bin/sopa-companion", "--config", str(here / "config.toml"),
                             "serve"]
    assert processes.foreign_engine(config, None, processes=lambda attrs: [typed]) == {"pid": 4301,
                                                                                      "data_dir": data_dir, "mine": True}
    # `python -m sopa_companion serve` started there is that copy's, as the launcher starts it.
    typed.info["cmdline"] = ["/env/bin/python3.12", "-m", "sopa_companion", "serve"]
    assert processes._install_root(typed) == here.resolve()


def starting(spawn, said: list) -> launcher.Services:
    """Services for a launch with no engine running: `spawn` starts its engine, and nothing else is asked."""
    services = launcher.Services(spawn=spawn, discover=lambda *a, **k: None, output=said.append, error=said.append,
                                 environ={}, sleep=lambda s: None)
    services.healthy = lambda record: False
    return services


def test_a_launch_whose_engine_loses_the_race_at_its_record_says_whose_engine_holds_the_folder(tmp_path, monkeypatch):
    """Another copy's engine can take the state folder after launch's two checks, as this copy's engine starts: that
    engine then refuses to start (exit 3, OtherEngine at its runtime file), and launch says whose engine holds the folder
    with its own exit 3, as when it refuses first. Any other exit stays a failed start."""
    a, b = copies(tmp_path)
    b["server"].update(host="127.0.0.1", port=8802)
    other = {"pid": 4141, "data_dir": str((tmp_path / "a-var").resolve()), "port": 8801}
    asked = []
    monkeypatch.setattr(launcher, "other_engine", lambda config: asked.append(1) or (other if len(asked) > 2 else None))
    monkeypatch.setattr(launcher, "spawn_engine", lambda config, path, services: services.spawn([]))
    said = []
    with pytest.raises(launcher.SharedStateFolder) as refused:
        launcher.ensure_engine(b, services=starting(lambda command, **k: said.append("spawned") or
                                                     types.SimpleNamespace(pid=4242, poll=lambda: 3), said))
    assert refused.value.exit_code == 3 and said == ["spawned"] and len(asked) == 3
    assert str(refused.value) == R.OTHER_ENGINE.format(pid=4141, state_dir=b["paths"]["state_dir"],
                                                       data_dir=other["data_dir"])
    # Exit 3 with no other engine (the state folder's move refused, say): launch's exit 3, its own words.
    monkeypatch.setattr(launcher, "other_engine", lambda config: None)
    for code, exit_code in ((3, 3), (1, 1)):
        with pytest.raises(DesktopError) as failed:
            launcher.ensure_engine(b, services=starting(lambda command, code=code, **k: types.SimpleNamespace(
                pid=4242, poll=lambda: code), said))
        assert str(failed.value) == f"Engine startup failed (exit {code})." and failed.value.exit_code == exit_code


def test_launch_checks_the_state_folder_again_under_its_lock(tmp_path, monkeypatch):
    """Another copy's engine that took the state folder while this launch waited for its lock: refused there, before
    any engine is started."""
    a, b = copies(tmp_path)
    other = {"pid": 4141, "data_dir": str((tmp_path / "a-var").resolve()), "port": 8801}
    asked = []
    monkeypatch.setattr(launcher, "other_engine", lambda config: asked.append(1) or (other if len(asked) > 1 else None))

    def never(*args, **kwargs):
        raise AssertionError("no engine is started")
    with pytest.raises(launcher.SharedStateFolder) as refused:
        launcher.ensure_engine(b, services=starting(never, []))
    assert refused.value.exit_code == 3 and len(asked) == 2 and "pid 4141" in str(refused.value)


def test_an_engine_refused_at_its_record_exits_3(tmp_path, engine_a, config, monkeypatch, capsys):
    """`serve` checks the state folder before it starts, and its record does again as the server starts (the lifespan):
    another copy's engine that took the folder between the two makes the command fail with exit 3 and the refusal, once
    the server has stopped. Nothing of that engine's record changes."""
    import uvicorn
    from sopa_companion import cli
    a, b = copies(tmp_path)
    R.write_runtime(a, TOKEN, "127.0.0.1", 8801, pid=engine_a.pid, argv=["python", "-m", "sopa_companion", "serve"])
    record = R.runtime_path(a)
    before = record.read_bytes()
    settings = {**config, "paths": {**config["paths"], **b["paths"]}}
    monkeypatch.setattr("sopa_companion.engine.config.load_config", lambda path=None, *args, **kwargs: settings)
    real, checks = R.other_engine, []
    # The first check (serve's own) finds nothing: the other engine takes the folder after it.
    monkeypatch.setattr(R, "other_engine", lambda config, **kw: checks.append(1) or (real(config, **kw)
                                                                                  if len(checks) > 1 else None))
    started = []

    class Server:
        """uvicorn's server as far as the lifespan: its startup fails, and run() returns, as uvicorn's does."""
        def __init__(self, server_config):
            self.app = server_config.app

        def run(self, sockets=None):
            import asyncio

            async def start():
                try:
                    async with self.app.router.lifespan_context(self.app):
                        started.append("served")
                except Exception as exc:
                    started.append(type(exc).__name__)
            asyncio.run(start())
    monkeypatch.setattr(uvicorn, "Server", Server)
    assert cli.main(["serve", "--port", "0"]) == 3
    refusal = R.OTHER_ENGINE.format(pid=engine_a.pid, state_dir=b["paths"]["state_dir"],
                                    data_dir=str((tmp_path / "a-var").resolve()))
    assert started == ["OtherEngine"] and capsys.readouterr().err.endswith(f"OtherEngine: {refusal}\n")
    assert record.read_bytes() == before and len(checks) == 2


def test_a_refused_launch_shows_no_engine_log_of_the_other_copy(tmp_path, engine_a):
    """The log in a shared state folder is the other copy's engine's: launch's refusal says whose engine holds the
    folder, and shows no lines of that log as this copy's."""
    a, b = copies(tmp_path)
    b["server"] = {"token": "auto"}
    R.write_runtime(a, TOKEN, "127.0.0.1", 8801, pid=engine_a.pid, argv=["python", "-m", "sopa_companion", "serve"])
    log = tmp_path / "state" / "logs" / "server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("the other engine's own lines\n")
    said = []
    assert launcher.command(b, parser().parse_args(["launch", "--no-browser"]), services=quiet_services(said)) == 3
    assert said == [R.OTHER_ENGINE.format(pid=engine_a.pid, state_dir=b["paths"]["state_dir"],
                                          data_dir=str((tmp_path / "a-var").resolve()))]


def test_launch_without_a_display_starts_no_engine(tmp_path):
    """The window needs a display: without one (an SSH session) launch says what to use instead, before any engine
    starts (it started one, then refused, and left it running), and shows no engine log with it."""
    a, b = copies(tmp_path)
    said = []
    assert launcher.command(b, parser().parse_args(["launch"]), services=quiet_services(said)) == 1
    assert said == ["No display is available. Use launch --no-browser or launch --print-url for an SSH tunnel."]
    assert not (tmp_path / "state").exists() and not (tmp_path / "b-var").exists()
