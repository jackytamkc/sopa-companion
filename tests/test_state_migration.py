"""The previous version's state folder and window profile become this version's at the default desktop instance's first
start (desktop/migrate.py): moved whole, never copied, and only while nothing of the previous version can use them.

Every test has a home folder of its own and passes it on: the user's real folders are never looked at."""
from __future__ import annotations

from datetime import datetime, timezone
import errno
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import types

import pytest

from sopa_companion.desktop import migrate as M
from sopa_companion.desktop.processes import DesktopError


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def settings(home: Path, state: str = ".local/state/sopa-companion") -> dict:
    return {"paths": {"state_dir": str((home / state).resolve())}}


def previous(home: Path, *, state: bool = True, profile: bool = True) -> tuple[Path, Path]:
    """What the previous version leaves: its state folder (a log, a run folder) and its window's profile."""
    old_state, old_profile = home / ".local/state/proseg_live", home / ".local/share/proseg_live/chrome-profile"
    if state:
        (old_state / "logs").mkdir(parents=True)
        (old_state / "logs/server.log").write_text("Proseg Live: engine starting: pid 1, SoPA companion 0.1.0\n")
        (old_state / "run").mkdir()
    if profile:
        (old_profile / "Default").mkdir(parents=True)
        (old_profile / "Default/Preferences").write_text('{"window": "where it was"}')
    return old_state, old_profile


class Process:
    """A process as psutil.process_iter gives it (info), with what migrate asks of a `serve`."""

    def __init__(self, pid, cmdline, cwd="/", environ=None):
        self.pid = pid
        self.info = {"pid": pid, "cmdline": cmdline, "status": "sleeping",
                     "uids": types.SimpleNamespace(real=os.getuid())}
        self._cwd, self._environ = str(cwd), environ or {}

    def cmdline(self):
        return self.info["cmdline"]

    def cwd(self):
        return self._cwd

    def environ(self):
        return self._environ


def table(*processes):
    return lambda attrs=None: list(processes)


NOBODY = table()


def untouched(home: Path, before: dict) -> bool:
    return tree(home) == before


def tree(home: Path) -> dict:
    return {str(p.relative_to(home)): p.lstat().st_ino for p in sorted(home.rglob("*"))}


def test_the_previous_versions_folders_move_once_whole(home):
    old_state, old_profile = previous(home)
    inodes = (old_state.stat().st_ino, old_profile.stat().st_ino)
    new = home / ".local/state/sopa-companion"
    said = M.migrate(settings(home), home=home, processes=NOBODY)
    assert said == (f"SoPA companion: moved the previous version's state folder {old_state} and window profile "
                    f"{old_profile} into {new}")
    assert (new.stat().st_ino, (new / "chrome-profile").stat().st_ino) == inodes      # renamed, not copied
    assert (new / "logs/server.log").read_text().startswith("Proseg Live: engine starting")
    assert json.loads((new / "chrome-profile/Default/Preferences").read_text()) == {"window": "where it was"}
    assert not old_state.exists() and not old_profile.exists()
    assert M.migrate(settings(home), home=home, processes=NOBODY) is None             # once
    from sopa_companion.desktop.window import chrome_profile
    assert chrome_profile(settings(home)) == new / "chrome-profile"                   # where the window opens it now


@pytest.mark.parametrize("state, profile", [(True, False), (False, True)])
def test_a_state_folder_or_a_profile_alone_moves_too(home, state, profile):
    old_state, old_profile = previous(home, state=state, profile=profile)
    new = home / ".local/state/sopa-companion"
    said = M.migrate(settings(home), home=home, processes=NOBODY)
    assert said.startswith("SoPA companion: moved the previous version's ") and said.endswith(f"into {new}")
    assert (new / "logs/server.log").exists() == state
    assert (new / "chrome-profile/Default/Preferences").exists() == profile
    assert not old_state.exists() and not old_profile.exists()
    if not state:                                          # a fresh folder of this version's own, private
        assert (new.stat().st_mode & 0o777) == 0o700
        assert [p.name for p in new.parent.iterdir()] == ["sopa-companion"]   # the temporary holder renamed


def test_nothing_moves_when_there_is_nothing_to_move_or_it_is_not_the_default_instance(home, tmp_path):
    assert M.migrate(settings(home), home=home, processes=NOBODY) is None             # nothing of the previous version
    assert not (home / ".local").exists()
    old_state, old_profile = previous(home)
    before = tree(home)
    for config in (settings(home, ".local/state/sopa-companion-test"),               # a test's or the harness's own
                   settings(home, "runtime/sopa-companion/1f3a9c2e"),                # a notebook's session
                   {"server": {"port": 0}}, {"paths": {}}):                          # no state folder named
        assert M.migrate(config, home=home, processes=NOBODY) is None
        assert not M.default_instance(config, home)
    assert untouched(home, before)
    (home / ".local/state/sopa-companion").mkdir()                                    # this version's is there already
    assert M.migrate(settings(home), home=home, processes=NOBODY) is None
    assert old_state.exists() and old_profile.exists()


def test_the_move_is_refused_while_the_previous_version_may_use_the_folders(home, tmp_path):
    old_state, old_profile = previous(home)
    before = tree(home)

    def refused(why, **kwargs):
        with pytest.raises(M.MigrationRefused, match=why) as caught:
            M.migrate(settings(home), home=home, **{"processes": NOBODY, **kwargs})
        assert caught.value.exit_code == 3 and "Nothing was moved or copied" in str(caught.value)
        assert untouched(home, before) and not (home / ".local/state/sopa-companion").exists()

    # Its engine: the runtime file names a running process.
    runtime = old_state / "run/server.json"
    runtime.write_text(json.dumps({"pid": os.getpid(), "token": "x"}))
    before = tree(home)
    refused(f"its engine \\(pid {os.getpid()}\\) is running")
    runtime.unlink()
    before = tree(home)
    # Its engine marker, whose owner still runs.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        marker = old_state / "run/engine.alive"
        marker.write_text(json.dumps({"pid": child.pid, "started": datetime.now(timezone.utc).isoformat()}))
        before = tree(home)
        refused(f"its engine \\(pid {child.pid}\\) is running")
        marker.unlink()
        before = tree(home)
        # A serve of the previous version whose state folder cannot be told (its app folder neither).
        serving = Process(child.pid, [sys.executable, "-m", "proseg_live", "serve"], cwd=tmp_path)
        refused(f"an engine \\(pid {child.pid}, python -m proseg_live\\) may be using it", processes=table(serving))
        # ... and one whose settings name another state folder is not in the way.
        other = tmp_path / "other.toml"
        other.write_text(f'[paths]\nstate_dir = "{tmp_path / "elsewhere"}"\n')
        elsewhere = Process(child.pid, [sys.executable, "-m", "proseg_live", "--config", str(other), "serve"],
                            cwd=tmp_path)
        assert M.old_users(old_state, old_profile, entry=home / "none", processes=table(elsewhere)) == []
    finally:
        child.kill()
        child.wait()
    # A launch of it under way (its launch lock held).
    with open(old_state / "run/launch.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = tree(home)
        refused("it is being started")
    before = tree(home)
    # Its window: a Chrome on its profile, or the profile's lock naming a running Chrome of this computer.
    chrome = Process(4242, ["/opt/google/chrome/chrome", f"--user-data-dir={old_profile}", "--app=http://x"])
    refused("its app window is open", processes=table(chrome))
    (old_profile / "SingletonLock").symlink_to(f"{socket.gethostname()}-{os.getpid()}")
    before = tree(home)
    refused("its app window is open")
    (old_profile / "SingletonLock").unlink()
    (old_profile / "SingletonLock").symlink_to("another-computer-1")                  # not this computer's Chrome
    assert M.old_users(old_state, old_profile, entry=home / "none", processes=NOBODY) == []


def test_what_the_previous_version_left_after_it_ended_does_not_stop_the_move(home, tmp_path):
    old_state, _ = previous(home)
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    pid = int(dead.stdout)
    (old_state / "run/server.json").write_text(json.dumps({"pid": pid, "token": "x"}))           # a crash's leftovers
    (old_state / "run/engine.alive").write_text(json.dumps({"pid": pid, "started": "2026-10-20T10:00:00+00:00"}))
    (old_state / "run/launch.lock").write_text("")
    assert M.migrate(settings(home), home=home, processes=NOBODY).startswith("SoPA companion: moved")
    assert (home / ".local/state/sopa-companion/run/engine.alive").exists()   # the crash check finds it there


def test_the_previous_version_installed_elsewhere_stops_the_move(home, tmp_path):
    """Its desktop entry starts it from another folder that is still there: started again from its icon, it would make
    new, empty folders in place of the moved ones. Its entry naming this app's folder, or a folder gone, does not."""
    old_state, _ = previous(home)
    entry = home / ".local/share/applications/proseg-live.desktop"
    entry.parent.mkdir(parents=True)
    installed = tmp_path / "apps with space" / "proseg_live"
    installed.mkdir(parents=True)
    escaped = str(installed).replace("\\", "\\\\").replace(" ", "\\s")
    entry.write_text(f"[Desktop Entry]\nType=Application\nName=SoPA companion\nPath={escaped}\nIcon=proseg-live\n")
    with pytest.raises(M.MigrationRefused, match=f"the previous version is installed in {installed}"):
        M.migrate(settings(home), home=home, root=tmp_path / "this app", processes=NOBODY)
    assert old_state.exists() and not (home / ".local/state/sopa-companion").exists()
    # This app's own folder (the previous version updated in place): moves.
    assert M.old_users(old_state, home / "none", entry=entry, root=installed, processes=NOBODY) == []
    installed.rmdir()                                                                 # moved away since: moves
    assert M.migrate(settings(home), home=home, root=tmp_path / "this app", processes=NOBODY).startswith("SoPA")


def previous_entry(python: Path, *, pythonpath: Path | None = None, module: str = "proseg_live") -> str:
    """A desktop entry as the previous version's first installers wrote it: no Path= key, the interpreter quoted as
    desktop/install.py quotes it (`env "PYTHONPATH=..."` before it when given), and actions with commands of their own."""
    from sopa_companion.desktop.install import exec_arg, entry_value
    command = f"{exec_arg(python)} -m {module}"
    if pythonpath is not None:
        command = f"env {exec_arg(f'PYTHONPATH={pythonpath}')} {command}"
    return (f"[Desktop Entry]\nType=Application\nName=SoPA companion\nExec={command} launch\n"
            f"TryExec={entry_value(python)}\nIcon=proseg-live\nTerminal=false\nActions=stop;log;\n\n"
            f"[Desktop Action stop]\nName=Stop engine\nExec={command} stop --force --timeout 900\n\n"
            f"[Desktop Action log]\nName=Show engine log\nExec=xdg-open \"/elsewhere/server.log\"\n")


def test_an_entry_without_a_path_key_still_says_where_the_previous_version_is(home, tmp_path):
    """The first installers wrote no Path= key: their entry runs the app folder's own interpreter,
    `"<folder>/.venv/bin/python" -m proseg_live`, or puts the folder's parent first with `env "PYTHONPATH=<parent>"`.
    Either names the folder, which stops the move while it is there and is not this app's. An entry that names no
    folder stops it while its interpreter is there; the launcher its installer recorded on the desktop counts as its
    menu entry does; an entry of this version's own module does not count."""
    old_state, old_profile = previous(home)
    entry = home / ".local/share/applications/proseg-live.desktop"
    entry.parent.mkdir(parents=True)
    installed = tmp_path / "apps with space" / "proseg_live"
    python = installed / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    this_app = tmp_path / "this app"

    def users(root=this_app, **kwargs):
        return M.old_users(old_state, old_profile, entry=entry, root=root, processes=NOBODY, **kwargs)

    entry.write_text(previous_entry(python))                         # the shape of the entries written before Path=
    assert M._previous_start(entry) == (installed, python)
    assert users() == [f"the previous version is installed in {installed} (its desktop entry {entry}), whose icon "
                       "would start it again on new, empty folders"]
    with pytest.raises(M.MigrationRefused, match=f"the previous version is installed in {installed}"):
        M.migrate(settings(home), home=home, root=this_app, processes=NOBODY)
    assert old_state.exists() and old_profile.exists() and not (home / ".local/state/sopa-companion").exists()
    assert users(root=installed) == []                               # this app's own folder: updated in place
    # `env "PYTHONPATH=<parent>"` before an interpreter of no app folder: the package's folder in that parent.
    elsewhere = tmp_path / "conda env" / "bin" / "python3"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("")
    entry.write_text(previous_entry(elsewhere, pythonpath=installed.parent))
    assert M._previous_start(entry) == (installed, elsewhere)
    assert len(users()) == 1 and users(root=installed) == []
    # No folder named, its interpreter still there: it may start the previous version from anywhere.
    entry.write_text(previous_entry(elsewhere))
    assert M._previous_start(entry) == (None, elsewhere)
    assert users() == [f"the previous version's desktop entry {entry} still starts it with {elsewhere}, from a folder "
                       "it does not name, and its icon would start it again on new, empty folders"]
    elsewhere.unlink()
    assert users() == []
    # An entry that runs this version's module is not the previous version's.
    entry.write_text(previous_entry(python, module="sopa_companion"))
    assert M._previous_start(entry) is None and users() == []
    # The menu entry gone, the launcher on the desktop that its installer recorded still starts it.
    entry.unlink()
    launcher = home / "Desktop/SoPA-companion.desktop"
    launcher.parent.mkdir()
    launcher.write_text(previous_entry(python))
    record = home / ".local/share/proseg_live/desktop-install.json"
    record.write_text(json.dumps({"schema": 1, "paths": [str(entry), str(launcher), str(home / "icon.svg")]}))
    assert users() == [] and len(users(record=record)) == 1
    with pytest.raises(M.MigrationRefused, match=f"its desktop entry {launcher}"):
        M.migrate(settings(home), home=home, root=this_app, processes=NOBODY)
    # Its folder gone (moved, as the deploy that ships the new name moves it): the entries start nothing, it moves.
    python.unlink()
    for folder in (python.parent, python.parent.parent, installed):
        folder.rmdir()
    assert M.migrate(settings(home), home=home, root=this_app, processes=NOBODY).startswith("SoPA companion: moved")


def test_a_move_that_cannot_be_done_in_place_is_refused_and_undone(home, monkeypatch):
    old_state, old_profile = previous(home)
    before = tree(home)
    real = os.rename
    calls = []

    def rename(source, target):
        calls.append((Path(source), Path(target)))
        if Path(target) == home / ".local/state/sopa-companion":
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real(source, target)

    monkeypatch.setattr(M.os, "rename", rename)
    with pytest.raises(M.MigrationRefused, match="Invalid cross-device link") as caught:
        M.migrate(settings(home), home=home, processes=NOBODY)
    said = str(caught.value)
    assert f"mv {old_state} {home / '.local/state/sopa-companion'}" in said and "Nothing was moved or copied" in said
    assert calls[0] == (old_profile, old_state / "chrome-profile")
    assert calls[-1] == (old_state / "chrome-profile", old_profile)
    assert untouched(home, before)                                                    # the profile put back
    # The profile's own move refused (no permission there): nothing at all happened.
    calls.clear()

    def denied(source, target):
        calls.append((Path(source), Path(target)))
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(M.os, "rename", denied)
    with pytest.raises(M.MigrationRefused, match="Permission denied"):
        M.migrate(settings(home), home=home, processes=NOBODY)
    assert untouched(home, before) and len(calls) == 1
    # Neither the move nor the undo: the message says where the profile is.
    def stuck(source, target):
        if Path(source) == old_profile:
            return real(source, target)
        raise OSError(errno.EBUSY, "Device or resource busy")

    monkeypatch.setattr(M.os, "rename", stuck)
    with pytest.raises(M.MigrationRefused, match=f"it is in {old_state / 'chrome-profile'}"):
        M.migrate(settings(home), home=home, processes=NOBODY)


def test_another_start_moving_them_at_the_same_moment(home, monkeypatch):
    old_state, old_profile = previous(home)
    new = home / ".local/state/sopa-companion"
    real = os.rename

    def first(source, target):                       # the other start got there first and moved everything
        monkeypatch.setattr(M.os, "rename", real)
        real(old_profile, old_state / "chrome-profile")
        real(old_state, new)
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(source))

    monkeypatch.setattr(M.os, "rename", first)
    assert M.migrate(settings(home), home=home, processes=NOBODY) is None
    assert (new / "chrome-profile/Default/Preferences").exists()
    # A folder that went with no new one coming: refused, saying to start again.
    other = home / "other"
    other.mkdir()
    old_state, old_profile = previous(other)

    def gone(source, target):
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(source))

    monkeypatch.setattr(M.os, "rename", gone)
    with pytest.raises(M.MigrationRefused, match="another start is moving them"):
        M.migrate(settings(other), home=other, processes=NOBODY, wait_s=0.2)


def test_serve_and_launch_move_first_and_nothing_else_does(home, tmp_path, monkeypatch, capsys):
    """`serve` and `launch` move the folders before anything makes this version's state folder (and say so: the engine
    log, the terminal); a refused move stops them with exit code 3. A notebook's session never moves anything, nor do
    `status` and `stop`."""
    from sopa_companion import cli
    from sopa_companion.cli import parser
    from sopa_companion.desktop import launcher, logs
    from sopa_companion.server import runtime_file
    order = []
    state = home / ".local/state/sopa-companion"

    def moving(config, **kwargs):
        order.append(("migrate", state.exists()))
        return "SoPA companion: moved the previous version's state folder X into Y"

    monkeypatch.setattr(M, "migrate", moving)
    monkeypatch.setattr(logs, "install_rotation", lambda config: order.append(("rotation", state.exists())))

    def stop(config):
        raise RuntimeError("stop here")
    monkeypatch.setattr(runtime_file, "acquire_engine_lock", stop)
    with pytest.raises(RuntimeError, match="stop here"):
        cli.serve({**settings(home), "server": {"port": 0}}, "127.0.0.1", 0)
    assert order == [("migrate", False), ("rotation", False)]
    assert "SoPA companion: moved the previous version's state folder X into Y\n" in capsys.readouterr().out
    # launch: first, before the engine (whose launch lock makes the state folder); said, and written to its log.
    order.clear()
    said = []
    services = launcher.Services(output=said.append, environ={})

    def engine(config, **kwargs):
        order.append(("engine", state.exists()))
        raise DesktopError("stop here")
    monkeypatch.setattr(launcher, "ensure_engine", engine)
    config = {"paths": {**settings(home)["paths"], "data_dir": str(tmp_path / "var")},
              "server": {"port": 8799, "host": "127.0.0.1", "token": "a-session-token-of-this-test-0123456789ab"}}
    with pytest.raises(DesktopError, match="stop here"):      # (no display here: no window, which launch refuses first)
        launcher.launch(config, parser().parse_args(["launch", "--no-browser"]), services=services)
    assert order == [("migrate", False), ("engine", True)]
    assert said == ["SoPA companion: moved the previous version's state folder X into Y"]
    assert (state / "logs/server.log").read_text() == said[0] + "\n"
    # A refused move: the launcher's answer is the refusal, exit code 3; nothing else is tried.
    order.clear()
    monkeypatch.setattr(M, "migrate", lambda config, **kwargs: (_ for _ in ()).throw(M.MigrationRefused("not now")))
    errors = []
    services = launcher.Services(output=said.append, error=errors.append, environ={})
    assert launcher.command(config, parser().parse_args(["launch", "--no-browser"]), services=services) == 3
    assert errors[0].startswith("not now") and order == []
    # status, stop and a notebook's session never call it.
    monkeypatch.setattr(M, "migrate", lambda config, **kwargs: order.append("moved"))
    monkeypatch.setattr(launcher, "read_runtime", lambda config: None)
    services = launcher.Services(output=said.append, error=errors.append, environ={}, discover=lambda *a: None,
                                 which=lambda name: None)                 # status asks no real desktop tool here
    assert launcher.status(config, parser().parse_args(["status"]), services=services) in (0, 1, 3)
    assert launcher.stop(config, parser().parse_args(["stop"]), services=services) == 0
    session = tmp_path / "session.toml"
    session.write_text('[session]\nmode = "api"\n')
    from sopa_companion.server import api_serve
    monkeypatch.setattr(api_serve, "main", lambda argv: 0)
    assert cli.main(["serve", "--config", str(session)]) == 0
    assert order == []
