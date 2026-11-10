"""The desktop's side: the launcher opens the app where [desktop] browser says (the default browser, or the Chrome app
window, which the engine still recognises by its profile), checking that setting before any engine starts; no answer to
the save dialog discards; and the shipped defaults use the whole workstation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import types
from xml.etree import ElementTree

import psutil
import pytest

from sopa_companion.desktop import browser, launcher
from sopa_companion.desktop.launcher import browser_command
from sopa_companion.desktop.processes import DesktopError
from sopa_companion.desktop.window import AppWindow, chrome_profile
from sopa_companion.engine.config import ROOT, load_config
from sopa_companion.server import app as server


@pytest.fixture(autouse=True)
def own_home(tmp_path, monkeypatch):
    # The launcher looks for a snap browser's folders in the home folder (~/snap/*/common/sopa-companion): a test's own,
    # never the user's. And this is no Linux under Windows, wherever the tests run.
    home = tmp_path / "own-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(browser, "wsl", lambda environ=None: False)
    return home


# ---- which Chrome is the app's own ------------------------------------------------------------------------------
def test_the_engine_looks_for_the_chrome_the_launcher_starts(config, tmp_path):
    chrome = lambda name: "/usr/bin/google-chrome" if name == "google-chrome" else None
    command = browser_command(config, "http://127.0.0.1:8777/?code=c", which=chrome)
    assert f"--user-data-dir={chrome_profile(config)}" in command
    # One profile per engine instance ("auto"), in its state folder: the default instance's (its saved layer settings
    # and window size, moved there from the previous version's ~/.local/share/proseg_live/chrome-profile at its first
    # start: desktop/migrate.py) moves with that folder; another instance (a test engine with its own state_dir) gets
    # its own, so a window opened for it never joins the user's app window, its window check or its session cookie.
    assert load_config()["desktop"]["profile_dir"] == "auto"
    assert chrome_profile(load_config()) == Path.home() / ".local/state/sopa-companion/chrome-profile"
    assert chrome_profile(config) == Path(config["paths"]["state_dir"]) / "chrome-profile"
    # A state folder configured to the previous version's keeps that version's profile where it was.
    legacy = {"paths": {"state_dir": str(Path.home() / ".local/state/proseg_live")}}
    assert chrome_profile(legacy) == Path.home() / ".local/share/proseg_live/chrome-profile"
    dev = tmp_path / "dev.toml"
    dev.write_text('[server]\nport = 8799\n[paths]\nstate_dir = "~/.local/state/sopa_companion_dev"\n')
    assert chrome_profile(load_config(dev)) == Path.home() / ".local/state/sopa_companion_dev/chrome-profile"
    assert chrome_profile({}) == chrome_profile({"desktop": {"profile_dir": ""}}) == chrome_profile(load_config())
    assert chrome_profile({"desktop": {"profile_dir": "var/profile"}}) == ROOT / "var/profile"   # inside the app
    assert chrome_profile({"desktop": {"profile_dir": "~/p"}}) == Path.home() / "p"               # ~ expanded


def test_the_window_check_imports_on_its_own():
    # desktop.processes imports server.runtime_file (server/__init__ imports server.app, which imports the window
    # check): importing it first must not meet a partially initialised module.
    for module in ("sopa_companion.desktop.processes", "sopa_companion.desktop.window", "sopa_companion.server.app"):
        done = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=ROOT.parent, capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT.parent)})
        assert done.returncode == 0, done.stderr[-500:]


class FakeProcess:
    def __init__(self, cmdline, uid=None, status=psutil.STATUS_SLEEPING):
        self.info = {"cmdline": cmdline, "uids": types.SimpleNamespace(real=os.getuid() if uid is None else uid),
                     "status": status}
        self.running = True

    def is_running(self):
        return self.running

    def status(self):
        if self.info["status"] == "gone":
            raise psutil.NoSuchProcess(1)
        return self.info["status"]


def test_app_window_finds_only_the_apps_own_chrome(tmp_path):
    profile = tmp_path / "chrome-profile"
    flag = f"--user-data-dir={profile}"
    listed = [FakeProcess(["bash", "-c", f"google-chrome {flag} --app=http://x"]),     # a shell that mentions it
              FakeProcess(["/opt/google/chrome/chrome", f"{flag}-old"]),              # another profile
              FakeProcess(["/opt/google/chrome/chrome", flag], uid=os.getuid() + 1),  # another user's
              FakeProcess(["/opt/google/chrome/chrome", flag], status=psutil.STATUS_ZOMBIE),
              FakeProcess(None)]                                                       # unreadable
    scans = []
    window = AppWindow(profile, processes=lambda attrs: (scans.append(attrs), list(listed))[1])
    assert window() is False and len(scans) == 1
    renderer = FakeProcess(["/opt/google/chrome/chrome", "--type=renderer", flag, "--lang=en-GB"])
    listed.append(renderer)
    assert window() is True and window() is True and len(scans) == 2      # remembered: no scan while it lives
    browser = FakeProcess(["/opt/google/chrome/chrome", "--app=http://127.0.0.1:8777/", flag])
    listed[-1:] = [browser]
    renderer.running = False                                                # the page crashed; the window lives
    assert window() is True and len(scans) == 3 and window.found is browser
    browser.info["status"] = "gone"                                         # psutil lost it: look again
    listed.clear()
    assert window() is False and len(scans) == 4 and window.found is None


def test_app_window_finds_a_profile_whose_path_has_spaces(tmp_path):
    profile = tmp_path / "chrome with space" / "chrome-profile"
    flag = f"--user-data-dir={profile}"
    # Chrome rewrites its process title: /proc/<pid>/cmdline becomes one space-joined string that psutil splits on
    # spaces, so the flag arrives in pieces (a probe of a real Chrome 149: no exact element at all).
    title = f"/opt/google/chrome/chrome --type=renderer {flag} --lang=en-GB".split(" ")
    listed = [FakeProcess(["bash", "-c", f"google-chrome {flag} --app=http://x"]),      # a shell that mentions it
              FakeProcess(["/opt/google/chrome/chrome", f"--user-data-dir={tmp_path / 'chrome'}", "with", "space"])]
    window = AppWindow(profile, processes=lambda attrs: list(listed))
    assert window() is False
    listed.append(FakeProcess(title))
    assert window() is True
    browser = f"/opt/google/chrome/chrome --app=http://127.0.0.1:8777/ {flag}".split(" ")   # the flag comes last
    assert AppWindow(profile, processes=lambda attrs: [FakeProcess(browser)])() is True
    exact = ["/opt/google/chrome/chrome", flag, "--class=SoPACompanion"]                     # an untouched argv
    assert AppWindow(profile, processes=lambda attrs: [FakeProcess(exact)])() is True
    # A profile whose path is the start of the running one's is another profile.
    assert AppWindow(tmp_path / "chrome with", processes=lambda attrs: [FakeProcess(title)])() is False


def test_without_a_desktop_or_zenity_runs_are_discarded(monkeypatch, capsys):
    """Without a desktop or zenity the fallback is discard. Keep the log prefix that tools read."""
    sampling = {"id": "r1", "sample": "a sample", "iter": 812, "decided": False}
    finishing = {"id": "r2", "sample": "another", "iter": 400, "decided": True}
    assert server.ask_to_save([sampling, finishing], has_display=lambda: False) is False
    assert capsys.readouterr().out == ("SoPA companion: closing with 1 unsaved run(s): discarding (dialog answer: not asked, "
                                       "no desktop to ask on); waiting for 1 run(s) already being saved or closed\n")
    monkeypatch.setattr(server.shutil, "which", lambda name: None)          # a desktop, but no zenity
    assert server.ask_to_save([sampling], has_display=lambda: True) is False
    line = capsys.readouterr().out
    assert line == "SoPA companion: closing with 1 unsaved run(s): discarding (dialog answer: not asked, no zenity to ask on)\n"
    assert re.search(r"^SoPA companion: closing with (\d+) unsaved run\(s\): (\w+) \(dialog answer:? ?([^)]*)\)", line)


# ---- the save dialog --------------------------------------------------------------------------------------------
FAKE_ZENITY = """#!{python}
import ctypes, json, os, sys, time
open(os.environ["FAKE_ZENITY_ARGS"], "w").write(json.dumps(sys.argv[1:]))
death = ctypes.c_int(0)
ctypes.CDLL(None).prctl(2, ctypes.byref(death))          # PR_GET_PDEATHSIG: the signal it gets when the engine dies
open(os.environ["FAKE_ZENITY_ARGS"] + ".death", "w").write(str(death.value))
action = os.environ["FAKE_ZENITY"]
if action == "hang":
    time.sleep(30)
said, code = {{"save": ("", 0), "discard": ("Discard\\n", 1), "esc": ("", 1), "timeout": ("", 5),
               "error": ("", 255), "other": ("Something else\\n", 1), "gone": ("", 1)}}[action]
sys.stdout.write(said)
sys.stderr.write({{"error": "cannot open display\\n", "gone": "Gtk-WARNING **: Failed to open display\\n",
                  "esc": "libEGL warning: DRI3 error\\n"}}.get(action, ""))
sys.exit(code)
"""


@pytest.fixture
def fake_zenity(tmp_path, monkeypatch):
    folder = tmp_path / "bin"
    folder.mkdir()
    (folder / "zenity").write_text(FAKE_ZENITY.format(python=sys.executable))
    (folder / "zenity").chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_ZENITY_ARGS", str(tmp_path / "args.json"))
    return tmp_path / "args.json"


def test_only_the_save_button_saves(fake_zenity, monkeypatch, capsys):
    unsaved = [{"id": "r1", "sample": "sample-1", "iter": 812}]
    desktop = lambda: True                                                  # no DISPLAY needed
    # zenity 4: Save results / Enter -> 0; Discard -> "Discard" and 1; Esc or the window's close button -> 1, nothing
    # printed; --timeout -> 5; a display that has gone -> 1, nothing printed; anything else is a failure. Only the
    # Discard button deletes work.
    for action, save in (("discard", False), ("save", True), ("esc", False), ("timeout", False), ("error", False),
                         ("other", False), ("gone", False)):
        monkeypatch.setenv("FAKE_ZENITY", action)
        assert server.ask_to_save(unsaved, has_display=desktop) is save, action
    args = json.loads(fake_zenity.read_text())
    assert {"--ok-label=Save to the sample", "--extra-button=Discard", "--timeout=600"} <= set(args)
    assert "sample-1 (run, iteration 812)" in args[-1] and "discards it" in args[-1]
    log = capsys.readouterr().out.splitlines()
    assert log[0].endswith("discarding (dialog answer: Discard)")
    assert log[2].endswith("discarding (dialog answer: closed without choosing)")
    assert "discarding (dialog answer: exit 255: cannot open display)" in log[4]
    assert log[6].endswith("discarding (dialog answer: could not open: Gtk-WARNING **: Failed to open display)")
    monkeypatch.setattr(server, "DIALOG_MARGIN_S", 0.5)                     # a dialog that never answers: discard
    monkeypatch.setenv("FAKE_ZENITY", "hang")
    assert server.ask_to_save(unsaved, timeout_s=0.5, has_display=desktop) is False
    assert "discarding (dialog answer: failed: no answer within 1 s)" in capsys.readouterr().out
    monkeypatch.setenv("FAKE_ZENITY", "esc")
    assert server.ask_to_save(unsaved, timeout_s=0, has_display=desktop) is False
    assert not any(a.startswith("--timeout") for a in json.loads(fake_zenity.read_text()))   # 0: wait for an answer
    fake_zenity.unlink()
    assert server.ask_to_save(unsaved, has_display=lambda: False) is False    # nobody to ask: discard
    assert not fake_zenity.exists()                                          # ... without running zenity


def test_runs_already_being_saved_or_closed_are_named_never_asked_about(fake_zenity, monkeypatch, capsys):
    desktop = lambda: True
    monkeypatch.setenv("FAKE_ZENITY", "discard")
    finishing = {"id": "r2", "sample": "sample-2", "iter": 400, "decided": True}
    assert server.ask_to_save([finishing], has_display=desktop) is False      # nothing to decide: no dialog at all
    assert not fake_zenity.exists()
    sampling = {"id": "r1", "sample": "sample-1", "iter": 812, "decided": False}
    assert server.ask_to_save([sampling, finishing], has_display=desktop) is False   # Discard: for r1 only
    text = json.loads(fake_zenity.read_text())[-1]
    assert "Save this run to the sample first?\n  • sample-1 (run, iteration 812)\n\n" in text
    assert "Already being saved or closed (your earlier choice still applies):\n  • sample-2 (run, iteration 400)" in text
    assert "discards it." in text
    assert capsys.readouterr().out.strip().endswith(
        "closing with 1 unsaved run(s): discarding (dialog answer: Discard); waiting for 1 run(s) already being saved or closed")
    if shutil.which("setpriv"):        # the dialog dies with the engine: a killed engine leaves no dead buttons behind
        assert int(Path(str(fake_zenity) + ".death").read_text()) == signal.SIGTERM
    assert server.ask_to_save([sampling, finishing], has_display=lambda: False) is False
    assert ("closing with 1 unsaved run(s): discarding (dialog answer: not asked, no desktop to ask on); "
            "waiting for 1 run(s) already being saved or closed") in capsys.readouterr().out


# ---- sopa_companion stop --------------------------------------------------------------------------------------------
def test_stop_force_asks_the_engine_to_stop_without_a_dialog_and_waits_for_it(config, monkeypatch):
    """`stop --force` asks the engine to stop without the dialog (POST /shutdown): the
    runs nobody answered for are discarded, a save under way is finished first, and the command says which."""
    record = {"pid": 4242, "port": 8777, "token": "t", "base_url": "http://127.0.0.1:8777"}
    monkeypatch.setattr(launcher, "read_runtime", lambda cfg: record)
    calls = []

    def request(rec, method, path):
        calls.append((method, path))
        return {"discarding": ["r1"], "saving": ["r2"]}
    services = launcher.Services(alive=lambda pid: True, request=request, output=lambda m: calls.append(("say", m)),
                                 stopper=lambda cfg, pid, **kw: calls.append(("signal", pid, kw["force"], kw["timeout"])))
    services.healthy = lambda rec: calls.append(("verified", rec["pid"])) or True
    from sopa_companion.cli import parser                # the command line's own defaults
    launcher.stop(config, parser().parse_args(["stop", "--force"]), services=services)
    # The listener is identified before the token goes out; the engine then discards and saves without asking, and
    # the command waits long enough for a whole sample's results before it would kill anything.
    assert calls[:2] == [("verified", 4242), ("POST", "/api/v1/shutdown")]
    assert ("say", "SoPA companion is stopping: 1 run not saved to its sample is discarded; 1 run is being saved first "
                   "(up to 900 s).") in calls
    assert ("signal", 4242, True, launcher.STOP_TIMEOUT_S) in calls and launcher.STOP_TIMEOUT_S >= 900
    assert calls[-1] == ("say", "SoPA companion stopped.")
    assert launcher.stopping_words(["a", "b"], [], 900) == ("SoPA companion is stopping: 2 runs not saved to their "
                                                             "samples are discarded.")
    assert launcher.stopping_words([], ["a", "b"], 60) == "SoPA companion is stopping: 2 runs are being saved first (up to 60 s)."
    calls.clear()     # plain stop: no shutdown request; it asks which runs wait for an answer (refused while any does)
    launcher.stop(config, parser().parse_args(["stop"]), services=services)
    assert calls[:3] == [("verified", 4242), ("GET", "/api/v1/closing"), ("signal", 4242, False, 30)]
    calls.clear()
    launcher.stop(config, parser().parse_args(["stop", "--timeout", "5"]), services=services)
    assert calls[2] == ("signal", 4242, False, 5.0)
    calls.clear()
    services.healthy = lambda rec: False            # an engine already shutting down (its dialog open): SIGTERM only
    launcher.stop(config, parser().parse_args(["stop", "--force"]), services=services)
    assert calls[0] == ("signal", 4242, True, launcher.STOP_TIMEOUT_S)


# ---- whose engine it is ------------------------------------------------------------------------------------------
class Serve:
    """A `python -m sopa_companion serve` process as foreign_engine sees it: command line, owner, working folder and
    environment."""

    def __init__(self, pid, args=(), uid=None, cwd="/", environ=None, module="sopa_companion"):
        self.pid = pid
        self.info = {"pid": pid, "cmdline": ["/x/.venv/bin/python", "-m", module, *args, "serve"],
                     "uids": types.SimpleNamespace(real=os.getuid() if uid is None else uid)}
        self._cwd, self._environ = str(cwd), dict(environ or {})

    def cmdline(self):
        return self.info["cmdline"]

    def cwd(self):
        return self._cwd

    def environ(self):
        return self._environ


def app_folder(folder: Path, data_dir: str = "var") -> Path:
    """A minimal app folder (what _install_root recognises) whose config.toml names `data_dir`."""
    (folder / "engine").mkdir(parents=True)
    (folder / "cli.py").write_text("")
    (folder / "config.toml").write_text(f'[paths]\ndata_dir = "{data_dir}"\n')
    return folder


def test_another_users_or_another_installs_engine_is_never_this_ones(config, tmp_path, monkeypatch):
    """Another user's serve is never this install's engine, even with a --config naming this data folder (the icon offered
    to stop it, then refused); a serve without --config belongs to the app folder it runs from (its working folder, or
    the sopa_companion its PYTHONPATH finds), not to this install; whatever listens on the port is never an owner (another
    user's socket has no visible pid, and serve takes the next free port)."""
    from sopa_companion.desktop import processes
    data_dir = config["paths"]["data_dir"]
    named = tmp_path / "named.toml"
    named.write_text(f'[paths]\ndata_dir = "{data_dir}"\n')
    other = app_folder(tmp_path / "other" / "sopa_companion")                  # another install: its own var
    mine = app_folder(tmp_path / "mine" / "sopa_companion", data_dir)          # an install whose data folder is this one
    look = lambda *listed: processes.foreign_engine(config, None, processes=lambda attrs: list(listed))
    assert look(Serve(4001, ["--config", str(named)], uid=os.getuid() + 1)) is None          # another user's
    assert look(Serve(4002, uid=os.getuid() + 1, cwd=mine)) is None
    assert look(Serve(4003, cwd=other)) is None                             # another app folder, by its working folder
    assert look(Serve(4004, cwd=tmp_path / "other")) is None                # ... or the sopa_companion found from there
    assert look(Serve(4005, cwd="/", environ={"PYTHONPATH": str(tmp_path / "other")})) is None   # ... or on its path
    assert look(Serve(4006, cwd="/")) is None                               # its app folder cannot be told: not this
    for found in (Serve(4007, ["--config", str(named)]), Serve(4008, cwd=mine), Serve(4009, cwd=tmp_path / "mine"),
                  Serve(4010, environ={"PYTHONPATH": str(tmp_path / "mine")})):
        assert look(found) == {"pid": found.pid, "data_dir": data_dir, "mine": True}
    listener = types.SimpleNamespace(status=psutil.CONN_LISTEN, pid=None, laddr=types.SimpleNamespace(
        ip="127.0.0.1", port=config["server"]["port"]))
    monkeypatch.setattr(psutil, "net_connections", lambda kind="inet": [listener])
    assert look() is None                                                   # another user's listener on the port


def test_an_engine_of_the_previous_version_is_known_by_its_own_package_name(config, tmp_path):
    """`stop` and `status` of this version know an engine the previous version started (`-m proseg_live serve`, from a
    folder called proseg_live), and still nothing else."""
    from sopa_companion.desktop import processes
    for argv in (["py", "-m", "proseg_live", "serve"], ["py", "-m", "proseg_live", "--config", "/x.toml", "serve"],
                 ["py", "-m", "sopa_companion", "--config=/x.toml", "serve"]):
        assert processes.serve_command(argv), argv
    for argv in (["py", "-m", "proseg_live", "launch"], ["py", "-m", "proseg_live.harness", "serve"],
                 ["py", "-m", "proseg_livex", "serve"], ["py", "-m"], ["bash", "-c", "python -m proseg_live serve"]):
        assert not processes.serve_command(argv), argv
    data_dir = config["paths"]["data_dir"]
    old = app_folder(tmp_path / "old" / "proseg_live", data_dir)             # the previous version's install
    new = app_folder(tmp_path / "new" / "sopa_companion")
    look = lambda *listed: processes.foreign_engine(config, None, processes=lambda attrs: list(listed))
    found = {"pid": 4101, "data_dir": data_dir, "mine": True}
    assert look(Serve(4101, cwd=old, module="proseg_live")) == found                                   # its folder
    assert look(Serve(4101, cwd=tmp_path / "old", module="proseg_live")) == found                      # found there
    assert look(Serve(4101, environ={"PYTHONPATH": str(tmp_path / "old")}, module="proseg_live")) == found
    # By the package it runs: -m sopa_companion from the folder holding both is the new install, not the old one.
    (tmp_path / "both").mkdir()
    for name, folder in (("proseg_live", old), ("sopa_companion", new)):
        (tmp_path / "both" / name).symlink_to(folder)
    assert look(Serve(4102, cwd=tmp_path / "both", module="proseg_live")) == {**found, "pid": 4102}
    assert look(Serve(4103, cwd=tmp_path / "both")) is None
    process = Serve(4104, module="proseg_live")
    process.uids = lambda: types.SimpleNamespace(real=os.getuid(), effective=os.getuid())
    assert processes.verify_process(4104, process_factory=lambda pid: process) is process
    process.info["cmdline"] = ["py", "-m", "proseg_live", "launch"]
    with pytest.raises(DesktopError, match="not the app's serve"):
        processes.verify_process(4104, process_factory=lambda pid: process)


def test_a_fresh_clones_engine_is_found_by_the_settings_it_reads(config, tmp_path, monkeypatch):
    """A clone without config.toml runs on config.example.toml, and config.local.toml may name its data folder ("~" in
    it too): its engine is found by the data folder the files load_config merges give (such a folder was not taken for
    the app's, and config.local.toml went unread)."""
    from sopa_companion.desktop import processes
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    clone = tmp_path / "clone" / "sopa-companion"
    (clone / "engine").mkdir(parents=True)
    (clone / "cli.py").write_text("")
    (clone / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')       # a clone: a source tree
    (clone / "config.example.toml").write_text('[paths]\ndata_dir = "var"\n')
    assert processes._app_folder(clone) and processes._data_dir(clone) == (clone / "var").resolve()
    (clone / "config.local.toml").write_text('[paths]\ndata_dir = "~/sopa-data"\n')
    assert processes._data_dir(clone) == (home / "sopa-data").resolve()
    mine = {**config, "paths": {**config["paths"], "data_dir": str((home / "sopa-data").resolve())}}
    found = processes.foreign_engine(mine, None, processes=lambda attrs: [Serve(4011, cwd=clone)])
    assert found == {"pid": 4011, "data_dir": mine["paths"]["data_dir"], "mine": True}


def test_auto_folders_are_read_as_the_engine_reads_them(tmp_path, monkeypatch):
    """[paths] "auto" (or empty, or none) in an engine's settings, as its stop, status and the state folder's move read
    them (engine/config.py data_dir_of, state_dir_of): a source tree's var/, an installed copy's per-user data folder,
    and this version's state folder; the previous version's default stays its own. Never `<app folder>/auto`."""
    from sopa_companion.desktop import migrate, processes
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    clone = app_folder(tmp_path / "clone" / "sopa_companion", "auto")
    (clone / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    installed = app_folder(tmp_path / "env" / "site-packages" / "sopa_companion", "auto")
    data = home / ".local/share/sopa-companion"                  # whatever $XDG_DATA_HOME says (the conftest sets it)
    assert processes._data_dir(clone) == (clone / "var").resolve()
    assert processes._data_dir(installed) == data.resolve()
    (installed / "config.toml").write_text("[paths]\n")                             # no data_dir at all: "auto" too
    assert processes._data_dir(installed) == data.resolve()
    named = tmp_path / "named.toml"
    named.write_text('[paths]\ndata_dir = "auto"\n')
    assert processes._data_dir(None, named) is None                         # without an app folder, "auto" tells nothing
    this_version = (home / ".local/state/sopa-companion").resolve()
    for value in ('state_dir = "auto"', 'state_dir = ""', ""):
        (clone / "config.toml").write_text(f"[paths]\n{value}\n")
        assert migrate._engine_state(Serve(4201, cwd=clone), "sopa_companion") == this_version, value
    assert migrate._engine_state(Serve(4202, cwd=clone, module="proseg_live"), "proseg_live") == \
        (home / ".local/state/proseg_live").resolve()                       # the previous version's default
    (clone / "config.toml").write_text('[paths]\nstate_dir = "state"\n')
    assert migrate._engine_state(Serve(4203, cwd=clone), "sopa_companion") == (clone / "state").resolve()
    # The console script's engine (sopa-companion serve) is this version's, and the move knows it.
    old_state = home / ".local/state/proseg_live"
    old_state.mkdir(parents=True)
    (clone / "config.toml").write_text(f'[paths]\nstate_dir = "{old_state}"\n')
    script = Serve(4204, cwd=clone)
    script.info["cmdline"] = ["/env/bin/python3.12", "/env/bin/sopa-companion", "serve"]
    assert migrate.old_users(old_state, home / "none", entry=home / "none", processes=lambda attrs=None: [script]) == \
        ["an engine (pid 4204, sopa-companion) may be using it"]


def test_the_icon_never_offers_to_stop_another_users_engine(config):
    """A data folder whose engine lock another user's engine holds: the launch says so plainly (exit 3), and asks
    nothing (no restart question, no signal)."""
    asked = []
    services = launcher.Services(discover=lambda cfg, record=None: {"pid": 4242, "data_dir": cfg["paths"]["data_dir"],
                                                                    "mine": False},
                                 which=lambda name: f"/usr/bin/{name}", environ={"DISPLAY": ":99"},
                                 run=lambda *a, **k: asked.append(a), stopper=lambda *a, **k: asked.append(a))
    with pytest.raises(DesktopError, match="Another user's SoPA companion engine") as refused:
        launcher._handle_foreign(config, None, services)
    assert refused.value.exit_code == 3 and asked == []


# ---- the name users see ------------------------------------------------------------------------------------------
def test_users_see_sopa_companion_everywhere_but_the_tool_called_proseg_live(config):
    """The desktop entry, the launcher's dialogs and its messages say SoPA companion (Proseg Live is one of its two tools
    now); the window class SoPACompanion (the installed launchers group the window by it, the harness finds the window by
    it; the notebook's window has the same), the icon sopa-companion, the module sopa_companion and the menu actions."""
    from sopa_companion.desktop import install
    entry = install.entry_text("/usr/bin/python3", "/tmp/server.log")
    assert "\nName=SoPA companion\n" in entry
    assert entry.split("\nComment=")[1].splitlines()[0].startswith("Stain segmentation with Cellpose, and Proseg Live")
    assert "instanseg" not in entry.lower(), "InstanSeg is dormant: the menu never names it"
    for kept in ("StartupWMClass=SoPACompanion", "Icon=sopa-companion", " -m sopa_companion launch", "\nName=Stop engine\n",
                 "\nName=Show engine log\n"):
        assert kept in entry, kept
    chrome = lambda name: "/usr/bin/google-chrome" if name == "google-chrome" else None
    assert "--class=SoPACompanion" in browser_command(config, "http://127.0.0.1:8777/?code=c", which=chrome)
    # The notebook's window is opened by the same code (desktop/browser.py open_app, through api/window.py).
    from sopa_companion.api import window as notebook_window
    assert "browser.open_app(" in Path(notebook_window.__file__).read_text()
    assert 'window_class="SoPACompanion"' in Path(browser.__file__).read_text()
    spawned = []
    child = types.SimpleNamespace(stdin=types.SimpleNamespace(write=lambda data: None, close=lambda: None),
                                  wait=lambda timeout=None: 0, terminate=lambda: None)
    services = launcher.Services(which=lambda name: f"/usr/bin/{name}", spawn=lambda command, **kw: spawned.append(command) or child,
                                 environ={"DISPLAY": ":99"})
    with launcher._progress(services):                # "Starting the engine…" while the icon starts it
        pass
    assert "--title=SoPA companion" in spawned[0] and "--text=Starting the engine…" in spawned[0]
    # Every other dialog and message of the launcher and the installer: the new name, never the old one as the app's.
    assert "Proseg Live" not in Path(launcher.__file__).read_text()
    assert [line for line in Path(install.__file__).read_text().splitlines() if "Proseg Live" in line] == \
        [line for line in entry.splitlines() if line.startswith("Comment=")]


# ---- the launcher on the desktop and the icon --------------------------------------------------------------------
ICON = ROOT / "desktop/sopa-companion.svg"


def test_the_icon_is_plain_svg_drawn_on_the_pixel_grid():
    """Plain SVG: nothing outside the file (no links, images, text or fonts), only its own gradients. Drawn on the 48 px
    grid, so every outline lands on a whole pixel at 48 px (the cells' 3-unit voxels at 128 px too); the lit cell's
    transcripts glow in the suspect colour of the Go deeper view."""
    text = ICON.read_text()
    root = ElementTree.fromstring(text)
    svg = "{http://www.w3.org/2000/svg}"
    assert root.tag == svg + "svg" and root.get("viewBox") == "0 0 48 48"
    assert {element.tag.removeprefix(svg) for element in root.iter()} <= {
        "svg", "defs", "linearGradient", "radialGradient", "stop", "g", "rect", "path", "circle"}
    assert not re.search(r"href|font|@import|data:", text)
    ids = {element.get("id") for element in root.iter()} - {None}
    assert all(ref.startswith("#") and ref[1:] in ids for ref in re.findall(r"url\(([^)]*)\)", text))
    for element in root.iter(svg + "path"):
        assert all(float(number).is_integer() for number in re.findall(r"[\d.]+", element.get("d"))), element.get("d")
    assert "#ff4fa3" in text


def test_the_page_icon_is_the_desktop_icon():
    """The page's favicon (frontend/index.html) is the icon above as a data URI: the same elements
    with the same attributes in the same order, only the file's intrinsic size left out."""
    from urllib.parse import unquote
    page = (ROOT / "frontend/index.html").read_text()
    (uri,) = re.findall(r'<link rel="icon" href="data:image/svg\+xml,([^"]+)">', page)

    def shape(text):
        return [(element.tag, sorted((k, v) for k, v in element.attrib.items() if element.tag[-3:] != "svg" or k not in
                                     ("width", "height"))) for element in ElementTree.fromstring(text).iter()]
    assert shape(unquote(uri)) == shape(ICON.read_text()) and len(shape(unquote(uri))) > 40


class Installer:
    """`install-desktop --home <a temporary home>` with the desktop's tools faked: every command is recorded and
    succeeds, and there is no xdg-user-dir, so the desktop folder is <home>/Desktop."""

    def __init__(self, home):
        self.home, self.ran, self.said = home, [], []
        share = home / ".local/share"
        self.entry = share / "applications/sopa-companion.desktop"
        self.icon = share / "icons/hicolor/scalable/apps/sopa-companion.svg"
        self.record = share / "sopa-companion/desktop-install.json"
        self.launcher = home / "Desktop/SoPA-companion.desktop"
        self.old = home / "Desktop/Proseg-Live.desktop"
        # What the previous version (the package proseg_live) installed: its menu entry and icon, and its record.
        self.old_entry = share / "applications/proseg-live.desktop"
        self.old_icon = share / "icons/hicolor/scalable/apps/proseg-live.svg"
        self.legacy_record = share / "proseg_live/desktop-install.json"

    def __call__(self, *flags):
        from sopa_companion.cli import parser
        from sopa_companion.desktop import install
        self.ran.clear()
        self.said.clear()
        args = parser().parse_args(["install-desktop", "--home", str(self.home), *flags])
        config = {"paths": {"state_dir": str(self.home / "state")}}
        return install.install(config, args, run=self.run, which=self.which, output=self.said.append)

    def run(self, command, **kwargs):
        self.ran.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    @staticmethod
    def which(name):
        return None if name == "xdg-user-dir" else f"/usr/bin/{name}"

    def files(self):
        return {path.relative_to(self.home).as_posix(): path.read_bytes() for path in self.home.rglob("*") if path.is_file()}

    def recorded(self):
        return json.loads(self.record.read_text())["paths"]

    def round4(self, launcher=None):
        """What the previous version's installer left: its menu entry and icon (proseg-live) and the launcher on the
        desktop (named Proseg-Live.desktop in its round 4), all three in its record."""
        launcher = launcher or self.old
        for path, mode in ((self.old_entry, 0o644), (self.old_icon, 0o644), (launcher, 0o755)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("round 4\n")
            path.chmod(mode)
        self.legacy_record.parent.mkdir(parents=True, exist_ok=True)
        self.legacy_record.write_text(json.dumps({"schema": 1, "paths": sorted(map(str, (self.old_entry, self.old_icon,
                                                                                         launcher)))}))


@pytest.fixture
def installer(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))           # a slip to the real home would land here
    return Installer(home)


def test_the_launcher_on_the_desktop_is_sopa_companion(installer):
    """The launcher on the desktop is SoPA-companion.desktop, the app grid's entry sopa-companion.desktop, the icon
    sopa-companion.svg (Icon=sopa-companion), recorded in ~/.local/share/sopa-companion/desktop-install.json."""
    i = installer
    assert i() == 0
    assert sorted(i.files()) == [".local/share/applications/sopa-companion.desktop",
                                 ".local/share/icons/hicolor/scalable/apps/sopa-companion.svg",
                                 ".local/share/sopa-companion/desktop-install.json", "Desktop/SoPA-companion.desktop"]
    entry = i.launcher.read_text()
    assert i.entry.read_text() == entry and "\nName=SoPA companion\n" in entry and "\nIcon=sopa-companion\n" in entry
    assert i.icon.read_bytes() == ICON.read_bytes()
    assert [path.stat().st_mode & 0o777 for path in (i.entry, i.icon, i.launcher)] == [0o644, 0o644, 0o755]
    assert i.recorded() == sorted(map(str, (i.entry, i.icon, i.launcher)))
    assert [command[1] for command in i.ran if command[0].endswith("desktop-file-validate")] == [str(i.entry), str(i.launcher)]
    assert ["/usr/bin/gio", "set", str(i.launcher), "metadata::trusted", "true"] in i.ran    # the desktop may launch it
    assert i.said == ["SoPA companion desktop integration installed."]


def test_the_previous_versions_install_is_replaced_by_the_new_one(installer, tmp_path):
    """Over the previous version's install: the new entry, icon and launcher are written and recorded first; then what
    its record lists goes (its menu entry and icon, the old launcher on the desktop), and its record last. The dock's
    favourite named the old entry: the message says to pin the icon again."""
    i = installer
    i.round4()
    assert i() == 0
    assert i.launcher.exists() and not i.old.exists() and not i.old_entry.exists() and not i.old_icon.exists()
    assert i.recorded() == sorted(map(str, (i.entry, i.icon, i.launcher)))
    assert not i.legacy_record.exists() and not i.legacy_record.parent.exists()
    assert sorted(i.files()) == [".local/share/applications/sopa-companion.desktop",
                                 ".local/share/icons/hicolor/scalable/apps/sopa-companion.svg",
                                 ".local/share/sopa-companion/desktop-install.json", "Desktop/SoPA-companion.desktop"]
    assert i.said == ["SoPA companion desktop integration installed. The previous version's entry was replaced: pin "
                      "the new icon in the dock again if you had pinned it."]
    # The test's own folder is left out: the gate runs the tests under the harness's state folder, proseg_live_harness.
    entry = i.entry.read_text().replace(str(tmp_path), "")
    assert "-m sopa_companion launch" in entry and "proseg_live" not in entry
    # Its last install, with SoPA-companion.desktop on the desktop already: replaced in place, its window profile kept.
    later = Installer(tmp_path / "later")
    later.round4(launcher=later.launcher)
    (later.legacy_record.parent / "chrome-profile").mkdir()
    assert later() == 0 and later.recorded() == sorted(map(str, (later.entry, later.icon, later.launcher)))
    assert "\nIcon=sopa-companion\n" in later.launcher.read_text() and not later.legacy_record.exists()
    assert (later.legacy_record.parent / "chrome-profile").is_dir()     # never ours to remove: the state move's
    i.said.clear()
    # A Proseg-Live.desktop the record does not list (the user's own, or older than the record) is not ours to remove.
    i.old.write_text("the user's\n")
    assert i() == 0
    assert i.old.read_text() == "the user's\n" and i.said == ["SoPA companion desktop integration is already installed."]
    # An install stopped between its two record writes: all in place, the old launcher still listed. Removing it is
    # the one change, and it is said.
    i.record.write_text(json.dumps({"schema": 1, "paths": sorted(map(str, (i.entry, i.icon, i.old, i.launcher)))}))
    assert i() == 0
    assert not i.old.exists() and i.said == ["SoPA companion desktop integration installed."]
    # Listed, but deleted by hand since: nothing to remove, and the record forgets it.
    gone = Installer(tmp_path / "gone")
    gone.round4()
    gone.old.unlink()
    assert gone() == 0 and gone.recorded() == sorted(map(str, (gone.entry, gone.icon, gone.launcher)))


def test_without_a_launcher_on_the_desktop_the_old_one_stays_recorded(installer):
    """--no-desktop-icon writes nothing on the desktop, so nothing replaces the old launcher: it stays, still recorded,
    and uninstall removes it (the previous version's menu entry and icon are replaced all the same)."""
    i = installer
    i.round4()
    assert i("--no-desktop-icon") == 0
    assert i.old.exists() and not i.launcher.exists() and str(i.old) in i.recorded()
    assert not i.old_entry.exists() and not i.old_icon.exists() and not i.legacy_record.exists()
    assert i("--uninstall") == 0
    assert i.files() == {}


def test_uninstall_knows_every_name_either_version_installed(installer):
    i = installer
    i.round4()
    i.launcher.write_text("round 5\n")              # all recorded: an install stopped between its record writes
    for path in (i.entry, i.icon):
        path.write_text("new\n")
    record = sorted(map(str, (i.entry, i.icon, i.old, i.launcher, i.old_entry, i.old_icon)))
    i.record.parent.mkdir(parents=True)
    i.record.write_text(json.dumps({"schema": 1, "paths": record}))
    before = i.files()
    assert i("--uninstall", "--dry-run") == 0
    assert i.files() == before and not i.ran
    assert i.said == [f"Would remove {path}" for path in [*record, i.record, i.legacy_record]]
    assert i("--uninstall") == 0
    assert i.files() == {} and i.said == ["SoPA companion desktop integration removed."]
    assert [command[0] for command in i.ran] == ["/usr/bin/update-desktop-database", "/usr/bin/gtk-update-icon-cache"]
    # Only the previous version's record: what it lists, and it, go.
    i.round4()
    assert i("--uninstall") == 0 and i.files() == {}
    # A record naming anything else is refused before anything is removed.
    i.round4()
    i.record.write_text(json.dumps({"schema": 1, "paths": [str(i.entry), str(i.home / "Desktop/notes.txt")]}))
    before = i.files()
    with pytest.raises(DesktopError, match="Unexpected path"):
        i("--uninstall")
    assert i.files() == before


def test_a_dry_run_only_says_what_it_would_do(installer):
    i = installer
    i.round4()
    before = i.files()
    assert i("--dry-run") == 0
    assert i.files() == before and not i.ran           # nothing written, removed or run
    stale = sorted(map(str, (i.old_entry, i.old_icon, i.old)))
    assert i.said[:8] == [f"Would install {i.entry} (644)", f"Would install {i.icon} (644)",
                          f"Would install {i.launcher} (755)", *(f"Would remove {path}" for path in stale),
                          f"Would record {i.record}", f"Would remove {i.legacy_record}"]
    assert "\nName=SoPA companion\n" in i.said[8]


def test_installing_again_changes_nothing(installer):
    i = installer
    assert i() == 0
    files = {path: path.stat().st_ino for path in (i.entry, i.icon, i.launcher, i.record)}
    assert i() == 0
    assert {path: path.stat().st_ino for path in files} == files          # nothing rewritten (writes replace the file)
    assert i.said == ["SoPA companion desktop integration is already installed."]
    assert all(command[0].endswith("desktop-file-validate") for command in i.ran)     # checked again; no trust, no refresh


def test_the_launcher_passes_the_desktop_entry_validator(tmp_path, monkeypatch):
    """The real desktop-file-validate (install-desktop requires it) accepts the entries under their names."""
    if not (validator := shutil.which("desktop-file-validate")):
        pytest.skip("desktop-file-validate is not installed")
    from sopa_companion.cli import parser
    from sopa_companion.desktop import install
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    args = parser().parse_args(["install-desktop", "--home", str(home)])
    tools = lambda name: validator if name == "desktop-file-validate" else None       # nothing else runs
    assert install.install({"paths": {"state_dir": str(tmp_path / "state")}}, args, which=tools, output=lambda line: None) == 0
    assert (home / "Desktop/SoPA-companion.desktop").is_file()


# ---- full power -------------------------------------------------------------------------------------------------
def test_shipped_defaults_use_the_whole_workstation():
    """proseg's threads by the shipped rule, on the cores this engine may use: half of them, at most 32, for a run on a
    region, three quarters for a tiled run (32 and 96 of a 128-core workstation; 1 and 1 of a 2-core CI runner)."""
    from sopa_companion.engine.analysis import DEFAULT_MIN_INTERVAL
    from sopa_companion.engine.config import AUTO_THREADS, cores
    shipped = load_config()
    here = cores()
    assert shipped["proseg"]["tiled_threads"] == max(1, here * 3 // 4)
    assert shipped["proseg"]["threads"] == max(1, min(32, here // 2))
    assert (AUTO_THREADS["threads"](128), AUTO_THREADS["tiled_threads"](128)) == (32, 96)
    assert shipped["analysis"]["min_interval_s"] == 0 == DEFAULT_MIN_INTERVAL          # as fast as the GPU worker goes
    assert shipped["desktop"]["profile_dir"] and shipped["desktop"]["save_dialog_timeout_s"] == 600



def test_stop_checks_finished_unsaved_runs_and_ignores_saved_test_and_older_runs(config, monkeypatch):
    """A plain `stop` (and the icon's Stop engine) waits for an answer about this version's runs not saved to their
    samples; runs from earlier versions (no one-run marker) never block it: the app lists them at its next start."""
    from test_runs_save import ended_run
    from sopa_companion.desktop import processes
    from sopa_companion.desktop.processes import active_metadata
    ended_run(config, "pending", session_model=2)
    ended_run(config, "saved", zarr_save={"state": "saved"}, session_model=2)
    ended_run(config, "square", qc_test={"of": "pending"}, session_model=2)
    ended_run(config, "filtered", kind="stain", summary_model={"cells": 6}, summary={"cells": 0}, session_model=2)
    ended_run(config, "patches", kind="patches", session_model=2)
    ended_run(config, "older")                                     # made before the one-run model
    assert {r["id"] for r in active_metadata(config)["runs"]} == {"pending", "filtered"}
    process = types.SimpleNamespace(pid=4242, send_signal=lambda sig: pytest.fail("stop signalled a pending engine"))
    monkeypatch.setattr(processes, "verify_process", lambda pid, **kw: process)
    with pytest.raises(DesktopError, match="save or discard it in the app, or use stop --force to discard it") as refused:
        processes.stop_process(config, 4242)
    assert refused.value.exit_code == 2 and "run pending (done)" in str(refused.value)


def test_a_plain_stop_asks_the_engine_which_runs_wait_for_an_answer(config, monkeypatch):
    """A plain stop (the icon's Stop engine) counted every marked run.json, so a folder the engine
    left alone at its start (the last session's save still writing it), which the app does not show, refused it with
    "save or discard it in the app". A healthy engine says which runs wait (GET /closing: Runs.pending); only without
    its answer are the run.json files read, and a run an engine kept on disk as it closed (`closed`) never counts."""
    from test_runs_save import ended_run
    from sopa_companion.cli import parser
    from sopa_companion.desktop import processes
    record = {"pid": 4242, "port": 8777, "token": "t", "base_url": "http://127.0.0.1:8777"}
    monkeypatch.setattr(launcher, "read_runtime", lambda cfg: record)
    ended_run(config, "still-saving", session_model=2, zarr_save={"state": "saving"})   # left alone at the start
    ended_run(config, "kept", session_model=2, closed="save failed",                    # kept as the last one closed
              zarr_save={"state": "failed", "message": "No space left on device"})
    assert [r["id"] for r in processes.active_metadata(config)["runs"]] == ["still-saving"]
    signalled, pending = [], []
    process = types.SimpleNamespace(pid=4242, send_signal=signalled.append, wait=lambda timeout: None)
    monkeypatch.setattr(processes, "verify_process", lambda pid, **kw: process)
    monkeypatch.setattr(processes, "remove_runtime", lambda cfg, pid: None)
    services = launcher.Services(alive=lambda pid: True, output=lambda message: None, stopper=processes.stop_process,
                                 request=lambda rec, method, path: {"runs": pending} if path == "/api/v1/closing" else {})
    services.healthy = lambda rec: True
    launcher.stop(config, parser().parse_args(["stop"]), services=services)        # the engine: nothing waits
    assert signalled == [signal.SIGTERM]
    pending.append({"id": "r1", "kind": "proseg", "state": "done", "decided": False})   # a run its window shows
    with pytest.raises(DesktopError, match="save or discard it in the app") as refused:
        launcher.stop(config, parser().parse_args(["stop"]), services=services)
    assert "run r1 (done)" in str(refused.value) and "still-saving" not in str(refused.value)
    # No answer from the engine (no healthy engine, an older one without the route, a reply it cannot read): the
    # run.json files.
    for healthy, answer in ((False, None), (True, DesktopError("Engine request failed (HTTP 404).")), (True, [])):
        def request(rec, method, path, answer=answer):
            if isinstance(answer, Exception):
                raise answer
            return answer
        services.healthy, services.request = (lambda rec, healthy=healthy: healthy), request
        with pytest.raises(DesktopError, match=r"run still-saving \(done\)"):
            launcher.stop(config, parser().parse_args(["stop"]), services=services)
    assert signalled == [signal.SIGTERM]


def test_print_url_gives_a_one_use_link_never_the_token(config, monkeypatch):
    """launch --print-url printed the bearer token in its URL (?token=), which a browser keeps in
    its history, and which issues launch codes and stops the engine for as long as it runs. It prints a launch code
    asked for as a link (one use, 10 minutes: time for an SSH tunnel); the window's own code is asked for plain."""
    from sopa_companion.cli import parser
    record = {"pid": 4242, "port": 8777, "token": "t" * 43, "base_url": "http://127.0.0.1:8777"}
    monkeypatch.setattr(launcher, "ensure_engine", lambda config, **kw: record)
    asked, said, spawned = [], [], []
    services = launcher.Services(request=lambda rec, method, path: asked.append((method, path)) or {"code": "c" * 43},
                                 output=said.append, spawn=lambda command, **kw: spawned.append(command),
                                 which=lambda name: "/usr/bin/google-chrome" if name == "google-chrome" else None,
                                 environ={"DISPLAY": ":9"})
    launcher.launch(config, parser().parse_args(["launch", "--print-url"]), services=services)
    assert asked == [("POST", "/api/v1/session/launch-code?link=1")] and spawned == []
    assert said == ["http://127.0.0.1:8777/?code=" + "c" * 43]
    asked.clear()
    launcher.launch(config, parser().parse_args(["launch"]), services=services)
    assert asked == [("GET", "/api/v1/health"), ("POST", "/api/v1/session/launch-code")]
    # Neither the token nor the window's code is on Chrome's command line: it opens a private page that sends it on.
    assert not any("t" * 43 in a or "c" * 43 in a for a in spawned[0])
    page = Path(next(a for a in spawned[0] if a.startswith("--app=")).removeprefix("--app=file://"))
    assert "http://127.0.0.1:8777/?code=" + "c" * 43 in page.read_text()
    services.request = lambda rec, method, path: {}
    with pytest.raises(DesktopError, match="did not return a launch code"):
        launcher.launch(config, parser().parse_args(["launch", "--print-url"]), services=services)


def test_the_window_opens_a_private_page_never_a_launch_code_on_a_command_line(config, monkeypatch):
    """Any account on this computer can read every process's command line (/proc/<pid>/cmdline), and the first to use a
    launch code is signed in as the window. Chrome (browser = "auto": "chrome") starts on a page of the engine's private
    run folder (0700) instead, readable by this user only (0600), which sends it on to the code's address. A page whose
    code has expired goes at the next launch, every one at stop; a window that does not start takes its page with it,
    and the default browser is tried instead."""
    import stat
    from sopa_companion.cli import parser
    from sopa_companion.server.runtime_file import runtime_path
    record = {"pid": 4242, "port": 8777, "token": "t" * 43, "base_url": "http://127.0.0.1:8777"}
    monkeypatch.setattr(launcher, "ensure_engine", lambda config, **kw: record)
    codes, spawned = iter(["a" * 43, "b" * 43, "c" * 43, "d" * 43]), []
    services = launcher.Services(request=lambda rec, method, path: {} if path == "/api/v1/health" else {"code": next(codes)},
                                 output=lambda message: None,
                                 spawn=lambda command, **kw: spawned.append(command),
                                 which=lambda name: "/usr/bin/google-chrome" if name == "google-chrome" else None,
                                 environ={"DISPLAY": ":9"}, terminal=lambda: True)
    launcher.launch(config, parser().parse_args(["launch"]), services=services)
    command = spawned[0]
    assert not any("a" * 43 in a or "code=" in a for a in command)
    assert command[0] == "/usr/bin/google-chrome" and f"--user-data-dir={chrome_profile(config)}" in command
    url = next(a for a in command if a.startswith("--app=")).removeprefix("--app=")
    page = Path(url.removeprefix("file://"))
    folder = runtime_path(config).parent
    assert url.startswith("file://") and page.parent == folder and page.name == "open-window.html"
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700 and stat.S_IMODE(page.stat().st_mode) == 0o600
    text = page.read_text()
    assert 'location.replace("http://127.0.0.1:8777/?code=' + "a" * 43 + '&window=chrome")' in text
    assert ('<meta http-equiv="refresh" content="0;url=http://127.0.0.1:8777/?code=' + "a" * 43 + '&amp;window=chrome">'
            in text)
    # A launch at once leaves the first window's page, which it may not have read yet; one after its code's lifetime
    # removes it.
    launcher.launch(config, parser().parse_args(["launch"]), services=services)
    assert page.exists() and sorted(p.name for p in folder.glob("open-*.html")) == ["open-window-2.html",
                                                                                       "open-window.html"]
    old = page.stat().st_mtime - browser.PAGE_KEEP_S - 1
    os.utime(page, (old, old))
    def broken(command, **kw):
        raise OSError(2, "No such file or directory")
    said = []
    services.spawn, services.output = broken, said.append
    launcher.launch(config, parser().parse_args(["launch"]), services=services)     # no default browser here either
    assert said == [browser.WORDS["chrome_failed"].format(error="No such file or directory"), browser.WORDS["none"],
                    launcher.OPEN_BY_HAND, "http://127.0.0.1:8777/?code=" + "d" * 43]
    assert not page.exists() and len(list(folder.glob("open-*.html"))) == 1     # the second window's page is left
    monkeypatch.setattr(launcher, "read_runtime", lambda cfg: record)
    services.alive, services.healthy, services.stopper = (lambda pid: True), (lambda rec: False), (lambda *a, **kw: None)
    launcher.stop(config, parser().parse_args(["stop"]), services=services)
    assert not list(folder.glob("open-*.html"))


def test_the_window_opens_on_the_same_page_each_launch_so_chrome_finds_its_size_and_place(config, monkeypatch, tmp_path):
    """Chrome keeps an app window's size and place (maximised) under the address it opened: a page of a new name at
    every launch made it a new window each time. The page keeps its name (written anew, then put in place); a second
    launch while a window may still be reading the first one's page takes the next name, so each window reads its own
    code."""
    import stat
    from sopa_companion.cli import parser
    from sopa_companion.server.runtime_file import runtime_path
    record = {"pid": 4242, "port": 8777, "token": "t" * 43, "base_url": "http://127.0.0.1:8777"}
    monkeypatch.setattr(launcher, "ensure_engine", lambda config, **kw: record)
    codes, spawned = iter(c * 43 for c in "abcde"), []
    services = launcher.Services(request=lambda rec, method, path: {} if path == "/api/v1/health" else {"code": next(codes)},
                                 output=lambda message: None,
                                 spawn=lambda command, **kw: spawned.append(command),
                                 which=lambda name: "/usr/bin/google-chrome" if name == "google-chrome" else None,
                                 environ={"DISPLAY": ":9"})
    opened = lambda: next(a for a in spawned[-1] if a.startswith("--app=")).removeprefix("--app=")
    launcher.launch(config, parser().parse_args(["launch"]), services=services)
    first, folder = opened(), runtime_path(config).parent
    page = folder / "open-window.html"
    assert first == page.as_uri() and "a" * 43 in page.read_text()
    aged = page.stat().st_mtime - browser.PAGE_LOAD_S - 1                # its window has read it by now
    os.utime(page, (aged, aged))
    launcher.launch(config, parser().parse_args(["launch"]), services=services)
    assert opened() == first and "b" * 43 in page.read_text() and "a" * 43 not in page.read_text()
    assert sorted(p.name for p in folder.iterdir() if "open-" in p.name) == ["open-window.html"]
    launcher.launch(config, parser().parse_args(["launch"]), services=services)      # at once: the next name
    assert opened() == (folder / "open-window-2.html").as_uri() and "b" * 43 in page.read_text()
    assert "c" * 43 in (folder / "open-window-2.html").read_text()
    # The name is put in place, never followed: a link there is replaced, what it led to is left alone.
    elsewhere = tmp_path / "elsewhere.html"
    elsewhere.write_text("mine")
    page.unlink()
    page.symlink_to(elsewhere)
    os.utime(page, (aged, aged), follow_symlinks=False)
    launcher.launch(config, parser().parse_args(["launch"]), services=services)
    assert not page.is_symlink() and "d" * 43 in page.read_text() and elsewhere.read_text() == "mine"
    assert stat.S_IMODE(page.stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="plain word"):
        browser.open_page(folder, "http://127.0.0.1:1/", name="../x")


RECORD = {"pid": 4242, "port": 8777, "token": "t" * 43, "base_url": "http://127.0.0.1:8777"}


class Engine:
    """The engine as the launcher asks it, faked: launch codes (a new one each time, ?link=1 marked), and /health whose
    presence.windows is `windows` (a function of how often it was asked: the browser it opened signs in) and whose
    presence.connects is `connects` (any page's socket, another tab's too)."""

    def __init__(self, windows=lambda asked: 0, connects=lambda asked: 0):
        self.asked, self.windows, self.connects, self.health = [], windows, connects, 0

    def __call__(self, record, method, path):
        self.asked.append((method, path))
        if path == "/api/v1/health":
            self.health += 1
            return {"ok": True, "presence": {"pages": 0, "state": "none", "since_s": 0, "connects": self.connects(self.health),
                                             "windows": self.windows(self.health)}}
        return {"code": f"code-{len(self.asked)}" + ("-link" if path.endswith("?link=1") else "")}


def launching(config, monkeypatch, *, opened=None, engine=None, **services):
    """A launch with the engine, the browser opener and the clock faked, run from a terminal unless `terminal` says
    otherwise: -> (what it said, the opener's calls, the engine, the clock's sleeps)."""
    from sopa_companion.cli import parser
    monkeypatch.setattr(launcher, "ensure_engine", lambda config, **kw: RECORD)
    said, calls, sleeps, now = [], [], [], [0.0]
    engine = engine or Engine()

    def opener(url, **kwargs):
        calls.append((url, kwargs))
        return opened or browser.Opened("default", "Firefox", None)

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    fake = launcher.Services(request=engine, output=said.append, opener=opener, clock=lambda: now[0], sleep=sleep,
                             environ={"DISPLAY": ":9"}, **{"terminal": lambda: True, **services})
    return lambda *args: (launcher.launch(config, parser().parse_args(["launch", *args]), services=fake), said, calls,
                          engine, sleeps)


def test_launch_opens_the_default_browser_through_the_opener_and_waits_for_a_sign_in(config, monkeypatch):
    """The launcher hands the opener the code's address, its private folder, the settings, the Chrome window's profile
    (chrome_profile: desktop/browser.py never looks it up itself) and launch --browser's choice; a browser that took the
    page has 20 s to sign in (GET /health presence.windows grows: its launch code traded), else a link is printed too."""
    from sopa_companion.server.runtime_file import runtime_path
    config["desktop"]["browser"] = "default"
    run = launching(config, monkeypatch, engine=Engine(windows=lambda asked: 4 if asked >= 3 else 3))
    _, said, [(url, kwargs)], engine, sleeps = run("--browser", "default")
    assert url == "http://127.0.0.1:8777/?code=code-2" and said == []          # a window's code (45 s), never ?link=1
    assert kwargs["folder"] == runtime_path(config).parent and kwargs["profile"] == chrome_profile(config)
    assert kwargs["config"] is config and kwargs["choice"] == "default" and kwargs["environ"] == {"DISPLAY": ":9"}
    assert stat_mode(kwargs["folder"]) == 0o700 and kwargs["home"] == Path(os.environ["HOME"])
    assert engine.asked[0] == ("GET", "/api/v1/health") and sleeps == [.5, .5]     # signed in at the third look
    # Nobody signs in: after open_wait_s, NOT_YET and a ten-minute link.
    _, said, _, engine, sleeps = launching(config, monkeypatch)()
    assert said == [launcher.NOT_YET, "http://127.0.0.1:8777/?code=code-43-link"] and sum(sleeps) == 20
    config["desktop"]["open_wait_s"] = 0                                         # 0: no wait at all
    _, said, _, _, sleeps = launching(config, monkeypatch)()
    assert said == [] and sleeps == []
    # The app's own Chrome window is never waited for.
    config["desktop"]["open_wait_s"] = 20
    chrome_window = browser.Opened("chrome", "Google Chrome", None, object())
    _, said, _, _, sleeps = launching(config, monkeypatch, opened=chrome_window)("--browser", "chrome")
    assert said == [] and sleeps == []


def test_another_tab_connecting_is_no_sign_in_of_the_browser_launch_opened(config, monkeypatch):
    """While launch waits for the page of the browser it opened, another tab of the engine reconnecting (its socket
    counted in presence.connects) is not that page: only its launch code traded is (presence.windows)."""
    config["desktop"]["browser"] = "default"
    elsewhere = Engine(connects=lambda asked: asked)                  # a tab of the engine's connecting again and again
    _, said, _, _, sleeps = launching(config, monkeypatch, engine=elsewhere)()
    assert said == [launcher.NOT_YET, f"http://127.0.0.1:8777/?code=code-{len(elsewhere.asked)}-link"]
    assert sum(sleeps) == 20


def test_without_a_terminal_launch_prints_no_live_code(config, monkeypatch):
    """The icon's launch has no terminal (Terminal=false): what it prints goes to the desktop session's own log, where
    nobody reads it and other accounts may. No browser opened, or none signed in: the address and how to ask for a
    link, never a live code; a launch code is not even asked for."""
    why = browser.WORDS["none"]
    for opened, engine, words in ((browser.Opened("link", why=why), Engine(), [why]),
                                  (None, Engine(), [launcher.NOT_OPENED])):
        _, said, _, engine, _ = launching(config, monkeypatch, opened=opened, engine=engine, terminal=lambda: False)()
        assert said == [*words, "SoPA companion: http://127.0.0.1:8777\n"
                                 "For a private URL: python -m sopa_companion launch --print-url"]
        assert not any("code=" in line for line in said)
        assert ("POST", "/api/v1/session/launch-code?link=1") not in engine.asked
    # --print-url asks for a link by name: printed wherever its output goes (a script reads it).
    _, said, _, _, _ = launching(config, monkeypatch, terminal=lambda: False)("--print-url")
    assert said == ["http://127.0.0.1:8777/?code=code-1-link"]


def stat_mode(path):
    import stat
    return stat.S_IMODE(Path(path).stat().st_mode)


def test_without_a_browser_launch_says_why_and_prints_a_link(config, monkeypatch):
    """No browser could be opened (none found, a sandbox that cannot read the page, no Chrome for the window with no
    default browser either): launch says why, then prints a ten-minute link, as --print-url does."""
    why = browser.WORDS["confined"].format(name="Firefox")
    opened = browser.Opened("link", why=why, notice=browser.WORDS["no_chrome"])
    _, said, calls, engine, sleeps = launching(config, monkeypatch, opened=opened)()
    assert said == [browser.WORDS["no_chrome"], why, launcher.OPEN_BY_HAND, "http://127.0.0.1:8777/?code=code-3-link"]
    assert len(calls) == 1 and sleeps == []


def test_without_chrome_chrome_mode_opens_the_default_browser_and_says_so(config, monkeypatch):
    """browser = "auto" ("chrome") with no Chrome or Chromium: the default browser opens on the private page, saying so
    (desktop/browser.py open_app, the real one here, with the desktop's answers faked); a browser it starts never sees
    the code."""
    from sopa_companion.cli import parser
    monkeypatch.setattr(launcher, "ensure_engine", lambda config, **kw: RECORD)
    home = Path(os.environ["HOME"])
    apps = home / ".local/share/applications"
    apps.mkdir(parents=True)
    (apps / "firefox.desktop").write_text("[Desktop Entry]\nType=Application\nName=Firefox\nExec=firefox %u\n")
    said, started = [], []
    answers = {"mime": "Default application for \u201cx-scheme-handler/https\u201d: firefox.desktop\n"}
    services = launcher.Services(
        request=Engine(windows=lambda asked: asked - 1), output=said.append, environ={"DISPLAY": ":9", "HOME": str(home)},
        which=lambda name: {"gio": "/usr/bin/gio", "firefox": "/usr/bin/firefox"}.get(name),
        run=lambda argv, **kw: types.SimpleNamespace(returncode=0, stdout=answers.get(argv[1], "")),
        spawn=lambda argv, **kw: started.append((argv, kw)), clock=iter(range(100)).__next__, sleep=lambda s: None)
    launcher.launch(config, parser().parse_args(["launch"]), services=services)
    [(argv, kw)] = started
    assert said == [browser.WORDS["no_chrome"]] and argv[0] == "/usr/bin/firefox" and argv[1].startswith("file://")
    assert not any("code=" in arg or "code-2" in arg for arg in argv)
    assert "window=chrome" not in Path(argv[1][7:]).read_text()


def test_a_bad_setting_is_named_before_any_engine_starts_and_never_blocks_a_link(config, monkeypatch):
    """A mistyped [desktop] browser, Chrome window flags it cannot take, or a bad open_wait_s: launch says so in plain
    words before the state folder's move or any engine (nothing is started, no lock taken). --print-url and
    --no-browser open no browser, and work whatever the setting says."""
    from sopa_companion.cli import parser
    from sopa_companion.desktop import migrate
    from sopa_companion.server.runtime_file import runtime_path
    started = []
    monkeypatch.setattr(launcher, "ensure_engine", lambda config, **kw: started.append("engine") or RECORD)
    monkeypatch.setattr(migrate, "migrate", lambda config, **kw: started.append("migrate"))
    services = launcher.Services(request=Engine(), output=started.append, error=lambda message: None,
                                 environ={"DISPLAY": ":9"}, which=lambda name: None,
                                 opener=lambda *a, **k: pytest.fail("no browser opens"))
    for desktop, words in (({"browser": "firefox"}, browser.CHOICE_WORDS.format(value="firefox")),
                           ({"browser": "chrome", "chrome_flags": ["--user-data-dir=/x"]},
                            "Chrome's profile and app address are chosen by the companion."),
                           ({"browser": "default", "open_wait_s": -1}, "desktop.open_wait_s must be a number")):
        config["desktop"] = {**config["desktop"], "chrome_flags": [], "open_wait_s": 20, **desktop}
        with pytest.raises(DesktopError) as refused:
            launcher.launch(config, parser().parse_args(["launch"]), services=services)
        assert str(refused.value).startswith(words) and started == []
        assert not runtime_path(config).parent.exists()
        assert launcher.command(config, parser().parse_args(["launch"]), services=services) == 1
    # Flags only the Chrome window takes stop nothing else; a link and --no-browser open no browser at all.
    config["desktop"] = {**config["desktop"], "browser": "firefox", "chrome_flags": ["--user-data-dir=/x"]}
    launcher.launch(config, parser().parse_args(["launch", "--print-url"]), services=services)
    launcher.launch(config, parser().parse_args(["launch", "--no-browser"]), services=services)
    assert started[0] == "migrate" and started[1:3] == ["engine", "http://127.0.0.1:8777/?code=code-1-link"]
    assert started[-1] == "SoPA companion running (pid 4242) at http://127.0.0.1:8777"


def test_status_says_where_the_app_opens_and_who_is_there_and_never_fails_over_the_setting(config, monkeypatch):
    from sopa_companion.cli import parser
    monkeypatch.setattr(launcher, "read_runtime", lambda cfg: {**RECORD, "version": "x",
                                                               "started_at": "2026-10-25T00:00:00+00:00"})
    said = []
    services = launcher.Services(request=Engine(windows=lambda asked: 1, connects=lambda asked: 2), output=said.append,
                                 environ={"DISPLAY": ":9"},
                                 which=lambda name: None, discover=lambda *a: None)
    services.healthy = lambda record: True
    config["desktop"]["browser"] = "firefox"
    assert launcher.status(config, parser().parse_args(["status", "--json"]), services=services) == 0
    result = json.loads(said[-1])
    assert result["browser"] == {"opens": "link", "browser": None, "id": None, "confined": None,
                                 "why": browser.CHOICE_WORDS.format(value="firefox")}
    assert result["window"] == {"pages": 0, "state": "none", "since_s": 0, "connects": 2, "windows": 1}
    config["desktop"]["browser"] = "default"
    services.healthy = lambda record: False
    launcher.status(config, parser().parse_args(["status", "--json"]), services=services)
    result = json.loads(said[-1])
    assert result["browser"]["why"] == browser.WORDS["none"] and "window" not in result


def test_the_icons_stop_engine_never_discards_or_saves_without_asking():
    """The icon's Stop engine runs a plain `stop`: with a run not saved to its sample
    it refuses, and the launcher shows why; only `stop --force`, typed after that message, discards it."""
    from sopa_companion.desktop import install
    entry = install.entry_text("/usr/bin/python3", "/tmp/server.log")
    action = entry.split("[Desktop Action stop]\n")[1].splitlines()
    assert action[0] == "Name=Stop engine" and action[1].endswith(" -m sopa_companion stop") and "--force" not in entry


def test_launch_no_longer_accepts_a_run_argument():
    from sopa_companion.cli import parser
    with pytest.raises(SystemExit) as refused:
        parser().parse_args(["launch", "--run", "old-run"])
    assert refused.value.code == 2
