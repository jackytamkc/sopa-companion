"""Where the app opens (desktop/browser.py), against fakes only: desktop entries in a folder of the test's own, the
desktop's answers from a fake gio, xdg-settings and xdg-mime, and a fake start that records what it would run. No
browser, gio open or xdg-open ever runs here. What is checked: which browser is the default (as xdg-open finds it on
GNOME, then the other tools), its Exec line read as the Desktop Entry Specification says, a sandboxed browser given a
page it can read, the launch code never on a command line nor in an environment, Chrome's app window as before, the
macOS and Windows seams, and a bad setting named in plain words before anything is written or started."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types

import pytest

from sopa_companion.desktop import browser, chrome
from sopa_companion.engine.config import ROOT

CODE = "Ab_-9" * 8
URL = "http://127.0.0.1:41234/?code=" + CODE
CHROME_LINE = "/usr/bin/google-chrome-stable %U"
FIREFOX_LINE = "firefox %u"
SNAP_LINE = "env BAMF_DESKTOP_FILE_HINT=/var/lib/snapd/desktop/applications/firefox_firefox.desktop /snap/bin/firefox %u"


def entry(**fields) -> browser.Entry:
    values = {"id": "x.desktop", "path": Path("/usr/share/applications/x.desktop"), "name": "X", "exec": FIREFOX_LINE,
              "icon": "", "snap": None, "flatpak": None, "forwards": False, **fields}
    return browser.Entry(**values)


def finds(*names, **paths):
    """A `which` that finds `names` in /usr/bin and `paths` where they say."""
    known = {**{name: f"/usr/bin/{name}" for name in names}, **paths}
    return lambda name: known.get(name)


class Desktop:
    """A desktop of the test's own: its home, the applications folders of $XDG_DATA_HOME and $XDG_DATA_DIRS (the system's,
    the snaps' and the flatpaks', as on Ubuntu), and the desktop's answers (gio, xdg-settings, xdg-mime: `answers`, by
    the asked command) through a fake run that records each question."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path
        self.home = tmp_path / "home"
        self.user = self.home / ".local/share/applications"
        self.system = tmp_path / "usr/share/applications"
        self.snaps = tmp_path / "var/lib/snapd/desktop/applications"
        self.flatpaks = tmp_path / "var/lib/flatpak/exports/share/applications"
        for folder in (self.user, self.system, self.snaps, self.flatpaks):
            folder.mkdir(parents=True)
        self.environ = {"HOME": str(self.home), "XDG_DATA_HOME": str(self.home / ".local/share"),
                        "XDG_DATA_DIRS": ":".join(str(p.parent) for p in (self.system, self.snaps, self.flatpaks)),
                        "DISPLAY": ":0"}
        self.answers, self.asked = {}, []
        self.private = tmp_path / "run"
        self.private.mkdir(mode=0o700)

    def write(self, folder: Path, name: str, text: str) -> Path:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def app(self, folder: Path, name: str, exec_line: str, *extra: str, title: str | None = None) -> Path:
        return self.write(folder, name, "\n".join(["[Desktop Entry]", "Type=Application", f"Name={title or name}",
                                                   f"Exec={exec_line}", "Icon=web-browser", *extra]) + "\n")

    def run(self, argv, **kwargs):
        self.asked.append((argv, kwargs))
        answer = self.answers.get(" ".join(Path(argv[0]).name if i == 0 else a for i, a in enumerate(argv)))
        if isinstance(answer, BaseException):
            raise answer
        return types.SimpleNamespace(returncode=0 if answer is not None else 1, stdout=answer or "")


@pytest.fixture
def desk(tmp_path):
    return Desktop(tmp_path)


# ---- the Exec line ---------------------------------------------------------------------------------------------------
def test_exec_lines_of_real_browsers_become_their_commands_with_the_page_and_no_shell(tmp_path):
    which = finds("firefox", "env")
    uri, path = "file:///run/open-1.html", "/run/open-1.html"
    chrome_program = tmp_path / "google-chrome-stable"                   # Chrome's line names its program by its path
    chrome_line = CHROME_LINE.replace("/usr/bin/google-chrome-stable", str(chrome_program))
    assert browser.exec_argv(entry(exec=chrome_line), uri, path, which=which) is None        # not installed: none
    chrome_program.write_text("#!/bin/sh\n")
    chrome_program.chmod(0o755)
    assert browser.exec_argv(entry(exec=chrome_line), uri, path, which=which) == [str(chrome_program), uri]
    assert browser.exec_argv(entry(exec=FIREFOX_LINE), uri, path, which=which) == ["/usr/bin/firefox", uri]
    assert browser.exec_argv(entry(exec=SNAP_LINE), uri, path, which=which) == [
        "/usr/bin/env", "BAMF_DESKTOP_FILE_HINT=/var/lib/snapd/desktop/applications/firefox_firefox.desktop",
        "/snap/bin/firefox", uri]
    # The field codes: %f the path, %i the icon, %c the name, %k the entry's own file; deprecated ones go; %% is %; a
    # code inside a longer argument; and no code at all: the page goes last.
    e = entry(exec="firefox %f %i %c %k %d %m --x=100%% --app=%u", icon="fx", name="Firefox Web Browser",
              path=Path("/a/firefox.desktop"))
    assert browser.exec_argv(e, uri, path, which=which) == [
        "/usr/bin/firefox", path, "--icon", "fx", "Firefox Web Browser", "/a/firefox.desktop", "--x=100%", "--app=" + uri]
    assert browser.exec_argv(entry(exec="firefox %i --new-tab"), uri, path, which=which) == [
        "/usr/bin/firefox", "--new-tab", uri]                                # no icon: %i is nothing; the page at the end


def test_exec_quoting_and_escapes_are_undone_as_the_specification_says(tmp_path):
    program = tmp_path / "My Browser" / "browse"
    program.parent.mkdir()
    program.write_text("#!/bin/sh\n")
    program.chmod(0o755)
    uri = "file:///run/open-1.html"
    # A quoted program path with a space; inside quotes \\" \\$ and \\\\ (written \\\\\\\\ in a file: the string escape
    # is undone first) stand for the character; outside, \\s is a space of the value.
    line = f'"{program}" --title="say \\\\"hi\\\\" \\\\$5 a\\\\\\\\b" --x\\sy %u'
    assert browser.exec_argv(entry(exec=line), uri, "/p", which=lambda n: None) == [
        str(program), '--title=say "hi" $5 a\\b', "--x", "y", uri]
    assert browser.exec_argv(entry(exec='"/bin/true %u'), uri, "/p", which=lambda n: None) is None   # a quote left open
    assert browser.exec_argv(entry(exec=f"{tmp_path}/missing %u"), uri, "/p", which=lambda n: None) is None
    assert browser.exec_argv(entry(exec=""), uri, "/p", which=lambda n: "/usr/bin/x") is None


# ---- desktop entries -------------------------------------------------------------------------------------------------
def test_entries_are_found_in_the_specifications_order_and_a_hidden_one_hides_the_rest(desk):
    which = finds("firefox", "firefox-user", "konqueror")
    desk.app(desk.system, "firefox.desktop", FIREFOX_LINE, title="Firefox (system)")
    found = browser.find_entry("firefox.desktop", desk.environ, desk.home, which=which)
    assert found.name == "Firefox (system)" and found.path == desk.system / "firefox.desktop"
    desk.app(desk.user, "firefox.desktop", "firefox-user %u", title="Firefox (mine)")    # $XDG_DATA_HOME first
    assert browser.find_entry("firefox.desktop", desk.environ, desk.home, which=which).name == "Firefox (mine)"
    # An id with a "-" may be a file in a sub-folder (kde4-konqueror.desktop: kde4/konqueror.desktop).
    desk.app(desk.system / "kde4", "konqueror.desktop", "konqueror %u", title="Konqueror")
    assert browser.find_entry("kde4-konqueror.desktop", desk.environ, desk.home, which=which).name == "Konqueror"
    # Hidden in the user's folder: the app was taken away for this user, whatever the system has.
    desk.write(desk.user, "firefox.desktop", "[Desktop Entry]\nType=Application\nName=F\nExec=firefox %u\nHidden=true\n")
    assert browser.find_entry("firefox.desktop", desk.environ, desk.home, which=which) is None
    # A file that cannot be read (a link that leads nowhere) is passed over; a name that is a path is never looked up.
    (desk.user / "firefox.desktop").unlink()
    (desk.user / "firefox.desktop").symlink_to(desk.root / "nowhere.desktop")
    assert browser.find_entry("firefox.desktop", desk.environ, desk.home, which=which).name == "Firefox (system)"
    for odd in ("../firefox.desktop", "/usr/share/applications/firefox.desktop", "firefox", "", None):
        assert browser.find_entry(odd, desk.environ, desk.home, which=which) is None
    assert browser.find_entry("nothing.desktop", desk.environ, desk.home, which=which) is None


def test_entries_that_cannot_open_a_page_are_refused(desk):
    which = finds("firefox")
    base = ["[Desktop Entry]", "Name=B", "Exec=firefox %u"]
    for lines in (["Type=Link"], ["Type=Application", "Terminal=true"], ["Type=Application", "TryExec=not-installed"],
                  ["Type=Application", "Hidden=true"]):
        path = desk.write(desk.system, "b.desktop", "\n".join([*base, *lines]) + "\n")
        assert browser.parse_entry(path, which=which) is None, lines
    path = desk.write(desk.system, "b.desktop", "[Desktop Entry]\nType=Application\nName=B\n")
    assert browser.parse_entry(path, which=which) is None                              # no Exec
    # Only the [Desktop Entry] group counts, localised names are not the name, the first of a repeated key holds.
    path = desk.write(desk.system, "b.desktop", "# a comment\n[Desktop Action new]\nExec=other %u\n[Desktop Entry]\n"
                      "Type=Application\nName[fr]=Navigateur\nName=Browser\nName=Second\nExec=firefox %u\nTryExec=firefox\n"
                      "[Desktop Action private]\nExec=firefox --private %u\n")
    parsed = browser.parse_entry(path, which=which)
    assert (parsed.name, parsed.exec, parsed.snap, parsed.flatpak, parsed.forwards) == ("Browser", "firefox %u", None,
                                                                                        None, False)


def test_snap_and_flatpak_browsers_are_known_by_their_entries(desk):
    which = finds("firefox", "env")
    snap = desk.app(desk.snaps, "firefox_firefox.desktop", SNAP_LINE, "X-SnapInstanceName=firefox")
    assert browser.parse_entry(snap, which=which).snap == "firefox"
    plain = desk.app(desk.snaps, "chromium_chromium.desktop", "/snap/bin/chromium %U")    # its folder says it
    assert browser.parse_entry(plain, which=which).snap == "chromium"
    elsewhere = desk.app(desk.user, "brave.desktop", "/snap/bin/brave.browser %U")          # its program says it
    assert browser.parse_entry(elsewhere, which=which).snap == "brave"
    line = ("/usr/bin/flatpak run --branch=stable --arch=x86_64 --command=firefox --file-forwarding org.mozilla.firefox "
            "@@u %u @@")
    flat = desk.app(desk.flatpaks, "org.mozilla.firefox.desktop", line, "X-Flatpak=org.mozilla.firefox")
    parsed = browser.parse_entry(flat, which=which)
    assert (parsed.flatpak, parsed.forwards, parsed.snap) == ("org.mozilla.firefox", True, None)
    bare = desk.app(desk.user, "web.desktop", "flatpak run --branch=stable org.example.Web %u")
    parsed = browser.parse_entry(bare, which=which)
    assert (parsed.flatpak, parsed.forwards) == ("org.example.Web", False)


# ---- the default browser ---------------------------------------------------------------------------------------------
GIO = ("Default application for “x-scheme-handler/{scheme}”: {id}\nRegistered applications:\n\t{id}\n"
       "\tgoogle-chrome.desktop\nRecommended applications:\n\t{id}\n")


def test_the_default_browser_is_the_one_gio_names_as_xdg_open_finds_it_on_gnome(desk):
    """This workstation, measured: gio (what xdg-open asks on GNOME) answers firefox.desktop, while xdg-settings and
    xdg-mime fall through to Chrome. The browser a user sees as the default is gio's."""
    which = finds("gio", "xdg-settings", "xdg-mime", "firefox", "google-chrome-stable")
    desk.app(desk.system, "firefox.desktop", FIREFOX_LINE, title="Firefox Web Browser")
    desk.app(desk.system, "google-chrome.desktop", CHROME_LINE, title="Google Chrome")
    desk.answers = {"gio mime x-scheme-handler/https": GIO.format(scheme="https", id="firefox.desktop"),
                    "xdg-settings get default-web-browser": "google-chrome.desktop\n",
                    "xdg-mime query default x-scheme-handler/https": "google-chrome.desktop\n"}
    found = browser.default_entry(desk.environ, run=desk.run, home=desk.home, which=which)
    assert found.id == "firefox.desktop" and found.name == "Firefox Web Browser"
    argv, kwargs = desk.asked[0]
    assert argv == ["/usr/bin/gio", "mime", "x-scheme-handler/https"] and len(desk.asked) == 1
    assert kwargs["env"]["LC_ALL"] == "C" and kwargs["env"]["HOME"] == str(desk.home)      # untranslated, same desktop
    assert kwargs["timeout"] == 3 and kwargs["capture_output"] is True


def test_a_stale_id_is_passed_over_and_each_tool_is_asked_in_turn(desk):
    which = finds("gio", "xdg-settings", "xdg-mime", "firefox", "google-chrome-stable", "epiphany")
    desk.app(desk.system, "google-chrome.desktop", CHROME_LINE, title="Google Chrome")
    desk.app(desk.system, "org.gnome.Epiphany.desktop", "epiphany %U", title="Web")
    # gio names a snap that is not installed (left in mimeapps.list): xdg-settings' answer is next.
    desk.answers = {"gio mime x-scheme-handler/https": GIO.format(scheme="https", id="firefox_firefox.desktop"),
                    "xdg-settings get default-web-browser": "google-chrome.desktop\n"}
    assert browser.default_entry(desk.environ, run=desk.run, home=desk.home, which=which).id == "google-chrome.desktop"
    # gio fails and takes too long for xdg-settings: xdg-mime, for https then http.
    desk.answers = {"gio mime x-scheme-handler/https": subprocess.TimeoutExpired("gio", 3),
                    "gio mime x-scheme-handler/http": "No default applications for “x-scheme-handler/http”\n",
                    "xdg-settings get default-web-browser": subprocess.TimeoutExpired("xdg-settings", 3),
                    "xdg-mime query default x-scheme-handler/http": "org.gnome.Epiphany.desktop\n"}
    desk.asked.clear()
    assert browser.default_entry(desk.environ, run=desk.run, home=desk.home, which=which).name == "Web"
    assert [" ".join(argv[1:]) for argv, _ in desk.asked] == [
        "mime x-scheme-handler/https", "mime x-scheme-handler/http", "get default-web-browser",
        "query default x-scheme-handler/https", "query default x-scheme-handler/http"]
    # No desktop tools at all: $BROWSER (Python's convention), a desktop id or a command; xdg-open never.
    nothing = finds("firefox", "epiphany")
    assert browser.default_entry(desk.environ, run=desk.run, home=desk.home, which=nothing) is None
    env = {**desk.environ, "BROWSER": "org.gnome.Epiphany.desktop:firefox"}
    assert browser.default_entry(env, run=desk.run, home=desk.home, which=nothing).name == "Web"
    env["BROWSER"] = "firefox --new-window %s"
    command = browser.default_entry(env, run=desk.run, home=desk.home, which=nothing)
    assert browser.exec_argv(command, "file:///p.html", "/p.html", which=nothing) == [
        "/usr/bin/firefox", "--new-window", "file:///p.html"]
    env["BROWSER"] = "xdg-open"
    assert browser.default_entry(env, run=desk.run, home=desk.home, which=nothing) is None


# ---- where a sandboxed browser reads the page ------------------------------------------------------------------------
def test_a_snap_reads_the_page_in_its_own_folder_and_a_flatpak_only_through_its_forwarding(desk):
    snap = entry(snap="firefox", name="Firefox")
    assert browser.page_folder(snap, desk.private, home=desk.home) is None              # never started: no folder
    common = desk.home / "snap/firefox/common"
    common.mkdir(parents=True)
    folder = browser.page_folder(snap, desk.private, home=desk.home)
    assert folder == common / "sopa-companion" and stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert browser.confined_folders(desk.home) == [folder]
    other = desk.home / "snap/chromium"
    other.mkdir(parents=True)
    (other / "common").symlink_to(desk.root, target_is_directory=True)                # never through a link
    assert browser.page_folder(entry(snap="chromium"), desk.private, home=desk.home) is None
    assert browser.page_folder(entry(flatpak="org.mozilla.firefox", forwards=True), desk.private, home=desk.home) == \
        desk.private
    assert browser.page_folder(entry(flatpak="org.example.Web"), desk.private, home=desk.home) is None
    assert browser.page_folder(entry(), desk.private, home=desk.home) == desk.private
    assert browser.confined_folders(desk.root / "no-home") == []


# ---- opening the app -------------------------------------------------------------------------------------------------
class Starts:
    """A fake Popen: records each command and how it was started; `fail` makes it raise as a missing program would."""

    def __init__(self, fail: BaseException | None = None):
        self.started, self.fail = [], fail

    def __call__(self, argv, **kwargs):
        self.started.append((argv, kwargs))
        if self.fail is not None:
            raise self.fail
        return types.SimpleNamespace(pid=4242, argv=argv)


def opener(desk, *, config=None, starts=None, which=None, **kwargs):
    starts = starts or Starts()
    opened = browser.open_app(URL, folder=desk.private, config=config or {"desktop": {"browser": "default"}},
                              profile=desk.root / "chrome-profile", environ=desk.environ, run=desk.run, spawn=starts,
                              which=which or finds("gio", "firefox"), platform="linux", home=desk.home, **kwargs)
    return opened, starts


def test_the_default_browser_opens_a_private_page_and_never_sees_the_code(desk, monkeypatch):
    monkeypatch.setattr(browser, "wsl", lambda environ=None: False)
    desk.app(desk.system, "firefox.desktop", FIREFOX_LINE, title="Firefox Web Browser")
    desk.answers = {"gio mime x-scheme-handler/https": GIO.format(scheme="https", id="firefox.desktop")}
    opened, starts = opener(desk)
    assert (opened.how, opened.browser, opened.why, opened.notice, opened.process) == ("default", "Firefox Web Browser",
                                                                                       None, None, None)
    [(argv, kwargs)] = starts.started
    assert argv == ["/usr/bin/firefox", opened.page.as_uri()]
    assert not any(CODE in arg or "code=" in arg for arg in argv)                       # never on a command line
    assert not any(CODE in value for value in kwargs["env"].values())                  # nor in an environment
    assert kwargs["env"] == desk.environ and kwargs["start_new_session"] and kwargs["close_fds"]
    assert kwargs["stdin"] == subprocess.DEVNULL
    page = opened.page
    assert page.parent == desk.private and re.fullmatch(r"open-[0-9a-f]{32}\.html", page.name)
    assert stat.S_IMODE(page.stat().st_mode) == 0o600 and stat.S_IMODE(desk.private.stat().st_mode) == 0o700
    assert f"location.replace({json.dumps(URL)})" in page.read_text() and "window=chrome" not in page.read_text()
    # A browser that cannot be started takes its page with it, and a link is printed instead, saying why.
    opened, _ = opener(desk, starts=Starts(OSError(2, "No such file or directory")))
    assert opened.how == "link" and opened.why == "Firefox Web Browser could not be started (No such file or directory)."
    assert list(desk.private.glob("open-*.html")) == [page]
    opened, starts = opener(desk, which=finds("gio"))                                   # its program is gone
    assert opened.how == "link" and "could not be started (its program was not found)" in opened.why
    assert starts.started == [] and list(desk.private.glob("open-*.html")) == [page]


def test_a_browser_named_by_the_setting_a_sandbox_and_nothing_found_each_say_why(desk, monkeypatch):
    monkeypatch.setattr(browser, "wsl", lambda environ=None: False)
    named = {"desktop": {"browser": "firefox.desktop"}}
    opened, starts = opener(desk, config=named)
    assert (opened.how, opened.why) == ("link", "No browser named firefox.desktop was found: [desktop] browser names it.")
    assert desk.asked == [] and starts.started == []                                   # the desktop is not asked
    desk.app(desk.user, "firefox.desktop", "firefox --profile x %u", title="Firefox (mine)")
    opened, starts = opener(desk, config=named)
    assert opened.how == "default" and starts.started[0][0][:3] == ["/usr/bin/firefox", "--profile", "x"]
    opened, starts = opener(desk)                                                       # nothing gio or $BROWSER names
    assert (opened.how, opened.why) == ("link", "No default web browser was found.")
    desk.app(desk.snaps, "firefox_firefox.desktop", SNAP_LINE, "X-SnapInstanceName=firefox", title="Firefox")
    desk.answers = {"gio mime x-scheme-handler/https": GIO.format(scheme="https", id="firefox_firefox.desktop")}
    opened, starts = opener(desk, which=finds("gio", "env"))
    assert opened.how == "link" and opened.why == ("Your default browser (Firefox) runs in a sandbox that cannot read "
                                                   "the app's private opening page.")
    (desk.home / "snap/firefox/common").mkdir(parents=True)                             # the snap has started once
    opened, starts = opener(desk, which=finds("gio", "env"))
    assert opened.how == "default" and opened.page.parent == desk.home / "snap/firefox/common/sopa-companion"
    assert starts.started[0][0][-1] == opened.page.as_uri()


def test_chrome_mode_is_the_app_window_as_before_and_without_chrome_the_default_browser(desk, monkeypatch):
    monkeypatch.setattr(browser, "wsl", lambda environ=None: False)
    flags = ["--password-store=basic"]
    config = {"desktop": {"browser": "auto", "chrome_flags": flags}}
    opened, starts = opener(desk, config=config, which=finds("google-chrome", "gio", "firefox"))
    [(argv, kwargs)] = starts.started
    page = desk.private / "open-window.html"
    assert (opened.how, opened.browser, opened.page, opened.process.pid) == ("chrome", "Google Chrome", page, 4242)
    assert argv == chrome.command("/usr/bin/google-chrome", page.as_uri(), desk.root / "chrome-profile", flags,
                                  window_class="SoPACompanion")
    assert desk.asked == [] and "env" not in kwargs                                     # no default browser asked
    # The page knows it is the app's own window (what it says when it cannot draw names chrome_flags).
    assert f"location.replace({json.dumps(URL + '&window=chrome')})" in page.read_text()
    # Flags it cannot take: refused in their words before anything is written.
    with pytest.raises(ValueError, match="chosen by the companion"):
        opener(desk, config={"desktop": {"browser": "chrome", "chrome_flags": ["--app=x"]}})
    page.unlink()
    # No Chrome: the default browser, saying so, and its page has no window flag.
    desk.app(desk.system, "firefox.desktop", FIREFOX_LINE, title="Firefox")
    desk.answers = {"gio mime x-scheme-handler/https": GIO.format(scheme="https", id="firefox.desktop")}
    opened, starts = opener(desk, config=config)
    assert opened.how == "default" and opened.notice == browser.WORDS["no_chrome"]
    assert "window=chrome" not in opened.page.read_text() and not page.exists()
    # Chrome that fails to start: the same, with why.
    opened, starts = opener(desk, config=config, which=finds("google-chrome", "gio", "firefox"),
                            starts=FailingFirst(OSError(13, "Permission denied")))
    assert opened.how == "default" and opened.notice == ("Google Chrome could not be started for the app window "
                                                         "(Permission denied): SoPA companion opens in your default browser.")
    assert not page.exists() and starts.started[0][0][0] == "/usr/bin/google-chrome"
    assert starts.started[1][0][0] == "/usr/bin/firefox"


def test_chrome_mode_never_takes_a_snaps_chromium_which_cannot_read_the_windows_page(desk, monkeypatch, tmp_path):
    """A snap cannot read the app window's profile and opening page, which live in hidden folders: Chrome mode passes
    over a Chrome or Chromium that is a snap's (under /snap/, a link that leads there or to the snap program, or a
    script that starts one, as Ubuntu's chromium-browser is) for the next one found, else the default browser, saying
    so. Google Chrome's own script, and a program not there to look at, are no snap's."""
    monkeypatch.setattr(browser, "wsl", lambda environ=None: False)
    tools = tmp_path / "bin"
    tools.mkdir()
    linked, launcher_link, wrapper = tools / "chromium", tools / "chromium-snap", tools / "chromium-browser"
    linked.symlink_to("/snap/bin/chromium")                                   # dangling here: no snap installed
    launcher_link.symlink_to("/usr/bin/snap")                                 # what /snap/bin/chromium itself is
    wrapper.write_text('#!/bin/sh\nif ! [ -x /snap/bin/chromium ]; then\n  exit 1\nfi\nexec /snap/bin/chromium "$@"\n')
    google = tools / "google-chrome"
    google.write_text('#!/bin/bash\n# Copyright The Chromium Authors\nexec -a "$0" "$HERE/chrome" "$@"\n')
    for executable in ("/snap/bin/chromium", linked, launcher_link, wrapper):
        assert chrome.snapped(str(executable)), executable
    for executable in (google, tools / "not-there", "/usr/lib/chromium/chromium"):
        assert not chrome.snapped(str(executable)), executable
    assert chrome.find(finds(chromium=str(linked), **{"chromium-browser": str(wrapper)})) is None
    assert chrome.find(finds(**{"google-chrome": "/snap/bin/chromium", "chromium": str(google)})) == str(google)
    # Chrome mode with a snap's Chromium alone: the default browser, saying so; status says the same.
    desk.app(desk.system, "firefox.desktop", FIREFOX_LINE, title="Firefox")
    desk.answers = {"gio mime x-scheme-handler/https": GIO.format(scheme="https", id="firefox.desktop")}
    snap_only = finds("gio", "firefox", chromium="/snap/bin/chromium")
    opened, starts = opener(desk, config={"desktop": {"browser": "chrome"}}, which=snap_only)
    assert opened.how == "default" and opened.notice == browser.WORDS["no_chrome"]
    assert [argv[0] for argv, _ in starts.started] == ["/usr/bin/firefox"]
    said = browser.describe({"desktop": {"browser": "chrome"}}, desk.environ, run=desk.run, home=desk.home,
                            which=snap_only, platform="linux")
    assert (said["opens"], said["why"]) == ("default", browser.WORDS["no_chrome"])
    assert "not a snap" in browser.WORDS["no_chrome"]


class FailingFirst(Starts):
    """The first start fails (Chrome's), the next ones succeed."""

    def __call__(self, argv, **kwargs):
        self.started.append((argv, kwargs))
        if len(self.started) == 1:
            raise self.fail
        return types.SimpleNamespace(pid=4243, argv=argv)


def test_a_bad_setting_is_named_before_anything_is_written_or_started(desk):
    for value in ("firefox", "", 3, "../firefox.desktop"):
        with pytest.raises(ValueError) as refused:
            opener(desk, config={"desktop": {"browser": value}})
        assert str(refused.value) == browser.CHOICE_WORDS.format(value=value)
        assert repr(value) in str(refused.value) and "such as firefox.desktop" in str(refused.value)
    assert list(desk.private.iterdir()) == [] and desk.asked == []
    assert browser.choice({}) == ("default", None) and browser.choice({"desktop": {}}) == ("default", None)
    assert browser.choice({"desktop": {"browser": "auto"}}) == browser.choice({"desktop": {"browser": "chrome"}}) == (
        "chrome", None)
    assert browser.choice({"desktop": {"browser": "brave-browser.desktop"}}) == ("default", "brave-browser.desktop")
    assert browser.choice({"desktop": {"browser": "brave-browser.desktop"}}, "default") == ("default", None)
    assert browser.choice({"desktop": {"browser": "firefox"}}, "chrome") == ("chrome", None)   # launch --browser


def test_macos_hands_the_address_to_osascript_on_its_input_and_windows_and_wsl_print_a_link(desk):
    asked = []

    def run(argv, **kwargs):
        asked.append((argv, kwargs))
        return types.SimpleNamespace(returncode=0)
    opened = browser.open_app(URL, folder=desk.private, config={}, profile=desk.root / "p", environ={}, run=run,
                              spawn=Starts(), which=finds(), platform="darwin", home=desk.home)
    assert (opened.how, opened.browser, opened.page) == ("default", "your default browser", None)
    [(argv, kwargs)] = asked
    assert argv == ["/usr/bin/osascript"] and kwargs["input"] == f'open location "{URL}"\n' and kwargs["check"] is True
    assert list(desk.private.iterdir()) == []
    def failing(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)
    opened = browser.open_app(URL, folder=desk.private, config={}, profile=desk.root / "p", environ={}, run=failing,
                              spawn=Starts(), which=finds(), platform="darwin", home=desk.home)
    assert opened.how == "link" and "could not be started (exit 1)" in opened.why
    with pytest.raises(ValueError):
        browser.open_app(URL + '"', folder=desk.private, config={}, profile=desk.root / "p", environ={}, run=run,
                         spawn=Starts(), which=finds(), platform="darwin", home=desk.home)
    for platform, environ, why in (("win32", {}, browser.WORDS["system"]),
                                   ("linux", {"WSL_DISTRO_NAME": "Ubuntu"}, browser.WORDS["wsl"])):
        starts = Starts()
        opened = browser.open_app(URL, folder=desk.private, config={}, profile=desk.root / "p", environ=environ,
                                  run=desk.run, spawn=starts, which=finds("gio"), platform=platform, home=desk.home)
        assert (opened.how, opened.why) == ("link", why) and starts.started == [] and desk.asked == []
    with pytest.raises(NotImplementedError):
        browser._windows_open(desk.private / "open-x.html")


def test_status_says_where_the_app_would_open_and_never_raises(desk, monkeypatch):
    monkeypatch.setattr(browser, "wsl", lambda environ=None: False)
    said = browser.describe({"desktop": {"browser": "firefox"}}, desk.environ, run=desk.run, home=desk.home,
                            which=finds("gio"), platform="linux")
    assert said == {"opens": "link", "browser": None, "id": None, "confined": None,
                    "why": browser.CHOICE_WORDS.format(value="firefox")}
    chrome_too = finds("google-chrome", "gio", "firefox")
    assert browser.describe({"desktop": {"browser": "auto"}}, desk.environ, run=desk.run, home=desk.home,
                            which=chrome_too, platform="linux")["opens"] == "chrome"
    desk.app(desk.snaps, "firefox_firefox.desktop", SNAP_LINE, "X-SnapInstanceName=firefox", title="Firefox")
    desk.answers = {"gio mime x-scheme-handler/https": GIO.format(scheme="https", id="firefox_firefox.desktop")}
    said = browser.describe({"desktop": {"browser": "auto"}}, desk.environ, run=desk.run, home=desk.home,
                            which=finds("gio", "env"), platform="linux")
    assert said == {"opens": "link", "browser": "Firefox", "id": "firefox_firefox.desktop", "confined": "snap",
                    "why": browser.WORDS["confined"].format(name="Firefox")}
    (desk.home / "snap/firefox/common").mkdir(parents=True)
    said = browser.describe({}, desk.environ, run=desk.run, home=desk.home, which=finds("gio", "env"), platform="linux")
    assert said["opens"] == "default" and said["why"] is None
    assert not (desk.home / "snap/firefox/common/sopa-companion").exists()           # nothing written

    def broken(*args, **kwargs):
        raise RuntimeError("the desktop cannot be asked")
    monkeypatch.setattr(browser, "default_entry", broken)
    said = browser.describe({}, desk.environ, run=desk.run, home=desk.home, which=finds("gio"), platform="linux")
    assert said["opens"] == "link" and "could not be looked up (RuntimeError)" in said["why"]


def test_someone_at_the_screen_is_a_display_on_linux_and_no_ssh_elsewhere():
    assert browser.desktop_session({"DISPLAY": ":0"}, "linux") and browser.desktop_session({"WAYLAND_DISPLAY": "w"}, "linux")
    assert not browser.desktop_session({}, "linux") and not browser.desktop_session({"SSH_CONNECTION": "x"}, "linux")
    for platform in ("darwin", "win32"):
        assert browser.desktop_session({}, platform)
        for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
            assert not browser.desktop_session({name: "1"}, platform)
    assert browser.wsl({"WSL_DISTRO_NAME": "Ubuntu"}) and not browser.wsl({}, interop=Path("/no/such/file"))


def test_old_pages_go_from_every_folder_given_and_new_ones_stay(tmp_path):
    folders = [tmp_path / "run", tmp_path / "snap"]
    for folder in folders:
        folder.mkdir(mode=0o700)
    old = [browser.open_page(folder, URL) for folder in folders]
    for page in old:
        os.utime(page, (page.stat().st_mtime - browser.PAGE_KEEP_S - 1,) * 2)
    new = browser.open_page(folders[0], URL, name="window")
    (folders[0] / "notes.txt").write_text("not a page")
    browser.remove_pages(folders, older_than=browser.PAGE_KEEP_S)
    assert not any(page.exists() for page in old) and new.exists()
    browser.remove_pages(folders[0])                                                    # one folder, every page
    assert sorted(p.name for p in folders[0].iterdir()) == ["notes.txt"]
    browser.remove_pages([tmp_path / "missing"])


def test_the_opening_page_is_this_users_alone_and_only_sends_the_window_on(tmp_path):
    """desktop/browser.py open_page, which the launcher and a notebook's session share: the page is made new (0600) in
    a folder of this user's that nobody else can open (0700), never in another; its script (or, without scripts, a
    refresh) sends the browser on to the code's address, which cannot end the script or the attribute early; Chrome's
    command opens it."""
    folder = tmp_path / "private"
    folder.mkdir()
    folder.chmod(0o700)
    page = browser.open_page(folder, URL)
    assert page.parent == folder and stat.S_IMODE(page.stat().st_mode) == 0o600
    text = page.read_text()
    assert f'<script>location.replace("{URL}")</script>' in text
    assert f'<noscript><meta http-equiv="refresh" content="0;url={URL}"></noscript>' in text
    assert browser.open_page(folder, URL) != page                                       # a new page each time
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    (tmp_path / "link").symlink_to(folder, target_is_directory=True)
    for place in (shared, tmp_path / "link"):
        with pytest.raises(ValueError, match="not a private folder"):
            browser.open_page(place, URL)
    assert not list(shared.iterdir())
    odd = browser.open_page(folder, 'http://127.0.0.1:1/?a="</script><b>').read_text()
    assert odd.count("</script>") == 1 and '"</' not in odd and "<b>" not in odd
    with pytest.raises(ValueError, match="plain word"):
        browser.open_page(folder, URL, name="../x")
    command = chrome.command("/usr/bin/google-chrome", page.as_uri(), tmp_path / "profile", ["--password-store=basic"],
                             window_class="SoPACompanion")
    assert command == ["/usr/bin/google-chrome", "--password-store=basic", "--app=" + page.as_uri(),
                       "--user-data-dir=" + str(tmp_path / "profile"), "--class=SoPACompanion", "--no-first-run",
                       "--no-default-browser-check"]
    assert chrome.flags_problem(["--app=http://elsewhere"]) and chrome.flags_problem(["-x"])
    assert chrome.flags_problem("--x") and chrome.flags_problem(["--password-store=basic"]) is None


# ---- who reads the setting, and what the kernel imports --------------------------------------------------------------
def test_the_engine_never_reads_where_the_app_opens():
    """Only the side that opens the browser reads [desktop] browser (the launcher, a notebook's kernel): the engine's
    own code has no reader of it, so nothing it does can follow another file than the one the window was opened by."""
    readers = re.compile(r"""browser\.choice\(|get\(\s*["']browser["']|\[\s*["']browser["']\s*\]""")
    found = [f"{path.relative_to(ROOT)}:{n}" for folder in ("server", "engine") for path in sorted((ROOT / folder).glob("*.py"))
             for n, line in enumerate(path.read_text().splitlines(), 1) if readers.search(line)]
    assert found == []


def test_the_browser_module_stays_inside_the_notebooks_import_fence():
    script = '''
import sys
sys.path.insert(0, sys.argv[1])
import sopa_companion.desktop.browser
unexpected = {name.split(".")[0] for name in sys.modules} - sys.stdlib_module_names - {"__main__", "sopa_companion"}
assert not unexpected, sorted(unexpected)
assert "psutil" not in sys.modules and "sopa_companion.desktop.window" not in sys.modules
assert not any(name.startswith(("sopa_companion.engine", "sopa_companion.server")) for name in sys.modules)
'''
    done = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(ROOT.parent)], capture_output=True,
                          text=True, timeout=30)
    assert done.returncode == 0, done.stderr[-1000:]
