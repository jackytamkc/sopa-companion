"""Where the app opens, for the desktop launcher and the notebook's kernel (api/window.py) alike. [desktop] browser says
"default" (the desktop's own web browser, in a new tab), "chrome" (an app window of its own in Google Chrome or a
Chromium: desktop/chrome.py; "auto", the previous versions' value, is "chrome") or, on Linux, the name of a browser's
.desktop file. Only the side that opens the browser reads that setting, and checks it before it starts anything; the
engine never does, so whatever it does about windows works the same in every browser.

Standard library only, with Chrome's command (desktop/chrome.py) and private files (private.py): the notebook's kernel
imports it, and nothing else of the desktop (desktop/window.py brings psutil and the engine's settings) or the server.

A launch code is never on a command line, nor in an environment: every account on this computer can read every
process's command line (/proc/<pid>/cmdline), and whoever uses a one-use code first is signed in. The browser starts on
a page written into a folder only this user can read (open_page), whose one job is to send it on to the engine's
address with the code: only the page's path is on the browser's command line. A browser in a sandbox gets the page where it can
read it (a snap: a folder in its own ~/snap/<name>/common; a flatpak: its file forwarding), or a link is printed
instead. On macOS the address goes through osascript's standard input (an Apple event), never through a command line.

On Linux the default browser is the desktop's handler of https addresses, asked as xdg-open asks on GNOME (gio), then
xdg-settings, then xdg-mime, then $BROWSER: the handler of web pages (text/html) is never used, as it may be an editor.
Its desktop entry is read as the Desktop Entry Specification says (find_entry, parse_entry, exec_argv) and started with
no shell."""
from __future__ import annotations

from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlencode, urlsplit, urlunsplit

from . import chrome
from ..private import is_mine, private_folder, write_private

PAGES = ("open-*.html", ".open-*.html")      # the opening pages, and one being written (open_page with a name)
# How long an opening page is kept: the longest a window's launch code lives (server/security.py LAUNCH_CODE_MAX_S). By
# then its code signs nobody in, used or not.
PAGE_KEEP_S = 60
# A window has read its opening page within this long of being started: an older page of the same name may be replaced.
PAGE_LOAD_S = 10
# A snap browser's own folder for the opening pages: ~/snap/<snap>/common/<this>, which the snap can read.
CONFINED = "sopa-companion"
ASK_S = 3.0                                  # the longest gio, xdg-settings or xdg-mime is given to answer

WORDS = {
    "none": "No default web browser was found.",
    "confined": "Your default browser ({name}) runs in a sandbox that cannot read the app's private opening page.",
    "failed": "{name} could not be started ({error}).",
    "wsl": "Under WSL, open the link in your Windows browser.",
    "system": "Opening a browser is not available on this system yet.",
    "no_chrome": "Google Chrome, or a Chromium that is not a snap, was not found for the app window: SoPA companion "
                 "opens in your default browser.",
    "chrome_failed": "Google Chrome could not be started for the app window ({error}): SoPA companion opens in your "
                     "default browser.",
    "chrome_linux": "The app's own Chrome window is for Linux for now: SoPA companion opens in your default browser.",
    "not_found": "No browser named {id} was found: [desktop] browser names it.",
}
CHOICE_WORDS = ("desktop.browser must be default, chrome, auto or the name of a browser's .desktop file (such as "
                "firefox.desktop), not {value!r}.")
DESKTOP_ID = re.compile(r"[A-Za-z0-9][\w.+-]*\.desktop", re.ASCII)      # a file's name, never a path
DEFAULT_LINE = re.compile(r"^Default application for .*: (\S+\.desktop)\s*$")
NOT_FOUND = "its program was not found"
SSH = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")


@dataclass(frozen=True)
class Entry:
    """A Linux desktop entry of a browser, as far as starting it needs. `exec` is its Exec value as the file has it
    (string escapes and quoting still in: exec_argv undoes them)."""
    id: str
    path: Path
    name: str
    exec: str
    icon: str
    snap: str | None          # X-SnapInstanceName, else the snap a /var/lib/snapd/... entry or /snap/bin program names
    flatpak: str | None       # X-Flatpak, else the app id of a flatpak export or of a `flatpak run` Exec
    forwards: bool            # its Exec forwards files into its sandbox (flatpak's @@u ... @@)


@dataclass(frozen=True)
class Opened:
    """What open_app did: `how` "default" (a browser was started on the page), "chrome" (the app window) or "link" (no
    browser: a link is to be printed, `why` says why); `browser`, its name for words; `page`, the opening page written
    (removed after PAGE_KEEP_S); `process`, the Chrome window's process (a notebook's close ends it); `notice`, what was
    said on the way (no Chrome for the app window: the default browser instead)."""
    how: str
    browser: str | None = None
    page: Path | None = None
    process: object | None = None
    why: str | None = None
    notice: str | None = None


def choice(config, override=None) -> tuple[str, str | None]:
    """("default" | "chrome", a desktop id that [desktop] browser names, or None). "auto" is "chrome"; a missing value
    is "default"; anything else raises ValueError(CHOICE_WORDS). `override`: launch --browser's, for one launch."""
    return _choice(config, override)


def _choice(config, override) -> tuple[str, str | None]:
    value = override if override is not None else (config.get("desktop") or {}).get("browser", "default")
    if value == "default":
        return "default", None
    if value in ("chrome", "auto"):
        return "chrome", None
    if isinstance(value, str) and DESKTOP_ID.fullmatch(value):
        return "default", value
    raise ValueError(CHOICE_WORDS.format(value=value))


def desktop_session(environ=None, platform=None) -> bool:
    """Someone sits at this computer's screen: on Linux a DISPLAY or WAYLAND_DISPLAY; on macOS and Windows, unless the
    session came over SSH (SSH_CONNECTION, SSH_CLIENT or SSH_TTY). The launcher, the notebook's open="auto", the save
    dialog and the Cellpose GUI ask this one function."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if platform in ("darwin", "win32"):
        return not any(environ.get(name) for name in SSH)
    return bool(environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"))


def wsl(environ=None, *, interop=Path("/proc/sys/fs/binfmt_misc/WSLInterop")) -> bool:
    """Whether this is Linux under Windows (WSL), where the browser to open is Windows'."""
    environ = os.environ if environ is None else environ
    return bool(environ.get("WSL_DISTRO_NAME")) or interop.exists()


# ---- Linux: desktop entries ------------------------------------------------------------------------------------------
def _data_dirs(environ, home: Path) -> list[Path]:
    """$XDG_DATA_HOME then each of $XDG_DATA_DIRS, + /applications; a relative path is not used (the base directory
    specification's rule)."""
    first = environ.get("XDG_DATA_HOME") or str(home / ".local/share")
    rest = (environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share").split(":")
    return [Path(folder) / "applications" for folder in (first, *rest) if folder and os.path.isabs(folder)]


def _id_paths(desktop_id: str) -> list[str]:
    """The file names an id may have in an applications folder: itself, then each '-' turned into '/' from the left, one
    at a time (kde4-konqueror.desktop may be kde4/konqueror.desktop)."""
    names, name = [desktop_id], desktop_id
    while "-" in name:
        name = name.replace("-", "/", 1)
        names.append(name)
    return names


def _group(path: Path) -> dict | None:
    """The [Desktop Entry] group's unlocalised keys, their values as the file has them (the first of a repeated key);
    None when the file cannot be read."""
    try:
        text = path.read_text(errors="replace")
    except (OSError, ValueError):                 # unreadable, a link that leads nowhere or round in a loop
        return None
    keys, inside = {}, False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            if inside:
                break
            inside = line == "[Desktop Entry]"
            continue
        key, equals, value = line.partition("=")
        key = key.strip()
        if inside and equals and "[" not in key and key not in keys:
            keys[key] = value.strip()
    return keys


def _unescape(value: str) -> str:
    """A string value's escapes (\\s \\n \\t \\r \\\\) undone; any other backslash stays, for the Exec key's quoting."""
    return re.sub(r"\\([sntr\\])", lambda m: {"s": " ", "n": "\n", "t": "\t", "r": "\r", "\\": "\\"}[m[1]], value)


def _split(line: str) -> list[str] | None:
    """An Exec value (its string escapes undone) as arguments: spaces and tabs separate them outside quotes; a quoted
    argument runs to the next unescaped double quote, inside which \\" \\` \\$ and \\\\ stand for the character. None
    for a quote left open. No shell is ever used."""
    args, current, started, quoted, i = [], [], False, False, 0
    while i < len(line):
        c = line[i]
        if quoted:
            if c == "\\" and i + 1 < len(line) and line[i + 1] in '"`$\\':
                current.append(line[i + 1])
                i += 2
                continue
            if c == '"':
                quoted = False
            else:
                current.append(c)
        elif c in " \t":
            if started:
                args.append("".join(current))
                current, started = [], False
        elif c == '"':
            quoted = started = True
        else:
            current.append(c)
            started = True
        i += 1
    if quoted:
        return None
    if started:
        args.append("".join(current))
    return args


def _program(args: list[str]) -> str | None:
    """The program an Exec runs, past an `env` and its VAR=value settings."""
    rest = list(args)
    if rest and Path(rest[0]).name == "env":
        rest = rest[1:]
        while rest and ("=" in rest[0] or rest[0].startswith("-")):
            rest = rest[1:]
    return rest[0] if rest else None


def _flatpak_app(args: list[str]) -> str | None:
    """The app a `flatpak run [options] APP ...` Exec runs."""
    if len(args) < 2 or Path(args[0]).name != "flatpak" or args[1] != "run":
        return None
    for arg in args[2:]:
        if arg in ("@@", "@@u"):
            break
        if not arg.startswith("-"):
            return arg.split("/", 1)[0]
    return None


def _found(program: str, which) -> str | None:
    """`program`'s full path: a bare name as `which` finds it, a path when it is an executable file."""
    if "/" not in program:
        return which(program)
    return program if os.path.isfile(program) and os.access(program, os.X_OK) else None


def parse_entry(path, desktop_id: str | None = None, *, which=shutil.which) -> Entry | None:
    """The browser `path`'s [Desktop Entry] group describes (`desktop_id`: its id, else its file name); None when it is
    not one that can be started: not an Application, a program for a terminal (a browser there cannot run the opening
    page's script), hidden, its TryExec not installed, or without an Exec."""
    path = Path(path)
    keys = _group(path)
    if keys is None or keys.get("Type") != "Application" or "true" in (keys.get("Terminal"), keys.get("Hidden")):
        return None
    line = keys.get("Exec", "")
    args = _split(_unescape(line)) if line else None
    if not args:
        return None
    if (tried := _unescape(keys.get("TryExec", ""))) and not _found(tried, which):
        return None
    desktop_id = desktop_id or path.name
    program, where = _program(args) or "", str(path)
    snap = _unescape(keys.get("X-SnapInstanceName", "")) or (
        desktop_id.split("_", 1)[0] if "/var/lib/snapd/desktop/applications/" in where else
        Path(program).name.split(".", 1)[0] if program.startswith("/snap/bin/") else None)
    flatpak = _unescape(keys.get("X-Flatpak", "")) or (
        desktop_id.removesuffix(".desktop") if "/flatpak/exports/share/applications/" in where else _flatpak_app(args))
    return Entry(id=desktop_id, path=path, name=_unescape(keys.get("Name", "")) or desktop_id, exec=line,
                 icon=_unescape(keys.get("Icon", "")), snap=snap or None, flatpak=flatpak or None,
                 forwards="@@u" in args or "@@" in args)


def find_entry(desktop_id: str, environ, home, *, which=shutil.which) -> Entry | None:
    """The desktop entry of `desktop_id`, looked for as the specification says: in $XDG_DATA_HOME/applications, then
    in each of $XDG_DATA_DIRS' applications folders, as the id itself and then in its sub-folder forms. The first file
    found decides: one that cannot be read (or a link round in a loop) is passed over, and one marked Hidden means the
    app was taken away for this user (it hides the same id further on). None when no file is found, or the one found
    cannot be started."""
    if not isinstance(desktop_id, str) or not DESKTOP_ID.fullmatch(desktop_id):
        return None
    for folder in _data_dirs(environ, Path(home)):
        for name in _id_paths(desktop_id):
            path = folder / name
            try:
                if not path.is_file():
                    continue
            except OSError:
                continue
            keys = _group(path)
            if keys is None:
                continue
            if keys.get("Hidden") == "true":
                return None
            return parse_entry(path, desktop_id, which=which)
    return None


def _ask(run, argv, environ) -> str:
    """What gio, xdg-settings or xdg-mime answers (untranslated: LC_ALL=C), "" when it fails or takes over ASK_S."""
    try:
        done = run(argv, env={**environ, "LC_ALL": "C"}, capture_output=True, text=True, timeout=ASK_S,
                   stdin=subprocess.DEVNULL)
    except (OSError, ValueError, subprocess.SubprocessError):
        return ""
    return (done.stdout or "") if done.returncode == 0 else ""


def default_entry(environ, *, run=subprocess.run, home, which=shutil.which) -> Entry | None:
    """The desktop's default web browser: the handler of https (then http) addresses as gio gives it (what xdg-open
    asks on GNOME, Cinnamon and MATE; gio reads the mimeapps.list files KDE writes too), then as xdg-settings and
    xdg-mime give it, then $BROWSER. The first id that names a browser that can be started wins: an id left behind by a
    browser since removed is passed over. None when there is none."""
    def ids():
        if gio := which("gio"):
            for scheme in ("https", "http"):
                for line in _ask(run, [gio, "mime", f"x-scheme-handler/{scheme}"], environ).splitlines():
                    if found := DEFAULT_LINE.match(line):
                        yield found[1]
                        break
        if tool := which("xdg-settings"):
            yield _ask(run, [tool, "get", "default-web-browser"], environ).strip()
        if tool := which("xdg-mime"):
            for scheme in ("https", "http"):
                yield _ask(run, [tool, "query", "default", f"x-scheme-handler/{scheme}"], environ).strip()

    for desktop_id in ids():
        if desktop_id and (entry := find_entry(desktop_id, environ, home, which=which)) is not None:
            return entry
    return _from_browser_variable(environ, home, which)


def _from_browser_variable(environ, home, which) -> Entry | None:
    """$BROWSER, Python's convention: its first entry, a desktop id or a command (its %s, where it has one, is where the
    address goes). xdg-open and gio are passed over: they would hand the page to the handler of web pages, which may be
    an editor."""
    first = (environ.get("BROWSER") or "").split(":", 1)[0].strip()
    if not first:
        return None
    if first.endswith(".desktop"):
        return find_entry(first, environ, home, which=which)
    args = _split(first) or []
    if not args or Path(args[0]).name in ("xdg-open", "gio"):
        return None
    line = first.replace("%s", "%u") if "%s" in first else first + " %u"
    program = args[0]
    return Entry(id=Path(program).name, path=Path(program), name=Path(program).name, exec=line, icon="",
                 snap=Path(program).name.split(".", 1)[0] if program.startswith("/snap/bin/") else None,
                 flatpak=_flatpak_app(args), forwards="@@u" in args or "@@" in args)


def exec_argv(entry: Entry, uri: str, path: str, *, which=shutil.which) -> list[str] | None:
    """The entry's command for one page, as the Desktop Entry Specification's Exec key says: the string escapes undone,
    split into arguments (its quoting undone), then the field codes: %u %U are `uri` and %f %F `path` (whole arguments,
    or within a longer one such as --app=%u); %i is --icon and the icon, %c the name, %k the entry's file; %% is %, and
    the deprecated or unknown codes go. Without %u %U %f %F, `uri` is added at the end. None when its program is not
    found."""
    args = _split(_unescape(entry.exec))
    if not args:
        return None
    codes = {"u": uri, "U": uri, "f": path, "F": path, "c": entry.name, "k": str(entry.path), "i": entry.icon, "%": "%"}
    argv, placed = [], False
    for arg in args:
        if arg in ("%u", "%U", "%f", "%F"):
            argv.append(uri if arg in ("%u", "%U") else path)
            placed = True
        elif arg == "%i":
            argv += ["--icon", entry.icon] if entry.icon else []
        elif re.fullmatch(r"%[dDnNvm]", arg):
            continue
        else:
            placed |= bool(re.search(r"%[uUfF]", arg.replace("%%", "")))
            argv.append(re.sub(r"%(.)", lambda m: codes.get(m[1], ""), arg))
    if not placed:
        argv.append(uri)
    program = _found(argv[0], which) if argv else None
    if program is None:
        return None
    return [program, *argv[1:]]


def _snap_common(entry: Entry, home) -> Path | None:
    """A snap's own ~/snap/<snap>/common, when it is there as a real folder of this user's (the snap made it at its
    first start); None otherwise."""
    common = Path(home) / "snap" / entry.snap / "common"
    try:
        if stat.S_ISDIR(common.lstat().st_mode) and is_mine(common):
            return common
    except OSError:
        pass
    return None


def page_folder(entry: Entry, private, *, home) -> Path | None:
    """Where `entry`'s browser can read a private opening page: `private` (the caller's own 0700 folder) for a browser
    in no sandbox, and for a flatpak that forwards files into its sandbox; for a snap, a 0700 folder of this user's in
    its ~/snap/<snap>/common, which the snap can read (none there yet: made); None otherwise (a snap that has never
    started, a flatpak that cannot be given the file)."""
    if entry.snap:
        if (common := _snap_common(entry, home)) is None:
            return None
        try:
            return private_folder(common / CONFINED, create=True)
        except ValueError:
            return None
    if entry.flatpak and not entry.forwards:
        return None
    return Path(private)


def confined_folders(home=None) -> list[Path]:
    """Every snap browser's folder for opening pages (~/snap/*/common/sopa-companion) that there is: real folders of
    this user's, never a link."""
    root = (Path(home) if home is not None else Path.home()) / "snap"
    try:
        snaps = sorted(root.iterdir())
    except OSError:
        return []
    found = []
    for snap in snaps:
        folder = snap / "common" / CONFINED
        try:
            if stat.S_ISDIR(folder.lstat().st_mode) and is_mine(folder):
                found.append(folder)
        except OSError:
            continue
    return found


# ---- the opening page -----------------------------------------------------------------------------------------------
def open_page(folder, url: str, *, name: str | None = None) -> Path:
    """A page in `folder` (this user's own, 0700: private.py), readable by this user only (0600, made new, never
    through a link or over another file), that sends the browser on to `url` at once: a script does, and without
    scripts a refresh does (only then, so the one-use code is asked for once). It is open-<random>.html, or with `name`
    open-<name>.html at every launch: Chrome keeps an app window's size and place under the address it opened, so the
    window opened on the same page again finds them. A page of that name that a window may still be reading (younger
    than PAGE_LOAD_S: a second launch at once) is left to it, and open-<name>-2.html (-3, …) is used. -> its path."""
    folder = private_folder(folder)
    target = json.dumps(url).replace("<", "\\u003c")
    text = ('<!doctype html><meta charset="utf-8"><meta name="referrer" content="no-referrer"><title>SoPA companion'
            f'</title><noscript><meta http-equiv="refresh" content="0;url={html.escape(url, quote=True)}"></noscript>'
            f'<script>location.replace({target})</script>\n')
    if name is None:
        return write_private(folder / f"open-{secrets.token_hex(16)}.html", text)
    if not re.fullmatch(r"[a-z]+", name):
        raise ValueError("An opening page's name is a plain word")
    now = time.time()
    for n in range(1, 10):
        path = folder / (f"open-{name}.html" if n == 1 else f"open-{name}-{n}.html")
        try:
            if now - path.lstat().st_mtime < PAGE_LOAD_S:
                continue                           # a window started a moment ago may not have read it yet
        except FileNotFoundError:
            pass
        written = write_private(folder / f".open-{secrets.token_hex(16)}.html", text)
        try:
            os.replace(written, path)              # the name, never what it may lead to: the folder is this user's alone
        except BaseException:
            written.unlink(missing_ok=True)
            raise
        return path
    return write_private(folder / f"open-{secrets.token_hex(16)}.html", text)


def remove_pages(folders, *, older_than: float | None = None) -> None:
    """Remove the opening pages in `folders` (one folder or several): all of them, or those written more than
    `older_than` seconds ago (a browser started meanwhile may still be reading its own)."""
    if isinstance(folders, (str, os.PathLike)):
        folders = [folders]
    now = time.time()
    for folder in folders:
        try:
            pages = [page for pattern in PAGES for page in Path(folder).glob(pattern)]
        except OSError:
            continue
        for page in pages:
            try:
                if older_than is None or now - page.lstat().st_mtime > older_than:
                    page.unlink()
            except OSError:
                pass


# ---- opening the app ------------------------------------------------------------------------------------------------
def _with(url: str, **fields) -> str:
    """`url` with `fields` added to its query."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(query="&".join(part for part in (parts.query, urlencode(fields)) if part)))


def _chrome_name(executable: str) -> str:
    return "Chromium" if "chromium" in Path(executable).name else "Google Chrome"


def _error(exc: BaseException) -> str:
    return getattr(exc, "strerror", None) or type(exc).__name__


def _mac_open(url: str, run, notice) -> Opened:
    """macOS: the address goes to LaunchServices through osascript's standard input (as the standard library's
    webbrowser does it), so it travels on a pipe and in an Apple event, never on a command line."""
    if '"' in url or "\\" in url:
        raise ValueError("The app's address cannot be handed to the browser")
    try:
        run(["/usr/bin/osascript"], input=f'open location "{url}"\n', text=True, timeout=10,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as exc:
        return Opened("link", why=WORDS["failed"].format(name="Your default browser", error=f"exit {exc.returncode}"),
                      notice=notice)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Opened("link", why=WORDS["failed"].format(name="Your default browser", error=_error(exc)), notice=notice)
    return Opened("default", "your default browser", notice=notice)


def _windows_open(page: Path) -> None:
    """Not written yet: on Windows the app prints a link to open by hand. What it is to do: read the ProgId of the
    https handler the user chose (HKCU\\Software\\Microsoft\\Windows\\Shell\\Associations\\UrlAssociations\\https\\
    UserChoice), then that ProgId's command (HKCR\\<ProgId>\\shell\\open\\command); put the page's file:///C:/...
    address in place of %1 (and %L, %*); start it with CreateProcess, never a shell. The page lives in
    %LOCALAPPDATA%\\sopa-companion\\open\\, under an access list that lets this user alone read it (private.py)."""
    raise NotImplementedError("Opening a browser is not available on Windows yet")


def open_app(url: str, *, folder, config, profile, choice=None, environ=None, which=shutil.which,
             spawn=subprocess.Popen, run=subprocess.run, platform=None, home=None, log=None) -> Opened:
    """Open the app at `url` (it carries a one-use code) where [desktop] browser says (`choice` overrides it, as
    launch --browser does). `folder`: the caller's private folder for the opening page; `profile`: the Chrome window's
    own profile folder, used in Chrome mode only (the launcher's chrome_profile(config), a notebook session's
    <state>/chrome-profile). ValueError (CHOICE_WORDS, or the chrome_flags' problem in Chrome mode) for a bad setting,
    before anything is written or started. `url` never goes into a command line or an environment."""
    mode, named = _choice(config, choice)
    flags = (config.get("desktop") or {}).get("chrome_flags", [])
    if mode == "chrome" and (problem := chrome.flags_problem(flags)):
        raise ValueError(problem + ".")
    environ = dict(os.environ if environ is None else environ)
    platform = sys.platform if platform is None else platform
    home = Path(home) if home is not None else Path(environ.get("HOME") or Path.home())
    output = log or subprocess.DEVNULL
    notice = None
    if mode == "chrome" and not platform.startswith("linux"):
        notice = WORDS["chrome_linux"]               # its window's profile and its closing are Linux's (api/window.py)
    elif mode == "chrome":
        if (executable := chrome.find(which)) is None:
            notice = WORDS["no_chrome"]
        else:
            # The page tells itself it is the app's own window (the words it shows when it cannot draw name the
            # window's chrome_flags), and keeps one name so that Chrome finds the window's size and place again.
            page = open_page(folder, _with(url, window="chrome"), name="window")
            try:
                process = spawn(chrome.command(executable, page.as_uri(), profile, flags, window_class="SoPACompanion"),
                                stdin=subprocess.DEVNULL, stdout=output, stderr=output, start_new_session=True,
                                close_fds=True)
            except OSError as exc:
                page.unlink(missing_ok=True)
                notice = WORDS["chrome_failed"].format(error=_error(exc))
            else:
                return Opened("chrome", _chrome_name(executable), page, process)
    if platform == "darwin":
        return _mac_open(url, run, notice)
    if platform == "win32":
        return Opened("link", why=WORDS["system"], notice=notice)
    if wsl(environ):
        return Opened("link", why=WORDS["wsl"], notice=notice)
    entry = find_entry(named, environ, home, which=which) if named else default_entry(environ, run=run, home=home,
                                                                                       which=which)
    if entry is None:
        return Opened("link", why=WORDS["not_found"].format(id=named) if named else WORDS["none"], notice=notice)
    target = page_folder(entry, folder, home=home)
    if target is None:
        return Opened("link", why=WORDS["confined"].format(name=entry.name), notice=notice)
    page = open_page(target, url)
    argv = exec_argv(entry, page.as_uri(), str(page), which=which)
    if argv is None:
        page.unlink(missing_ok=True)
        return Opened("link", why=WORDS["failed"].format(name=entry.name, error=NOT_FOUND), notice=notice)
    try:
        # Not waited for: a browser already running takes the page, and the process started here exits.
        spawn(argv, env=environ, stdin=subprocess.DEVNULL, stdout=output, stderr=output, start_new_session=True,
              close_fds=True)
    except OSError as exc:
        page.unlink(missing_ok=True)
        return Opened("link", why=WORDS["failed"].format(name=entry.name, error=_error(exc)), notice=notice)
    return Opened("default", entry.name, page, None, None, notice)


def describe(config, environ=None, *, run=subprocess.run, home=None, which=shutil.which, platform=None) -> dict:
    """For `status`: where launch would open the app, {"opens": "default" | "chrome" | "link", "browser": its name,
    "id": its desktop id, "confined": "snap" | "flatpak" | None, "why": why it opens no browser, or what it says on the
    way}. Nothing secret, nothing written or started. Never raises: a bad setting is "opens": "link" with its words."""
    result = {"opens": "link", "browser": None, "id": None, "confined": None, "why": None}
    try:
        mode, named = _choice(config, None)
    except ValueError as exc:
        return {**result, "why": str(exc)}
    environ = dict(os.environ if environ is None else environ)
    platform = sys.platform if platform is None else platform
    home = Path(home) if home is not None else Path(environ.get("HOME") or Path.home())
    if mode == "chrome" and not platform.startswith("linux"):
        result["why"] = WORDS["chrome_linux"]
    elif mode == "chrome":
        if problem := chrome.flags_problem((config.get("desktop") or {}).get("chrome_flags", [])):
            return {**result, "why": problem + "."}
        if executable := chrome.find(which):
            return {**result, "opens": "chrome", "browser": _chrome_name(executable)}
        result["why"] = WORDS["no_chrome"]
    try:
        if platform == "darwin":
            return {**result, "opens": "default", "browser": "your default browser"}
        if platform == "win32" or wsl(environ):
            return {**result, "why": WORDS["system" if platform == "win32" else "wsl"]}
        entry = find_entry(named, environ, home, which=which) if named else default_entry(environ, run=run, home=home,
                                                                                           which=which)
    except Exception as exc:                       # status says what it could not find out, and goes on
        return {**result, "why": f"The default browser could not be looked up ({type(exc).__name__})."}
    if entry is None:
        return {**result, "why": WORDS["not_found"].format(id=named) if named else WORDS["none"]}
    confined = "snap" if entry.snap else "flatpak" if entry.flatpak else None
    found = {"browser": entry.name, "id": entry.id, "confined": confined}
    readable = _snap_common(entry, home) is not None if entry.snap else not (entry.flatpak and not entry.forwards)
    if not readable:
        return {**result, **found, "why": WORDS["confined"].format(name=entry.name)}
    return {**result, **found, "opens": "default"}
