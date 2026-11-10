"""The notebook's way into the app: opening it where [desktop] browser says (desktop/browser.py: the default browser, or
an app window of its own in Chrome with the session's own profile), one-use links shown in the notebook, and the
session's Chrome window closed at the end. Of the desktop only the browser opener and Chrome's command (standard library
only); nothing of the server."""
from __future__ import annotations

import builtins
import getpass
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import sys
import time
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from ..desktop import browser, chrome

PAGE_KEEP_S = browser.PAGE_KEEP_S         # how long an opening page is kept (its code is used or expired by then)
# What a link shown in the notebook becomes once its code has signed a browser in: no code is left in the notebook.
USED = ("SoPA companion: a one-use link was shown here and used at {at}. If that was not you, close the app with its "
        "close().")
# Said with a link when a browser was opened and no page signed in after a while (api/handle.py OPEN_WAIT_S).
NOT_YET = ("SoPA companion has not opened in a browser yet. If no tab shows it, open this link in your browser (once, "
           "within 10 minutes):")
# What a link shown in the notebook becomes once it is no longer needed, before anyone used it: a newer link was shown
# below it, or the browser the app opened signed in after all. No code is left in the notebook either way.
TAKEN_OUT = {"newer": "SoPA companion: a newer link replaced the one shown here.",
             "opened": "SoPA companion: the app opened in a browser, so the link shown here was taken out."}


def open_mode(value: str) -> str:
    if value not in ("auto", "window", "link", "none"):
        raise ValueError('open must be "auto", "window", "link" or "none"')
    display = browser.desktop_session()
    if value == "auto":
        return "window" if display and not os.environ.get("SSH_CONNECTION") else "link"
    return "link" if value == "window" and not display else value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(port: int, token: str, path: str, *, method="GET", timeout=1.0) -> dict:
    origin = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer " + token, "Origin": origin, "Accept": "application/json"}
    req = Request(origin + path, data=b"{}" if method == "POST" else None, headers=headers, method=method)
    # Local credentials must not pass through an HTTP proxy or a redirect.
    with build_opener(ProxyHandler({}), _NoRedirect()).open(req, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("The app sent an unreadable reply")
    return value


def launch_url(port: int, token: str, *, link: bool = False, session: str | None = None, timeout=1.0) -> str:
    """A one-use opening link. `link`: one a person opens by hand (printed, or app.url), which the engine keeps for ten
    minutes (?link=1); else one for a window opened now, which lives seconds (server/security.py LaunchCodes).
    `session`: the session's id, which the link names (api=), so the tab keeps its storage for that session
    (frontend/src/storage-boot.mjs): in the user's own browser it would otherwise share the browser's."""
    path = "/api/v1/session/launch-code" + ("?link=1" if link else "")
    value = request(port, token, path, method="POST", timeout=timeout)
    code = value.get("code")
    if not isinstance(code, str) or not code:
        raise ValueError("The app did not send an opening link")
    return f"http://127.0.0.1:{port}/?" + urlencode([("code", code), *([("api", session)] if session else [])])


def check_flags(flags) -> None:
    """[desktop] chrome_flags as a window can use them, else ValueError in plain words."""
    if problem := chrome.flags_problem(flags):
        raise ValueError(problem)


def open_window(url: str, *, folder: Path, settings=None, log=None) -> browser.Opened:
    """The app on `url`, a one-use code's address, where [desktop] browser says (`settings`: the user's merged [desktop]
    table): the default browser, or the session's own Chrome window (its profile <folder>/chrome-profile, its
    chrome_flags). The code never goes on a command line (any account can read those): the browser opens a page in
    `folder`, the session's private one (or in a snap browser's own), that sends it on (desktop/browser.py open_page).
    -> what was opened; how "link" (with its why) when no browser could be, and no page is left then."""
    try:
        return browser.open_app(url, folder=folder, config={"desktop": dict(settings or {})},
                                profile=Path(folder) / "chrome-profile", log=log)
    except (OSError, ValueError) as exc:
        return browser.Opened("link", why=browser.WORDS["failed"].format(name="The browser", error=str(exc) or
                                                                         type(exc).__name__))


def _displayer():
    """IPython's display, in a notebook (a shell is running); None in a plain script."""
    get_shell = getattr(builtins, "get_ipython", None)
    if get_shell is None or get_shell() is None:
        return None
    try:
        from IPython.display import display
    except ImportError:
        return None
    return display


class Shown:
    """A one-use link shown in the notebook (show_link): `state` "shown" (its code is there), "taken out" or "used".
    used() replaces it, once a code has signed a browser in, with words that hold no code, so a notebook saved afterwards
    keeps none; in a plain script those words are printed. take_out() puts such words in its place before any use, once
    it is no longer needed (TAKEN_OUT; in a plain script nothing printed can be taken back); a use that comes later is
    still said."""

    def __init__(self, handle=None, schedule=None):
        self.handle, self.schedule, self.state = handle, schedule, "shown"

    def used(self, at: float) -> None:
        """A code signed a browser in at `at` (time.time()): this link's, or one that no longer shows."""
        if self.state == "used":
            return
        self.state = "used"
        self._say(USED.format(at=time.strftime("%H:%M", time.localtime(at))), printed=True)

    def take_out(self, text: str) -> None:
        """No longer needed: `text`, which holds no code, in its place while it still shows its code."""
        if self.state != "shown":
            return
        self.state = "taken out"
        self._say(text, printed=False)

    def _say(self, text: str, *, printed: bool) -> None:
        if self.handle is None:
            if printed:
                print(text, flush=True)
            return
        update = lambda: self.handle.update({"text/plain": text}, raw=True)
        try:
            if self.schedule is not None:
                self.schedule(update)      # on the kernel's loop, as every other change to the notebook
            else:
                update()
        except RuntimeError:
            pass                           # the kernel's loop is shutting down: nothing left to update


def show_link(url: str, port: int, *, why: str | None = None, schedule=None) -> Shown:
    """`why` no browser opened (when there is one) and the link `url`, with an SSH tunnel's command when the kernel runs
    over SSH. In a notebook it is one output that Shown.used replaces later (through `schedule`, the kernel's loop);
    in a plain script it is printed."""
    lines = [*([why] if why else []), f"SoPA companion: {url}"]
    if os.environ.get("SSH_CONNECTION"):
        target = shlex.quote(getpass.getuser() + "@" + socket.gethostname())
        lines.append(f"To open it on your computer: ssh -N -L {port}:127.0.0.1:{port} {target}")
    display = _displayer()
    if display is not None:
        try:
            return Shown(display({"text/plain": "\n".join(lines)}, raw=True, display_id=True), schedule)
        except Exception:              # a notebook front end that cannot show it: printed instead
            pass
    for line in lines:
        print(line, flush=True)
    return Shown(None, schedule)


def remove_pages(folders, *, older_than: float | None = None) -> None:
    """The opening pages open_window wrote (desktop/browser.py remove_pages): in `folders` (one or several)."""
    browser.remove_pages(folders, older_than=older_than)


def profile_processes(profile: Path, *, proc=Path("/proc")) -> list[int]:
    """Only this user's processes carrying the exact private profile argument."""
    flag, found = "--user-data-dir=" + str(profile), []
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = path.joinpath("cmdline").read_bytes().rstrip(b"\0").decode(errors="replace").split("\0")
            matches = flag in args or any(args[i:i + 2] == ["--user-data-dir", str(profile)] for i in range(len(args)))
            if len(args) == 1:
                # Chrome can rewrite argv as one title; only parse a Chrome title,
                # never a shell command that happens to mention a profile.
                pieces, wanted = args[0].split(" "), flag.split(" ")
                if Path(pieces[0]).name in {"chrome", "google-chrome", "chromium", "chromium-browser"}:
                    n = len(wanted)
                    matches |= any(pieces[i:i + n] == wanted and (i + n == len(pieces) or pieces[i + n].startswith("--"))
                                   for i in range(len(pieces)))
            if matches:
                found.append(int(path.name))
        except OSError:
            continue
    return found


def _signal_profile(profile: Path, sig: int) -> None:
    for pid in profile_processes(profile):
        fd = None
        try:
            if hasattr(os, "pidfd_open"):
                fd = os.pidfd_open(pid)
            # Recheck after acquiring the process reference, so a reused pid is never signalled.
            if pid not in profile_processes(profile):
                continue
            if fd is not None:
                signal.pidfd_send_signal(fd, sig)
            else:
                os.kill(pid, sig)
        except (OSError, ProcessLookupError):
            pass
        finally:
            if fd is not None:
                os.close(fd)


def close_profile(profile: Path) -> None:
    """End the Chrome window of a session's own profile. Linux only: the window is opened there alone
    (desktop/browser.py, chrome_linux), and its processes are found in /proc."""
    if not sys.platform.startswith("linux"):
        return
    _signal_profile(profile, signal.SIGTERM)
    deadline = time.monotonic() + 0.5
    while profile_processes(profile) and time.monotonic() < deadline:
        time.sleep(0.02)
    _signal_profile(profile, signal.SIGKILL)
