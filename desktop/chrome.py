"""The app window in Chrome ([desktop] browser = "chrome"), for the desktop launcher and the notebook's kernel alike
(desktop/browser.py opens it): which Chrome or Chromium, and its command line. Standard library only: the kernel imports
nothing else of the desktop or the server.

The window has a profile of its own (--user-data-dir: desktop/window.py chrome_profile, a notebook session's
<state>/chrome-profile), so it keeps the window's size and place and the page's settings away from the user's everyday
profile, takes the [desktop] chrome_flags a remote desktop may need, and is the Chrome the engine looks for (its
process carries that profile's flag). It opens a private page that sends it on to the engine's address with a launch
code (desktop/browser.py open_page): a code on Chrome's command line could be read, and used first, by any account on
this computer. A snap's Chromium cannot read that page or the profile, which live in hidden folders of this user's: it
is passed over (snapped)."""
from __future__ import annotations

import os
from pathlib import Path

BROWSERS = ("google-chrome", "chromium", "chromium-browser")
# What the companion sets itself: a configured flag never replaces the window's profile or the page it opens.
OWN_FLAGS = frozenset({"--user-data-dir", "--app", "--profile-directory"})
SNAP_PROGRAM = Path("/usr/bin/snap")       # what every program in /snap/bin is a link to
SCRIPT_BYTES = 4096                        # of a script, what is read to see whether it starts a snap


def find(which) -> str | None:
    """The first Chrome or Chromium `which` (shutil.which, or a test's) finds that is no snap's (snapped)."""
    for name in BROWSERS:
        if (executable := which(name)) and not snapped(executable):
            return executable
    return None


def snapped(executable: str) -> bool:
    """Whether `executable` is a snap's: under /snap/, or a link that leads there or to the snap program, or a script
    that starts one (as Ubuntu's chromium-browser does). A program that cannot be looked at is taken as no snap's."""
    path = Path(executable)
    for _ in range(40):                    # each link it leads through, as far as a loop would let it
        if str(path).startswith("/snap/") or path == SNAP_PROGRAM:
            return True
        try:
            if not path.is_symlink():
                break
            target = Path(os.readlink(path))
        except OSError:
            return False
        path = target if target.is_absolute() else path.parent / target
    try:
        with open(path, "rb") as stream:
            head = stream.read(SCRIPT_BYTES)
    except OSError:
        return False
    return head.startswith(b"#!") and b"/snap/" in head


def flags_problem(flags) -> str | None:
    """Why [desktop] chrome_flags cannot be used, in plain words; None when they can."""
    if not isinstance(flags, (list, tuple)) or not all(isinstance(f, str) and f.startswith("--") for f in flags):
        return "desktop.chrome_flags must be a list of flags beginning with --"
    if any(f.split("=", 1)[0] in OWN_FLAGS for f in flags):
        return "Chrome's profile and app address are chosen by the companion"
    return None


def command(executable: str, url: str, profile, flags=(), *, window_class: str) -> list[str]:
    """Chrome's app window on `url` with its own `profile`; `window_class` groups the window on the desktop. The app's
    own callers pass an opening page's file:// address (desktop/browser.py open_page), never an address with a code."""
    return [executable, *flags, "--app=" + url, "--user-data-dir=" + str(profile), "--class=" + window_class,
            "--no-first-run", "--no-default-browser-check"]
