"""The app window's own Chrome ([desktop] browser = "chrome"): its profile, and whether it is still running. Imported by
the engine (server/app.py: the Cellpose GUI opens on that window's desktop) and the launcher; it imports nothing from
either, so neither can form an import cycle through it. The notebook's kernel never imports it (psutil, the engine's
settings): desktop/browser.py is handed the profile instead."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psutil

from ..engine.config import ROOT
from ..names import LEGACY_PROFILE, LEGACY_STATE_DIR, STATE_DIR

# Every engine instance's Chrome window has its own profile, <state_dir>/chrome-profile, so two engines never share a
# browser or a window check. The default instance's (state folder ~/.local/state/sopa-companion) holds the page's
# remembered settings and the window's size; the previous version kept that one apart, in
# ~/.local/share/proseg_live/chrome-profile, which the first start of this version moves in (desktop/migrate.py). A
# state folder configured to the previous version's (~/.local/state/proseg_live) keeps that profile where it is.
DEFAULT_STATE_DIR = STATE_DIR


def chrome_profile(config: dict) -> Path:
    """The app window's own Chrome profile (browser = "chrome"): the launcher starts Chrome with --user-data-dir=<this>,
    and the engine recognises the app's Chrome by it. "auto" (or empty, the default): <state_dir>/chrome-profile, but
    the previous version's own place for its state folder. Otherwise `~` is expanded and a relative path is inside the
    app folder."""
    value = config.get("desktop", {}).get("profile_dir") or "auto"
    if value == "auto":
        state = Path(config.get("paths", {}).get("state_dir") or DEFAULT_STATE_DIR).expanduser().resolve()
        if state == Path(LEGACY_STATE_DIR).expanduser().resolve():
            return Path(LEGACY_PROFILE).expanduser()
        return state / "chrome-profile"
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


class AppWindow:
    """Whether the app's own Chrome is still running: a process of this user whose command line carries exactly
    --user-data-dir=<profile> (the browser and its renderer/GPU/utility children all do). Chrome rewrites its process
    title as one string of space-separated arguments, which psutil splits on spaces, so a profile path with spaces
    arrives as consecutive pieces: those match when the next argument is another switch (or there is none), never as a
    substring of one argument (a shell command that mentions the flag) nor as the start of a longer path. The process
    found is remembered, so asking again while it lives costs one lookup rather than a scan of every process."""

    def __init__(self, profile: str | Path, *, processes=psutil.process_iter, uid: int | None = None):
        self.flag = "--user-data-dir=" + str(profile)
        self.pieces = self.flag.split(" ")
        # The app's own Chrome window is Linux's alone (desktop/browser.py chrome_linux): elsewhere there is none to find,
        # and Windows has no uids.
        self.linux = sys.platform.startswith("linux")
        self.processes, self.uid = processes, (os.getuid() if self.linux else None) if uid is None else uid
        self.found = None

    def carries(self, cmdline) -> bool:
        args, pieces = list(cmdline or ()), self.pieces
        if self.flag in args or len(pieces) == 1:      # an untouched argv: the flag is one element
            return self.flag in args
        n = len(pieces)
        return any(args[i:i + n] == pieces and (i + n == len(args) or args[i + n].startswith("-"))
                   for i in range(len(args) - n + 1))

    def __call__(self) -> bool:
        if not self.linux:
            return False
        if self.found is not None:
            try:
                if self.found.is_running() and self.found.status() != psutil.STATUS_ZOMBIE:
                    return True
            except psutil.Error:
                pass
            self.found = None
        for process in self.processes(["cmdline", "uids", "status"]):
            info = process.info
            uids = info.get("uids")
            if uids is not None and uids.real == self.uid and info.get("status") != psutil.STATUS_ZOMBIE \
                    and self.carries(info.get("cmdline")):
                self.found = process
                return True
        return False
