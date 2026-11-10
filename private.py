"""Folders and files of this user's alone: POSIX 0700 folders and 0600 files owned by this user, never reached through
a link. One place, so that another system's way is written here alone. On Windows a folder's privacy is its access
list: the app's folders there are in the user's own profile (platformdirs' folders under %LOCALAPPDATA%, the user's
temporary folder), whose list lets in this user, the system and the administrators only, and every folder and file made
in them inherits it; so this module checks there only that a folder is a real one, never a link or a junction (a
reparse point). Standard library only: the notebook's kernel imports it (through desktop/browser.py)."""
from __future__ import annotations

import os
from pathlib import Path
import stat

# A new file is made, never one that is there already, and never through a link; it is not handed to a child process
# (Windows' os.open makes it so: O_EXCL refuses a link there too, whose name is taken).
_NEW = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0) |
        getattr(os, "O_BINARY", 0))
WINDOWS = os.name == "nt"


def _linked(info) -> bool:
    """Whether a Windows lstat() result is a reparse point (a link or a junction)."""
    return bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def private_folder(path, *, create: bool = False) -> Path:
    """`path` when it is a real folder of this user's that nobody else can open (0700); with `create`, made 0700 when
    missing (its parents as mkdir makes them). ValueError otherwise."""
    folder = Path(path)
    if create:
        try:
            folder.mkdir(mode=0o700, parents=True)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ValueError(f"{folder} could not be made: {exc.strerror or exc}") from None
        else:
            folder.chmod(0o700)           # whatever the umask took away or left: this user's alone
    try:
        info = folder.lstat()
    except OSError:
        raise ValueError(f"{folder} is not a private folder of yours") from None
    if WINDOWS:
        if not stat.S_ISDIR(info.st_mode) or _linked(info):
            raise ValueError(f"{folder} is not a private folder of yours")
        return folder
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError(f"{folder} is not a private folder of yours")
    return folder


def write_private(path, text: str) -> Path:
    """`text` into a new file at `path`, readable by this user only (0600), made new and never through a link (the file
    is removed again should the write fail). -> path."""
    path = Path(path)
    fd = os.open(path, _NEW, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def is_mine(path) -> bool:
    """Whether `path` (not followed, when it is a link) belongs to this user; False when it cannot be looked at. On
    Windows, whatever can be looked at that is not a link or a junction (the app looks only in the user's own folders
    there: the module's docstring)."""
    try:
        info = Path(path).lstat()
    except OSError:
        return False
    if WINDOWS:
        return not _linked(info)
    return info.st_uid == os.getuid()
