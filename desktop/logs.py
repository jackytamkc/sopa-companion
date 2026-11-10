"""Private engine logs, bounded rotation and credential-free diagnostics."""
from __future__ import annotations

from collections import deque
import io
import os
from pathlib import Path
import re
import stat
import sys
import threading

from ..server.runtime_file import private_directory

MAX_BYTES = 20 * 1024 * 1024
BACKUPS = 3


def log_path(config: dict) -> Path:
    return Path(config["paths"]["state_dir"]) / "logs/server.log"


def _check_file(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("Engine log must be a regular file owned by the current user")
    path.chmod(0o600)


def rotate(path: str | Path) -> None:
    path = Path(path)
    for suffix in ("", *(f".{i}" for i in range(1, BACKUPS + 1))):
        _check_file(Path(str(path) + suffix))
    Path(str(path) + f".{BACKUPS}").unlink(missing_ok=True)
    for index in range(BACKUPS - 1, 0, -1):
        source = Path(str(path) + f".{index}")
        if source.exists():
            source.replace(str(path) + f".{index + 1}")
    if path.exists():
        path.replace(str(path) + ".1")


def open_log(path: str | Path, *, max_bytes: int = MAX_BYTES):
    path = Path(path)
    private_directory(path.parent)
    _check_file(path)
    if path.exists() and path.stat().st_size >= max_bytes:
        rotate(path)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, "ab", buffering=0)


class RotatingOutput(io.TextIOBase):
    """Share rotation across stdout/stderr, including their inherited descriptors."""
    def __init__(self, path, *, max_bytes=MAX_BYTES, redirect_fds=False):
        self.path, self.max_bytes, self.redirect_fds = Path(path), max_bytes, redirect_fds
        self.stream = open_log(path, max_bytes=max_bytes)
        self.lock = threading.RLock()
        self._redirect()

    def _redirect(self):
        if self.redirect_fds:
            for descriptor in (1, 2):
                os.dup2(self.stream.fileno(), descriptor)

    @property
    def encoding(self):
        return "utf-8"

    def fileno(self):
        return self.stream.fileno()

    def writable(self):
        return True

    def write(self, text):
        raw = text.encode("utf-8", errors="replace")
        with self.lock:
            if os.fstat(self.stream.fileno()).st_size + len(raw) > self.max_bytes:
                self.stream.close()
                rotate(self.path)
                self.stream = open_log(self.path, max_bytes=self.max_bytes)
                self._redirect()
            self.stream.write(raw)
        return len(text)

    def flush(self):
        if not self.stream.closed:
            self.stream.flush()

    def close(self):
        super().close()
        self.stream.close()


def install_rotation(config: dict) -> None:
    """Only launcher children replace their output streams."""
    if os.environ.get("SOPA_COMPANION_DESKTOP_LOG") == str(log_path(config)):
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = sys.stderr = RotatingOutput(log_path(config), redirect_fds=True)


def redact(text: str, secrets=()) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"(?i)([?&](?:token|code)=)[^\s&\"'<>]+", r"\1[redacted]", text)
    return re.sub(r"(?i)(Bearer\s+)[^\s\"'<>]+", r"\1[redacted]", text)


def log_tail(path: str | Path, *, secrets=()) -> str:
    try:
        with Path(path).open(errors="replace") as stream:
            return redact("".join(deque(stream, maxlen=30)).rstrip(), secrets)
    except OSError:
        return "No engine log is available."
