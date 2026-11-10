"""Idempotent installation of per-user desktop entries and the application icon."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

from .. import dirs
from ..engine.config import ROOT
from ..names import INSTALL_RECORD, LEGACY_INSTALL_RECORD, MODULE, STATE_DIR, home_path
from ..server.runtime_file import atomic_private_json, private_directory
from .logs import log_path
from .processes import DesktopError

# The launcher on the desktop, the applications menu's entry and the icon, named after the app.
DESKTOP_NAME = "SoPA-companion.desktop"
ENTRY_NAME, ICON_NAME = "sopa-companion.desktop", "sopa-companion.svg"
# What earlier installs left, removed where an installation record lists them once this one's files replace them: the
# previous version's menu entry and icon (named proseg-live), and an earlier name of the launcher on the desktop.
OLD_DESKTOP_NAME = "Proseg-Live.desktop"
OLD_NAMES = ("proseg-live.desktop", "proseg-live.svg")
RECORDED_NAMES = (ENTRY_NAME, DESKTOP_NAME, ICON_NAME, OLD_DESKTOP_NAME, *OLD_NAMES)
PACKAGE = MODULE                                # the module the entries run (python -m sopa_companion)


def exec_arg(value: str | Path) -> str:
    """Quote a literal Exec argument through both desktop-entry escape layers."""
    value = str(value).replace("%", "%%")
    if any(c in value for c in "\n\r\x00"):
        raise DesktopError("Desktop entry paths cannot contain newlines or NULs.")
    for character in ("\\", '"', "`", "$"):
        value = value.replace(character, "\\" + character)
    return '"' + value.replace("\\", "\\\\") + '"'


def entry_value(value: str | Path) -> str:
    """A literal path as a plain key's value (TryExec, Path): not an Exec command line, so only the value escapes."""
    value = str(value)
    if any(c in value for c in "\n\r\x00"):
        raise DesktopError("Desktop entry paths cannot contain newlines or NULs.")
    return value.replace("\\", "\\\\").replace(" ", "\\s")


def app_command(python: str, root: Path = ROOT) -> str:
    """The Exec command that starts this copy of the app whichever folder the desktop starts it from. A source tree
    whose folder is named as the package puts its parent on PYTHONPATH (as the engine does for every helper), ahead of
    any other copy the interpreter also knows (a .pth file naming one); another source tree relies on its editable
    install (pip install -e .), and an installed copy on its environment, which holds the package: its parent is that
    environment's site-packages, never put on a PYTHONPATH (the engine would hand it to every helper)."""
    command = [exec_arg(python), "-m", PACKAGE]
    if root.name == PACKAGE and dirs.source_tree(root):
        command = ["env", exec_arg(f"PYTHONPATH={root.parent}"), *command]
    return " ".join(command)


def entry_text(python: str, log: str | Path, root: Path = ROOT, workdir: str | Path | None = None) -> str:
    """The desktop entry. Its launcher starts in `workdir` (an installed copy's state folder), else in the app's folder:
    never in site-packages or the home folder, which `python -m` would put first on its path."""
    command = app_command(python, root)
    return f"""[Desktop Entry]
Type=Application
Name=SoPA companion
GenericName=Cell segmentation for SpatialData samples
Comment=Stain segmentation with Cellpose, and Proseg Live: watch proseg build cells one iteration at a time
Exec={command} launch
TryExec={entry_value(python)}
Path={entry_value(workdir or root)}
Icon=sopa-companion
Terminal=false
StartupNotify=true
StartupWMClass=SoPACompanion
Categories=Science;Education;DataVisualization;
Keywords=segmentation;spatial;transcriptomics;Xenium;proseg;sopa;Cellpose;
Actions=stop;log;

[Desktop Action stop]
Name=Stop engine
Exec={command} stop

[Desktop Action log]
Name=Show engine log
Exec=xdg-open {exec_arg(log)}
"""


def desktop_directory(home: Path, *, custom_home: bool = False, run=subprocess.run, which=shutil.which) -> Path:
    if executable := which("xdg-user-dir"):
        env = dict(os.environ)
        if custom_home:
            env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"))
        try:
            result = run([executable, "DESKTOP"], env=env, capture_output=True, text=True, timeout=5)
            candidate = Path(result.stdout.strip())
            if result.returncode == 0 and candidate.is_absolute() and candidate != home:
                if not custom_home or candidate.resolve().is_relative_to(home):
                    return candidate
        except (OSError, subprocess.TimeoutExpired):
            pass
    return home / "Desktop"


def _write_changed(path: Path, content: bytes, mode: int) -> bool:
    if path.is_symlink():
        raise DesktopError("Refusing to replace a symlink in the desktop installation.")
    if path.exists() and path.read_bytes() == content and stat.S_IMODE(path.stat().st_mode) == mode:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".sopa-companion-install-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return True


def install(config: dict, args, *, run=subprocess.run, which=shutil.which, output=None) -> int:
    output = output or (lambda value: sys.stdout.write(value + "\n"))
    home = Path(args.home).expanduser().resolve() if args.home else Path.home()
    share = home / ".local/share"
    manifest, legacy = home_path(INSTALL_RECORD, home), home_path(LEGACY_INSTALL_RECORD, home)
    # Without a record of its own, what the previous version's installer recorded (its files are this app's to replace).
    source = manifest if manifest.exists() else legacy if legacy.exists() else None
    previous = json.loads(source.read_text()) if source else {"schema": 1, "paths": []}
    if args.uninstall:
        paths = [Path(path) for path in previous["paths"]]
        for path in paths:
            if not path.is_absolute() or path.name not in RECORDED_NAMES:
                raise DesktopError("Unexpected path in desktop installation record; refusing to remove it.")
        for path in [*paths, manifest, *([legacy] if legacy.exists() else [])]:
            if args.dry_run:
                output("Would remove " + str(path))
            else:
                path.unlink(missing_ok=True)
        if not args.dry_run:
            _refresh(share, run, which)
            output("SoPA companion desktop integration removed.")
        return 0

    application = share / "applications" / ENTRY_NAME
    icon = share / "icons/hicolor/scalable/apps" / ICON_NAME
    desktop = None if args.no_desktop_icon else desktop_directory(home, custom_home=bool(args.home), run=run, which=which) / DESKTOP_NAME
    # Earlier files go only where the new ones replace them, and only those an installer recorded: the previous
    # version's menu entry and icon always, the old launcher on the desktop where the new one goes.
    stale = [Path(path) for path in previous["paths"] if Path(path).is_absolute() and
             (Path(path).name in OLD_NAMES or desktop and Path(path).name == OLD_DESKTOP_NAME)]
    # Preserve the venv path: resolving the interpreter's symlink loses the venv.
    python = os.path.abspath(sys.executable)
    workdir = None if dirs.source_tree(ROOT) else launcher_folder(config, home if args.home else None)
    if workdir is not None and not args.dry_run:
        private_directory(workdir)
    text = entry_text(python, log_path(config), ROOT, workdir=workdir).encode()
    targets = [(application, text, 0o644), (icon, Path(__file__).with_name(ICON_NAME).read_bytes(), 0o644)]
    if desktop:
        targets.append((desktop, text, 0o755))
    if args.dry_run:
        for path, _, mode in targets:
            output(f"Would install {path} ({mode:o})")
        for path in stale:
            output("Would remove " + str(path))
        output("Would record " + str(manifest))
        if source == legacy:
            output("Would remove " + str(legacy))
        output(text.decode().rstrip())
        return 0

    validator = which("desktop-file-validate")
    if not validator:
        raise DesktopError("desktop-file-validate is required to install the desktop entries.")
    changed = []
    for path, content, mode in targets:
        if _write_changed(path, content, mode):
            changed.append(path)
        if path.suffix == ".desktop":
            checked = run([validator, str(path)], capture_output=True, text=True, timeout=10)
            if checked.returncode:
                raise DesktopError("Desktop entry validation failed: " + checked.stdout + checked.stderr)
    paths = sorted(set(previous["paths"]) | {str(path) for path, _, _ in targets})
    value = {"schema": 1, "paths": paths}
    if value != previous or not manifest.exists():
        atomic_private_json(manifest, value)
    if stale:
        # Removed once the record lists the new files, so the record never misses a file of ours.
        for path in stale:
            path.unlink(missing_ok=True)
        changed += stale
        atomic_private_json(manifest, {"schema": 1, "paths": [path for path in paths if Path(path) not in stale]})
    if source == legacy:
        # The previous version's record goes last, once this one lists everything it listed that is still there.
        legacy.unlink(missing_ok=True)
        try:
            legacy.parent.rmdir()                    # its folder, when nothing else of the previous version is left in it
        except OSError:
            pass
    if desktop in changed and (gio := which("gio")):
        _optional(run, [gio, "set", str(desktop), "metadata::trusted", "true"])
    if changed:
        _refresh(share, run, which)
    said = "SoPA companion desktop integration installed." if changed else \
        "SoPA companion desktop integration is already installed."
    if any(path.name == "proseg-live.desktop" for path in stale):
        said += " The previous version's entry was replaced: pin the new icon in the dock again if you had pinned it."
    output(said)
    return 0


def launcher_folder(config: dict, home: Path | None = None) -> Path:
    """Where an installed copy's launcher starts: the engine's state folder (it holds no Python file); with --home and
    the default state folder, that home's own."""
    state = Path(config["paths"]["state_dir"])
    if home is not None and state == dirs.state_dir().resolve():
        return home_path(STATE_DIR, home)
    return state


def _optional(run, command):
    try:
        run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _refresh(share, run, which):
    if executable := which("update-desktop-database"):
        _optional(run, [executable, str(share / "applications")])
    if executable := which("gtk-update-icon-cache"):
        _optional(run, [executable, "-f", "-t", str(share / "icons/hicolor")])
