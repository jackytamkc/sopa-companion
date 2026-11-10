"""Names that cross versions of the app.

The app's package was called proseg_live before it was called sopa_companion, and both versions can meet on one
computer: an engine of the previous version (a desktop install not updated yet, a notebook package not reinstalled yet,
or a rollback) beside one of this version, saving into the same samples. What this version writes carries the new
names; what the previous version writes, holds or runs is still read, taken or recognised, through the names here.
Standard library only: the engine, the helpers in the other environments, the notebook's side and the harness import it.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

# What a save records in the sample: a proseg save's table's uns[PROVENANCE], a stain save's shapes' attrs[PROVENANCE].
# Readers take the first of PROVENANCE_KEYS there is: a sample keeps what either version saved into it.
PROVENANCE = "sopa_companion"
PROVENANCE_KEYS = (PROVENANCE, "proseg_live")

# The module an engine runs (`python -m <module> [--config FILE] serve`): status, stop and the state folder's move
# recognise an engine of either version.
MODULE = "sopa_companion"
MODULES = (MODULE, "proseg_live")
# The command pip installs for this version: an engine it starts runs as `<python> <...>/sopa-companion [--config FILE]
# serve` (the script's first line names its interpreter), which a later version's stop and status must know too. On
# Windows pip installs it as sopa-companion.exe, or as a launcher beside sopa-companion-script.py (or .pyw).
COMMAND = "sopa-companion"
COMMAND_FORMS = (COMMAND, COMMAND + ".exe", COMMAND + "-script.py", COMMAND + "-script.pyw")

# The engine log's prefix ("SoPA companion: engine starting: ..."). A log can hold both versions' lines (the previous
# version's engine wrote the log the first start of this one reads for its crash report): readers take either.
LOG_PREFIX = "SoPA companion"
LOG_PREFIXES = (LOG_PREFIX, "Proseg Live")

# Folders and files in the user's home are written from it ("~/..."): home_path gives them for any home folder.

# Zarr save locks, one file per zarr in each folder, taken in this order and held for a save's whole build and write.
# The previous version takes only the first, its only folder: its saves and this version's exclude each other both
# ways, and nobody holds the second without the first, so no two saves can deadlock. A later version can take the
# second alone once no engine of the previous one is left.
ZARR_LOCK_FOLDERS = ("~/.cache/proseg_live/zarr-locks", "~/.cache/sopa-companion/zarr-locks")
# The mixing check's computer-wide fit locks (one fit on GPU 0 at a time), tried in the same order, without blocking.
FIT_LOCKS = ("~/.cache/proseg_live/qc-fit.lock", "~/.cache/sopa-companion/qc-fit.lock")
# A private lock folder and fit lock file (tests, the harness's engines): when set, the only ones used; nothing outside
# them is created, locked or pruned. The previous version's names for them are refused when set alone
# (retired_isolation).
ZARR_LOCK_DIR_ENV = "SOPA_COMPANION_ZARR_LOCK_DIR"
FIT_LOCK_ENV = "SOPA_COMPANION_QC_FIT_LOCK"
RETIRED_ENV = {"PROSEG_LIVE_ZARR_LOCK_DIR": ZARR_LOCK_DIR_ENV, "PROSEG_LIVE_QC_FIT_LOCK": FIT_LOCK_ENV}

# What a save moves aside, beside the zarr, with its journal. Written under the previous version's name, the only folder
# that version rolls an interrupted save back from (renamed, an interrupted save of this version would be invisible to
# it, and its next save into that zarr could take a name the interrupted save had moved aside); both are read.
BACKUPS = ".proseg-live-backups"
BACKUP_FOLDERS = (BACKUPS, ".sopa-companion-backups")

# The GPU worker's per-frame folders on the RAM disk: <prefix><engine pid>-*. Dead engines' folders of either are swept.
BINS_PREFIX = "sopa-companion-bins-"
BINS_PREFIXES = (BINS_PREFIX, "proseg-live-bins-")

# The default desktop instance's state folder, and the previous version's folders, which the first start of this version
# moves into it (desktop/migrate.py): its state, and the window's Chrome profile, which it kept apart.
STATE_DIR = "~/.local/state/sopa-companion"
LEGACY_STATE_DIR = "~/.local/state/proseg_live"
LEGACY_PROFILE = "~/.local/share/proseg_live/chrome-profile"
# The desktop entry's installation record (desktop/install.py), and the previous version's, read when there is no new
# one; the previous version's entry in the applications menu.
INSTALL_RECORD = "~/.local/share/sopa-companion/desktop-install.json"
LEGACY_INSTALL_RECORD = "~/.local/share/proseg_live/desktop-install.json"
LEGACY_ENTRY = "~/.local/share/applications/proseg-live.desktop"


def home_path(value: str, home: str | Path | None = None) -> Path:
    """A "~/..." name above in `home`, else in the home folder as it is now (tests point HOME elsewhere)."""
    if not value.startswith("~/"):
        return Path(value)
    return Path(home if home is not None else Path.home()) / value[2:]


def command_script(arg) -> bool:
    """Whether a command line's argument is that command's script, by its file name (COMMAND_FORMS; any case on
    Windows): `.../bin/sopa-companion` is, `sopa-companion-tools` is not."""
    if not isinstance(arg, str) or not arg:
        return False
    name = re.split(r"[\\/]", arg)[-1]
    if os.name == "nt":
        return name.lower() in COMMAND_FORMS
    return name in COMMAND_FORMS


def retired_isolation(environ=None) -> str | None:
    """Why this process may not use the computer's shared lock files: a private lock variable of the previous version
    set without its new name. Silently ignored, it would put a test's or the harness's engine on the user's real locks.
    None when there is none."""
    environ = os.environ if environ is None else environ
    for old, new in RETIRED_ENV.items():
        if environ.get(old) and not environ.get(new):
            return f"{old} is no longer read: name the private folder or file with {new} instead"
    return None
