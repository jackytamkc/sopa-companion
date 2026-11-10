"""The app's code says its own name, sopa_companion, everywhere; the previous version's (proseg_live) only where this
version still reads, takes or moves what that version wrote or holds (names.py), and the patched proseg keeps its own
name, proseg-live."""
from __future__ import annotations

from pathlib import Path
import re

from sopa_companion import names
from sopa_companion.engine.config import ROOT

# The code that ships: the package's modules, its page and its settings (not its tests, docs, harness or native build).
SHIPPED = ("*.py", "*.toml", "analysis/**/*.py", "api/**/*.py", "desktop/**/*.py", "engine/**/*.py", "server/**/*.py",
           "zarr_bridge/**/*.py", "frontend/index.html", "frontend/src/*")
# Where the previous version's names are read, taken, moved or recognised on purpose, and why.
ALLOWED = {
    "names.py": "the names that cross versions",
    "desktop/migrate.py": "moves the previous version's state folder and profile",
    "desktop/install.py": "replaces the previous version's desktop entry and icon",
    "desktop/processes.py": "recognises an engine of the previous version",
    "server/runtime_file.py": "recognises an engine of the previous version (serve_command)",
    "desktop/window.py": "the previous version's profile, configured state folder",
    "engine/crashlog.py": "blanks the previous version's cookie",
    "zarr_bridge/helper.py": "reads the previous version's provenance",
    "frontend/src/storage-boot.mjs": "copies the previous version's storage keys",
}
# The previous version's names; not the patched proseg's (proseg-live, its own words "proseg-live: ...", and
# locate.proseg_live(), which finds that program), nor the learned models' format ids (proseg-live-mixing-...,
# proseg-live-kinds/...), nor the tool's name before its words.
OLD = re.compile(r"proseg_live(?!\()|PROSEG_LIVE_|prosegLive|ProsegLive|proseg-live(?::(?!\s)|-(?!mixing-|kinds/))|"
                 r"Proseg Live: (?!watch)")


def shipped_files():
    found = set()
    for pattern in SHIPPED:
        found.update(path for path in ROOT.glob(pattern) if path.is_file() and "__pycache__" not in path.parts)
    return sorted(found)


def test_the_previous_versions_names_appear_only_where_they_are_read_or_moved():
    files = shipped_files()
    assert len(files) > 100 and ROOT / "names.py" in files
    stray = [f"{path.relative_to(ROOT)}:{n}: {line.strip()}" for path in files
             if path.relative_to(ROOT).as_posix() not in ALLOWED
             for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1) if OLD.search(line)]
    assert not stray, "\n".join(stray)


def test_what_crosses_versions_is_named_once():
    assert names.PROVENANCE == names.PROVENANCE_KEYS[0] == names.MODULE == names.MODULES[0] == "sopa_companion"
    assert names.PROVENANCE_KEYS[1] == names.MODULES[1] == "proseg_live"
    assert names.ZARR_LOCK_FOLDERS[0].startswith("~/.cache/proseg_live/")           # the previous version's, first
    assert names.FIT_LOCKS[0].startswith("~/.cache/proseg_live/")
    assert names.BACKUPS == names.BACKUP_FOLDERS[0] == ".proseg-live-backups"
    assert names.LOG_PREFIXES == ("SoPA companion", "Proseg Live") and names.BINS_PREFIX == names.BINS_PREFIXES[0]
    assert names.home_path("~/.x/y", "/h") == names.home_path("~/.x/y", "/h/") == Path("/h/.x/y")
    assert names.retired_isolation({}) is None
    refused = names.retired_isolation({"PROSEG_LIVE_ZARR_LOCK_DIR": "/t"})
    assert refused.startswith("PROSEG_LIVE_ZARR_LOCK_DIR is no longer read")
    assert "SOPA_COMPANION_ZARR_LOCK_DIR" in refused
    assert names.retired_isolation({"PROSEG_LIVE_QC_FIT_LOCK": "/t", "SOPA_COMPANION_QC_FIT_LOCK": "/u"}) is None
