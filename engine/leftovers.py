"""What deleted and discarded runs leave in the runs folder, removed once nothing holds it.

On a disk mounted through FUSE (NTFS through ntfs-3g, for one) a file deleted while a process still has it open or
mapped is renamed to
.fuse_hidden<n> and only goes when that process lets go of it. rmtree cannot remove the folder around it, so the
run's folder survives the delete, and stays behind empty once the file goes: an engine that kept a run's transcript
index mapped left runs/<id>/live/ behind for every run deleted or discarded. Whoever holds a file is looked up in
/proc (this user's processes: open files, mappings, working folders), so nothing in use is ever touched; a file
still held is left for the next attempt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import threading
import time

HIDDEN = ".fuse_hidden"
# What the engine's own background builders write into a run's folder, and their temporary files (".<name>.<pid>" or
# ".<name>.<pid>.<thread>"): the transcript index (engine/transcript_index.py) and the prior shapes as JSON and as
# vectors (runs.py). They are derived from the run's own files and worthless without them; a build still writing when
# its run was deleted or discarded (engine exit included) leaves them in a folder that has no run.json any more. So can
# a stain run's stored cell brightness (engine/stain_exclusions.py: stain/brightness/<outline file>-c<channel>-<measure>-
# <time>.npy) when the run is deleted while it is being written.
DERIVED = re.compile(r"\.?(?:transcripts\.sorted\.bin|transcripts\.grid\.json|vectors\.v\d+\.bin\.zst|boundaries\.json)"
                     r"(?:\.\d+){0,2}|(?:tile|resolved)-\d+(?:\.npz)?-c\d+-[a-z0-9]+-\d+\.npy(?:\.tmp)?")
UNFINISHED = re.compile(r"\.(?:transcripts\.sorted\.bin|transcripts\.grid\.json|vectors\.v\d+\.bin\.zst|boundaries\.json)"
                        r"(?:\.\d+){1,2}")     # a build's temporary file, left in a kept run by an engine that died
# Compare's results beside a run (engine/compare.py, analysis/worker.py): compare/<job>/result.json, cells-<column>.npy
# and the points files points-<stage>-<column>.bin, with their temporary files (".result.json.<pid>",
# ".points-<s>-<c>.bin.<pid>", "points-<s>-<c>.bin.tmp"). Derived from the run and its sample: a comparison still being
# scored when its run was deleted leaves them in a folder that has no run.json any more.
COMPARE_JOB = re.compile(r"[0-9a-f]{8,64}")
COMPARE_DERIVED = re.compile(r"\.?(?:result\.json|cells-\d+\.npy|points-\d+-\d+\.bin)(?:\.\d+|\.tmp)?")


def holders(folders) -> set[str]:
    """The paths in or under `folders` that a process of this user has open, mapped or as its working folder."""
    prefixes = [os.path.realpath(f) for f in folders]

    def inside(path: str) -> bool:
        return any(path == p or path.startswith(p + os.sep) for p in prefixes)

    found: set[str] = set()
    if os.name == "nt":                   # no /proc, nor uids: a held file there refuses its removal by itself
        return found
    uid = os.getuid()
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:                       # no /proc (macOS): nothing can be told apart
        return found
    for pid in pids:
        proc = f"/proc/{pid}"
        try:
            if os.stat(proc).st_uid != uid:
                continue
            links = [f"{proc}/cwd", *(f"{proc}/fd/{fd}" for fd in os.listdir(f"{proc}/fd"))]
        except OSError:                   # gone meanwhile, or not ours to read
            continue
        for link in links:
            try:
                target = os.readlink(link).removesuffix(" (deleted)")    # as maps paths: an unlinked file still held
            except OSError:
                continue
            if inside(target):
                found.add(target)
        try:
            with open(f"{proc}/maps", "rb") as maps:
                for line in maps:
                    fields = line.rstrip(b"\n").split(maxsplit=5)
                    if len(fields) == 6:
                        path = os.fsdecode(fields[5]).removesuffix(" (deleted)")
                        if inside(path):
                            found.add(path)
        except OSError:
            continue
    return found


def named(folders) -> set[str]:
    """The folders (as given) that a process of this user, other than this one, names on its command line: a save
    helper is given a run's output folder there (engine/zarr_samples.py helper; an older engine's, inside its JSON
    arguments) and holds nothing in it before it reads it, so only its command line tells that it will (the start's
    check for a save still at work, engine/runs.py _in_use). Read from /proc/<pid>/cmdline, as API mode's sweeper finds Chrome; a name counts from an
    argument's start (a path that only ends the same, /copy/data/runs/a, is another folder) up to a path boundary
    (runs/a never matches runs/a-2), plain or JSON-escaped."""
    forms = {}
    for folder in folders:
        text = str(folder).rstrip(os.sep) or os.sep
        forms[str(folder)] = [re.compile(r'(?:^|(?<=[\x00\s"\'=,:(\[]))' + re.escape(form) + r'(?=$|[/\\"\s\x00])')
                              for form in {text, json.dumps(text)[1:-1]}]
    found: set[str] = set()
    if not forms:
        return found
    if not os.path.isdir("/proc"):
        return _named_elsewhere(forms)
    uid, me = os.getuid(), os.getpid()
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return found
    for pid in pids:
        if pid == me:
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as stream:
                line = os.fsdecode(stream.read())
        except OSError:                   # gone meanwhile, or not ours to read
            continue
        for folder, patterns in forms.items():
            if folder not in found and any(p.search(line) for p in patterns):
                found.add(folder)
        if len(found) == len(forms):
            break
    return found


def _named_elsewhere(forms: dict) -> set[str]:
    """named() on macOS and Windows, which have no /proc: this user's processes' command lines from psutil."""
    from .. import portable
    found: set[str] = set()
    try:
        import psutil
        processes = list(psutil.process_iter(["pid", "cmdline"]))
    except Exception:
        return found
    for process in processes:
        try:
            if process.info["pid"] == os.getpid() or not portable.owned_by_me(process):
                continue
            line = "\x00".join(process.info["cmdline"] or ())
        except Exception:                 # gone meanwhile, or not ours to read
            continue
        for folder, patterns in forms.items():
            if folder not in found and any(p.search(line) for p in patterns):
                found.add(folder)
        if len(found) == len(forms):
            break
    return found


def remove_folder(folder: Path, held=holders) -> bool:
    """Remove a deleted run's folder, whatever is left in it, unless a process still holds something there.
    True once it is gone."""
    folder = Path(folder)
    if not folder.exists():
        return True
    if held([folder]):
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return not folder.exists()


def remove_later(folder: Path, held=holders, delays=(1, 5, 30, 120)):
    """Large crash leftovers go off the startup path; files still held are retried."""
    def remove():
        if remove_folder(folder, held=held):
            return
        for delay in delays:
            time.sleep(delay)
            if remove_folder(folder, held=held):
                return
    worker = threading.Thread(target=remove, name=f"remove-{folder.name}", daemon=True)
    worker.start()
    return worker


def trash(folder: Path) -> Path:
    """Move a marked run out of the registry before slow deletion starts."""
    root = folder.parent / ".trash"
    root.mkdir(exist_ok=True)
    destination = root / f"{folder.name}-{time.time_ns()}"
    folder.rename(destination)
    return destination


def _only_leftovers(folder: Path) -> bool:
    """Nothing in the folder but folders, .fuse_hidden* files and what the engine derives (DERIVED; COMPARE_DERIVED in
    compare/<job>/): what a delete leaves, never a run's own content."""
    for place, _, files in os.walk(folder):
        place = Path(place)
        compared = place.parent == Path(folder) / "compare" and COMPARE_JOB.fullmatch(place.name) is not None
        if any(not (name.startswith(HIDDEN) or DERIVED.fullmatch(name) or (compared and COMPARE_DERIVED.fullmatch(name)))
               for name in files):
            return False
    return True


def sweep(root: Path, held=holders) -> dict:
    """At engine start (the engine lock is held: no other engine uses these runs): .fuse_hidden* files and unfinished
    builds' temporary files (UNFINISHED) that nobody holds, in run folders and their live/ and xenium/ ("hidden"), and
    run folders without run.json holding nothing but empty folders, such files and derived ones (DERIVED, and Compare's
    COMPARE_DERIVED; "folders").
    A folder without run.json that holds anything else is left alone and listed in "left" (Runs logs it)."""
    root = Path(root)
    result: dict[str, list[str]] = {"hidden": [], "folders": [], "left": []}
    try:
        folders = [entry for entry in os.scandir(root) if entry.is_dir(follow_symlinks=False)]
    except OSError:
        return result
    hidden, orphans = [], []
    for entry in folders:
        if entry.name == ".trash":
            continue
        folder = Path(entry.path)
        if not (folder / "run.json").exists():
            if _only_leftovers(folder):
                orphans.append(folder)
            else:
                result["left"].append(str(folder))
            continue
        for place in (folder, folder / "live", folder / "xenium"):
            try:
                hidden += [Path(e.path) for e in os.scandir(place)
                           if (e.name.startswith(HIDDEN) or UNFINISHED.fullmatch(e.name)) and e.is_file()]
            except OSError:
                pass
    if not hidden and not orphans:
        return result
    busy = held([*orphans, *{p.parent for p in hidden}])
    for path in hidden:
        if str(path) not in busy and os.path.realpath(path) not in busy:
            try:
                path.unlink()
                result["hidden"].append(str(path))
            except OSError:
                pass
    for folder in orphans:
        real = os.path.realpath(folder)
        if any(p == real or p.startswith(real + os.sep) for p in busy):
            result["left"].append(str(folder))
            continue
        shutil.rmtree(folder, ignore_errors=True)
        (result["left"] if folder.exists() else result["folders"]).append(str(folder))
    return result
