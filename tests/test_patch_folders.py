"""Training patches in a folder chosen at every export (engine/patch_folders.py): the checks a typed path passes, the
export dialog's listing and suggestion, the list of folders exported into, manifests read from a folder others can
write to, and an export's own writes into such a folder. The patch helper is a double here: it writes an export's files
into the staging folder it is given (its real exports: test_stain_patches.py)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from conftest import outside
from test_stain import Images, SETTINGS, stain_cpu  # noqa: F401  (the stain engine's fixtures)
from sopa_companion.engine import patch_folders as F
from sopa_companion.engine.config import ROOT
from sopa_companion.engine.runs import RunError, Runs, user_cache
from sopa_companion.engine.samples import Samples

CHANNELS = ["DAPI", "Boundary", None]
BASE, TOKEN = "http://127.0.0.1:8799", "t" * 40


def helper(state, action, **values):
    """The patch helper's answers: boxes as sent, an export's files written into its staging folder (named by the
    manifest's stem, as the real helper names them), and every patch untouched."""
    if action == "boxes":
        return {"patches": [{"id": i, "box": box} for i, box in enumerate(values["boxes"], 1)], "message": ""}
    if action == "inspect":
        return [{"id": p["id"], "state": "untouched", "cells": p["cells"]} for p in values["manifest"]["patches"]]
    assert action == "export"
    manifest, staging = values["manifest"], Path(values["staging"])
    patches = []
    for patch in manifest["patches"]:
        name = f"{manifest['stem']}-{patch['id']:02d}"
        (staging / f"{name}.tif").write_bytes(f"image {name}".encode())
        (staging / f"{name}_seg.npy").write_bytes(f"mask {name}".encode())
        patches.append({**patch, "image": f"{name}.tif", "mask": f"{name}_seg.npy",
                        "export_hash": hashlib.sha256(name.encode()).hexdigest(), "cells": 0})
    written = {**manifest, "patches": patches, "sample": state.meta["sample"], "channels": CHANNELS}
    (staging / "patches.json").write_text(json.dumps(written))
    return written


@pytest.fixture
def engine(config, tmp_path, monkeypatch):
    """A stain engine whose data and state folders are outside pytest's (refused by name only where they are)."""
    for folder in outside("patch-engine-"):
        config["paths"].update(data_dir=str(folder / "var"), state_dir=str(folder / "state"),
                               cache_dir=str(folder / "var" / "cache"))
        config["proseg"]["keep_unsaved"] = True             # several items at once
        config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain", gpu=1)
        runs = Runs(config, Samples(config), Images(tmp_path / "unwritten.zarr"))
        monkeypatch.setattr(runs.stain.worker, "call", helper)
        yield runs
        runs.close(save=False)


def item(runs, sample=None, name=None, boxes=([0, 0, 64, 64], [64, 0, 128, 64])):
    """A Training patches item with its boxes; `sample` and `name` stand for another sample's id and name."""
    run = runs.get(runs.create("image-sample", kind="patches", region={"kind": "full"},
                               stain={**SETTINGS, "start_from": "empty", "count": len(boxes), "size": 64})["id"])
    if sample:
        runs._set(run, sample=sample, sample_name=name)
    runs.stain.patches.change(run, {"version": 0, "boxes": [list(b) for b in boxes]})
    return run


def export(runs, run, folder):
    manager = runs.stain.patches
    return manager.export(run, {"version": manager.draft(run)["version"], "folder": str(folder)})


def plant(folder, sample="image-sample", name="Test", level=0, channels=CHANNELS, patches=2, created=None):
    """A valid set of this app's in `folder`'s records, as another export (or someone else) left it."""
    set_id, stem = os.urandom(16).hex(), f"{name}-20260101-000000"
    manifest = {"set": set_id, "stem": stem, "sample": sample, "sample_name": name, "level": level,
                "channels": list(channels), "layout": "flat", "created": created or "2026-01-01T00:00:00+00:00",
                "patches": [{"id": i, "box": [0, 0, 64, 64], "image": f"{stem}-{i:02d}.tif",
                             "mask": f"{stem}-{i:02d}_seg.npy", "export_hash": "0" * 64, "cells": 0}
                            for i in range(1, patches + 1)]}
    sets = Path(folder) / F.META / "sets"
    sets.mkdir(parents=True, exist_ok=True)
    (sets / f"{set_id}.json").write_text(json.dumps(manifest))
    return manifest


def tree(folder) -> dict:
    return {str(p.relative_to(folder)): p.lstat().st_mtime_ns for p in Path(folder).rglob("*")}


def refusal(folders, value):
    with pytest.raises(RunError) as caught:
        folders.check(value)
    assert caught.value.status == 422 and caught.value.extra == {"field": "folder"}
    return str(caught.value)


# ---- the checks -------------------------------------------------------------------------------------------------------
def test_every_rule_refuses_in_its_own_words_with_the_folder_named(engine, chosen, monkeypatch):
    folders = engine.stain.patches.folders
    home = Path.home()
    store = chosen / "sample" / "data"
    store.mkdir(parents=True)
    (chosen / "sample" / "zarr.json").write_text("{}")
    old = chosen / "old"
    old.mkdir()
    (old / "patches.json").write_text("{}")
    records = chosen / "records"
    records.mkdir()
    (records / F.META).write_text("")
    (chosen / "a file").write_text("")
    runtime = chosen / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("XDG_CACHE_HOME", str(chosen / "caches"))
    environment, prefix = chosen / "env", chosen / "venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin" / "python").write_text("")
    engine.stain.python = str(environment / "bin" / "python")
    # The engine's own environment (a venv in a hidden folder here would be refused as hidden first).
    monkeypatch.setattr(F, "sys", SimpleNamespace(prefix=str(prefix), base_prefix=str(Path.home())))
    cases = {
        "": F.words("empty"), None: F.words("empty"),
        "/" + "a" * 256: F.words("long"), "/a" * 2100: F.words("long"),
        "/tmp/a\nb": F.words("control"), "/tmp/a\x7fb": F.words("control"), "/tmp/\udc80": F.words("control"),
        "patches/here": F.words("relative"), "~nobody-here-at-all/x": F.words("relative"),
        str(chosen / "a[1]"): F.words("glob"), str(chosen / "a*"): F.words("glob"),
        "/": F.words("top", name="/"), "~": F.words("top", name="your home folder"), str(home): F.words("top", name="your home folder"),
        tempfile.gettempdir(): F.words("top", name=os.path.realpath(tempfile.gettempdir())),
        str(chosen / ".hidden" / "x"): F.words("hidden", name=".hidden"),
        str(chosen / "x" / F.META): F.words("hidden", name=F.META),
        str(chosen / "sopa-companion-1a2b3c4d" / "x"): F.words("session"),
        str(runtime / "sopa-companion" / "x"): F.words("session"),
        str(engine.root.parent / "patches"): F.words("app", name="patches"),
        str(engine.state_dir): F.words("app", name=engine.state_dir.name),
        str(ROOT / "frontend"): F.words("app", name="frontend"),
        str(prefix / "patches"): F.words("app", name="patches"),
        str(environment / "patches"): F.words("app", name="patches"),
        str(user_cache() / "patches"): F.words("app", name="patches"),
        str(chosen / "Sample.ZARR" / "images"): F.words("zarr", name="Sample.ZARR"),
        str(store / "images"): F.words("zarr", name="sample"),
        str(chosen / "a file"): F.words("file", name="a file"),
        str(chosen / "a file" / "below"): F.words("file", name="a file"),
        str(old): F.words("old"),
        str(records): F.words("records", name="records"),
    }
    for value, said in cases.items():
        assert refusal(folders, value) == said, value
    assert not (chosen / "a[1]").exists() and not (chosen / "Sample.ZARR").exists()


def test_links_resolve_to_the_real_path_and_a_folder_in_the_temporary_folder_is_taken(engine, chosen):
    folders = engine.stain.patches.folders
    real = chosen / "real" / "patches"
    real.mkdir(parents=True)
    (chosen / "link").symlink_to(chosen / "real", target_is_directory=True)
    (chosen / "patches-link").symlink_to(real, target_is_directory=True)
    for typed in (chosen / "link" / "patches", chosen / "patches-link"):
        assert folders.check(str(typed)) == F.Target(Path(os.path.realpath(real)), True)
    new = folders.check(str(chosen / "link" / "new" / "deeper"))
    assert new == F.Target(Path(os.path.realpath(real.parent / "new" / "deeper")), False)
    assert not (real.parent / "new").exists()
    assert folders.check(f"{chosen}/x/../y").real == Path(os.path.realpath(chosen / "y"))


def test_a_folder_that_cannot_be_written_is_refused_also_through_its_nearest_parent(engine, chosen):
    folders = engine.stain.patches.folders
    locked = chosen / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    if os.access(locked, os.W_OK):
        pytest.skip("this account or filesystem writes into a folder of mode 500 all the same")
    assert refusal(folders, str(locked)) == F.words("write", name="locked")
    assert refusal(folders, str(locked / "new" / "deeper")) == F.words("write", name="locked")


def test_checks_and_listings_write_nothing(engine, chosen):
    run = item(engine)
    manager = engine.stain.patches
    (chosen / "kept").mkdir()
    plant(chosen / "kept")
    before, data = tree(chosen), set(tree(engine.root.parent))     # the engine's own files: none made
    for value in (chosen / "kept", chosen / "new" / "deeper", chosen / "a[1]", chosen / ".hidden"):
        manager.folder_listing(run, str(value))
        try:
            manager.folders.check(str(value))
        except RunError:
            pass
    manager.folder_listing(run, None)
    assert tree(chosen) == before and set(tree(engine.root.parent)) == data
    assert not F.default_home().exists() or not list(F.default_home().iterdir())


# ---- the dialog's listing ---------------------------------------------------------------------------------------------
def test_a_listing_names_folders_in_order_and_says_what_the_folder_holds(engine, chosen):
    run = item(engine)
    manager = engine.stain.patches
    for name in ("beta", "Alpha", "gamma", ".hidden", "data.zarr", "store"):
        (chosen / name).mkdir()
    (chosen / "store" / ".zgroup").write_text("{}")
    (chosen / "a file").write_text("")
    plant(chosen / "gamma")
    (chosen / "Alpha-link").symlink_to(chosen / "Alpha", target_is_directory=True)
    listed = manager.folder_listing(run, str(chosen))
    assert [e["name"] for e in listed["entries"]] == ["Alpha", "Alpha-link", "beta", "gamma"]
    assert [e["patches"] for e in listed["entries"]] == [False, False, False, True]
    assert listed["entries"][2]["path"] == str(Path(os.path.realpath(chosen)) / "beta")
    assert listed["exists"] and listed["parent"] == str(Path(os.path.realpath(chosen)).parent)
    assert listed["refused"] is None and listed["holds"] is None and listed["mixes"] is None
    assert listed["new"] == "Test-scale0"
    (chosen / "Test-scale0").mkdir()
    (chosen / "Test-scale0-2").write_text("")
    assert manager.folder_listing(run, str(chosen))["new"] == "Test-scale0-3"
    new = manager.folder_listing(run, str(chosen / "new" / "deeper"))
    assert (new["exists"], new["entries"], new["new"], new["refused"]) == (False, [], "Test-scale0", None)
    assert new["parent"] == str(Path(os.path.realpath(chosen)) / "new")
    root = manager.folder_listing(run, "/")
    assert root["parent"] is None and root["refused"] == F.words("top", name="/")


def test_a_listing_shows_at_most_2000_folders(engine, chosen):
    run = item(engine)
    for n in range(2001):
        (chosen / f"f{n:05d}").mkdir()
    entries = engine.stain.patches.folder_listing(run, str(chosen))["entries"]
    assert len(entries) == 2000 and entries[-1]["name"] == "f01999"


def test_holds_counts_the_sets_patches_and_samples_and_mixes_names_resolution_then_channels(engine, chosen):
    run = item(engine)
    manager = engine.stain.patches
    folder = chosen / "shared"
    folder.mkdir()
    plant(folder, patches=3)
    plant(folder, sample="other", name="Other", patches=1)
    listed = manager.folder_listing(run, str(folder))
    assert listed["holds"] == {"patches": 4, "sets": 2, "samples": ["Other", "Test"], "more": 0}
    assert listed["mixes"] is None                     # another sample alone is no mix: it is named in holds
    for name in ("Third", "Fourth", "Fifth"):
        plant(folder, sample=name.lower(), name=name, patches=1)
    assert manager.folder_listing(run, str(folder))["holds"] == {
        "patches": 7, "sets": 5, "samples": ["Fifth", "Fourth", "Other"], "more": 2}
    plant(folder, channels=["DAPI", None, None])
    assert manager.folder_listing(run, str(folder))["mixes"] == "channels"
    plant(folder, level=1)
    assert manager.folder_listing(run, str(folder))["mixes"] == "resolution"


def test_shortcuts_are_home_then_up_to_four_folders_exported_into(engine, chosen):
    run = item(engine)
    folders = engine.stain.patches.folders
    key = folders.key(run, engine.stain.patches.draft(run))
    made = []
    for n in range(6):
        made.append(chosen / f"used{n}")
        made[-1].mkdir()
        folders.used(Path(os.path.realpath(made[-1])), key)
    (chosen / "used5").rmdir()                          # the newest is gone: not offered
    listed = engine.stain.patches.folder_listing(run, str(chosen / "used4"))
    home = Path(os.path.realpath(Path.home()))
    assert listed["shortcuts"][0] == {"kind": "home", "name": home.name, "path": str(home)}
    assert [s["name"] for s in listed["shortcuts"][1:]] == ["used3", "used2", "used1", "used0"]
    assert all(s["kind"] == "earlier" for s in listed["shortcuts"][1:])


def test_a_path_refused_before_it_resolves_is_echoed_and_an_unreadable_folder_says_so(engine, chosen):
    run = item(engine)
    manager = engine.stain.patches
    listed = manager.folder_listing(run, "not/absolute")
    assert listed == {"path": "not/absolute", "exists": False, "parent": None, "entries": [],
                      "shortcuts": listed["shortcuts"], "new": None, "holds": None, "mixes": None,
                      "refused": F.words("relative")}
    closed = chosen / "closed"
    closed.mkdir()
    closed.chmod(0o300)
    if os.access(closed, os.R_OK):
        pytest.skip("this account or filesystem reads a folder of mode 300 all the same")
    listed = manager.folder_listing(run, str(closed))
    assert listed["entries"] == [] and listed["refused"].startswith("closed could not be read: ")


def test_the_listing_route_answers_any_path_and_refuses_runs_that_are_not_image_runs(config, chosen, monkeypatch):
    from sopa_companion.server import create_app
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain", gpu=1)
    app = create_app(config, token=TOKEN)
    runs = app.state.runs
    runs.zarr = Images(chosen / "unwritten.zarr")
    monkeypatch.setattr(runs.stain.worker, "call", helper)
    with TestClient(app, base_url=BASE, headers={"Authorization": "Bearer " + TOKEN, "Origin": BASE}) as client:
        run = item(runs)
        url = f"/api/v1/runs/{run.meta['id']}/stain/patches/folders"
        answer = client.get(url, params={"path": str(chosen)})
        assert answer.status_code == 200 and answer.headers["cache-control"] == "no-store"
        assert answer.json()["path"] == os.path.realpath(chosen)
        assert client.get(url, params={"path": "/x\ny"}).json()["refused"] == F.words("control")
        assert client.get(url).json()["path"] == os.path.realpath(F.default_home() / "Test-scale0")
        assert client.get("/api/v1/runs/none/stain/patches/folders").status_code == 404
        assert client.get(url, headers={"Authorization": "Bearer wrong"}).status_code == 401
        runs.closing = True
        try:
            assert client.get(url).status_code == 503
        finally:
            runs.closing = False


# ---- the suggestion ---------------------------------------------------------------------------------------------------
def test_the_suggestion_is_the_last_folder_of_this_key_else_a_new_one_named_after_the_sample(engine, chosen):
    run = item(engine)
    folders = engine.stain.patches.folders
    key = folders.key(run, engine.stain.patches.draft(run))
    assert key == ["image-sample", 0, CHANNELS]
    home = Path(os.path.realpath(F.default_home()))
    assert folders.suggest(run) == home / "Test-scale0"
    plant(home / "Test-scale0", level=1)                 # another resolution's patches there: the next name
    assert folders.suggest(run) == home / "Test-scale0-2"
    plant(home / "Test-scale0-2")                        # this key's: it takes more
    assert folders.suggest(run) == home / "Test-scale0-2"
    other, mine = chosen / "other", chosen / "mine"
    other.mkdir()
    mine.mkdir()
    folders.used(Path(os.path.realpath(mine)), key)
    folders.used(Path(os.path.realpath(other)), ["other-sample", 0, CHANNELS])   # newer, of another key
    assert folders.suggest(run) == Path(os.path.realpath(mine))
    mine.rmdir()
    assert folders.suggest(run) == home / "Test-scale0-2"


def test_a_notebooks_patches_dir_is_the_suggestions_base(engine, chosen, monkeypatch):
    run = item(engine)
    monkeypatch.setattr(engine, "patches_dir", str(chosen / "kept"))
    assert engine.stain.patches.folders.suggest(run) == Path(os.path.realpath(chosen)) / "kept" / "Test-scale0"


def test_the_suggestion_falls_back_to_the_home_folder_which_the_dialog_refuses(engine, chosen, monkeypatch):
    run = item(engine)
    monkeypatch.setenv(F.HOME_ENV, str(chosen / ".hidden"))
    assert engine.stain.patches.folders.suggest(run) == Path(os.path.realpath(Path.home()))
    assert engine.stain.patches.folder_listing(run)["refused"] == F.words("top", name="your home folder")


# ---- the folders exported into ----------------------------------------------------------------------------------------
def test_the_folders_exported_into_are_kept_newest_first_and_bounded(engine, chosen, capsys):
    run = item(engine)
    folders = engine.stain.patches.folders
    export(engine, run, chosen / "first")
    entry = folders.registry()[0]
    assert entry["folder"] == os.path.realpath(chosen / "first") and entry["keys"] == [["image-sample", 0, CHANNELS]]
    for n in range(101):
        folders.used(Path(f"/elsewhere/{n}"), ["s", n, []])
    entries = folders.registry()
    assert len(entries) == 100 and entries[0]["folder"] == "/elsewhere/100"
    for n in range(21):
        folders.used(Path("/elsewhere/many"), ["s", n, []])
    assert folders.registry()[0]["keys"] == [["s", n, []] for n in range(20, 0, -1)]
    folders.path.write_text("not json")
    assert folders.registry() == []
    capsys.readouterr()
    folders.used(Path("/elsewhere/again"), ["s", 0, []])
    assert [e["folder"] for e in folders.registry()] == ["/elsewhere/again"]
    assert capsys.readouterr().out.count("SoPA companion: ") == 1


# ---- manifests from a folder others can write to ----------------------------------------------------------------------
def test_a_manifest_names_paths_only_when_valid(chosen):
    manifest = plant(chosen)
    assert F.valid_manifest(manifest, set=manifest["set"], stem=manifest["stem"]) is manifest
    patch = manifest["patches"][0]
    for change in ({"image": "../x.tif"}, {"mask": "numpy.py"}, {"image": "Other-20260101-000000-01.tif"},
                   {"id": 65}, {"id": True}, {"box": [0, 0, 64]}, {"export_hash": "x" * 64}, {"cells": -1}):
        assert F.valid_manifest({**manifest, "patches": [{**patch, **change}]}) is None, change
    assert F.valid_manifest({**manifest, "patches": [patch, patch]}) is None            # one number, two patches
    assert F.valid_manifest(manifest, set="0" * 32) is None
    assert F.valid_manifest(manifest, stem="other") is None
    assert F.valid_manifest({**manifest, "stem": "../up"}) is None
    assert F.valid_manifest({**manifest, "set": "../../x"}) is None
    assert F.valid_manifest([manifest]) is None
    old = {k: v for k, v in manifest.items() if k != "stem"}
    assert F.valid_manifest(old) is None                     # an old set's names start with "patch"
    path = chosen / "big.json"
    path.write_text(json.dumps({**manifest, "padding": "x" * F.MAX_MANIFEST}))
    with pytest.raises(ValueError):
        F.read_small(path)
    (chosen / "link.json").symlink_to(next((chosen / F.META / "sets").iterdir()))
    with pytest.raises(OSError):
        F.read_small(chosen / "link.json")
    os.mkfifo(chosen / "pipe.json")
    with pytest.raises(ValueError):
        F.read_small(chosen / "pipe.json")                   # never waits on a pipe
    (chosen / "deep.json").write_text("[" * 100000 + "]" * 100000)
    with pytest.raises(ValueError):
        F.read_small(chosen / "deep.json")


def test_a_real_exports_manifest_is_valid_and_carries_no_exports(engine, chosen):
    run = item(engine)
    result = export(engine, run, chosen / "patches")
    folder = Path(result["exported"]["folder"])
    saved = json.loads((folder / F.META / "sets" / f"{result['set']}.json").read_text())
    assert F.valid_manifest(saved, set=result["set"], stem=saved["stem"]) is not None
    assert "exports" not in saved and saved["layout"] == "flat" and saved["sample_name"] == "Test"
    assert engine.stain.patches.draft(run)["exports"] == {result["set"]: str(folder)}


# ---- the export's writes ----------------------------------------------------------------------------------------------
def test_an_export_goes_beside_what_is_there_under_names_of_its_own(engine, chosen):
    run = item(engine)
    folder = chosen / "patches"
    folder.mkdir()
    (folder / "notes.txt").write_text("mine")
    result = export(engine, run, folder)
    stem = result["exported"]["stem"]
    assert stem.startswith("Test-") and result["exported"]["folder"] == os.path.realpath(folder)
    names = sorted(p.name for p in folder.iterdir())
    assert names == sorted([F.META, "notes.txt", f"{stem}-01.tif", f"{stem}-01_seg.npy", f"{stem}-02.tif",
                            f"{stem}-02_seg.npy"])
    assert (folder / f"{stem}-01.tif").read_bytes() == f"image {stem}-01".encode()
    assert not list(engine.stain.patches.staging.iterdir())


def test_a_name_that_appears_meanwhile_is_never_replaced(engine, chosen, monkeypatch):
    run = item(engine)
    folder = chosen / "patches"

    def planting(state, action, **values):
        written = helper(state, action, **values)
        if action == "export":
            Path(values["target"]).mkdir(parents=True, exist_ok=True)
            (Path(values["target"]) / f"{values['manifest']['stem']}-02.tif").write_text("someone else's")
        return written
    monkeypatch.setattr(engine.stain.worker, "call", planting)
    with pytest.raises(RunError) as caught:
        export(engine, run, folder)
    stem = caught.value.args[0].split("named ")[1].split("-02.tif")[0]
    assert caught.value.status == 409 and caught.value.extra == {"field": "folder"}
    assert str(caught.value) == F.words("clash", file=f"{stem}-02.tif", name="patches")
    assert sorted(p.name for p in folder.iterdir()) == [F.META, f"{stem}-02.tif"]
    assert (folder / f"{stem}-02.tif").read_text() == "someone else's"
    assert not list((folder / F.META / "sets").iterdir()) and not list(engine.stain.patches.staging.iterdir())
    assert engine.stain.patches.draft(run)["version"] == 1 and engine.stain.patches.exporting == set()


def test_records_that_are_a_link_are_refused_before_and_during_an_export(engine, chosen, monkeypatch):
    run = item(engine)
    elsewhere = chosen / "elsewhere"
    elsewhere.mkdir()
    folder = chosen / "patches"
    folder.mkdir()
    (folder / F.META).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(RunError, match="holds a .sopa-companion entry that is not a folder"):
        export(engine, run, folder)
    (folder / F.META).unlink()

    def swapping(state, action, **values):              # the records swapped for a link after the check
        written = helper(state, action, **values)
        if action == "export":
            (Path(values["target"]) / F.META).symlink_to(elsewhere, target_is_directory=True)
        return written
    monkeypatch.setattr(engine.stain.worker, "call", swapping)
    with pytest.raises(RunError) as caught:
        export(engine, run, folder)
    assert caught.value.status == 422 and str(caught.value) == F.words("records", name="patches")
    assert not list(elsewhere.iterdir()) and sorted(p.name for p in folder.iterdir()) == [F.META]


def test_nothing_is_left_behind_after_a_success_a_refusal_or_a_failed_helper(engine, chosen, monkeypatch):
    run = item(engine)
    manager = engine.stain.patches
    folder = chosen / "patches"

    def left():
        return [p.name for p in folder.rglob("*.part")] + [p.name for p in manager.staging.iterdir()]
    export(engine, run, folder)
    assert left() == []
    version = manager.draft(run)["version"]

    def changing(state, action, **values):              # the boxes change while the helper works
        written = helper(state, action, **values)
        if action == "export":
            manager.save(run, {**manager.draft(run), "version": version + 1})
        return written
    monkeypatch.setattr(engine.stain.worker, "call", changing)
    with pytest.raises(RunError) as caught:
        export(engine, run, folder)
    assert caught.value.status == 409 and caught.value.extra == {"reason": "changed"} and left() == []
    assert len(list((folder / F.META / "sets").iterdir())) == 1

    def failing(state, action, **values):
        if action == "export":
            raise RunError("The patches could not be read. Try opening the run again", 503)
        return helper(state, action, **values)
    monkeypatch.setattr(engine.stain.worker, "call", failing)
    with pytest.raises(RunError) as caught:
        export(engine, run, folder)
    assert caught.value.status == 503 and left() == []
    monkeypatch.setattr(engine.stain.worker, "call", lambda state, action, **values: {})   # writes no manifest
    with pytest.raises(RunError, match="could not be exported") as caught:
        export(engine, run, folder)
    assert caught.value.status == 503 and left() == []


def test_staging_left_by_a_crash_goes_when_the_engine_starts(config, engine):
    leftover = engine.stain.patches.staging / ("0" * 32)
    leftover.mkdir(parents=True)
    (leftover / "half.tif").write_text("")
    again = Runs(config, Samples(config), engine.zarr)
    try:
        assert not again.stain.patches.staging.exists()
    finally:
        again.close(save=False)


def test_a_planted_log_link_is_never_written_through(engine, chosen):
    run = item(engine)
    manager = engine.stain.patches
    result = export(engine, run, chosen / "patches")
    target = chosen / "someone-elses.txt"
    target.write_text("theirs")
    log = Path(result["exported"]["folder"]) / F.META / "cellpose.log"
    log.symlink_to(target)
    manager.window, manager.launcher = None, "sopa_companion.tests.fake_stain_gui"
    with pytest.raises(RunError, match="could not be started: its log, cellpose.log, is not a plain file"):
        manager.open_gui(run, {"set": result["set"]})
    assert target.read_text() == "theirs" and log.is_symlink() and not manager.processes


def test_a_planted_log_pipe_is_refused_at_once_and_never_written_into(engine, chosen):
    """A FIFO named as the GUI's log: with no reader, Open answers at once instead of waiting for one while it holds
    the manager's lock (every poll and export of every run waits on that lock); with a reader, nothing is written into
    it. Nothing starts either way."""
    run = item(engine)
    manager = engine.stain.patches
    result = export(engine, run, chosen / "patches")
    folder = Path(result["exported"]["folder"])
    log = folder / F.META / "cellpose.log"
    os.mkfifo(log)
    manager.window, manager.launcher = None, "sopa_companion.tests.fake_stain_gui"
    answers = []

    def press():
        try:
            answers.append(manager.open_gui(run, {"set": result["set"]}))
        except RunError as exc:
            answers.append(exc)
    try:
        pressed = threading.Thread(target=press, daemon=True)
        pressed.start()
        pressed.join(10)
        waited = pressed.is_alive()
        if waited:                          # it waits for a reader: one comes and goes, so the press ends
            os.close(os.open(log, os.O_RDONLY | os.O_NONBLOCK))
            pressed.join(10)
        assert not waited, "Open waited for a reader of the planted log"
        said = "The Cellpose GUI could not be started: its log, cellpose.log, is not a plain file"
        assert isinstance(answers[0], RunError) and (answers[0].status, str(answers[0])) == (503, said)
        reader = os.open(log, os.O_RDONLY | os.O_NONBLOCK)
        try:
            with pytest.raises(RunError) as caught:
                manager.open_gui(run, {"set": result["set"]})
            assert (caught.value.status, str(caught.value)) == (503, said)
            assert os.read(reader, 1024) == b""              # no writer left, and nothing written
        finally:
            os.close(reader)
        assert stat.S_ISFIFO(os.lstat(log).st_mode) and not manager.processes
        assert not (folder / "gui-launch.json").exists()
    finally:
        for process in manager.processes.values():
            process.wait(timeout=10)


def test_two_exports_in_one_second_get_stems_of_their_own_and_a_stem_whose_names_are_taken_is_skipped(
        engine, chosen, monkeypatch):
    monkeypatch.setattr(F, "stamp", lambda: "20261024-101010")
    folders = engine.stain.patches.folders
    folder = Path(os.path.realpath(chosen))
    first, second = folders.reserve(folder, "Test", 2), folders.reserve(folder, "Test", 2)
    assert (first, second) == ("Test-20261024-101010", "Test-20261024-101010-2")
    folders.release(folder, first)
    folders.release(folder, second)
    (folder / "Test-20261024-101010-02_seg.npy").write_text("")
    assert folders.reserve(folder, "Test", 2) == "Test-20261024-101010-2"
    assert folders.reserve(folder, "Test", 1) == "Test-20261024-101010"     # only -01's names count for one patch
    run = item(engine)
    one = export(engine, run, chosen / "patches")["exported"]["stem"]
    two = export(engine, run, chosen / "patches")["exported"]["stem"]
    assert one != two and two == one + "-2"


def test_safe_names_start_and_end_with_a_letter_or_digit():
    assert F.safe_name("My sample #3 (left)") == "My-sample-3-left"
    assert F.safe_name("_.-x-._") == "x" and F.safe_name("..") == "sample" and F.safe_name(None) == "sample"
    assert len(F.safe_name("a" * 100)) == 40 and F.STEM.fullmatch(F.safe_name("é" * 50) + "-20261024-101010")
    assert F.export_channels({"nucleus": "DAPI"}) == ["DAPI", None, None]
    assert F.export_channels({"method": "instanseg", "channels": ["a", "b", "c", "d"]}) == ["a", "b", "c", "d"]
    assert F.export_channels({"channels": ["a"]}) == ["a", None, None]


def test_the_helper_names_files_by_the_engines_own_stem_rule():
    helper_source = (ROOT / "zarr_bridge" / "stain_patches.py").read_text()
    assert f're.compile(r"{F.STEM.pattern}")' in helper_source


def test_sources_read_sets_in_the_folders_exported_into_the_default_place_and_patches_dir(engine, chosen, monkeypatch):
    run = item(engine)
    manager = engine.stain.patches
    exported = export(engine, run, chosen / "used")
    home = F.default_home()
    plant(home / "Test-scale0", created="2026-01-02T00:00:00+00:00")
    monkeypatch.setattr(engine, "patches_dir", str(chosen / "kept"))
    plant(chosen / "kept" / "Other-scale1", sample="other", name="Other", created="2026-01-01T00:00:00+00:00")
    plant(chosen / "kept", created="2025-12-31T00:00:00+00:00")
    found = manager.sources()
    assert [s["manifest"]["sample"] for s in found] == ["image-sample", "other", "image-sample", "image-sample"]
    assert found[-1]["set"] == exported["set"] and found[-1]["folder"] == Path(exported["exported"]["folder"])
    assert [s["folder"].name for s in manager.sources("other")] == ["Other-scale1"]
    legacy = manager.root / "old-run" / ("a" * 32)
    legacy.mkdir(parents=True)
    (legacy / "patches.json").write_text(json.dumps({"set": "a" * 32, "sample": "image-sample",
                                                     "created": "2025-01-01T00:00:00+00:00"}))
    assert manager.sources()[0] == legacy and F.legacy_sets(manager.root) == [legacy]


def test_the_guis_working_folder_is_the_helpers_cache_on_the_desktop_and_the_users_cache_in_a_notebook(
        engine, chosen, monkeypatch):
    manager = engine.stain.patches
    assert manager.gui_cache() == engine.root.parent / "stain-cache" and manager.gui_cache().is_dir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(chosen / "caches"))
    monkeypatch.setattr(engine, "session", object())          # a notebook's session: its data folder goes at close
    assert manager.gui_cache() == user_cache() / "stain-gui" == chosen / "caches" / "sopa-companion" / "stain-gui"
    assert manager.gui_cache().is_dir()


def test_the_guis_record_replaces_a_planted_link_and_records_that_became_a_link_say_the_set_is_gone(engine, chosen):
    run = item(engine)
    manager = engine.stain.patches
    result = export(engine, run, chosen / "patches")
    folder = Path(result["exported"]["folder"])
    victim = chosen / "victim.txt"
    victim.write_text("theirs")
    (folder / F.META / "gui.json").symlink_to(victim)
    (folder / "keep-gui").touch()
    manager.window, manager.launcher = None, "sopa_companion.tests.fake_stain_gui"
    try:
        assert manager.open_gui(run, {"set": result["set"]})["running"]
        record = folder / F.META / "gui.json"
        assert not record.is_symlink() and json.loads(record.read_text())["pid"] == manager.processes[str(folder)].pid
        assert victim.read_text() == "theirs"
    finally:
        (folder / "keep-gui").unlink()
        manager.processes[str(folder)].wait(timeout=10)
    manager.processes.clear()
    elsewhere = chosen / "elsewhere"
    (folder / F.META).rename(elsewhere)
    (folder / F.META).symlink_to(elsewhere, target_is_directory=True)
    before = sorted(p.name for p in elsewhere.iterdir())
    with pytest.raises(RunError) as caught:
        manager.open_gui(run, {"set": result["set"]})
    assert caught.value.status == 404 and str(caught.value) == F.words("gone", folder=folder)
    assert sorted(p.name for p in elsewhere.iterdir()) == before and not manager.processes


def test_an_export_whose_draft_cannot_be_saved_takes_back_its_files(engine, chosen, monkeypatch):
    run = item(engine)
    manager = engine.stain.patches
    folder = chosen / "patches"
    real = manager.save

    def failing(run, value):
        if value.get("exports"):                       # the export's commit: its draft
            raise OSError(28, "No space left on device")
        return real(run, value)
    monkeypatch.setattr(manager, "save", failing)
    with pytest.raises(OSError):
        export(engine, run, folder)
    assert sorted(p.name for p in folder.iterdir()) == [F.META] and not list((folder / F.META / "sets").iterdir())
    assert "exports" not in manager.draft(run) and manager.exporting == set() and not list(manager.staging.iterdir())
