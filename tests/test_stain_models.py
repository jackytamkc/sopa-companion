"""Model discovery and loading use small fake weight files, never Cellpose's shared cache or a GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conftest import sopa_python
from sopa_companion.zarr_bridge import stain


def test_custom_model_label_comes_from_gui_patch_name_without_reading_weights(tmp_path):
    from sopa_companion.engine.patch_folders import legacy_sets
    from sopa_companion.engine.stain import StainRuns
    from test_cellpose_settings import gui_file
    folder = tmp_path / "run" / "set"
    folder.mkdir(parents=True)
    (folder / "patches.json").write_text(json.dumps(dict(level=1, factor=2, um_per_px=.425)))
    gui_file(folder, model_path="/missing/learned_model")
    older = tmp_path / "run" / "older-set"                  # exported before Resolution: cut at scale0
    older.mkdir()
    (older / "patches.json").write_text(json.dumps(dict(um_per_px=.2125)))
    gui_file(older, model_path="/missing/older_model")
    manager = SimpleNamespace(patches=SimpleNamespace(sources=lambda sample=None: legacy_sets(tmp_path)))
    listed = [dict(name="cpsam", kind="builtin"), dict(name="learned_model", kind="custom", path="/missing/learned_model"),
              dict(name="older_model", kind="custom", path="/missing/older_model")]
    result = StainRuns._model_labels(manager, listed)
    assert result[1]["label"] == "learned_model · learned at scale1"
    # The level as the set recorded it, which the models route sends and the picker names (stain.mjs modelLabel).
    assert result[1]["resolution"] == dict(level=1, factor=2)
    assert result[2]["resolution"] == dict(level=0, factor=1) and result[2]["label"] == "older_model · learned at scale0"
    assert result[0] == listed[0] and "label" not in listed[1] and "resolution" not in listed[1]


def test_of_two_sets_naming_one_model_the_newest_gives_its_level(tmp_path):
    """The sets are read oldest first (by the time their manifests were made), so the level a model learned at is the
    newest set's that names it, whichever folder holds which."""
    from sopa_companion.engine.patch_folders import legacy_sets
    from sopa_companion.engine.stain import StainRuns
    from test_cellpose_settings import gui_file
    for name, level, created in (("b-newer", 2, "2026-10-22T10:00:00+00:00"), ("a-older", 1, "2026-10-21T10:00:00+00:00")):
        folder = tmp_path / "run" / name
        folder.mkdir(parents=True)
        (folder / "patches.json").write_text(json.dumps(dict(level=level, factor=2 ** level, created=created)))
        gui_file(folder, model_path="/missing/learned_model", mtime=(3 - level) * 10**18)   # the older set's file is newer
    manager = SimpleNamespace(patches=SimpleNamespace(sources=lambda sample=None: legacy_sets(tmp_path)))
    labelled = StainRuns._model_labels(manager, [dict(name="learned_model", kind="custom", path="/missing/learned_model")])
    assert labelled[0]["resolution"] == dict(level=2, factor=4) and labelled[0]["label"].endswith("learned at scale2")


@pytest.fixture
def models(tmp_path):
    directory = tmp_path / "cellpose"
    directory.mkdir()
    return SimpleNamespace(MODEL_DIR=directory, MODEL_LIST_PATH=directory / "gui_models.txt",
                           MODEL_NAMES=["cpsam_v2", "cpdino", "cpdino-vitb", "cpsam"])


def test_discovery_reads_installed_names_registry_and_unlisted_weights(models):
    root = models.MODEL_DIR
    for name in ("cpsam", "cpsam_20250102", "unlisted.pth", "notes.txt", "settings.json", ".partial"):
        (root / name).write_bytes(b"weights")
    (root / "folder").mkdir()
    registered = root.parent / "registered_20260327"
    registered.write_bytes(b"custom")
    models.MODEL_LIST_PATH.write_text(f"cpsam_20250102\n{registered}\nmissing\ncpsam\ncpsam_20250102\n\n")
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in root.iterdir() if p.is_file()}
    found = stain.discover_models(models)
    assert [m["name"] for m in found[:4]] == models.MODEL_NAMES
    assert {m["name"] for m in found[4:]} == {"cpsam_20250102", registered.name, "unlisted.pth"}
    assert [m["created"] for m in found[4:]] == sorted([m["created"] for m in found[4:]],reverse=True)
    assert [m["kind"] for m in found] == ["builtin"] * 4 + ["custom"] * 3
    assert found[0]["path"] is None and found[3]["path"] == str(root / "cpsam")
    by_name = {m["name"]: m for m in found}
    assert by_name["cpsam_20250102"]["date"] == "2025-01-02" and by_name[registered.name]["date"] == "2026-03-27"
    assert by_name["unlisted.pth"]["date"] and by_name[registered.name]["path"] == str(registered)
    assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in root.iterdir() if p.is_file()}
    models.MODEL_NAMES = ["another"]
    assert stain.discover_models(models)[0]["name"] == "another"


def test_discovery_without_a_cache_still_offers_builtins(models):
    models.MODEL_DIR.rmdir()
    assert stain.discover_models(models) == [
        dict(name=n, kind="builtin", path=None, date=None) for n in models.MODEL_NAMES]
    assert not models.MODEL_DIR.exists()


def test_gui_model_in_temporary_home_appears_without_loading_weights(tmp_path):
    home = tmp_path / "home"
    model = home / ".cellpose/models/gui-trained_20261020"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"discovery only")
    env = {**os.environ, "HOME": str(home), "CUDA_VISIBLE_DEVICES": "",
           "MPLCONFIGDIR": str(tmp_path / "mpl"), "NUMBA_CACHE_DIR": str(tmp_path / "numba"),
           "XDG_CACHE_HOME": str(tmp_path / "cache"), "PYTHONDONTWRITEBYTECODE": "1"}
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "CELLPOSE_LOCAL_MODELS_PATH"):
        env.pop(key, None)
    result = subprocess.run([sopa_python(), "-B", "-m", "sopa_companion.zarr_bridge.stain", "--models"],
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    found = next(m for m in json.loads(result.stdout)["models"] if m["name"] == model.name)
    assert found["kind"] == "custom" and found["source"] == "gui"
    assert found["path"] == str(model) and found["date"] == "2026-10-20"
    assert model.read_bytes() == b"discovery only"


@pytest.fixture
def out(tmp_path):
    path = tmp_path / "run"
    (path / "stain").mkdir(parents=True)
    return path


def test_custom_model_is_hashed_and_changed_resume_is_refused(models, out):
    path = models.MODEL_DIR / "cpsam_20250102"
    path.write_bytes(b"trained weights")
    settings = dict(model=path.name, model_kind="custom", model_path=str(path))
    record = stain.model_file(settings, out, models)
    assert record == dict(model=path.name, model_kind="custom", model_path=str(path),
                          model_sha256=hashlib.sha256(b"trained weights").hexdigest())
    (out / "stain/model.json").write_text(json.dumps(record))
    assert stain.model_file(settings, out, models) == record
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="file has changed"):
        stain.model_file(settings, out, models)
    path.unlink()
    with pytest.raises(ValueError, match="no longer available"):
        stain.model_file(settings, out, models)


@pytest.mark.parametrize("name", ["unknown", "../cpsam", "/tmp/cpsam"])
def test_helper_rejects_unknown_models_without_loading_or_downloading(models, out, name):
    models.cache_model_path = Mock()
    with pytest.raises(ValueError, match="no longer available"):
        stain.model_file(dict(model=name), out, models)
    models.cache_model_path.assert_not_called()


def test_custom_model_path_is_pinned_to_the_discovered_file(models, out):
    path = models.MODEL_DIR / "trained"
    path.write_bytes(b"weights")
    with pytest.raises(ValueError, match="file has moved"):
        stain.model_file(dict(model="trained", model_path="/elsewhere/trained"), out, models)


def test_a_missing_builtin_downloads_once_into_cellposes_own_folder_and_reports_progress(models, out, monkeypatch):
    """Into Cellpose's model folder (MODEL_DIR, normally ~/.cellpose/models: every later run and session, Cellpose's
    window and sopa's own Cellpose find it there), through a staging folder inside it (the same disk: one rename
    publishes it whole), MODEL_DIR given back after; never once per run (1.15 GB for cpsam)."""
    events = []
    monkeypatch.setattr(stain, "emit", lambda ev, **kw: events.append((ev, kw)))
    directory = models.MODEL_DIR
    choices = stain.discover_models(models)
    choices[1]["path"] = str(directory / "cpdino")  # a shared file disappeared after discovery
    monkeypatch.setattr(stain, "discover_models", lambda _: choices)

    def download(name):
        assert models.MODEL_DIR.parent == directory and models.MODEL_DIR.name.startswith(".download-")
        assert models.MODEL_DIR.is_dir()
        path = models.MODEL_DIR / name
        path.write_bytes(b"downloaded")
        return str(path)

    models.cache_model_path = Mock(side_effect=download)
    first = stain.model_file(dict(model="cpdino"), out, models)
    assert first["model_path"] == str((directory / "cpdino").resolve())
    assert Path(first["model_path"]).read_bytes() == b"downloaded"
    assert events == [("progress", {"message": "Downloading the model…"})]
    assert models.MODEL_DIR == directory and [p.name for p in directory.iterdir()] == ["cpdino"]
    assert not (out / "stain/models").exists()
    second = out.parent / "another-run"
    (second / "stain").mkdir(parents=True)
    assert stain.model_file(dict(model="cpdino"), second, models) == first
    models.cache_model_path.assert_called_once_with("cpdino")


def test_a_run_with_its_own_downloaded_model_keeps_it(models, out):
    """A run from before (its built-in downloaded into the run) resumes on that copy, downloading nothing."""
    own = out / "stain/models/cpdino"
    own.parent.mkdir(parents=True)
    own.write_bytes(b"the run's own copy")
    models.cache_model_path = Mock()
    assert stain.model_file(dict(model="cpdino"), out, models)["model_path"] == str(own.resolve())
    models.cache_model_path.assert_not_called()
    assert list(models.MODEL_DIR.iterdir()) == []


def test_a_builtin_downloads_into_the_run_where_cellposes_folder_cannot_be_written(models, out, tmp_path):
    """Where Cellpose's folder cannot be made (here a file is in its way), the run gets the copy, as before."""
    blocked = tmp_path / "a-file"
    blocked.write_bytes(b"")
    models.MODEL_DIR = blocked / "models"

    def download(name):
        assert models.MODEL_DIR.parent == out / "stain/models"
        path = models.MODEL_DIR / name
        path.write_bytes(b"downloaded")
        return str(path)

    models.cache_model_path = Mock(side_effect=download)
    record = stain.model_file(dict(model="cpdino"), out, models)
    assert record["model_path"] == str((out / "stain/models/cpdino").resolve())
    assert models.MODEL_DIR == blocked / "models"
    assert [p.name for p in (out / "stain/models").iterdir()] == ["cpdino"]


def test_failed_download_has_plain_error_and_removes_partial_weights(models, out):
    directory = models.MODEL_DIR

    def fail(name):
        models.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        (models.MODEL_DIR / name).write_bytes(b"partial")
        raise OSError("network details")

    models.cache_model_path = Mock(side_effect=fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Could not download the model. Check the connection and try again."):
            stain.model_file(dict(model="cpdino"), out, models)
        assert not (out / "stain/models").exists()
        assert list(directory.iterdir()) == []          # neither the partial weights nor the staging folder
        assert models.MODEL_DIR == directory
    assert models.cache_model_path.call_count == 2


@pytest.mark.parametrize("name", ["cpsam", "cpsam_v2", "cpdino", "cpdino-vitb", "trained"])
def test_selected_weights_are_loaded_and_recorded(models, out, monkeypatch, name):
    path = models.MODEL_DIR / name
    path.write_bytes(b"weights")
    models.CellposeModel = Mock()
    models.cache_model_path = Mock()
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, OutOfMemoryError=RuntimeError),
                            device=lambda name: name)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "cellpose", SimpleNamespace(models=models))
    monkeypatch.setattr(stain.importlib.metadata, "version", lambda name: "test")
    emit = Mock()
    monkeypatch.setattr(stain, "emit", emit)
    image = {"key": "image", "channels": ["DAPI"], "to_frame": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
             "levels": [{"path": "images/image/scale0/image", "shape": [1, 64, 64], "dims": ["c", "y", "x"],
                         "origin": [0, 0], "pixel": [1, 1]}]}
    wrapper = stain.Cellpose(dict(settings=dict(model=name), image=image), out)     # the run's job, as the helper has it
    models.CellposeModel.assert_called_once_with(gpu=False, pretrained_model=str(path), device="cpu")
    models.cache_model_path.assert_not_called()
    record = json.loads((out / "stain/model.json").read_text())
    assert record["model"] == name and record["model_path"] == str(path)
    if name == "trained":
        assert record["model_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert emit.call_args.kwargs["model"] == name and emit.call_args.kwargs["level"] == 0
    path.unlink()
    with pytest.raises(ValueError, match="file is missing"):
        wrapper.load()
    assert models.CellposeModel.call_count == 1
