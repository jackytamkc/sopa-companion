import os
import json
import pickle
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from conftest import needs_sopa, sopa_python
from sopa_companion.engine import cellpose_settings as C

MODELS = [{"name": "cpsam", "kind": "builtin"}, {"name": "cpsam_custom_1", "kind": "custom"}]


def test_gui_settings_convert_each_sets_pixels_before_comparing_and_return_raw(tmp_path):
    from sopa_companion.engine.stain_levels import offered
    from test_stain_levels import pyramid
    fine, coarse = tmp_path / "fine", tmp_path / "coarse"
    fine.mkdir()
    coarse.mkdir()
    for folder, pixel, level, diameter, block in ((fine, .2125, 0, 40, 128), (coarse, .425, 1, 20, 64)):
        (folder / "patches.json").write_text(json.dumps(dict(image="image", um_per_px=pixel, level=level, factor=2**level)))
        gui_file(folder, mtime=(level + 1) * 10**18, diameter=diameter,
                 normalize_params={**C.NORMALIZE_DEFAULT, "tile_norm_blocksize": block})
    images = [dict(key="image", levels=4, um_per_px=.2125, scales=offered(pyramid()))]
    found = C.gui_settings([fine, coarse], MODELS, images=images)
    assert found["settings"]["diameter"] == 40 and found["raw"]["diameter"] == 20
    assert found["settings"]["normalize"]["tile_norm_blocksize"] == 128
    assert found["settings"]["level"] == 1 and found["resolution"]["um_per_px"] == .425
    assert "diameter" not in found["differences"] and "normalize" not in found["differences"]


def test_gui_settings_unavailable_level_still_converts_to_full_pixels(tmp_path):
    from test_stain_levels import pyramid
    (tmp_path / "patches.json").write_text(json.dumps(dict(image="image", um_per_px=.3, level=1, factor=2)))
    gui_file(tmp_path, diameter=15, normalize_params={"tile_norm_blocksize": 7})
    image = pyramid((1,))
    image["to_frame"][0][0] = image["to_frame"][1][1] = .2
    found = C.gui_settings(tmp_path, MODELS, images=[image])
    assert found["settings"]["diameter"] == pytest.approx(22.5) and "level" not in found["settings"]
    assert found["settings"]["normalize"]["tile_norm_blocksize"] == 10


def gui_file(folder, name="patch-01", mtime=None, **values):
    """A `_seg.npy` shaped like the Cellpose GUI's 2-D save (gui/io.py `_save_sets`)."""
    data = {"outlines": np.zeros((8, 8), np.uint16), "colors": np.zeros((1, 3), np.uint8),
            "masks": np.zeros((8, 8), np.uint16), "filename": str(folder / f"{name}.tif"), "flows": [],
            "ismanual": np.array([True]), "manual_changes": [], "model_path": "/home/x/.cellpose/models/cpsam_custom_1",
            "flow_threshold": .55, "cellprob_threshold": -1.5,
            "normalize_params": {**C.NORMALIZE_DEFAULT, "percentile": np.array([2., 98.]), "tile_norm_blocksize": 128},
            "restore": None, "ratio": 1., "diameter": 23.}
    data.update(values)
    path = folder / f"{name}_seg.npy"
    np.save(path, data)
    if mtime is not None:
        os.utime(path, ns=(mtime, mtime))
    return path


def test_validate_fills_cellpose_defaults_and_checks_ranges():
    result = C.validate({})
    assert result["flow_threshold"] == .4 and result["cellprob_threshold"] == 0 and result["diameter"] is None
    assert result["min_size"] == 15 and result["niter"] is None and result["augment"] is False
    assert result["normalize"] == C.NORMALIZE_DEFAULT
    tuned = C.validate({"diameter": 23, "niter": 300, "normalize": {"percentile": [1, 99.5], "tile_norm_blocksize": 64,
                                                                      "invert": True}})
    assert tuned["diameter"] == 23. and tuned["niter"] == 300
    assert tuned["normalize"]["percentile"] == [1., 99.5] and tuned["normalize"]["tile_norm_blocksize"] == 64
    fixed = C.validate({"normalize": {"lowhigh": [10, 2000]}})["normalize"]
    assert fixed["lowhigh"] == [10., 2000.] and fixed["percentile"] is None
    for bad in ({"normalize": {"lowhigh": [5, 5]}}, {"normalize": {"lowhigh": [1, "x"]}}, {"flow_threshold": "x"}, {"cellprob_threshold": 50}, {"niter": 2.5}, {"augment": 1},
                {"normalize": {"percentile": [99, 1]}}, {"normalize": {"percentile": [1]}},
                {"normalize": {"tile_norm_blocksize": -1}}, {"diameter": float("nan")}):
        with pytest.raises(C.SettingsError):
            C.validate(bad)


def test_eval_kwargs_match_cellpose_eval_names():
    kwargs = C.eval_kwargs(C.validate({"flow_threshold": .6, "normalize": {"sharpen_radius": 2}}))
    assert set(kwargs) == {"diameter", "flow_threshold", "cellprob_threshold", "niter", "min_size", "augment",
                           "normalize"}
    assert kwargs["normalize"]["sharpen_radius"] == 2 and kwargs["normalize"]["normalize"] is True


def test_gui_settings_reads_the_newest_gui_save_and_lists_differences(tmp_path):
    gui_file(tmp_path, "patch-01", mtime=1_000_000_000_000_000_000, flow_threshold=.5)
    gui_file(tmp_path, "patch-02", mtime=2_000_000_000_000_000_000)
    found = C.gui_settings(tmp_path, MODELS)
    s = found["settings"]
    assert found["from"] == "patch-02_seg.npy" and found["files"] == 2 and found["unknown_model"] is None
    assert s["model"] == "cpsam_custom_1" and s["flow_threshold"] == .55 and s["cellprob_threshold"] == -1.5
    assert s["diameter"] == 23. and s["normalize"]["percentile"] == [2., 98.]
    assert s["normalize"]["tile_norm_blocksize"] == 128
    assert set(found["differences"]) == {"flow_threshold"}
    name = tmp_path.name
    assert found["differences"]["flow_threshold"] == {f"{name}/patch-02_seg.npy": .55, f"{name}/patch-01_seg.npy": .5}
    assert found["set"] == name and found["run"] == tmp_path.parent.name
    assert C.validate({k: v for k, v in s.items() if k not in ("model", "model_path")})   # usable as run settings


def test_gui_settings_ignores_app_exports_and_reports_unknown_models(tmp_path):
    export = {"masks": np.zeros((8, 8), np.uint16), "outlines": np.zeros((8, 8), np.uint16), "filename": "x.tif",
              "ismanual": np.zeros(0, bool), "flow_threshold": .4}
    np.save(tmp_path / "patch-01_seg.npy", export)
    assert C.gui_settings(tmp_path, MODELS) is None
    gui_file(tmp_path, "patch-02", model_path="/somewhere/else/mystery_model", diameter=0.)
    found = C.gui_settings(tmp_path, MODELS)
    assert found["settings"]["model"] is None and found["unknown_model"] == "/somewhere/else/mystery_model"
    assert found["settings"]["diameter"] is None                                  # the GUI's 0 means "automatic"


class Boom:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_read_seg_refuses_arbitrary_pickled_objects(tmp_path):
    path = tmp_path / "evil_seg.npy"
    np.save(path, {"masks": np.zeros((2, 2)), "manual_changes": [], "x": Boom()})
    with pytest.raises(pickle.UnpicklingError):
        C.read_seg(path)
    assert C.gui_settings(tmp_path, MODELS) is None
    np.save(tmp_path / "plain_seg.npy", np.zeros((3, 3)))
    with pytest.raises(ValueError):
        C.read_seg(tmp_path / "plain_seg.npy")


def test_read_seg_never_waits_on_a_pipe_nor_reads_through_a_link(tmp_path):
    """A mask in a folder others can write to may be swapped after it was listed: a FIFO in its place is refused at
    once (a model list working out its labels would otherwise wait for a writer, holding its lock), and a link is not
    read through."""
    gui_file(tmp_path, "patch-01")
    pipe = tmp_path / "pipe_seg.npy"
    os.mkfifo(pipe)
    raised = []

    def read():
        try:
            C.read_seg(pipe)
        except (OSError, ValueError) as exc:
            raised.append(exc)
    reading = threading.Thread(target=read, daemon=True)
    reading.start()
    reading.join(10)
    waited = reading.is_alive()
    if waited:                              # it waits for a writer: one comes and goes, so the read ends
        os.close(os.open(pipe, os.O_WRONLY | os.O_NONBLOCK))
        reading.join(10)
    assert not waited and isinstance(raised[0], ValueError)
    (tmp_path / "link_seg.npy").symlink_to(tmp_path / "patch-01_seg.npy")
    with pytest.raises(OSError):
        C.read_seg(tmp_path / "link_seg.npy")
    assert C.read_seg(tmp_path / "patch-01_seg.npy")["flow_threshold"] == .55


@needs_sopa
def test_settings_saved_by_the_installed_cellpose_gui(tmp_path):
    check = Path(__file__).with_name("cellpose_gui_save_check.py")
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"} | {"QT_QPA_PLATFORM": "offscreen",
                                                                     "MPLCONFIGDIR": str(tmp_path / "mpl")}
    done = subprocess.run([sopa_python(), "-I", str(check), str(tmp_path),
                           str(Path.home() / ".cellpose/models/cpsam_custom_1")],
                          capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    found = C.gui_settings(tmp_path, MODELS)
    s = found["settings"]
    assert s["model"] == "cpsam_custom_1" and s["flow_threshold"] == .55 and s["cellprob_threshold"] == -1.5
    assert s["diameter"] == 23. and s["normalize"]["percentile"] == [2., 98.] and s["normalize"]["sharpen_radius"] == 1


def test_gui_files_are_read_once_and_oversized_or_linked_files_are_skipped(tmp_path, monkeypatch):
    gui_file(tmp_path, "patch-01")
    reads = []
    real = C.read_seg
    monkeypatch.setattr(C, "read_seg", lambda path: reads.append(path.name) or real(path))
    assert C.gui_settings(tmp_path, MODELS)["files"] == 1
    assert C.gui_settings(tmp_path, MODELS)["files"] == 1 and reads == ["patch-01_seg.npy"]       # cached
    gui_file(tmp_path, "patch-01", mtime=3_000_000_000_000_000_000, flow_threshold=.7)              # changed: re-read
    assert C.gui_settings(tmp_path, MODELS)["settings"]["flow_threshold"] == .7 and len(reads) == 2
    (tmp_path / "link_seg.npy").symlink_to(tmp_path / "patch-01_seg.npy")
    monkeypatch.setattr(C, "MAX_SEG_BYTES", 10)
    big = gui_file(tmp_path, "patch-02", mtime=4_000_000_000_000_000_000)
    found = C.gui_settings(tmp_path, MODELS)
    assert found["files"] == 1 and found["from"] == "patch-01_seg.npy" and big.stat().st_size > 10


def chosen_set(folder, stem, level, run, created, patches=2):
    """A set in a chosen folder as the engine's sources give it ({folder, manifest, set, run}), with its manifest."""
    manifest = {"set": os.urandom(16).hex(), "stem": stem, "run": run, "created": created, "image": "image",
                "level": level, "factor": 2 ** level, "um_per_px": .2125 * 2 ** level,
                "patches": [{"id": i, "box": [0, 0, 8, 8], "image": f"{stem}-{i:02d}.tif", "mask": f"{stem}-{i:02d}_seg.npy",
                             "export_hash": "0" * 64, "cells": 0} for i in range(1, patches + 1)]}
    return {"folder": folder, "manifest": manifest, "set": manifest["set"], "run": run}


def test_sets_side_by_side_in_one_folder_each_read_with_their_own_manifest(tmp_path):
    """Two exports in one chosen folder: each GUI file takes its resolution from its own set's manifest, and the answer
    names the newest file's set and run; a file no set lists (another tool's, or Cellpose's own) is not read."""
    from sopa_companion.engine.stain_levels import offered
    from test_stain_levels import pyramid
    fine = chosen_set(tmp_path, "Test-20261024-100000", 0, "run-fine", "2026-10-24T10:00:00+00:00")
    coarse = chosen_set(tmp_path, "Test-20261024-110000", 1, "run-coarse", "2026-10-24T11:00:00+00:00")
    gui_file(tmp_path, "Test-20261024-100000-01", mtime=10**18, diameter=40)
    gui_file(tmp_path, "Test-20261024-110000-02", mtime=2 * 10**18, diameter=20)
    gui_file(tmp_path, "unlisted", mtime=3 * 10**18, diameter=99)
    images = [dict(key="image", levels=4, um_per_px=.2125, scales=offered(pyramid()))]
    found = C.gui_settings([fine, coarse], MODELS, images=images)
    assert found["files"] == 2 and found["from"] == "Test-20261024-110000-02_seg.npy"
    assert found["set"] == coarse["set"] and found["run"] == "run-coarse"
    assert found["resolution"] == {"level": 1, "factor": 2, "um_per_px": .425} and found["raw"]["diameter"] == 20
    assert found["settings"]["diameter"] == 40 and found["settings"]["level"] == 1
    alone = C.gui_settings(fine, MODELS, images=images)
    assert alone["set"] == fine["set"] and alone["resolution"]["level"] == 0 and alone["files"] == 1


def test_a_listed_mask_that_was_deleted_is_passed_over(tmp_path):
    source = chosen_set(tmp_path, "Test-20261024-100000", 0, "run", "2026-10-24T10:00:00+00:00")
    gui_file(tmp_path, "Test-20261024-100000-02", flow_threshold=.6)
    found = C.gui_settings([source], MODELS)                    # -01 was never saved, or was deleted
    assert found["files"] == 1 and found["settings"]["flow_threshold"] == .6
    (tmp_path / "Test-20261024-100000-02_seg.npy").unlink()
    assert C.gui_settings([source], MODELS) is None


def test_normalisation_can_be_switched_off():
    off = C.validate({"normalize": {"normalize": False}})["normalize"]
    assert off["normalize"] is False and C.eval_kwargs({"normalize": off})["normalize"]["normalize"] is False
    assert C.validate({})["normalize"]["normalize"] is True
    with pytest.raises(C.SettingsError):
        C.validate({"normalize": {"normalize": "no"}})
