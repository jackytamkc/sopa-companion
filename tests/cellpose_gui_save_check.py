"""Run in the sopa env: save a `_seg.npy` through the installed Cellpose GUI's own `_save_sets`, headless.

    python cellpose_gui_save_check.py <folder> <model_path> [<image name>]

The image is the patch whose mask is saved (patch-01.tif unless named: an export's names start with its stem).
"""
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from cellpose.gui import io as gui_io  # noqa: E402


def main(folder, model_path, image="patch-01.tif"):
    folder = Path(folder)
    masks = np.zeros((32, 32), np.uint16)
    masks[4:12, 4:12] = 1
    masks[16:28, 16:28] = 2
    parent = SimpleNamespace(
        filename=str(folder / image), NZ=1, restore=None, ratio=1.,
        segmentation_settings=SimpleNamespace(flow_threshold=.55, cellprob_threshold=-1.5, diameter=23.),
        RGBDropDown=SimpleNamespace(name_map={0: "RGB", 1: "red"}), color=0,
        outpix=np.zeros((1, 32, 32), np.uint16), cellcolors=np.zeros((3, 3), np.uint8), cellpix=masks[None],
        flows=[], ismanual=np.array([False, True]), track_changes=[], current_model_path=model_path,
        get_normalize_params=lambda: {**gui_io.normalize_default, "percentile": [2., 98.], "tile_norm_blocksize": 128,
                                      "sharpen_radius": 1, "smooth_radius": 0, "invert": False},
        logger=logging.getLogger("cellpose-gui-check"), ncells=SimpleNamespace(get=lambda: 2))
    gui_io._save_sets(parent)
    assert (folder / f"{Path(image).stem}_seg.npy").is_file()


if __name__ == "__main__":
    main(*sys.argv[1:4])
