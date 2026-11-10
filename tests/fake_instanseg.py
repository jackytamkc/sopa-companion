"""The real inference worker protocol, with a deterministic model and no torch or downloads."""
import os
from pathlib import Path
import sys
import time

import numpy as np

from sopa_companion.zarr_bridge import stain_instanseg as worker


class FakeModel:
    def __init__(self, settings, out):
        self.settings = settings
        assert settings["pixel_size"] == .5
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert "DISPLAY" not in os.environ
        for key in ("INSTANSEG_MODEL_PATH", "INSTANSEG_TORCHSCRIPT_PATH"):
            assert os.environ[key] == os.environ["INSTANSEG_BIOIMAGEIO_PATH"]
        worker.emit("progress", message="Downloading model…")
        if os.environ.get("FAKE_INSTANSEG_OFFLINE"):
            raise RuntimeError("Could not download the InstanSeg model. Check the connection and try again.")
        worker.emit("hello", model=settings["model"], device="cpu", cuda_visible_devices="1",
                    nice=os.getpriority(os.PRIO_PROCESS, 0), display=os.environ.get("DISPLAY"))

    def __call__(self, pixels):
        assert pixels.shape[-1] == len(self.settings["channels"])
        assert pixels.shape[-1] == 4
        assert np.array_equal(pixels[..., 0] * 2, pixels[..., 3])
        time.sleep(.1)
        return pixels[..., 0].astype(np.uint32)


worker.InstanSeg = FakeModel
worker.discover_models = lambda: [dict(name=n, kind="builtin", date=None) for n in worker.MODEL_NAMES]

if __name__ == "__main__":
    sys.exit(worker.main())
