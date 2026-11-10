"""Only the image read is replaced; patch geometry, export, watching and freezing use the real helper."""
import numpy as np
from sopa_companion.zarr_bridge import stain_patches


class CropModel:
    def __init__(self, job, out):
        import os
        from sopa_companion.zarr_bridge.stain import emit
        self.out = out
        self.job = job
        emit("hello", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
             nice=os.getpriority(os.PRIO_PROCESS, 0), device="cpu")

    def __call__(self, pixels):
        import json
        import time
        if (self.out / "stain/fail-export").exists():
            raise ValueError("Test model failed")
        slow = self.out / "stain/slow-export"          # seconds a patch takes (a test of the export's progress)
        if slow.exists():
            time.sleep(float(slow.read_text()))
        with (self.out / "crop-evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(list(pixels.shape))+"\n")
        mask = np.zeros(pixels.shape[:2], np.uint32)
        mask[20:40,20:40] = 1
        return mask


class CropInstanSeg(CropModel):
    def __init__(self, job):
        from pathlib import Path
        super().__init__(job, Path(job["out"]))
    def __enter__(self): return self
    def __exit__(self, *args): pass


def pixels(job, window, step=1):
    x, y, xx, yy = window
    channels = len(job["settings"]["channels"]) if job["settings"].get("method") == "instanseg" else 3
    image = np.zeros((yy-y, xx-x, channels), np.float32)
    image[..., 0] = np.arange(xx-x)[None, :] % 23 + 1
    if channels > 1:
        image[..., 1] = np.arange(yy-y)[:, None] % 17 + 1
    return image[::step, ::step]


if __name__ == "__main__":
    import sys
    from sopa_companion.zarr_bridge import stain, stain_instanseg
    if len(sys.argv) > 1 and sys.argv[1] != "--export":
        # A CPU-only InstanSeg protocol double for real tiling tests, including pyramid pixel sizes.
        class Model:
            def __init__(self, settings, out):
                import os
                assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and "DISPLAY" not in os.environ
                self.settings = settings
                stain_instanseg.emit("hello", model=settings["model"], device="cpu", model_pixel_size=.5)

            def __call__(self, pixels):
                assert pixels.shape[-1] == len(self.settings["channels"])
                return pixels[..., 0].astype(np.uint32)

        stain_instanseg.InstanSeg = Model
        sys.exit(stain_instanseg.main())
    stain.Cellpose = CropModel
    stain_instanseg.InstanSegProcess = CropInstanSeg
    stain_patches.image_pixels = pixels
    stain_patches.main()
