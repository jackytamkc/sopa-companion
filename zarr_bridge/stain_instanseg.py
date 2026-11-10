"""InstanSeg inference in its own environment; the sopa helper owns tiles and geometry."""
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.request
import zipfile

import numpy as np

from .devices import choose_device, empty_cache
from .stain import atomic_json, check_tile, emit, resolution_info

MODEL_NAMES = ("fluorescence_nuclei_and_cells", "single_channel_nuclei", "brightfield_nuclei")


def model_index():
    spec = importlib.util.find_spec("instanseg")
    if spec is None:
        raise RuntimeError("The InstanSeg software is unavailable")
    index = Path(spec.origin).parent / "bioimageio_models/model-index.json"
    entries = json.loads(index.read_text())
    return [next(m for m in entries if m["name"] == name) for name in MODEL_NAMES]


def discover_models():
    return [dict(name=m["name"], kind="builtin", date=None) for m in model_index()]


def model_file(settings, out, load):
    """Stage only the official TorchScript weights, validate them, then publish under a cache lock."""
    name = settings["model"]
    entry = next((m for m in model_index() if m["name"] == name), None)
    if entry is None:
        raise ValueError("Choose a model from the list")
    version = entry["version"]
    if not isinstance(version, str) or not version or any(c not in "0123456789." for c in version):
        raise ValueError("The model version is invalid")
    url = entry["url"]
    if not url.startswith("https://github.com/instanseg/instanseg/releases/download/"):
        raise ValueError("The model download address is invalid")
    root = Path(os.environ["INSTANSEG_BIOIMAGEIO_PATH"])
    for key in ("INSTANSEG_MODEL_PATH", "INSTANSEG_TORCHSCRIPT_PATH"):
        os.environ[key] = str(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / name / version / "instanseg.pt"
    if not path.is_file():
        emit("progress", message="Downloading model…")
    with (root / f".{name}.lock").open("a") as lock:
        from .. import portable
        portable.lock(lock)
        if not path.is_file():
            try:
                with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as staging:
                    staging = Path(staging)
                    with urllib.request.urlopen(url, timeout=30) as response, (staging / "model.zip").open("wb") as stream:
                        shutil.copyfileobj(response, stream)
                    with zipfile.ZipFile(staging / "model.zip") as archive:
                        with archive.open("instanseg.pt") as source, (staging / "instanseg.pt").open("wb") as target:
                            shutil.copyfileobj(source, target)
                    load(staging / "instanseg.pt")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging / "instanseg.pt", path)
            except Exception as exc:
                raise RuntimeError("Could not download the InstanSeg model. Check the connection and try again.") from exc
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    record = dict(method="instanseg", model=name, model_kind="builtin", model_version=version,
                  model_path=str(path.resolve()), model_sha256=digest)
    previous = out / "stain/model.json"
    if previous.is_file() and json.loads(previous.read_text()) != record:
        raise ValueError("The model file has changed. Start a new run to use it.")
    return record


class InstanSeg:
    def __init__(self, settings, out):
        import torch
        from instanseg import InstanSeg as Model
        self.settings, self.torch, self.factory = settings, torch, Model
        record = model_file(settings, out, lambda p: torch.jit.load(str(p), map_location="cpu"))
        self.path = record["model_path"]
        self.device, self.gpu, skipped = choose_device(torch)
        if not self.gpu:
            emit("fallback", device="cpu", reason=skipped or "The stain GPU is unavailable")
        try:
            self.model = self.load()
        except RuntimeError as exc:
            if not self.gpu:
                raise
            self.fallback(exc)
        atomic_json(out / "stain/model.json", record)
        self.model_pixel_size = float(getattr(getattr(self.model, "instanseg", None), "pixel_size", .5))
        if not math.isfinite(self.model_pixel_size) or self.model_pixel_size <= 0:
            raise ValueError("The model's pixel size is invalid")
        emit("hello", **record, instanseg=importlib.metadata.version("instanseg-torch"),
             model_pixel_size=self.model_pixel_size,
             device=self.device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
             nice=os.getpriority(os.PRIO_PROCESS, 0))

    def load(self):
        weights = self.torch.jit.load(self.path, map_location="cpu")
        return self.factory(weights, device=self.device, verbosity=0, image_reader="skimage.io")

    def fallback(self, error):
        was = getattr(self, "device", "cuda:0" if self.gpu else "cpu")
        self.gpu, self.device, self.model = False, "cpu", None
        empty_cache(self.torch, was)
        emit("fallback", device="cpu", reason=str(error))
        self.model = self.load()

    def evaluate(self, pixels):
        # InstanSeg squeezes inputs and treats the shortest axis as channels, even for tensors.
        # Pad narrow edge tiles so that axis choice is unambiguous, then crop the returned masks.
        height, width, channels = pixels.shape
        padding = ((0, max(0, channels+1-height)), (0, max(0, channels+1-width)), (0, 0))
        if any(high for _, high in padding):
            pixels = np.pad(pixels, padding, mode="edge")
        check_tile(pixels.shape, {**self.settings, "method": "instanseg"},
                   model_pixel_size=getattr(self, "model_pixel_size", .5))
        image = self.torch.from_numpy(np.moveaxis(pixels, -1, 0).copy()).unsqueeze(0)
        dual = self.settings["model"] == "fluorescence_nuclei_and_cells"
        resolve = dual and self.settings.get("resolve_cells", False)
        with self.torch.inference_mode():
            labels = self.model.eval_medium_image(image, pixel_size=self.settings["pixel_size"],
                target="all_outputs" if resolve else self.settings["output"], return_image_tensor=False,
                resolve_cell_and_nucleus=resolve)
            channel = 1 if resolve and self.settings["output"] == "cells" else 0
            masks = labels[0, channel].detach().cpu().numpy()
        if masks.shape != pixels.shape[:2] or not np.isfinite(masks).all() or (masks < 0).any():
            raise ValueError("The model returned an invalid mask")
        return masks[:height, :width].astype(np.uint32)

    def __call__(self, pixels):
        try:
            return self.evaluate(pixels)
        except RuntimeError as exc:
            if not self.gpu:
                raise
            self.fallback(exc)
            return self.evaluate(pixels)


class InstanSegProcess:
    """One model per run, one request at a time. Temporary arrays are never tile commits."""
    def __init__(self, job):
        self.job, self.process = job, None
        self.root = Path(job["out"]) / "stain"
        self.model_pixel_size = .5

    def __enter__(self):
        return self

    def start(self):
        if self.process is None:
            launcher = ["setpriv", "--pdeathsig", "TERM"] if shutil.which("setpriv") else []
            self.process = subprocess.Popen(launcher + [self.job["instanseg_python"], "-m",
                self.job["instanseg_module"], str(self.root / "job.json")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)
            for line in self.process.stdout:
                event = json.loads(line)
                kind = event.pop("ev")
                if kind == "error":
                    raise RuntimeError(event["message"])
                if kind == "hello":
                    self.model_pixel_size = float(event.get("model_pixel_size", .5))
                    if not math.isfinite(self.model_pixel_size) or self.model_pixel_size <= 0:
                        raise ValueError("The model's pixel size is invalid")
                    emit(kind, **{**event, **resolution_info(self.job), "model_pixel_size": self.model_pixel_size})
                    return
                emit(kind, **event)
            raise RuntimeError("The InstanSeg helper ended before loading the model. Details are in the Log.")

    def __call__(self, pixels):
        self.start()
        check_tile(pixels.shape, {**self.job["settings"], "method": "instanseg",
                   "pixel_size": resolution_info(self.job)["um_per_px"]}, model_pixel_size=self.model_pixel_size)
        np.save(self.root / "instanseg-input.npy", pixels, allow_pickle=False)
        self.process.stdin.write("tile\n")
        self.process.stdin.flush()
        for line in self.process.stdout:
            event = json.loads(line)
            kind = event.pop("ev")
            if kind == "mask":
                return np.load(self.root / "instanseg-output.npy", allow_pickle=False)
            if kind == "error":
                raise RuntimeError(event["message"])
            emit(kind, **event)
        raise RuntimeError("The InstanSeg helper ended without a mask. Details are in the Log.")

    def __exit__(self, *args):
        if self.process is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdout.close()
        for name in ("instanseg-input.npy", "instanseg-output.npy"):
            (self.root / name).unlink(missing_ok=True)


def main():
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    try:
        with redirect_stdout(sys.stderr):
            if sys.argv[1] == "--models":
                emit("models", models=discover_models())
            else:
                job = json.loads(Path(sys.argv[1]).read_text())
                out = Path(job["out"])
                model = InstanSeg({**job["settings"], "pixel_size": resolution_info(job)["um_per_px"]}, out)
                for line in sys.stdin:
                    if line.strip() != "tile":
                        raise ValueError("The image request is invalid")
                    pixels = np.load(out / "stain/instanseg-input.npy", allow_pickle=False)
                    np.save(out / "stain/instanseg-output.npy", model(pixels), allow_pickle=False)
                    emit("mask")
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit("error", message=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
