"""Cellpose settings shared by the picker, the stain helper and the Cellpose GUI's own `_seg.npy` files.

The Cellpose GUI (4.2.1.1, gui/io.py `_save_sets`) stores the settings it segmented with next to every saved mask:
model_path, diameter, flow_threshold, cellprob_threshold and normalize_params. `gui_settings` reads them back so a
run can use exactly what the user tuned in the GUI. The files are pickled dicts; they are read with an unpickler that
only rebuilds numpy arrays and plain Python values, never arbitrary objects.
"""
import io
import json
import math
import os
import pickle
from pathlib import Path
import stat

import numpy as np

# Cellpose's own defaults (models.normalize_default and CellposeModel.eval in 4.2.1.1).
NORMALIZE_DEFAULT = {"lowhigh": None, "percentile": None, "normalize": True, "norm3D": True, "sharpen_radius": 0,
                     "smooth_radius": 0, "tile_norm_blocksize": 0, "tile_norm_smooth3D": 1, "invert": False}
# name: (default, low, high); None as default means "Cellpose decides".
NUMBERS = {"diameter": (None, 10, 1000), "flow_threshold": (.4, 0, 10), "cellprob_threshold": (0., -20, 20),
           "niter": (None, 1, 10000), "min_size": (15, -1, 100000)}
NORMALIZE_NUMBERS = {"sharpen_radius": (0, 0, 1000), "smooth_radius": (0, 0, 1000),
                     "tile_norm_blocksize": (0, 0, 100000)}
GUI_KEYS = ("model_path", "diameter", "flow_threshold", "cellprob_threshold", "normalize_params")


class SettingsError(ValueError):
    pass


def _number(name, value, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise SettingsError(f"Choose a number between {low} and {high} for {name.replace('_', ' ')}")
    if integer and value != int(value):
        raise SettingsError(f"Choose a whole number for {name.replace('_', ' ')}")
    return int(value) if integer else float(value)


def validate(value):
    """The Cellpose settings of a run, checked and completed with Cellpose's defaults."""
    if not isinstance(value, dict):
        raise SettingsError("Check the Cellpose settings")
    result = {}
    for key, (default, low, high) in NUMBERS.items():
        number = value.get(key, default)
        if number is None or (key == "diameter" and number == "auto"):
            result[key] = None
            continue
        result[key] = _number(key, number, low, high, integer=key in ("niter", "min_size"))
    augment = value.get("augment", False)
    if type(augment) is not bool:
        raise SettingsError("Choose whether to augment")
    result["augment"] = augment
    result["normalize"] = normalize(value.get("normalize"))
    return result


def normalize(value):
    """Cellpose's normalisation subset the app exposes: percentile low/high or a fixed intensity range (lowhigh), tile
    size, sharpen, smooth, invert."""
    result = dict(NORMALIZE_DEFAULT)
    if value is None:
        return result
    if not isinstance(value, dict):
        raise SettingsError("Check the normalisation settings")
    percentile = value.get("percentile")
    if percentile is not None:
        if (not isinstance(percentile, (list, tuple)) or len(percentile) != 2):
            raise SettingsError("Choose a low and a high percentile")
        low, high = (_number("percentile", p, 0, 100) for p in percentile)
        if not low < high:
            raise SettingsError("The low percentile must be below the high one")
        result["percentile"] = [low, high]
    lowhigh = value.get("lowhigh")
    if lowhigh is not None:
        if not isinstance(lowhigh, (list, tuple)) or len(lowhigh) != 2:
            raise SettingsError("Choose a fixed low and high intensity, or none")
        low, high = (_number("intensity range", v, -1e9, 1e9) for v in lowhigh)
        if not low < high:
            raise SettingsError("The low intensity must be below the high one")
        result["lowhigh"] = [low, high]
    for key, (default, low, high) in NORMALIZE_NUMBERS.items():
        number = value.get(key, default)
        result[key] = _number(key, number, low, high, integer=key == "tile_norm_blocksize")
    invert = value.get("invert", False)
    if type(invert) is not bool:
        raise SettingsError("Choose whether to invert the image")
    result["invert"] = invert
    enabled = value.get("normalize", True)
    if type(enabled) is not bool:
        raise SettingsError("Choose whether to adjust the image brightness")
    result["normalize"] = enabled                      # False: Cellpose sees the raw intensities
    return result


def eval_kwargs(settings):
    """Keyword arguments for CellposeModel.eval from validated settings."""
    kwargs = {k: settings.get(k, NUMBERS[k][0]) for k in NUMBERS}
    kwargs["augment"] = settings.get("augment", False)
    kwargs["normalize"] = dict(settings.get("normalize") or NORMALIZE_DEFAULT)
    return kwargs


class _SafeUnpickler(pickle.Unpickler):
    """Only what a `_seg.npy` dict needs: numpy arrays/scalars and plain containers."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar"), ("builtins", "slice"), ("collections", "OrderedDict")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"{module}.{name} is not allowed in a mask file")


def read_seg(path):
    """The dict a Cellpose GUI `_seg.npy` holds, without running arbitrary pickled code. Its folder may be one others
    can write to, and the file may be swapped after it was listed: it is never read through a link, and it is opened
    without waiting, so a FIFO or a device in its place is refused before a byte is read."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0) | \
        getattr(os, "O_BINARY", 0)                       # (Windows has the last alone)
    with open(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("This is not a Cellpose mask file")
        version = np.lib.format.read_magic(stream)
        header = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if header is None:
            raise ValueError("This is not a Cellpose mask file")
        shape, _, dtype = header(stream, max_header_size=10000)
        if dtype != np.dtype(object) or shape != ():
            raise ValueError("This is not a Cellpose mask file")
        value = _SafeUnpickler(io.BytesIO(stream.read())).load()
    value = value.item() if isinstance(value, np.ndarray) else value
    if not isinstance(value, dict):
        raise ValueError("This is not a Cellpose mask file")
    return value


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _from_gui(data, models):
    found = {}
    path = data.get("model_path")
    if isinstance(path, (str, Path)) and str(path):
        name = Path(str(path)).name
        match = next((m for m in models if m["name"] == name), None)
        found["model"] = match["name"] if match else None
        found["model_path"] = str(path)
    for key in ("diameter", "flow_threshold", "cellprob_threshold"):
        value = _plain(data.get(key))
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            found[key] = None if key == "diameter" and value <= 0 else float(value)
    params = _plain(data.get("normalize_params"))
    if isinstance(params, dict):
        picked = {k: params[k] for k in ("percentile", "lowhigh", "sharpen_radius", "smooth_radius",
                                         "tile_norm_blocksize", "invert") if k in params}
        for key in ("percentile", "lowhigh"):
            if picked.get(key) is None:
                picked.pop(key, None)
        if picked.get("tile_norm_blocksize") is not None:
            picked["tile_norm_blocksize"] = int(picked["tile_norm_blocksize"])
        try:
            found["normalize"] = normalize(picked)
        except SettingsError:
            pass
    return found


MAX_SEG_BYTES = 64 * 1024 * 1024          # the patch reader's limit (zarr_bridge/stain_patches.read_seg)
_seen: dict[tuple, dict | None] = {}       # (path, size, mtime_ns) -> settings saved by the GUI, or None


def _saved_by_gui(path, models, info=None):
    """The settings in one GUI-saved file (None: not saved by the GUI, unreadable, a link or too large), cached by the
    file's size and time so repeated requests read each file once. `info`: the file's lstat, when it was taken."""
    try:
        info = os.lstat(path) if info is None else info
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    key = (str(path), info.st_size, info.st_mtime_ns)
    if key not in _seen:
        value = None
        if info.st_size <= MAX_SEG_BYTES:
            try:
                data = read_seg(path)
                if "manual_changes" in data:
                    value = data
            except (OSError, ValueError, pickle.UnpicklingError, EOFError):
                pass
        for old in [k for k in _seen if k[0] == key[0]]:
            del _seen[old]
        _seen[key] = None if value is None else {k: value.get(k) for k in GUI_KEYS}
    return None if _seen[key] is None else _from_gui(_seen[key], models)


def gui_settings(sources, models, images=None):
    """Settings saved by the Cellpose GUI in exported patch sets (one or several): the newest file's values, and where
    the files disagree.

    A source is a set folder of the earlier layout (a path: its `*_seg.npy` files, its patches.json beside them) or a
    set in a chosen folder, {folder, manifest, set, run}: only the masks its validated manifest lists, in that folder
    (other sets' files and files no set lists are not its). Each file is looked at once; one that cannot be is passed
    over.
    Only files the GUI saved count: its 2-D save always writes `manual_changes`, which the app's own exports never do
    (they may carry the run's settings, but those are not the user's). Returns None when nothing was saved there."""
    sources = [sources] if isinstance(sources, (str, Path, dict)) else sources
    files = []                                  # (mtime_ns, path, lstat, manifest, set, run)
    for source in sources:
        if isinstance(source, dict):
            folder, manifest = Path(source["folder"]), source.get("manifest") or {}
            paths = [folder / patch["mask"] for patch in manifest.get("patches") or []]
            where = (source.get("set"), source.get("run"))
        else:
            folder = Path(source)
            try:
                manifest = json.loads((folder / "patches.json").read_text())
            except (OSError, ValueError):
                manifest = {}
            manifest = manifest if isinstance(manifest, dict) else {}
            try:
                paths = list(folder.glob("*_seg.npy"))
            except OSError:
                paths = []
            where = (folder.name, folder.parent.name)
        for path in paths:
            try:
                info = os.lstat(path)
            except OSError:
                continue
            files.append((info.st_mtime_ns, path, info, manifest, *where))
    files.sort(key=lambda file: file[0], reverse=True)
    from .stain_levels import full_from_gui, offered
    seen, raw, resolutions, origins = [], {}, {}, {}
    for _, path, info, manifest, name, run in files:
        found = _saved_by_gui(path, models, info)
        if found is not None:
            raw[path], origins[path] = found, (name, run)
            image_key = manifest.get("image") or (manifest.get("settings") or {}).get("image")
            image_key = image_key.get("key") if isinstance(image_key, dict) else image_key
            image = next((im for im in images or [] if im["key"] == image_key), None)
            if image is None and images:
                image = images[0]
            choices = offered(image) if image else []
            full_pixel = choices[0]["um_per_px"] if choices else (image or {}).get("um_per_px", 1.)
            pixel = manifest.get("um_per_px") or (manifest.get("run_pixel_to_sample") or [0, 0, full_pixel])[2]
            converted = full_from_gui(found, pixel, full_pixel)
            match = min(choices, key=lambda c: abs(c["um_per_px"] / pixel - 1)) if choices else None
            if match and abs(match["um_per_px"] / pixel - 1) <= .05:
                converted["level"] = match["level"]
            resolutions[path] = {"level": manifest.get("level", 0), "factor": manifest.get("factor", 1.),
                                 "um_per_px": pixel}
            seen.append((path, converted))
    if not seen:
        return None
    newest_path, newest = seen[0]
    differences = {}
    for key in newest:
        values = {f"{path.parent.name}/{path.name}": s.get(key) for path, s in seen if key in s}
        if len({repr(v) for v in values.values()}) > 1:
            differences[key] = values
    unknown = newest.get("model") is None and "model_path" in newest
    return {"settings": newest, "raw": raw[newest_path], "resolution": resolutions[newest_path],
            "from": newest_path.name, "set": origins[newest_path][0],
            "run": origins[newest_path][1], "files": len(seen), "differences": differences,
            "unknown_model": newest.get("model_path") if unknown else None}
