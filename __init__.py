"""SoPA companion: stain segmentation and live proseg for SpatialData, on the desktop or from a notebook.

From a notebook: ``import sopa_companion; app = sopa_companion.launch(sdata)``. The notebook's names (api/) are imported
on first use, so `import sopa_companion` itself loads only the standard library: the desktop engine and the helpers in
other environments (python -m sopa_companion..., python -m sopa_companion.zarr_bridge.helper ...) never load the
notebook's side, and a kernel loads nothing of the engine.
"""
from __future__ import annotations

# 0.3.0: the first wheel without proseg-live inside it (the proseg-live package installs it) and with the
# sopa-companion command. pip leaves an installed wheel of the same version in place, so a wheel whose files change
# carries a higher version (tests/test_packaging.py LAST_SHIPPED). 0.4.0: the pink suspect test with nine of proseg's
# own settings and their "?" demos, Compare's own maps; it needs proseg-live 3.2.0.1 or newer (proseg's own count).
# 0.4.1: the first published release (one pip line from it), the mixing check's trainer logging each line before it
# sends it. 0.4.2: the mixing check waits for the room its sample's panel needs on the graphics card (not a fixed 6 GB),
# and says how much it needs and how much is free. 0.5.0: macOS on Apple silicon and Windows x64 besides Linux, for the
# alpha (portable.py; release.yml installs and runs the notebook route on all three). 0.5.1: the first Mac test's
# fixes: an export of training patches waits for its patches (900 s, not 60), and Cellpose's window opens on macOS and
# Windows (no X11 there). 0.5.2: the second Mac test's fixes: Apple's GPU is used only where a short test in a process
# of its own passes (PyTorch 2.15 on macOS 14 aborts in every linear layer: Cellpose then runs on the CPU, and its
# window starts with "use GPU" off), and a built-in Cellpose model is downloaded once, into Cellpose's own folder.
# 0.6.0: the mixing check learns on a Mac's own GPU (Apple's MPS) where it passes the probe's test, its models kept
# apart from a CUDA GPU's (equivalent results, not the same cells' numbers). 0.6.1: the install brings PyTorch 2.14 on
# macOS 14 with Apple silicon (2.15 needs macOS 15 for Apple's GPU; later macOS takes the newest), an export with the
# model's outlines is timed per patch and says each patch in its dialog, and the model list shows a spinner as it loads.
# 0.6.2: the first public release: the app's own code under the BSD 3-Clause licence (LICENSE; proseg-live stays GPL v3).
# 0.7.0: on PyPI, with proseg-live 3.2.0.3 (the same program, its package without the classifier that forbade the
# upload): pip install "sopa-companion[all]" is the whole install, and the release workflow publishes both there.
__version__ = "0.7.0"
PROTOCOL = "sopa-companion 1.0"
# What `sopa_companion.<name>` gives a notebook (api/__init__.py), and sopa's companion extra (sopa.companionapp).
_API = ("launch", "CompanionApp", "CompanionError", "CompanionStartError", "CompanionWarning", "SOPA_API")
__all__ = list(_API)


def __getattr__(name: str):
    if name in _API:
        from . import api
        value = getattr(api, name)
        globals()[name] = value              # later reads skip this function
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_API})
