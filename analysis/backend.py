"""What each backend of the analysis worker needs (analysis/worker.py pick_backend), looked up without importing it:
the engine asks the same of its own Python for [analysis] python = "auto". Standard library only, so the engine and the
worker read it before any of these packages is loaded."""
from __future__ import annotations

from importlib.util import find_spec

GPU_MODULES = ("cupy", "cuml", "cugraph", "cudf")
CPU_MODULES = ("numcodecs", "scipy", "sklearn", "igraph", "umap", "numba")
DISTRIBUTIONS = {"sklearn": "scikit-learn", "umap": "umap-learn"}   # where a module's name is not its package's
BACKEND_ENV = "SOPA_COMPANION_ANALYSIS_BACKEND"   # [analysis] backend as the engine passes it: auto | gpu | cpu
CPU_CACHE_ENV = "SOPA_COMPANION_CPU_CACHE"        # the folder numba keeps the CPU path's compiled code in


def missing(modules) -> list[str]:
    """The modules of `modules` this Python cannot import (find_spec; a module whose parent fails counts as missing).
    Found is not loadable: a package whose import fails is not missing here (the worker loads them before its hello)."""
    lacking = []
    for name in modules:
        try:
            found = find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            lacking.append(name)
    return lacking


def distributions(modules) -> list[str]:
    """Their package names, as pip and the page name them (DISTRIBUTIONS, else the module's name)."""
    return [DISTRIBUTIONS.get(name, name) for name in modules]
