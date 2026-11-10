"""Errors shared by the small public API (results are saved into SpatialData)."""
from __future__ import annotations


class CompanionError(RuntimeError):
    """The companion could not start or finish closing."""


class CompanionStartError(CompanionError):
    """The engine did not become ready; the message includes its last log lines."""


class CompanionWarning(UserWarning):
    """An element or an optional tool is unavailable in this session."""
