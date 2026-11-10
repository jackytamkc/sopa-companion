"""Open the SoPA companion on a SpatialData sample."""
from .handle import CompanionApp
from .launch import launch
from .result import CompanionError, CompanionStartError, CompanionWarning

SOPA_API = 1

__all__ = ["launch", "CompanionApp", "CompanionError", "CompanionStartError", "CompanionWarning", "SOPA_API"]
