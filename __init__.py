# src/local_qlbm/__init__.py

from .grid import LBMGrid
from .types import BoundaryCondition, RegionOfInterest, VelocitySet

# This defines what is available when someone does: from local_qlbm import *
__all__ = [
    "LBMGrid",
    "BoundaryCondition",
    "RegionOfInterest",
    "VelocitySet",
]
