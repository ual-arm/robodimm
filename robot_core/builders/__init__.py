"""
robot_core/builders - Robot Model Builders
======================================

This package provides robot model builders for CR4 and CR6 robots.
"""

from . import base
from . import cr4
from . import cr6

from .cr4 import build_cr4_real, build_cr4_pink, apply_cr4_motor_stator_masses
from .cr6 import build_cr6_real, build_cr6_pink

__all__ = [
    "base",
    "cr4",
    "cr6",
    "build_cr4_real",
    "build_cr4_pink",
    "apply_cr4_motor_stator_masses",
    "build_cr6_real",
    "build_cr6_pink",
]
