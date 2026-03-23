"""
robot_core.py - Compatibility Wrapper
====================================

This file provides backward compatibility by re-exporting all functions
from the new modular robot_core package.

The original robot_core.py has been refactored into a modular package structure:
- robot_core/constants.py: Constants and joint mappings
- robot_core/conversions.py: Joint configuration conversions
- robot_core/builders/: Robot model builders (CR4, CR6)
- robot_core/kinematics.py: Forward and inverse kinematics
- robot_core/interpolation.py: Trajectory interpolation
- robot_core/dynamics/: Inverse dynamics with constraints
- robot_core/actuators.py: Actuator selection

All imports from 'import robot_core' will continue to work without modification.
"""

# Re-export everything from the new modular package
from robot_core import *

# This ensures __all__ is properly propagated
__all__ = robot_core.__all__
