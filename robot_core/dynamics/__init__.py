"""
robot_core/dynamics - Inverse Dynamics for Closed-Loop Mechanisms
============================================================

This package provides inverse dynamics functions for robots with
closed kinematic chains (e.g., parallelogram mechanisms).
"""

from . import constrained
from . import trajectory
from . import comparison

from .constrained import (
    compute_constrained_inverse_dynamics,
    correct_parallelogram_torques,
    compute_motor_inverse_dynamics
)
from .trajectory import (
    compute_constrained_inverse_dynamics_trajectory,
    compute_inverse_dynamics_trajectory
)
from .comparison import compare_dynamics_methods

__all__ = [
    'constrained',
    'trajectory',
    'comparison',
    'compute_constrained_inverse_dynamics',
    'correct_parallelogram_torques',
    'compute_motor_inverse_dynamics',
    'compute_constrained_inverse_dynamics_trajectory',
    'compute_inverse_dynamics_trajectory',
    'compare_dynamics_methods'
]
