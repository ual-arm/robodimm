"""
robot_core - Robot Model and Kinematics Core
==========================================

This package provides the core robot functionality using Pinocchio:

ROBOT MODELS:
- CR6: Spherical wrist with parallelogram elbow mechanism
- CR4: Palletizer with parallelogram and horizontal TCP

PARALLELOGRAM MECHANISM:
Both robots use a parallelogram linkage for the elbow joint. The motor (J3real)
is located at the shoulder but drives the elbow through a 4-bar linkage. This:
- Reduces arm inertia (motor mass stays at base)
- Improves dynamics and reduces J2 torque requirements
- Requires special handling in inverse dynamics (see correct_parallelogram_torques)

INTERNAL MODEL STRUCTURE:
  6-DOF: 9 internal DOF (6 active + 3 passive parallelogram joints)
  4-DOF: 8 internal DOF (4 active + 4 passive joints including J_aux)

ACTIVE JOINTS (require actuators):
  6-DOF: J1, J2, J3real→J3, J4, J5, J6
  4-DOF: J1, J2, J3real→J3, J4

SCALING:
  All dimensions scale linearly with the 'scale' parameter.
  Reference: 4-DOF arm length = 945mm at scale=1.0
             6-DOF arm length = 705mm at scale=1.0

Dependencies:
- pinocchio: Rigid body dynamics library
- hppfcl: Collision detection (for geometry primitives)
- pink: Inverse kinematics solver
- numpy: Numerical operations
"""

from .constants import (
    solve_constraints,
    ACTIVE_JOINTS_CR6,
    ACTIVE_JOINTS_CR4,
    PARALLELOGRAM_TORQUE_COMBINATION_CR6,
    PARALLELOGRAM_TORQUE_COMBINATION_CR4,
)

from .conversions import q_real_to_pink, q_real_to_robotstudio, q_pink_to_real

from .builders import (
    build_cr4_real,
    build_cr4_pink,
    build_cr6_real,
    build_cr6_pink,
    apply_cr4_motor_stator_masses,
)
from .builders.base import normalize_payload_inertia

from .kinematics import get_end_effector_position, run_ik_linear, run_ik_pose

from .interpolation import (
    interpolate_joint,
    interpolate_circular,
    interpolate_cartesian,
)

from .dynamics import (
    compute_constrained_inverse_dynamics,
    compute_constrained_inverse_dynamics_trajectory,
    compute_inverse_dynamics_trajectory,
    compare_dynamics_methods,
    correct_parallelogram_torques,
    compute_motor_inverse_dynamics,
)

from .actuators import (
    analyze_trajectory_requirements,
    select_actuators,
    get_actuator_masses,
)

import pinocchio as pin
import numpy as np
from typing import Dict, Any, Optional


def _expand_active_joint_params(
    robot_type: str, model: pin.Model, values, default: float = 0.0
):
    """Expand actuator-space vectors to full model.nv vectors."""
    nv_obj = getattr(model, "nv", 0)
    nv = int(nv_obj) if nv_obj is not None else 0
    if nv <= 0:
        return []

    def _new_default_array():
        return np.array([float(default) for _ in range(nv)], dtype=float)

    if values is None:
        return [float(default) for _ in range(nv)]

    arr = np.array(values, dtype=float).reshape(-1)
    if arr.size == nv:
        return arr.tolist()

    if robot_type == "CR4" and arr.size >= 4:
        out = _new_default_array()
        active_idx = [0, 1, 2, 7]
        out[active_idx] = arr[:4]
        return out.tolist()

    if robot_type == "CR6" and arr.size >= 6:
        out = _new_default_array()
        active_idx = [0, 1, 2, 6, 7, 8]
        out[active_idx] = arr[:6]
        return out.tolist()

    out = _new_default_array()
    upto = min(nv, arr.size)
    out[:upto] = arr[:upto]
    return out.tolist()


def build_robot(
    robot_type: str = "CR4",
    scale: float = 1.0,
    payload_kg: float = 0.0,
    payload_inertia: Optional[dict] = None,
    friction_coeffs: Optional[list] = None,
    reflected_inertia: Optional[list] = None,
    coulomb_friction: Optional[list] = None,
    motor_masses: Optional[list] = None,
    motor_layout: str = "concentric_j2_j3act",
    iref_model_mode: str = "diag",
    structural_mass_scale_exp: float = 3.0,
    structural_inertia_scale_exp: Optional[float] = None,
    visualize: bool = True,
):
    """
    Build robot model with all components.

    Parameters
    ----------
    robot_type : str
        "CR4" or "CR6"
    scale : float
        Scaling factor for all dimensions
    payload_kg : float
        Payload mass in kg at TCP
    payload_inertia : dict, optional
        Custom inertia tensor with keys 'Ixx', 'Iyy', 'Izz'
    friction_coeffs : list, optional
        Viscous friction coefficients per joint
    visualize : bool
        Whether to include visualization (default: True)

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': Pinocchio model
        - 'data': Pinocchio data
        - 'constraint_model': Parallelogram constraint model
        - 'model_pink': Pink model for IK
        - 'data_pink': Pink model data
        - 'geom_model': Geometry model
        - 'geom_data': Geometry data
        - 'viz': Visualization object (None if visualize=False)
        - 'q': Initial joint configuration
        - 'robot_type': Robot type
        - 'scale': Scale factor
        - 'payload_kg': Payload mass
        - 'payload_inertia': Payload inertia
        - 'friction_coeffs': Friction coefficients
        - 'ee_always_down': Whether EE is always down (CR4)
        - 'dimensions': Dictionary of dimensions
    """
    payload_inertia = normalize_payload_inertia(
        payload_kg=payload_kg,
        payload_inertia=payload_inertia,
        robot_type=robot_type,
    )

    if robot_type == "CR4":
        model, geom_model, constraint_model = build_cr4_real(
            scale,
            payload_kg,
            payload_inertia,
            structural_mass_scale_exp=structural_mass_scale_exp,
            structural_inertia_scale_exp=structural_inertia_scale_exp,
        )
        model_pink = build_cr4_pink(scale)
        ee_always_down = True
    else:  # Default CR6
        model, geom_model, constraint_model = build_cr6_real(
            scale, payload_kg, payload_inertia
        )
        model_pink = build_cr6_pink(scale)
        ee_always_down = False

    motor_masses_axis = None
    if robot_type == "CR4" and motor_masses is not None:
        mm = np.array(motor_masses, dtype=float).reshape(-1)
        if mm.size >= 5:
            motor_masses_axis = [float(mm[0]), float(mm[1]), float(mm[2]), float(mm[4])]
        elif mm.size >= 4:
            motor_masses_axis = [float(mm[0]), float(mm[1]), float(mm[2]), float(mm[3])]
        if motor_masses_axis is not None:
            apply_cr4_motor_stator_masses(model, motor_masses_axis, layout=motor_layout)

    if robot_type == "CR4" and payload_kg > 0.0:
        payload_data = payload_inertia or {}
        j4_id = model.getJointId("J4")
        I_payload = np.diag(
            [
                payload_data.get("Ixx", 0.0),
                payload_data.get("Iyy", 0.0),
                payload_data.get("Izz", 0.0),
            ]
        )
        com_payload = np.array(
            payload_data.get("com_from_tcp", [0.0, 0.0, 0.0]), dtype=float
        ).reshape(3)
        model.appendBodyToJoint(
            j4_id,
            pin.Inertia(float(payload_kg), np.zeros(3), I_payload),
            pin.SE3(np.eye(3), com_payload),
        )

    data = model.createData()
    geom_data = geom_model.createData()
    data_pink = model_pink.createData()

    # Initial configuration
    q = pin.neutral(model)
    # Solve constraints for initial q
    q, _ = solve_constraints(model, data, constraint_model, q)

    friction_coeffs = _expand_active_joint_params(
        robot_type, model, friction_coeffs, default=0.0
    )
    reflected_inertia = _expand_active_joint_params(
        robot_type, model, reflected_inertia, default=0.0
    )
    coulomb_friction = _expand_active_joint_params(
        robot_type, model, coulomb_friction, default=0.0
    )

    if robot_type == "CR4" and str(iref_model_mode or "diag").lower() in {
        "q5_physical",
        "q3_q5_physical",
    }:
        j4_id = model.getJointId("J4")
        j4_idx_v = model.joints[j4_id].idx_v
        I = np.array(model.inertias[j4_id].inertia, dtype=float)
        I[2, 2] += float(reflected_inertia[j4_idx_v])
        model.inertias[j4_id] = pin.Inertia(
            model.inertias[j4_id].mass,
            model.inertias[j4_id].lever,
            I,
        )
        reflected_inertia[j4_idx_v] = 0.0

    # Visualization (not implemented in this refactor)
    viz = None

    return {
        "model": model,
        "data": data,
        "constraint_model": constraint_model,
        "model_pink": model_pink,
        "data_pink": data_pink,
        "geom_model": geom_model,
        "geom_data": geom_data,
        "viz": viz,
        "q": q,
        "robot_type": robot_type,
        "scale": scale,
        "payload_kg": payload_kg,
        "payload_inertia": payload_inertia,
        "friction_coeffs": friction_coeffs,
        "reflected_inertia": reflected_inertia,
        "coulomb_friction": coulomb_friction,
        "motor_masses": motor_masses_axis,
        "motor_layout": motor_layout,
        "iref_model_mode": iref_model_mode,
        "structural_mass_scale_exp": structural_mass_scale_exp,
        "structural_inertia_scale_exp": structural_inertia_scale_exp,
        "ee_always_down": ee_always_down,
        "dimensions": {},  # Dimensions are now implicit in the model
    }


__all__ = [
    # Constants
    "solve_constraints",
    "ACTIVE_JOINTS_CR6",
    "ACTIVE_JOINTS_CR4",
    "PARALLELOGRAM_TORQUE_COMBINATION_CR6",
    "PARALLELOGRAM_TORQUE_COMBINATION_CR4",
    # Conversions
    "q_real_to_pink",
    "q_real_to_robotstudio",
    "q_pink_to_real",
    # Builders
    "build_cr4_real",
    "build_cr4_pink",
    "build_cr6_real",
    "build_cr6_pink",
    # Kinematics
    "get_end_effector_position",
    "run_ik_linear",
    "run_ik_pose",
    # Interpolation
    "interpolate_joint",
    "interpolate_circular",
    "interpolate_cartesian",
    # Dynamics
    "compute_constrained_inverse_dynamics",
    "compute_constrained_inverse_dynamics_trajectory",
    "compute_inverse_dynamics_trajectory",
    "compare_dynamics_methods",
    "correct_parallelogram_torques",
    "compute_motor_inverse_dynamics",
    # Actuators
    "analyze_trajectory_requirements",
    "select_actuators",
    "get_actuator_masses",
    "normalize_payload_inertia",
    # Main builder
    "build_robot",
]
