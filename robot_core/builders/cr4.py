"""
robot_core/builders/cr4.py - CR4 Robot Builders
===========================================

This module provides robot model builders for the CR4 4-DOF palletizer robot.
Uses standardized inertial parameters from geometric shapes.
"""

import pinocchio as pin
import numpy as np
from .base import (
    add_geom,
    get_rotation_matrices,
    create_tcp_axes,
    add_static_base,
    create_parallelogram_constraint,
    add_payload,
)
from ..inertial_params import (
    get_cr4_closed_loop_joint_inertial_params,
    get_cr4_serial5_reference_params,
)
from ..constants import solve_constraints


def _append_point_mass_at_parent_side(model, target_joint_id: int, mass: float) -> None:
    """Append a stator mass on the parent side of a target joint."""
    if mass <= 0.0:
        return
    parent_joint_id = model.parents[target_joint_id]
    if parent_joint_id < 0:
        return
    placement_parent_to_target = model.jointPlacements[target_joint_id]
    inertia_eps = pin.Inertia(float(mass), np.zeros(3), np.diag([1e-12, 1e-12, 1e-12]))
    model.appendBodyToJoint(parent_joint_id, inertia_eps, placement_parent_to_target)


def apply_cr4_motor_stator_masses(
    model: pin.Model, motor_masses_axis4, layout: str = "concentric_j2_j3act"
) -> None:
    """
    Apply CR4 stator masses with validated placement rules.

    motor_masses_axis4 order: [J1, J2, J3act, J4].
    J1 motor is on static base and is not added to moving links.
    """
    m = np.asarray(motor_masses_axis4, dtype=float).reshape(-1)
    if m.size < 4:
        return

    name_to_joint_id = {model.names[i]: i for i in range(1, len(model.names))}
    if layout not in {"concentric_j2_j3act", "serial_like"}:
        raise ValueError("layout must be 'concentric_j2_j3act' or 'serial_like'")

    # J4 motor on parent side of wrist joint.
    _append_point_mass_at_parent_side(model, name_to_joint_id["J4"], float(m[3]))

    if layout == "concentric_j2_j3act":
        # J2 and J3act concentric at shoulder.
        _append_point_mass_at_parent_side(model, name_to_joint_id["J2"], float(m[1]))
        _append_point_mass_at_parent_side(
            model, name_to_joint_id["J3real"], float(m[2])
        )
    else:
        # Serial-like placement: J3 motor around elbow branch.
        _append_point_mass_at_parent_side(model, name_to_joint_id["J2"], float(m[1]))
        _append_point_mass_at_parent_side(model, name_to_joint_id["J3"], float(m[2]))


def _static_dh_se3(a: float, alpha: float, d: float):
    ca, sa = np.cos(alpha), np.sin(alpha)
    R = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=float)
    return pin.SE3(R, np.array([a, 0.0, d], dtype=float))


def _build_cr4_serial_reference_model(
    scale: float,
    structural_mass_scale_exp: float = 3.0,
    structural_inertia_scale_exp: float | None = None,
):
    p = get_cr4_serial5_reference_params(
        scale,
        structural_mass_scale_exp=structural_mass_scale_exp,
        structural_inertia_scale_exp=structural_inertia_scale_exp,
    )
    dh_a = np.array([0.0, 0.540, 0.600, -0.125, 0.0], dtype=float) * float(scale)
    dh_alpha = np.array([-np.pi / 2, 0.0, 0.0, np.pi / 2, 0.0], dtype=float)
    dh_d = np.array([0.400, 0.0, 0.0, 0.0, 0.135], dtype=float) * float(scale)

    model = pin.Model()
    parent = 0
    joint_ids = []
    for i in range(5):
        placement = (
            pin.SE3.Identity()
            if i == 0
            else _static_dh_se3(dh_a[i - 1], dh_alpha[i - 1], dh_d[i - 1])
        )
        jid = model.addJoint(parent, pin.JointModelRZ(), placement, f"serial_j{i + 1}")
        joint_ids.append(jid)
        inertia = pin.Inertia(float(p["masses"][i]), p["coms"][i], p["inertias"][i])
        model.appendBodyToJoint(
            jid, inertia, _static_dh_se3(dh_a[i], dh_alpha[i], dh_d[i])
        )
        parent = jid
    return model, joint_ids


def _apply_serial_joint_transfer_inertias(
    model,
    constraint_model,
    scale: float,
    structural_mass_scale_exp: float = 3.0,
    structural_inertia_scale_exp: float | None = None,
) -> None:
    from ..conversions import q_pink_to_real

    serial_model, serial_joint_ids = _build_cr4_serial_reference_model(
        scale,
        structural_mass_scale_exp=structural_mass_scale_exp,
        structural_inertia_scale_exp=structural_inertia_scale_exp,
    )
    serial_data = serial_model.createData()
    data = model.createData()

    q_serial_home = np.array([0.0, -np.pi / 2, np.pi / 2, np.pi, 0.0], dtype=float)
    q_real_home = q_pink_to_real(
        "CR4", model, data, constraint_model, np.zeros(5, dtype=float)
    )

    pin.forwardKinematics(serial_model, serial_data, q_serial_home)
    pin.updateFramePlacements(serial_model, serial_data)
    pin.forwardKinematics(model, data, q_real_home)
    pin.updateFramePlacements(model, data)

    name_to_joint_id = {model.names[i]: i for i in range(1, len(model.names))}
    pairs = [
        ("J1", serial_joint_ids[0]),
        ("J2", serial_joint_ids[1]),
        ("J3", serial_joint_ids[2]),
        ("J_aux", serial_joint_ids[3]),
        ("J4", serial_joint_ids[4]),
    ]
    for real_name, serial_jid in pairs:
        real_jid = name_to_joint_id[real_name]
        I_serial = serial_model.inertias[serial_jid]
        T_rs = data.oMi[real_jid].inverse() * serial_data.oMi[serial_jid]
        R = T_rs.rotation
        t = T_rs.translation
        com_real = R @ I_serial.lever + t
        inertia_real = R @ I_serial.inertia @ R.T
        model.inertias[real_jid] = pin.Inertia(I_serial.mass, com_real, inertia_real)

    m_exp = float(structural_mass_scale_exp)
    i_exp = float(structural_inertia_scale_exp) if structural_inertia_scale_exp is not None else (m_exp + 2.0)
    light_I = np.diag([1e-5, 1e-5, 1e-5]) * (float(scale) ** i_exp)
    for name, mass_ref in (("J3real", 0.001), ("J1p", 0.001), ("J2p", 0.02)):
        jid = name_to_joint_id[name]
        model.inertias[jid] = pin.Inertia(
            mass_ref * (float(scale) ** m_exp), np.zeros(3), light_I
        )


def build_cr4_real(
    scale=1.0,
    payload_kg=0.0,
    payload_inertia=None,
    structural_mass_scale_exp: float = 3.0,
    structural_inertia_scale_exp: float | None = None,
):
    """
    Build the CR4 real robot model with parallelogram mechanism.

    Uses standardized inertial parameters from geometric shapes.

    Parameters
    ----------
    scale : float
        Scaling factor for all dimensions
    payload_kg : float
        Payload mass in kg at TCP
    payload_inertia : dict, optional
        Custom inertia tensor for payload

    Returns
    -------
    model : pin.Model
        The robot model
    geom_model : pin.GeometryModel
        The geometry model
    constraint_model : pin.RigidConstraintModel
        The parallelogram constraint model
    """
    model = pin.Model()
    model.name = "CR4"
    geom_model = pin.GeometryModel()

    # Dimensions (CR4 Specific - 1:1 Scale)
    L1_X = 0.0 * scale  # Concentric shoulder
    L1_Z = 0.400 * scale  # 400mm
    L2_Z = 0.540 * scale  # 540mm
    L_CRANK = 0.200 * scale  # 200mm
    L_ROD = 0.540 * scale  # Matches L2
    D_PARA = 0.200 * scale  # Parallelogram offset
    L3_X = 0.600 * scale  # 600mm
    L4_X = 0.125 * scale  # 125mm
    L4_Z = -0.135 * scale  # 135mm vertical drop

    inertial = get_cr4_closed_loop_joint_inertial_params(
        scale,
        structural_mass_scale_exp=structural_mass_scale_exp,
        structural_inertia_scale_exp=structural_inertia_scale_exp,
    )

    R = get_rotation_matrices()

    # Helper to create Inertia from tensor
    def make_inertia(mass, I_tensor, com):
        """Create pin.Inertia from mass, inertia tensor, and COM."""
        return pin.Inertia(mass, com, I_tensor)

    # --- Static Base (Universe) ---
    add_static_base(geom_model, scale)

    # --- J1: Base (Z) ---
    j1_id = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "J1")
    model.appendBodyToJoint(
        j1_id,
        make_inertia(
            inertial["J1"]["mass"], inertial["J1"]["inertia"], inertial["J1"]["com"]
        ),
        pin.SE3.Identity(),
    )
    add_geom(
        "L1",
        j1_id,
        pin.SE3.Identity(),
        pin.hppfcl.Cylinder(0.02 * scale, L1_Z),
        [0.9, 0.47, 0.08, 1.0],
        geom_model,
    )

    # --- J2: Shoulder (Y) ---
    j2_placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, L1_Z]))
    j2_id = model.addJoint(j1_id, pin.JointModelRY(), j2_placement, "J2")
    model.appendBodyToJoint(
        j2_id,
        make_inertia(
            inertial["J2"]["mass"], inertial["J2"]["inertia"], inertial["J2"]["com"]
        ),
        pin.SE3.Identity(),
    )
    add_geom(
        "L2",
        j2_id,
        pin.SE3(np.eye(3), np.array([0, 0, L2_Z / 2])),
        pin.hppfcl.Cylinder(0.02 * scale, L2_Z),
        [0.9, 0.47, 0.08, 1.0],
        geom_model,
    )

    # --- J3real: Elbow motor (Y) ---
    j3real_placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, L1_Z]))
    j3real_id = model.addJoint(j1_id, pin.JointModelRY(), j3real_placement, "J3real")
    model.appendBodyToJoint(
        j3real_id,
        make_inertia(
            inertial["J3real"]["mass"],
            inertial["J3real"]["inertia"],
            inertial["J3real"]["com"],
        ),
        pin.SE3.Identity(),
    )
    add_geom(
        "Crank",
        j3real_id,
        pin.SE3(R["Rnx"], np.array([-L_CRANK / 2, 0, 0])),
        pin.hppfcl.Cylinder(0.02 * scale, L_CRANK),
        [0.2, 0.5, 1.0, 1.0],
        geom_model,
    )

    # --- J1p: Crank (Y) ---
    j1p_placement = pin.SE3(np.eye(3), np.array([-L_CRANK, 0.0, 0.0]))
    j1p_id = model.addJoint(j3real_id, pin.JointModelRY(), j1p_placement, "J1p")
    model.appendBodyToJoint(
        j1p_id,
        make_inertia(
            inertial["J1p"]["mass"], inertial["J1p"]["inertia"], inertial["J1p"]["com"]
        ),
        pin.SE3.Identity(),
    )
    add_geom(
        "Rod",
        j1p_id,
        pin.SE3(np.eye(3), np.array([0, 0, L_ROD / 2])),
        pin.hppfcl.Cylinder(0.02 * scale, L_ROD),
        [0.2, 0.5, 1.0, 1.0],
        geom_model,
    )

    # --- J3: Elbow (Y) ---
    j3_placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, L2_Z]))
    j3_id = model.addJoint(j2_id, pin.JointModelRY(), j3_placement, "J3")
    model.appendBodyToJoint(
        j3_id,
        make_inertia(
            inertial["J3"]["mass"], inertial["J3"]["inertia"], inertial["J3"]["com"]
        ),
        pin.SE3.Identity(),
    )
    add_geom(
        "L3",
        j3_id,
        pin.SE3(R["Rnx"], np.array([L3_X / 2, 0, 0])),
        pin.hppfcl.Cylinder(0.02 * scale, L3_X),
        [1.0, 0.9, 0.0, 1.0],
        geom_model,
    )

    # --- J2p: Connecting rod (Y) ---
    j2p_placement = pin.SE3(np.eye(3), np.array([-D_PARA, 0.0, 0.0]))
    j2p_id = model.addJoint(j3_id, pin.JointModelRY(), j2p_placement, "J2p")
    model.appendBodyToJoint(
        j2p_id,
        make_inertia(
            inertial["J2p"]["mass"], inertial["J2p"]["inertia"], inertial["J2p"]["com"]
        ),
        pin.SE3.Identity(),
    )

    # --- J_aux: Passive joint to maintain horizontality (Y) ---
    j_aux_placement = pin.SE3(np.eye(3), np.array([L3_X, 0.0, 0.0]))
    j_aux_id = model.addJoint(j3_id, pin.JointModelRY(), j_aux_placement, "J_aux")
    model.appendBodyToJoint(
        j_aux_id,
        make_inertia(
            inertial["J_aux"]["mass"],
            inertial["J_aux"]["inertia"],
            inertial["J_aux"]["com"],
        ),
        pin.SE3.Identity(),
    )
    add_geom(
        "L4",
        j_aux_id,
        pin.SE3(np.eye(3), np.array([L4_X / 2, 0, L4_Z / 2])),
        pin.hppfcl.Box(L4_X, 0.05 * scale, abs(L4_Z)),
        [0.5, 0.5, 0.5, 1],
        geom_model,
    )

    # --- J4: TCP Rotation (Z) ---
    j4_placement = pin.SE3(np.eye(3), np.array([L4_X, 0.0, L4_Z]))
    j4_id = model.addJoint(j_aux_id, pin.JointModelRZ(), j4_placement, "J4")
    model.appendBodyToJoint(
        j4_id,
        make_inertia(
            inertial["J4"]["mass"], inertial["J4"]["inertia"], inertial["J4"]["com"]
        ),
        pin.SE3.Identity(),
    )

    # --- Frame TCP ---
    R_tcp = pin.utils.rotate("y", np.pi)  # Z pointing down
    tool0_placement = pin.SE3(R_tcp, np.zeros(3))
    model.addFrame(pin.Frame("tool0", j4_id, tool0_placement, pin.FrameType.OP_FRAME))
    model.addFrame(
        pin.Frame("end_effector", j4_id, tool0_placement, pin.FrameType.OP_FRAME)
    )

    # --- TCP Axes ---
    create_tcp_axes(j4_id, scale, geom_model=geom_model)

    # --- Parallelogram Constraint ---
    constraint_model = create_parallelogram_constraint(model, j1p_id, j2p_id, L_ROD)

    _apply_serial_joint_transfer_inertias(
        model,
        constraint_model,
        scale,
        structural_mass_scale_exp=structural_mass_scale_exp,
        structural_inertia_scale_exp=structural_inertia_scale_exp,
    )

    return model, geom_model, constraint_model


def build_cr4_pink(scale=1.0):
    """
    Build the CR4 Pink model (simplified serial chain for IK).

    Parameters
    ----------
    scale : float
        Scaling factor for all dimensions

    Returns
    -------
    model : pin.Model
        The simplified robot model
    """
    model = pin.Model()
    model.name = "CR4_Pink"

    # Dimensions (CR4 Specific - Same as above)
    L1_X = 0.0 * scale
    L1_Z = 0.400 * scale  # 400mm
    L2_Z = 0.540 * scale
    L3_X = 0.600 * scale
    L4_X = 0.125 * scale
    L4_Z = -0.135 * scale

    # J1
    j1_id = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "J1")
    model.appendBodyToJoint(
        j1_id, pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.1), pin.SE3.Identity()
    )

    # J2
    j2_pl = pin.SE3(np.eye(3), np.array([0, 0, L1_Z]))  # L1_X=0
    j2_id = model.addJoint(j1_id, pin.JointModelRY(), j2_pl, "J2")
    model.appendBodyToJoint(
        j2_id, pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.1), pin.SE3.Identity()
    )

    # J3 (Codo)
    j3_pl = pin.SE3(np.eye(3), np.array([0, 0, L2_Z]))
    j3_id = model.addJoint(j2_id, pin.JointModelRY(), j3_pl, "J3")
    model.appendBodyToJoint(
        j3_id, pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.1), pin.SE3.Identity()
    )

    # J_aux (J3p)
    jaux_pl = pin.SE3(np.eye(3), np.array([L3_X, 0, 0]))
    jaux_id = model.addJoint(j3_id, pin.JointModelRY(), jaux_pl, "J_aux")
    model.appendBodyToJoint(
        jaux_id, pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.1), pin.SE3.Identity()
    )

    # J4
    j4_pl = pin.SE3(np.eye(3), np.array([L4_X, 0, L4_Z]))
    j4_id = model.addJoint(jaux_id, pin.JointModelRZ(), j4_pl, "J4")
    model.appendBodyToJoint(
        j4_id, pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.1), pin.SE3.Identity()
    )

    # TCP Frame (for Pink tasks)
    R_tcp = pin.utils.rotate("y", np.pi)
    tool0_pl = pin.SE3(R_tcp, np.zeros(3))
    model.addFrame(pin.Frame("end_effector", j4_id, tool0_pl, pin.FrameType.OP_FRAME))

    return model
