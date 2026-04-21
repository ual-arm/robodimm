"""
robot_core/builders/base.py - Common Builder Functions
================================================

This module provides common helper functions for building robot models.
"""

import pinocchio as pin
import hppfcl
import numpy as np


def _payload_box_defaults_cr4(payload_kg: float):
    base_mass = 30.0
    base_size = np.array([0.60, 0.40, 0.30], dtype=float)
    if payload_kg <= 0.0:
        size = base_size.copy()
    else:
        size = base_size * ((float(payload_kg) / base_mass) ** (1.0 / 3.0))
    com = np.array([0.25 * size[0], 0.0, 0.50 * size[2]], dtype=float)
    return size, com


def _payload_box_inertia_diag(
    payload_kg: float, box_size_xyz_m, com_from_tcp
) -> tuple[np.ndarray, np.ndarray]:
    size = np.asarray(box_size_xyz_m, dtype=float).reshape(3)
    com = np.asarray(com_from_tcp, dtype=float).reshape(3)
    sx, sy, sz = size.tolist()
    inertia_at_com = np.diag(
        [
            float(payload_kg) * (sy * sy + sz * sz) / 12.0,
            float(payload_kg) * (sx * sx + sz * sz) / 12.0,
            float(payload_kg) * (sx * sx + sy * sy) / 12.0,
        ]
    )
    shift = float(payload_kg) * ((np.dot(com, com) * np.eye(3)) - np.outer(com, com))
    inertia = inertia_at_com + shift
    return np.diag(inertia), size


def normalize_payload_inertia(
    payload_kg=0.0, payload_inertia=None, robot_type: str = "generic"
):
    if payload_kg <= 0.0:
        return None

    data = dict(payload_inertia or {})
    com_from_tcp_raw = data.get("com_from_tcp")
    com_from_tcp = (
        np.array(com_from_tcp_raw, dtype=float).reshape(3)
        if com_from_tcp_raw is not None
        else None
    )
    box_size = data.get("box_size_xyz_m") or data.get("size_xyz_m")

    if box_size is not None:
        box_size_arr = np.asarray(box_size, dtype=float).reshape(3)
        if com_from_tcp is None:
            _, com_from_tcp = _payload_box_defaults_cr4(float(payload_kg))
            com_from_tcp = np.array(
                [0.25 * box_size_arr[0], 0.0, 0.50 * box_size_arr[2]], dtype=float
            )
        inertia_diag, box_size_arr = _payload_box_inertia_diag(
            payload_kg, box_size_arr, com_from_tcp
        )
        return {
            "model": "box",
            "box_size_xyz_m": box_size_arr.tolist(),
            "com_from_tcp": com_from_tcp.tolist(),
            "Ixx": float(inertia_diag[0]),
            "Iyy": float(inertia_diag[1]),
            "Izz": float(inertia_diag[2]),
        }

    if all(k in data for k in ("Ixx", "Iyy", "Izz")):
        if com_from_tcp is None:
            com_from_tcp = np.zeros(3, dtype=float)
        return {
            "model": data.get("model", "explicit"),
            "com_from_tcp": com_from_tcp.tolist(),
            "Ixx": float(data.get("Ixx", 0.0)),
            "Iyy": float(data.get("Iyy", 0.0)),
            "Izz": float(data.get("Izz", 0.0)),
        }

    if str(robot_type).upper() == "CR4":
        default_size, default_com = _payload_box_defaults_cr4(float(payload_kg))
        if com_from_tcp is None:
            com_from_tcp = default_com
        inertia_diag, box_size_arr = _payload_box_inertia_diag(
            payload_kg, default_size, com_from_tcp
        )
        return {
            "model": "box_auto",
            "box_size_xyz_m": box_size_arr.tolist(),
            "com_from_tcp": com_from_tcp.tolist(),
            "Ixx": float(inertia_diag[0]),
            "Iyy": float(inertia_diag[1]),
            "Izz": float(inertia_diag[2]),
        }

    if com_from_tcp is None:
        com_from_tcp = np.zeros(3, dtype=float)
    r = (3.0 * float(payload_kg) / (4.0 * np.pi * 500.0)) ** (1.0 / 3.0)
    i_val = 0.4 * float(payload_kg) * r**2
    return {
        "model": "sphere_auto",
        "com_from_tcp": com_from_tcp.tolist(),
        "Ixx": float(i_val),
        "Iyy": float(i_val),
        "Izz": float(i_val),
    }


def add_geom(name, joint_id, placement, shape, color=None, geom_model=None):
    """
    Helper function to add geometry with color to a Pinocchio model.

    Parameters
    ----------
    name : str
        Name of the geometry object
    joint_id : int
        ID of the joint to attach geometry to
    placement : pin.SE3
        Placement (position + orientation) of the geometry
    shape : hppfcl.Shape
        Geometry shape (Cylinder, Box, Sphere, etc.)
    color : list or np.ndarray, optional
        RGBA color values [r, g, b, a]
    geom_model : pin.GeometryModel
        Geometry model to add object to

    Returns
    -------
    geom : pin.GeometryObject
        The created geometry object
    """
    geom = pin.GeometryObject(name, joint_id, placement, shape)
    if color is not None:
        geom.meshColor = np.array(color)
    if geom_model is not None:
        geom_model.addGeometryObject(geom)
    return geom


def get_rotation_matrices():
    """
    Get standard rotation matrices for cylinder orientation.

    Returns
    -------
    dict
        Dictionary with rotation matrices:
        - Rx: Z → X rotation
        - Ry: Z → Y rotation
        - Rnx: Z → -X rotation
    """
    Rx = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # Z → X
    Ry = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)  # Z → Y
    Rnx = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)  # Z → -X
    return {"Rx": Rx, "Ry": Ry, "Rnx": Rnx}


def create_tcp_axes(
    joint_id, scale=1.0, tcp_axis_len=0.15, tcp_axis_r=0.006, geom_model=None
):
    """
    Create TCP (Tool Center Point) axis visualization.

    Parameters
    ----------
    joint_id : int
        Joint ID to attach TCP axes to
    scale : float
        Scaling factor for dimensions
    tcp_axis_len : float
        Length of axis cylinders
    tcp_axis_r : float
        Radius of axis cylinders
    geom_model : pin.GeometryModel
        Geometry model to add axes to

    Returns
    -------
    None
        Axes are added directly to geom_model
    """
    R = get_rotation_matrices()
    tool0_placement = pin.SE3(pin.utils.rotate("y", np.pi), np.zeros(3))

    # X axis (red)
    add_geom(
        "tcp_x",
        joint_id,
        tool0_placement * pin.SE3(R["Rx"], np.array([tcp_axis_len / 2, 0, 0])),
        pin.hppfcl.Cylinder(tcp_axis_r * scale, tcp_axis_len * scale),
        [1, 0, 0, 1],
        geom_model,
    )

    # Y axis (green)
    add_geom(
        "tcp_y",
        joint_id,
        tool0_placement * pin.SE3(R["Ry"], np.array([0, tcp_axis_len / 2, 0])),
        pin.hppfcl.Cylinder(tcp_axis_r * scale, tcp_axis_len * scale),
        [0, 1, 0, 1],
        geom_model,
    )

    # Z axis (blue)
    add_geom(
        "tcp_z",
        joint_id,
        tool0_placement * pin.SE3(np.eye(3), np.array([0, 0, tcp_axis_len / 2])),
        pin.hppfcl.Cylinder(tcp_axis_r * scale, tcp_axis_len * scale),
        [0, 0, 1, 1],
        geom_model,
    )


def add_static_base(geom_model, scale=1.0):
    """
    Add static base geometry to the model.

    Parameters
    ----------
    geom_model : pin.GeometryModel
        Geometry model to add base to
    scale : float
        Scaling factor for dimensions
    """
    add_geom(
        "static_base",
        0,
        pin.SE3(np.eye(3), np.array([0, 0, 0.01 * scale])),
        pin.hppfcl.Cylinder(0.2 * scale, 0.02 * scale),
        [0.8, 0.8, 0.8, 1],
        geom_model,
    )


def create_parallelogram_constraint(model, j1p_id, j2p_id, L_ROD):
    """
    Create parallelogram constraint model for closed-loop mechanisms.

    Parameters
    ----------
    model : pin.Model
        The robot model
    j1p_id : int
        Joint ID for the crank end
    j2p_id : int
        Joint ID for the connecting rod end
    L_ROD : float
        Length of the connecting rod

    Returns
    -------
    constraint_model : pin.RigidConstraintModel
        The constraint model for the parallelogram
    """
    constraint_model = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_3D,
        model,
        j1p_id,
        pin.SE3(np.eye(3), np.array([0.0, 0.0, L_ROD])),
        j2p_id,
        pin.SE3.Identity(),
        pin.ReferenceFrame.LOCAL,
    )
    constraint_model.name = "parallelogram_closure"
    kp = 10.0
    kd = 2.0 * np.sqrt(kp)

    # Pinocchio exposes Baumgarte gains through different APIs across versions.
    if hasattr(constraint_model, "corrector"):
        constraint_model.corrector.Kp[:] = kp
        constraint_model.corrector.Kd[:] = kd
    elif hasattr(constraint_model, "m_baumgarte_parameters"):
        constraint_model.m_baumgarte_parameters.Kp = kp
        constraint_model.m_baumgarte_parameters.Kd = kd

    return constraint_model


def add_payload(
    model, joint_id, payload_kg=0.0, payload_inertia=None, robot_type: str = "generic"
):
    """
    Add payload inertia to a joint.

    Parameters
    ----------
    model : pin.Model
        The robot model
    joint_id : int
        Joint ID to attach payload to
    payload_kg : float
        Payload mass in kg
    payload_inertia : dict, optional
        Custom inertia tensor with keys 'Ixx', 'Iyy', 'Izz'

    Returns
    -------
    None
        Payload is added directly to the model
    """
    if payload_kg <= 0:
        return

    payload = normalize_payload_inertia(
        payload_kg=payload_kg,
        payload_inertia=payload_inertia,
        robot_type=robot_type,
    )
    if payload is None:
        return

    I_pay = pin.Inertia(
        float(payload_kg),
        np.zeros(3),
        np.diag([payload["Ixx"], payload["Iyy"], payload["Izz"]]),
    )
    placement = pin.SE3(
        np.eye(3), np.array(payload["com_from_tcp"], dtype=float).reshape(3)
    )
    model.appendBodyToJoint(joint_id, I_pay, placement)
