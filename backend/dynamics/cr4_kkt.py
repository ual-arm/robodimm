from __future__ import annotations
import math
import hashlib
import json
import threading
import numpy as np
import pinocchio as pin
from typing import Dict, List, Tuple, Any, Optional, Callable

from backend.dynamics.pinocchio_utils import (
    rigid_inertia,
    rod_x_inertia,
    cylinder_z_tensor,
    triangle_plate_inertia_xz_rotated_to_edge_frame,
    payload_inertia,
    distance,
    frame_rotation,
    local_point,
    unit,
    se3
)

# In-memory cache for built Pinocchio models to avoid rebuilding on every sample
_MODEL_CACHE: Dict[str, BuiltClosedModel] = {}
_MODEL_CACHE_LOCK = threading.Lock()

G = 9.80665

# These defaults are selected by the frozen common-step sensitivity ladder.
# Keep both operations explicit even though the selected values are equal.
DEFAULT_MAPPING_FD_STEP = 1e-4
DEFAULT_DIRECTIONAL_FD_STEP = 1e-4

POSITION_RESIDUAL_TARGET_M = 1e-8
VELOCITY_RESIDUAL_TARGET_M_S = 1e-8
ACCELERATION_RESIDUAL_TARGET_M_S2 = 1e-6
PASSIVE_TORQUE_RESIDUAL_TARGET_NM = 1e-8

# Extended precision is used only while finite-differencing the geometric
# user-to-cut-tree map. Pinocchio still receives float64 state vectors.
_FD_DTYPE = (
    np.longdouble
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
    else np.float64
)

# Default masses matching the moderated Robodimm/essay CR4 preset.
BODY_MASSES = {
    'SWING': 90.0,
    'P_ARM': 35.0,
    'LOWER_ARM': 75.0,
    'P_LINK': 25.0,
    'UPPER_ARM': 40.0,
    'LOWER_LINK': 20.0,
    'LINK_PLATE': 15.0,
    'UPPER_LINK': 15.0,
    'TILT': 15.0,
    'DISK': 10.0
}

class Cr4GeometryContext:
    def __init__(self, geometry: dict[str, list[float]]):
        self.points = {key: np.array(value, dtype=float) for key, value in geometry.items()}
        self.lengths = {
            "OB": distance(self.points, "O", "B"),
            "OC": distance(self.points, "O", "C"),
            "BP": distance(self.points, "B", "P"),
            "PC": distance(self.points, "P", "C"),
            "PH": distance(self.points, "P", "H"),
            "DE": distance(self.points, "D", "E"),
            "FG": distance(self.points, "F", "G"),
            "HG": distance(self.points, "H", "G"),
        }

class BuiltClosedModel:
    def __init__(
        self,
        model: pin.Model,
        constraints: list[pin.RigidConstraintModel],
        constraint_datas: list[pin.RigidConstraintData],
        joint_ids: dict[str, int],
        geom: Cr4GeometryContext
    ):
        self.model = model
        self.constraints = constraints
        self.constraint_datas = constraint_datas
        self.joint_ids = joint_ids
        self.geom = geom


def get_robot_hash(robot: dict) -> str:
    serialized = json.dumps(robot, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def body_mass(robot: dict[str, Any], body: str) -> float:
    inertial = robot.get("inertials", {}).get(body, {})
    return float(inertial.get("massKg", BODY_MASSES.get(body, 0.0)))


def get_body_inertia(
    robot: dict[str, Any],
    body: str,
    default_com: np.ndarray,
    default_inertia_tensor: np.ndarray
) -> pin.Inertia:
    """
    Constructs pin.Inertia using custom spec parameters if defined, otherwise falling back
    to computed/default geometric values.
    """
    inertial = robot.get("inertials", {}).get(body, {})
    mass = float(inertial.get("massKg", BODY_MASSES.get(body, 0.0)))
    
    # Custom COM override
    if "comM" in inertial and inertial["comM"] is not None:
        com = np.array(inertial["comM"], dtype=float)
    else:
        com = default_com
        
    # Custom Inertia Matrix override
    if "inertiaKgM2" in inertial and inertial["inertiaKgM2"] is not None:
        inertia_tensor = np.array(inertial["inertiaKgM2"], dtype=float)
    else:
        inertia_tensor = default_inertia_tensor
        
    return pin.Inertia(mass, com, inertia_tensor)


def j4_vertical_axis_frame_in_hgee(geom: Cr4GeometryContext) -> pin.SE3:
    parent_from_hgee = frame_rotation(geom.points["H"], geom.points["G"])
    vertical_parent = parent_from_hgee.T @ np.array([0.0, 0.0, 1.0])
    z_axis = unit(vertical_parent)
    x_seed = np.array([1.0, 0.0, 0.0])
    x_axis = x_seed - z_axis * float(x_seed @ z_axis)
    if np.linalg.norm(x_axis) <= 1e-12:
        x_axis = np.array([0.0, 1.0, 0.0]) - z_axis * float(np.array([0.0, 1.0, 0.0]) @ z_axis)
    x_axis = unit(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return pin.SE3(rotation, local_point(geom.points, "J4", "H", "G"))


def add_ry_body(model: pin.Model, parent: int, name: str, placement: pin.SE3, inertia: pin.Inertia) -> int:
    joint_id = model.addJoint(parent, pin.JointModelRY(), placement, name)
    model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())
    return joint_id


def build_closed_pinocchio_model(robot: dict[str, Any]) -> BuiltClosedModel:
    geom = Cr4GeometryContext(robot["geometry"])
    model = pin.Model()
    model.gravity.linear = np.array([0.0, 0.0, -G])
    joint_ids: dict[str, int] = {}

    # 1. J1 Swing Base Link
    j1 = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "J1")
    joint_ids["J1"] = j1
    o_from_a = geom.points["O"] - geom.points["A"]
    
    # Swing custom override with offset: com_pin = (O - A) + com_swing
    swing_mass = body_mass(robot, "SWING")
    default_swing_com = np.array([0.18, 0.0, 0.25])
    default_swing_inertia = cylinder_z_tensor(swing_mass, 0.045, 0.035)
    swing_inertia = get_body_inertia(robot, "SWING", default_swing_com, default_swing_inertia)
    
    # Shift COM by (O - A) in Pinocchio swing frame
    swing_inertia.lever = o_from_a + swing_inertia.lever
    model.appendBodyToJoint(j1, swing_inertia, pin.SE3.Identity())

    # 2. OB (P_ARM)
    ob_mass = body_mass(robot, "P_ARM")
    default_ob_com = np.array([geom.lengths["OB"] / 2.0, 0.0, 0.0])
    default_ob_inertia = rod_x_inertia(ob_mass, geom.lengths["OB"]).inertia
    ob_inertia = get_body_inertia(robot, "P_ARM", default_ob_com, default_ob_inertia)
    ob = add_ry_body(model, j1, "OB", se3(o_from_a), ob_inertia)

    # 3. OC (LOWER_ARM)
    oc_mass = body_mass(robot, "LOWER_ARM")
    default_oc_com = np.array([geom.lengths["OC"] / 2.0, 0.0, 0.0])
    default_oc_inertia = rod_x_inertia(oc_mass, geom.lengths["OC"]).inertia
    oc_inertia = get_body_inertia(robot, "LOWER_ARM", default_oc_com, default_oc_inertia)
    oc = add_ry_body(model, j1, "OC", se3(o_from_a), oc_inertia)

    # 4. BP (P_LINK)
    bp_mass = body_mass(robot, "P_LINK")
    default_bp_com = np.array([geom.lengths["BP"] / 2.0, 0.0, 0.0])
    default_bp_inertia = rod_x_inertia(bp_mass, geom.lengths["BP"]).inertia
    bp_inertia = get_body_inertia(robot, "P_LINK", default_bp_com, default_bp_inertia)
    bp = add_ry_body(model, ob, "BP", se3([geom.lengths["OB"], 0.0, 0.0]), bp_inertia)

    # 5. PCH (UPPER_ARM) - Joint origin is at P. So CAD COM (rel to C) is translated: com_pin = [L_PC, 0, 0] + com_cad
    pch_mass = body_mass(robot, "UPPER_ARM")
    # Default midpoint rel to C is (L_CH - L_PC)/2. rel to P is PH/2 = (L_CH + L_PC)/2
    default_pch_com_cad = np.array([(geom.lengths["PH"] - 2 * geom.lengths["PC"]) / 2.0, 0.0, 0.0]) # rel to C
    default_pch_inertia = rod_x_inertia(pch_mass, geom.lengths["PH"]).inertia
    pch_inertia_cad = get_body_inertia(robot, "UPPER_ARM", default_pch_com_cad, default_pch_inertia)
    # Translate COM relative to joint origin P:
    pch_inertia_cad.lever = np.array([geom.lengths["PC"], 0.0, 0.0]) + pch_inertia_cad.lever
    pch = add_ry_body(model, bp, "PCH", se3([geom.lengths["BP"], 0.0, 0.0]), pch_inertia_cad)

    # 6. DE (LOWER_LINK)
    de_mass = body_mass(robot, "LOWER_LINK")
    default_de_com = np.array([geom.lengths["DE"] / 2.0, 0.0, 0.0])
    default_de_inertia = rod_x_inertia(de_mass, geom.lengths["DE"]).inertia
    de_inertia = get_body_inertia(robot, "LOWER_LINK", default_de_com, default_de_inertia)
    de = add_ry_body(model, j1, "DE", se3(geom.points["D"] - geom.points["A"]), de_inertia)

    # 7. CEF (LINK_PLATE) - Triangle vertices C, E, F in CAD frame
    cef_mass = body_mass(robot, "LINK_PLATE")
    vertices_cef = np.array([
        [0.0, 0.0],
        (geom.points["E"] - geom.points["C"])[[0, 2]],
        (geom.points["F"] - geom.points["C"])[[0, 2]]
    ], dtype=float)
    default_cef_inertia = triangle_plate_inertia_xz_rotated_to_edge_frame(
        cef_mass,
        vertices_xz=vertices_cef,
        canonical_from_edge=frame_rotation(geom.points["C"], geom.points["E"])
    )
    # CEF CAD frame has origin at C, aligned with C->E at q=0, which matches child joint frame
    cef_inertia = get_body_inertia(robot, "LINK_PLATE", default_cef_inertia.lever, default_cef_inertia.inertia)
    cef = add_ry_body(model, oc, "CEF", se3([geom.lengths["OC"], 0.0, 0.0]), cef_inertia)

    # 8. FG (UPPER_LINK)
    fg_mass = body_mass(robot, "UPPER_LINK")
    default_fg_com = np.array([geom.lengths["FG"] / 2.0, 0.0, 0.0])
    default_fg_inertia = rod_x_inertia(fg_mass, geom.lengths["FG"]).inertia
    fg_inertia = get_body_inertia(robot, "UPPER_LINK", default_fg_com, default_fg_inertia)
    fg = add_ry_body(model, cef, "FG", se3(local_point(geom.points, "F", "C", "E")), fg_inertia)

    # 9. HGEE (TILT) - Triangle vertices H, J4, G in CAD frame
    tilt_mass = body_mass(robot, "TILT")
    vertices_tilt = np.array([
        [0.0, 0.0],
        (geom.points["J4"] - geom.points["H"])[[0, 2]],
        (geom.points["G"] - geom.points["H"])[[0, 2]]
    ], dtype=float)
    default_tilt_inertia = triangle_plate_inertia_xz_rotated_to_edge_frame(
        tilt_mass,
        vertices_xz=vertices_tilt,
        canonical_from_edge=frame_rotation(geom.points["H"], geom.points["G"])
    )
    tilt_inertia = get_body_inertia(robot, "TILT", default_tilt_inertia.lever, default_tilt_inertia.inertia)
    hgee = add_ry_body(model, pch, "HGEE", se3([geom.lengths["PH"], 0.0, 0.0]), tilt_inertia)

    # 10. DISK (J4 Disk)
    disk_mass = body_mass(robot, "DISK")
    default_disk_inertia = cylinder_z_tensor(disk_mass, 0.045, 0.040)
    disk_inertia = get_body_inertia(robot, "DISK", np.zeros(3), default_disk_inertia)
    hgee_from_j4 = j4_vertical_axis_frame_in_hgee(geom)
    j4 = model.addJoint(hgee, pin.JointModelRZ(), hgee_from_j4, "J4")
    model.appendBodyToJoint(j4, disk_inertia, pin.SE3.Identity())

    # 11. Payload
    payload = robot.get("payload", {})
    payload_mass = float(payload.get("massKg", 0.0))
    if payload_mass > 0.0:
        tcp_from_j4_hgee = local_point(geom.points, "TCP", "H", "G") - local_point(geom.points, "J4", "H", "G")
        tcp_from_j4 = hgee_from_j4.rotation.T @ tcp_from_j4_hgee
        payload_com = np.array(payload.get("comM") or [0.0, 0.0, 0.0], dtype=float)
        p_inertia = payload_inertia(payload_mass, payload_com, payload.get("inertiaKgM2"))
        model.appendBodyToJoint(j4, p_inertia, se3(tcp_from_j4))

    joint_ids.update({
        "OB": ob, "OC": oc, "BP": bp, "PCH": pch,
        "DE": de, "CEF": cef, "FG": fg, "HGEE": hgee, "J4": j4
    })

    # Link loop closure constraints
    constraints = [
        pin.RigidConstraintModel(pin.ContactType.CONTACT_3D, model, pch, se3([geom.lengths["PC"], 0.0, 0.0]), oc, se3([geom.lengths["OC"], 0.0, 0.0])),
        pin.RigidConstraintModel(pin.ContactType.CONTACT_3D, model, cef, se3(local_point(geom.points, "E", "C", "E")), de, se3([geom.lengths["DE"], 0.0, 0.0])),
        pin.RigidConstraintModel(pin.ContactType.CONTACT_3D, model, hgee, se3(local_point(geom.points, "G", "H", "G")), fg, se3([geom.lengths["FG"], 0.0, 0.0])),
    ]

    return BuiltClosedModel(model, constraints, [constraint.createData() for constraint in constraints], joint_ids, geom)


def get_or_build_model(robot: dict[str, Any]) -> BuiltClosedModel:
    robot_hash = get_robot_hash(robot)
    cached = _MODEL_CACHE.get(robot_hash)
    if cached is not None:
        return cached
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(robot_hash)
        if cached is None:
            cached = build_closed_pinocchio_model(robot)
            _MODEL_CACHE[robot_hash] = cached
    return cached


def _unit_preserving_precision(vector: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.sum(vector * vector))
    if norm <= 1e-12:
        raise ValueError("Zero-length vector")
    return vector / norm


def _frame_rotation_preserving_precision(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    x_axis = _unit_preserving_precision(second - first)
    y_seed = np.array([0.0, 1.0, 0.0], dtype=x_axis.dtype)
    z_axis = _unit_preserving_precision(np.cross(x_axis, y_seed))
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def closed_chain_points(geom: Cr4GeometryContext, j2: float, j3: float) -> dict[str, np.ndarray]:
    home = geom.points
    dtype = np.result_type(j2, j3, np.float64)
    points = {key: value.astype(dtype, copy=True) for key, value in home.items()}
    o = points["O"]
    
    # 2D Rotations around -Y (Y is vertical-right, standard Pinocchio RY rotation)
    c2, s2 = np.cos(j2), np.sin(j2)
    ry2 = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]])
    c3, s3 = np.cos(j3), np.sin(j3)
    ry3 = np.array([[c3, 0.0, s3], [0.0, 1.0, 0.0], [-s3, 0.0, c3]])
    
    points["C"] = o + ry2 @ (home["C"] - o)
    points["B"] = o + ry3 @ (home["B"] - o)
    points["P"] = points["B"] + (points["C"] - o)
    points["E"] = points["D"] + (points["C"] - o)
    
    cp = points["C"] - points["P"]
    points["H"] = points["P"] + _unit_preserving_precision(cp) * geom.lengths["PH"]
    
    rot_ce = _frame_rotation_preserving_precision(points["C"], points["E"])
    points["F"] = points["C"] + rot_ce @ local_point(home, "F", "C", "E")
    
    # linkage circle intersection for point G
    pref_side = (home["H"][0] - home["F"][0]) * (home["G"][2] - home["F"][2]) - (home["H"][2] - home["F"][2]) * (home["G"][0] - home["F"][0])
    points["G"] = circle_intersection_xz(
        points["F"], geom.lengths["FG"], 
        points["H"], geom.lengths["HG"], 
        home["G"], pref_side
    )
    
    rot_hg = _frame_rotation_preserving_precision(points["H"], points["G"])
    points["J4"] = points["H"] + rot_hg @ local_point(home, "J4", "H", "G")
    points["EE"] = points["H"] + rot_hg @ local_point(home, "EE", "H", "G")
    points["TCP"] = points["H"] + rot_hg @ local_point(home, "TCP", "H", "G")
    return points


def circle_intersection_xz(
    center_a: np.ndarray, radius_a: float, 
    center_b: np.ndarray, radius_b: float, 
    prefer: np.ndarray, prefer_side: float
) -> np.ndarray:
    delta = center_b - center_a
    dxz = np.array([delta[0], delta[2]])
    dist = np.sqrt(np.sum(dxz * dxz))
    if dist <= 1e-12:
        raise ValueError("Linkage circle centers are coincident")
    a = (radius_a * radius_a - radius_b * radius_b + dist * dist) / (2.0 * dist)
    h = np.sqrt(np.maximum(radius_a * radius_a - a * a, 0.0))
    ex = dxz / dist
    base = np.array([center_a[0] + a * ex[0], center_a[2] + a * ex[1]])
    perp = np.array([-ex[1], ex[0]])
    candidates = (base + h * perp, base - h * perp)
    side_sign = np.sign(prefer_side)
    filtered = []
    for candidate in candidates:
        area = dxz[0] * (candidate[1] - center_a[2]) - dxz[1] * (candidate[0] - center_a[0])
        if side_sign == 0.0 or np.sign(area) == side_sign:
            filtered.append(candidate)
    if not filtered:
        filtered = list(candidates)
    preferred = np.array([prefer[0], prefer[2]])
    best = min(
        filtered,
        key=lambda candidate: float(np.sum((candidate - preferred) ** 2)),
    )
    return np.array([best[0], 0.0, best[1]], dtype=best.dtype)


def angle(a: np.ndarray, b: np.ndarray) -> np.floating:
    d = b - a
    return np.arctan2(d[2], d[0])


def closed_full_configuration(geom: Cr4GeometryContext, q_user: np.ndarray) -> np.ndarray:
    _j1, j2, j3, j4 = q_user
    points = closed_chain_points(geom, j2, j3)
    theta_ob = angle(points["O"], points["B"])
    theta_oc = angle(points["O"], points["C"])
    theta_bp = angle(points["B"], points["P"])
    theta_pch = angle(points["P"], points["H"])
    theta_de = angle(points["D"], points["E"])
    theta_cef = angle(points["C"], points["E"])
    theta_fg = angle(points["F"], points["G"])
    theta_hgee = angle(points["H"], points["G"])
    return np.array([
        q_user[0],
        -theta_ob,
        -theta_oc,
        -(theta_bp - theta_ob),
        -(theta_pch - theta_bp),
        -theta_de,
        -(theta_cef - theta_oc),
        -(theta_fg - theta_cef),
        -(theta_hgee - theta_pch),
        q_user[3],
    ], dtype=np.result_type(q_user.dtype, np.float64))


def _fd_step(options: Dict[str, Any], name: str, default: float) -> float:
    """Read and validate one finite-difference step from solver options."""
    value = options.get(name, default)
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def resolve_fd_steps(options: Optional[Dict[str, Any]]) -> tuple[float, float]:
    """Resolve mapping and directional FD steps, retaining legacy defaults.

    ``fd_step`` is a convenience for the common-step sensitivity campaign.
    Explicit operation-specific options take precedence, which also permits
    reproducing the historical 1e-6/1e-5 pair.
    """
    values = options or {}
    common = values.get("fd_step")
    mapping_default = DEFAULT_MAPPING_FD_STEP if common is None else common
    directional_default = DEFAULT_DIRECTIONAL_FD_STEP if common is None else common
    mapping_step = _fd_step(values, "mapping_fd_step", mapping_default)
    directional_step = _fd_step(values, "directional_fd_step", directional_default)
    return mapping_step, directional_step


def mapped_jacobian(
    map_fn: Callable[[np.ndarray], np.ndarray],
    q_user: np.ndarray,
    fd_step: float = DEFAULT_MAPPING_FD_STEP,
) -> np.ndarray:
    fd_step = float(fd_step)
    if not np.isfinite(fd_step) or fd_step <= 0.0:
        raise ValueError("fd_step must be a finite positive number")
    q0 = np.asarray(q_user, dtype=_FD_DTYPE)
    f0 = map_fn(q0)
    jacobian = np.zeros((f0.size, q0.size))
    for index in range(q0.size):
        step = np.zeros_like(q0)
        step[index] = fd_step
        pair = np.unwrap(np.vstack((map_fn(q0 - step), map_fn(q0 + step))), axis=0)
        jacobian[:, index] = (pair[1] - pair[0]) / (2.0 * fd_step)
    return jacobian


def mapped_state(
    map_fn: Callable[[np.ndarray], np.ndarray],
    q_user: np.ndarray,
    qd_user: np.ndarray,
    qdd_user: np.ndarray,
    mapping_fd_step: float = DEFAULT_MAPPING_FD_STEP,
    directional_fd_step: float = DEFAULT_DIRECTIONAL_FD_STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q0 = np.asarray(q_user, dtype=_FD_DTYPE)
    qd = np.asarray(qd_user, dtype=_FD_DTYPE)
    qdd = np.asarray(qdd_user, dtype=_FD_DTYPE)
    f0 = map_fn(q0)
    jacobian = mapped_jacobian(map_fn, q0, mapping_fd_step)
    jacobian_plus = mapped_jacobian(
        map_fn, q0 + directional_fd_step * qd, mapping_fd_step
    )
    jacobian_minus = mapped_jacobian(
        map_fn, q0 - directional_fd_step * qd, mapping_fd_step
    )
    jdot_qd = ((jacobian_plus - jacobian_minus) / (2.0 * directional_fd_step)) @ qd
    return f0, jacobian @ qd, jacobian @ qdd + jdot_qd


def closed_torque_to_user(
    geom: Cr4GeometryContext,
    q_user: np.ndarray,
    tau_actuated_pin: np.ndarray,
    mapping_fd_step: float = DEFAULT_MAPPING_FD_STEP,
) -> np.ndarray:
    def actuated_cut_configuration(value: np.ndarray) -> np.ndarray:
        q_closed = closed_full_configuration(geom, value)
        return q_closed[[0, 1, 2, 9]]

    jacobian = mapped_jacobian(actuated_cut_configuration, q_user, mapping_fd_step)
    tau_user = jacobian.T @ tau_actuated_pin
    # Simscape reports J4 actuation torque with the opposite sign after vertical re-alignment
    tau_user[3] *= -1.0
    return tau_user


def _constraint_kinematics(
    built: BuiltClosedModel,
    q_closed: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the analytic CONTACT_3D Jacobian and actual contact positions.

    Pinocchio's ``getConstraintsJacobian`` is deliberately used without row
    filtering.  The three CONTACT_3D constraints therefore always contribute
    their nine analytic rows, including rows that may be numerically small at
    a particular configuration.
    """
    model = built.model
    data = model.createData()
    constraint_datas = [constraint.createData() for constraint in built.constraints]

    pin.forwardKinematics(model, data, q_closed)
    pin.computeJointJacobians(model, data, q_closed)
    data.q_in = q_closed
    for constraint_model, constraint_data in zip(built.constraints, constraint_datas):
        constraint_model.calc(model, data, constraint_data)

    jacobian_c = np.asarray(
        pin.getConstraintsJacobian(model, data, built.constraints, constraint_datas),
        dtype=float,
    )
    expected_shape = (3 * len(built.constraints), model.nv)
    if jacobian_c.shape != expected_shape:
        raise ValueError(
            "Pinocchio CONTACT_3D Jacobian has unexpected shape "
            f"{jacobian_c.shape}; expected {expected_shape}"
        )

    # oMc1/oMc2 are the world placements of the two actual contact frames
    # after calc(), rather than reconstructed hardpoint coordinates.
    position_residuals = [
        np.asarray(constraint_data.oMc1.translation - constraint_data.oMc2.translation, dtype=float)
        for constraint_data in constraint_datas
    ]
    return jacobian_c, position_residuals


def _constraint_jacobian_directional_derivative(
    built: BuiltClosedModel,
    q_closed: np.ndarray,
    v_closed: np.ndarray,
    fd_step: float,
) -> np.ndarray:
    """Differentiate the analytic constraint Jacobian along the actual state."""
    jacobian_plus, _ = _constraint_kinematics(
        built, q_closed + fd_step * v_closed
    )
    jacobian_minus, _ = _constraint_kinematics(
        built, q_closed - fd_step * v_closed
    )
    return (jacobian_plus - jacobian_minus) / (2.0 * fd_step)


def compute_cr4_kkt_dynamics(
    robot_spec: Dict[str, Any],
    q: List[float],
    qd: List[float],
    qdd: List[float],
    options: Dict[str, Any] = None
) -> Tuple[List[float], List[float], Dict[str, Any], List[str]]:
    """
    Computes inverse dynamics for a single sample of CR4 using closed-chain KKT.
    Returns:
        tauNm: computed active joint torques in user space
        powerW: computed joint power
        diagnostics: dictionary containing KKT diagnostics
        warnings: warning messages generated during calculation
    """
    options = options or {}
    mapping_fd_step, directional_fd_step = resolve_fd_steps(options)

    built = get_or_build_model(robot_spec)
    model = built.model
    data = model.createData()

    q_user = np.array(q, dtype=float)
    qd_user = np.array(qd, dtype=float)
    qdd_user = np.array(qdd, dtype=float)

    # 1. State Mapping: User Space -> Pinocchio Cut Tree space
    q_closed, v_closed, a_closed = mapped_state(
        lambda val: closed_full_configuration(built.geom, val), 
        q_user,
        qd_user,
        qdd_user,
        mapping_fd_step=mapping_fd_step,
        directional_fd_step=directional_fd_step,
    )
    q_closed = np.asarray(q_closed, dtype=np.float64)
    v_closed = np.asarray(v_closed, dtype=np.float64)
    a_closed = np.asarray(a_closed, dtype=np.float64)

    # 2. Pinocchio Open-Loop dynamics
    tau_open = np.asarray(pin.rnea(model, data, q_closed, v_closed, a_closed), dtype=float)

    # 3. Kinematic constraints calculation.  Pinocchio supplies the analytic
    # 9x10 CONTACT_3D Jacobian; no small-row filtering is applied.
    pin.forwardKinematics(model, data, q_closed, v_closed, a_closed)
    pin.computeJointJacobians(model, data, q_closed)
    data.q_in = q_closed

    constraint_datas = [constraint.createData() for constraint in built.constraints]
    for constraint_model, constraint_data in zip(built.constraints, constraint_datas):
        constraint_model.calc(model, data, constraint_data)
        
    jacobian_c = np.asarray(
        pin.getConstraintsJacobian(model, data, built.constraints, constraint_datas),
        dtype=float,
    )
    expected_jacobian_shape = (9, model.nv)
    if jacobian_c.shape != expected_jacobian_shape:
        raise ValueError(
            "Pinocchio CONTACT_3D Jacobian has unexpected shape "
            f"{jacobian_c.shape}; expected {expected_jacobian_shape}"
        )

    position_residuals = [
        np.asarray(constraint_data.oMc1.translation - constraint_data.oMc2.translation, dtype=float)
        for constraint_data in constraint_datas
    ]
    velocity_residual = jacobian_c @ v_closed
    constraint_jdot = _constraint_jacobian_directional_derivative(
        built, q_closed, v_closed, directional_fd_step
    )
    acceleration_residual = jacobian_c @ a_closed + constraint_jdot @ v_closed

    # 4. KKT system solver: solve Lagrange multipliers forcing passive joint torques to zero
    actuated = [0, 1, 2, model.joints[built.joint_ids["J4"]].idx_v]
    passive = [idx for idx in range(model.nv) if idx not in actuated]

    jacobian_p = jacobian_c[:, passive]
    solved_system = jacobian_p.T
    singular_values = np.linalg.svd(jacobian_p, compute_uv=False)
    solved_singular_values = np.linalg.svd(solved_system, compute_uv=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    rank_tolerance = max(solved_system.shape) * np.finfo(np.float64).eps * sigma_max
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    if sigma_max > 0.0:
        lstsq_rcond = rank_tolerance / sigma_max
    else:
        lstsq_rcond = 0.0
    lambdas = np.linalg.lstsq(
        solved_system, -tau_open[passive], rcond=lstsq_rcond
    )[0]
    
    # Restored torques
    tau_restored = tau_open + jacobian_c.T @ lambdas
    tau_actuated = tau_restored[actuated]

    # 5. Torque Projection back to User Space
    tau_user = np.asarray(
        closed_torque_to_user(
            built.geom, q_user, tau_actuated, mapping_fd_step=mapping_fd_step
        ),
        dtype=np.float64,
    )

    # Add joint viscous friction in user space
    limits_by_name = {limit["name"]: limit for limit in robot_spec.get("limits", [])}
    for i, name in enumerate(["J1", "J2", "J3", "J4"]):
        limit = limits_by_name.get(name, {})
        friction_coeff = float(limit.get("frictionCoeffNmSPerRad", 0.0))
        tau_user[i] += friction_coeff * qd_user[i]

    power_user = tau_user * qd_user

    # 6. Diagnostics metrics.  The passive residual is evaluated from the
    # solved passive-torque equation and is not the old algebraic zero check
    # on the actuated torque reconstruction.
    passive_residual_vector = tau_open[passive] + solved_system @ lambdas
    passive_residual = np.linalg.norm(passive_residual_vector)
    condition: float | str = (
        float(solved_singular_values[0] / solved_singular_values[-1])
        if rank == len(passive) and solved_singular_values.size == len(passive)
        else "infinity"
    )
    position_norms = [float(np.linalg.norm(residual)) for residual in position_residuals]
    velocity_norm = float(np.linalg.norm(velocity_residual))
    acceleration_norm = float(np.linalg.norm(acceleration_residual))

    diagnostic_failures = []
    finite_arrays = (
        position_residuals,
        velocity_residual,
        acceleration_residual,
        passive_residual_vector,
        singular_values,
    )
    if not all(np.all(np.isfinite(value)) for value in finite_arrays):
        diagnostic_failures.append("one or more diagnostic values are non-finite")
    if isinstance(condition, float) and not math.isfinite(condition):
        diagnostic_failures.append("solved-system condition number is non-finite")
    if rank != len(passive):
        diagnostic_failures.append(f"Jc_passive rank is {rank}; expected {len(passive)}")
    if max(position_norms, default=0.0) > POSITION_RESIDUAL_TARGET_M:
        diagnostic_failures.append(
            f"position residual exceeds {POSITION_RESIDUAL_TARGET_M:.0e} m"
        )
    if velocity_norm > VELOCITY_RESIDUAL_TARGET_M_S:
        diagnostic_failures.append(
            f"velocity residual exceeds {VELOCITY_RESIDUAL_TARGET_M_S:.0e} m/s"
        )
    if acceleration_norm > ACCELERATION_RESIDUAL_TARGET_M_S2:
        diagnostic_failures.append(
            f"acceleration residual exceeds {ACCELERATION_RESIDUAL_TARGET_M_S2:.0e} m/s^2"
        )
    if passive_residual > PASSIVE_TORQUE_RESIDUAL_TARGET_NM:
        diagnostic_failures.append(
            f"passive-torque residual exceeds {PASSIVE_TORQUE_RESIDUAL_TARGET_NM:.0e} Nm"
        )

    diagnostics = {
        # Backward-compatible v1 field. It remains the deprecated algebraic
        # identity and is never used as loop-closure evidence.
        "constraint_residual_norm": 0.0,
        "position_residual_vectors": [residual.tolist() for residual in position_residuals],
        "position_residual_norms": position_norms,
        "position_residual_stacked_norm": float(np.linalg.norm(position_residuals)),
        "position_residual_max_norm": max(position_norms, default=0.0),
        "velocity_closure_residual": velocity_residual.tolist(),
        "velocity_closure_residual_norm": velocity_norm,
        "acceleration_closure_residual": acceleration_residual.tolist(),
        "acceleration_closure_residual_norm": acceleration_norm,
        "passive_torque_residual": passive_residual_vector.tolist(),
        "passive_torque_residual_norm": float(passive_residual),
        "rank": rank,
        "rank_tolerance": float(rank_tolerance),
        "singular_values": singular_values.tolist(),
        "condition_number": condition,
        "mapping_fd_step": mapping_fd_step,
        "directional_fd_step": directional_fd_step,
        "diagnostics_pass": not diagnostic_failures,
        "diagnostic_failures": diagnostic_failures,
    }

    warnings = [
        f"CR4 KKT diagnostic target failed: {failure}"
        for failure in diagnostic_failures
    ]
    return tau_user.tolist(), power_user.tolist(), diagnostics, warnings


def compute_cr4_kkt_batch(
    robot_spec: Dict[str, Any],
    samples: List[Dict[str, Any]],
    options: Dict[str, Any] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Computes inverse dynamics for a batch trajectory of CR4 using closed-chain KKT.
    Loads/caches the model once, then evaluates for all samples.
    """
    get_or_build_model(robot_spec)
    
    out_samples = []
    out_diags = []
    all_warnings = []

    for idx, s in enumerate(samples):
        # We reuse the cached model built in the batch loop
        tau, power, diags, warnings = compute_cr4_kkt_dynamics(robot_spec, s["q"], s["qd"], s["qdd"], options)
        
        out_samples.append({
            "time_s": s["time_s"],
            "q": s["q"],
            "velocity": s["qd"],
            "acceleration": s["qdd"],
            "tau": tau,
            "power": power
        })
        out_diags.append(diags)
        all_warnings.extend(warnings)

    # Unique warnings only
    unique_warnings = list(dict.fromkeys(all_warnings))
    return out_samples, out_diags, unique_warnings
