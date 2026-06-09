from __future__ import annotations
import math
import hashlib
import json
import numpy as np
import pinocchio as pin
from typing import Dict, List, Tuple, Any, Optional

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

G = 9.80665

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
    if robot_hash not in _MODEL_CACHE:
        _MODEL_CACHE[robot_hash] = build_closed_pinocchio_model(robot)
    return _MODEL_CACHE[robot_hash]


def closed_chain_points(geom: Cr4GeometryContext, j2: float, j3: float) -> dict[str, np.ndarray]:
    home = geom.points
    points = {key: value.copy() for key, value in home.items()}
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
    points["H"] = points["P"] + unit(cp) * geom.lengths["PH"]
    
    rot_ce = frame_rotation(points["C"], points["E"])
    points["F"] = points["C"] + rot_ce @ local_point(home, "F", "C", "E")
    
    # linkage circle intersection for point G
    pref_side = (home["H"][0] - home["F"][0]) * (home["G"][2] - home["F"][2]) - (home["H"][2] - home["F"][2]) * (home["G"][0] - home["F"][0])
    points["G"] = circle_intersection_xz(
        points["F"], geom.lengths["FG"], 
        points["H"], geom.lengths["HG"], 
        home["G"], pref_side
    )
    
    rot_hg = frame_rotation(points["H"], points["G"])
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
    dist = float(np.linalg.norm(dxz))
    if dist <= 1e-12:
        raise ValueError("Linkage circle centers are coincident")
    a = (radius_a * radius_a - radius_b * radius_b + dist * dist) / (2.0 * dist)
    h = float(math.sqrt(max(radius_a * radius_a - a * a, 0.0)))
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
    best = min(filtered, key=lambda candidate: float(np.linalg.norm(candidate - preferred)))
    return np.array([best[0], 0.0, best[1]], dtype=float)


def angle(a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    return float(np.arctan2(d[2], d[0]))


def closed_full_configuration(geom: Cr4GeometryContext, q_user: np.ndarray) -> np.ndarray:
    _j1, j2, j3, j4 = map(float, q_user)
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
    ], dtype=float)


def mapped_jacobian(map_fn, q_user: np.ndarray) -> np.ndarray:
    q0 = np.asarray(q_user, dtype=float)
    f0 = map_fn(q0)
    jacobian = np.zeros((f0.size, q0.size))
    eps = 1e-6
    for index in range(q0.size):
        step = np.zeros_like(q0)
        step[index] = eps
        pair = np.unwrap(np.vstack((map_fn(q0 - step), map_fn(q0 + step))), axis=0)
        jacobian[:, index] = (pair[1] - pair[0]) / (2.0 * eps)
    return jacobian


def mapped_state(map_fn, q_user: np.ndarray, qd_user: np.ndarray, qdd_user: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q0 = np.asarray(q_user, dtype=float)
    qd = np.asarray(qd_user, dtype=float)
    qdd = np.asarray(qdd_user, dtype=float)
    f0 = map_fn(q0)
    jacobian = mapped_jacobian(map_fn, q0)
    eps = 1e-5
    jacobian_plus = mapped_jacobian(map_fn, q0 + eps * qd)
    jacobian_minus = mapped_jacobian(map_fn, q0 - eps * qd)
    jdot_qd = ((jacobian_plus - jacobian_minus) / (2.0 * eps)) @ qd
    return f0, jacobian @ qd, jacobian @ qdd + jdot_qd


def closed_torque_to_user(geom: Cr4GeometryContext, q_user: np.ndarray, tau_actuated_pin: np.ndarray) -> np.ndarray:
    def actuated_cut_configuration(value: np.ndarray) -> np.ndarray:
        q_closed = closed_full_configuration(geom, value)
        return q_closed[[0, 1, 2, 9]]

    jacobian = mapped_jacobian(actuated_cut_configuration, q_user)
    tau_user = jacobian.T @ tau_actuated_pin
    # Simscape reports J4 actuation torque with the opposite sign after vertical re-alignment
    tau_user[3] *= -1.0
    return tau_user


def compute_cr4_kkt_dynamics(
    robot_spec: Dict[str, Any],
    q: List[float],
    qd: List[float],
    qdd: List[float],
    options: Dict[str, Any] = None
) -> Tuple[List[float], List[float], Dict[str, float], List[str]]:
    """
    Computes inverse dynamics for a single sample of CR4 using closed-chain KKT.
    Returns:
        tauNm: computed active joint torques in user space
        powerW: computed joint power
        diagnostics: dictionary containing KKT diagnostics
        warnings: warning messages generated during calculation
    """
    built = get_or_build_model(robot_spec)
    model = built.model
    data = model.createData()

    q_user = np.array(q, dtype=float)
    qd_user = np.array(qd, dtype=float)
    qdd_user = np.array(qdd, dtype=float)

    # 1. State Mapping: User Space -> Pinocchio Cut Tree space
    q_closed, v_closed, a_closed = mapped_state(
        lambda val: closed_full_configuration(built.geom, val), 
        q_user, qd_user, qdd_user
    )

    # 2. Pinocchio Open-Loop dynamics
    tau_open = np.asarray(pin.rnea(model, data, q_closed, v_closed, a_closed), dtype=float)

    # 3. Kinematic constraints calculation
    pin.forwardKinematics(model, data, q_closed, v_closed, a_closed)
    pin.computeJointJacobians(model, data, q_closed)
    data.q_in = q_closed

    for constraint_model, constraint_data in zip(built.constraints, built.constraint_datas):
        constraint_model.calc(model, data, constraint_data)
        
    jacobian_c = np.asarray(pin.getConstraintsJacobian(model, data, built.constraints, built.constraint_datas), dtype=float)
    jacobian_c = jacobian_c[np.linalg.norm(jacobian_c, axis=1) > 1e-10]

    # 4. KKT system solver: solve Lagrange multipliers forcing passive joint torques to zero
    actuated = [0, 1, 2, model.joints[built.joint_ids["J4"]].idx_v]
    passive = [idx for idx in range(model.nv) if idx not in actuated]

    lambdas = np.linalg.lstsq(jacobian_c[:, passive].T, -tau_open[passive], rcond=None)[0]
    
    # Restored torques
    tau_restored = tau_open + jacobian_c.T @ lambdas
    tau_actuated = tau_restored[actuated]

    # 5. Torque Projection back to User Space
    tau_user = closed_torque_to_user(built.geom, q_user, tau_actuated)

    # Add joint viscous friction in user space
    limits_by_name = {limit["name"]: limit for limit in robot_spec.get("limits", [])}
    for i, name in enumerate(["J1", "J2", "J3", "J4"]):
        limit = limits_by_name.get(name, {})
        friction_coeff = float(limit.get("frictionCoeffNmSPerRad", 0.0))
        tau_user[i] += friction_coeff * qd_user[i]

    power_user = tau_user * qd_user

    # 6. Diagnostics metrics
    constraint_residual = np.linalg.norm(jacobian_c[:, actuated].T @ lambdas + tau_open[actuated] - tau_restored[actuated])
    passive_residual = np.linalg.norm(tau_restored[passive])
    
    # SVD for condition number of constraints Jacobian
    singular_values = np.linalg.svd(jacobian_c, compute_uv=False)
    cond = float(singular_values[0] / singular_values[-1]) if len(singular_values) > 0 and singular_values[-1] > 1e-10 else 1.0

    diagnostics = {
        "constraint_residual_norm": float(constraint_residual),
        "passive_torque_residual_norm": float(passive_residual),
        "condition_number": cond
    }

    return tau_user.tolist(), power_user.tolist(), diagnostics, []


def compute_cr4_kkt_batch(
    robot_spec: Dict[str, Any],
    samples: List[Dict[str, Any]],
    options: Dict[str, Any] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]], List[str]]:
    """
    Computes inverse dynamics for a batch trajectory of CR4 using closed-chain KKT.
    Loads/caches the model once, then evaluates for all samples.
    """
    built = get_or_build_model(robot_spec)
    
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
    unique_warnings = list(set(all_warnings))
    return out_samples, out_diags, unique_warnings
