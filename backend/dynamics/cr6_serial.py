from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Mapping

import numpy as np

SERIAL6_SCHEMA_VERSION = "kineforge.serial6.v1"

@dataclass(frozen=True)
class SerialInertialSpec:
    mass_kg: float = 0.0
    com_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia_kg_m2: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    frame: str = "cad"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SerialInertialSpec:
        mass = float(data.get("massKg", 0.0))
        com = _tuple3(data.get("comM", (0.0, 0.0, 0.0)))
        inertia = _matrix3(
            data.get(
                "inertiaKgM2",
                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            )
        )
        frame = str(data.get("frame", "cad"))
        return cls(mass_kg=mass, com_m=com, inertia_kg_m2=inertia, frame=frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "massKg": self.mass_kg,
            "comM": list(self.com_m),
            "inertiaKgM2": [list(row) for row in self.inertia_kg_m2],
            "frame": self.frame,
        }


@dataclass(frozen=True)
class DHJointSpec:
    name: str
    a_m: float = 0.0
    alpha_rad: float = 0.0
    d_m: float = 0.0
    theta_offset_rad: float = 0.0
    lower_limit_rad: float = -math.inf
    upper_limit_rad: float = math.inf
    friction_coeff: float = 0.0
    inertial: SerialInertialSpec = field(default_factory=SerialInertialSpec)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], limit_data: Mapping[str, Any] = None) -> DHJointSpec:
        limit = limit_data or {}
        return cls(
            name=str(data.get("name")),
            a_m=float(data.get("a_m", 0.0)),
            alpha_rad=float(data.get("alpha_rad", 0.0)),
            d_m=float(data.get("d_m", 0.0)),
            theta_offset_rad=float(data.get("theta_offset_rad", 0.0)),
            lower_limit_rad=float(limit.get("lowerLimitRad", -math.inf)),
            upper_limit_rad=float(limit.get("upperLimitRad", math.inf)),
            friction_coeff=float(limit.get("frictionCoeffNmSPerRad", 0.0)),
            inertial=SerialInertialSpec.from_dict(data.get("inertial", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "a_m": self.a_m,
            "alpha_rad": self.alpha_rad,
            "d_m": self.d_m,
            "theta_offset_rad": self.theta_offset_rad,
            "lower_limit_rad": self.lower_limit_rad,
            "upper_limit_rad": self.upper_limit_rad,
            "inertial": self.inertial.to_dict(),
        }


@dataclass(frozen=True)
class Serial6FK:
    q: np.ndarray
    joint_origins: tuple[np.ndarray, ...]
    joint_axes: tuple[np.ndarray, ...]
    joint_body_transforms: tuple[np.ndarray, ...]
    link_transforms: tuple[np.ndarray, ...]
    tcp_transform: np.ndarray


@dataclass
class Serial6Template:
    joints: tuple[DHJointSpec, ...]
    tool_transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.80665)
    payload: SerialInertialSpec = field(default_factory=SerialInertialSpec)
    schema_version: str = SERIAL6_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.joints) != 6:
            raise ValueError(f"Serial6Template requires 6 joints, got {len(self.joints)}")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    def clamp_configuration(self, q: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        q_clamped = _vecn(q, 6).copy()
        for index, joint in enumerate(self.joints):
            if math.isfinite(joint.lower_limit_rad) and math.isfinite(joint.upper_limit_rad):
                q_clamped[index] = np.clip(q_clamped[index], joint.lower_limit_rad, joint.upper_limit_rad)
        return q_clamped

    def forward_kinematics(self, q: np.ndarray | list[float] | tuple[float, ...]) -> Serial6FK:
        q_array = self.clamp_configuration(q)
        transform = np.eye(4, dtype=float)
        origins: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        joint_body_transforms: list[np.ndarray] = []
        link_transforms: list[np.ndarray] = []
        for index, joint in enumerate(self.joints):
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, 2].copy())
            theta = float(q_array[index]) + joint.theta_offset_rad
            joint_body_transforms.append((transform @ _rotz(theta)).copy())
            transform = transform @ _standard_dh(
                joint.a_m,
                joint.alpha_rad,
                joint.d_m,
                theta,
            )
            link_transforms.append(transform.copy())
        return Serial6FK(
            q=q_array,
            joint_origins=tuple(origins),
            joint_axes=tuple(axes),
            joint_body_transforms=tuple(joint_body_transforms),
            link_transforms=tuple(link_transforms),
            tcp_transform=transform @ self.tool_transform,
        )

    def inverse_dynamics(
        self,
        q: np.ndarray | list[float] | tuple[float, ...],
        qd: np.ndarray | list[float] | tuple[float, ...],
        qdd: np.ndarray | list[float] | tuple[float, ...],
    ) -> np.ndarray:
        q_array = self.clamp_configuration(q)
        qd_array = _vecn(qd, 6)
        qdd_array = _vecn(qdd, 6)
        tau = np.zeros(6, dtype=float)

        fk = self.forward_kinematics(q_array)
        gravity = np.asarray(self.gravity_m_s2, dtype=float)
        origins = fk.joint_origins
        axes = fk.joint_axes
        origin_velocities = _serial_origin_velocities(origins, axes, qd_array)
        axis_derivatives = _serial_axis_derivatives(axes, qd_array)

        body_inertials = [joint.inertial for joint in self.joints]
        body_transforms = [transform for transform in fk.link_transforms]
        if self.payload.mass_kg != 0.0 or np.any(self.payload.inertia_kg_m2):
            body_inertials.append(self.payload)
            body_transforms.append(fk.tcp_transform)

        for body_index, (inertial, link_transform) in enumerate(
            zip(body_inertials, body_transforms, strict=True)
        ):
            if inertial.mass_kg == 0.0 and not np.any(inertial.inertia_kg_m2):
                continue
            link_index = min(body_index, 5)
            if body_index < 6 and inertial.frame == "cad":
                inertial = self._cad_inertial_to_link_inertial(body_index, inertial)

            com_local = np.asarray(inertial.com_m, dtype=float)
            com_world = (link_transform @ np.array([*com_local, 1.0], dtype=float))[:3]
            rotation_world_link = link_transform[:3, :3]
            inertia_world = (
                rotation_world_link
                @ np.asarray(inertial.inertia_kg_m2, dtype=float)
                @ rotation_world_link.T
            )

            jacobian_v = np.zeros((3, 6), dtype=float)
            jacobian_w = np.zeros((3, 6), dtype=float)
            jacobian_v_dot_qd = np.zeros(3, dtype=float)
            jacobian_w_dot_qd = np.zeros(3, dtype=float)
            for joint_index in range(link_index + 1):
                axis = axes[joint_index]
                origin = origins[joint_index]
                radius = com_world - origin
                jacobian_v[:, joint_index] = np.cross(axis, radius)
                jacobian_w[:, joint_index] = axis

            com_velocity = jacobian_v @ qd_array
            for joint_index in range(link_index + 1):
                axis_dot = axis_derivatives[joint_index]
                radius = com_world - origins[joint_index]
                radius_dot = com_velocity - origin_velocities[joint_index]
                jacobian_v_dot_qd += qd_array[joint_index] * (
                    np.cross(axis_dot, radius) + np.cross(axes[joint_index], radius_dot)
                )
                jacobian_w_dot_qd += qd_array[joint_index] * axis_dot

            com_acceleration = jacobian_v @ qdd_array + jacobian_v_dot_qd
            angular_velocity = jacobian_w @ qd_array
            angular_acceleration = jacobian_w @ qdd_array + jacobian_w_dot_qd
            force = inertial.mass_kg * (com_acceleration - gravity)
            moment = inertia_world @ angular_acceleration + np.cross(
                angular_velocity, inertia_world @ angular_velocity
            )
            tau += jacobian_v.T @ force + jacobian_w.T @ moment

        # Add joint viscous friction
        for i, joint in enumerate(self.joints):
            tau[i] += joint.friction_coeff * qd_array[i]

        return tau

    def _cad_inertial_to_link_inertial(
        self, link_index: int, inertial: SerialInertialSpec
    ) -> SerialInertialSpec:
        home_fk = self.forward_kinematics(np.zeros(6, dtype=float))
        cad_transform = home_fk.joint_body_transforms[link_index]
        cad_transform = cad_transform.copy()
        cad_transform[:3, :3] = np.eye(3, dtype=float)
        link_transform = home_fk.link_transforms[link_index]
        converted = _transform_inertial_frame(
            inertial,
            from_transform=cad_transform,
            to_transform=link_transform,
            frame="link",
        )
        return converted


# --- Helper math functions ---

def _rotz(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array(
        ((c, -s, 0.0, 0.0), (s, c, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        dtype=float,
    )


def _standard_dh(
    a_m: float, alpha_rad: float, d_m: float, theta_rad: float
) -> np.ndarray:
    ct = math.cos(theta_rad)
    st = math.sin(theta_rad)
    ca = math.cos(alpha_rad)
    sa = math.sin(alpha_rad)
    return np.array(
        (
            (ct, -st * ca, st * sa, a_m * ct),
            (st, ct * ca, -ct * sa, a_m * st),
            (0.0, sa, ca, d_m),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=float,
    )


def _vec3(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"Expected 3-vector, got shape {vector.shape}")
    return vector


def _vecn(value: Any, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (size,):
        raise ValueError(f"Expected {size}-vector, got shape {vector.shape}")
    return vector


def _tuple3(value: Any) -> tuple[float, float, float]:
    vector = _vec3(value)
    return float(vector[0]), float(vector[1]), float(vector[2])


def _matrix3(value: Any) -> tuple[tuple[float, float, float], ...]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {matrix.shape}")
    return tuple(tuple(float(cell) for cell in row) for row in matrix)


def _transform_inertial_frame(
    inertial: SerialInertialSpec,
    *,
    from_transform: np.ndarray,
    to_transform: np.ndarray,
    frame: str,
) -> SerialInertialSpec:
    com_from = np.asarray(inertial.com_m, dtype=float)
    inertia_from = np.asarray(inertial.inertia_kg_m2, dtype=float)
    rotation_from = from_transform[:3, :3]
    rotation_to = to_transform[:3, :3]
    com_world = rotation_from @ com_from + from_transform[:3, 3]
    com_to = rotation_to.T @ (com_world - to_transform[:3, 3])
    rotation_to_from = rotation_to.T @ rotation_from
    inertia_to = rotation_to_from @ inertia_from @ rotation_to_from.T
    com_to[np.abs(com_to) < 1e-12] = 0.0
    inertia_to[np.abs(inertia_to) < 1e-12] = 0.0
    return SerialInertialSpec(
        mass_kg=inertial.mass_kg,
        com_m=(float(com_to[0]), float(com_to[1]), float(com_to[2])),
        inertia_kg_m2=tuple(tuple(float(cell) for cell in row) for row in inertia_to),
        frame=frame,
    )


def _serial_origin_velocities(
    origins: tuple[np.ndarray, ...], axes: tuple[np.ndarray, ...], qd: np.ndarray
) -> tuple[np.ndarray, ...]:
    velocities: list[np.ndarray] = []
    for joint_index, origin in enumerate(origins):
        velocity = np.zeros(3, dtype=float)
        for parent_index in range(joint_index):
            velocity += qd[parent_index] * np.cross(
                axes[parent_index], origin - origins[parent_index]
            )
        velocities.append(velocity)
    return tuple(velocities)


def _serial_axis_derivatives(
    axes: tuple[np.ndarray, ...], qd: np.ndarray
) -> tuple[np.ndarray, ...]:
    derivatives: list[np.ndarray] = []
    angular_velocity = np.zeros(3, dtype=float)
    for joint_index, axis in enumerate(axes):
        derivatives.append(np.cross(angular_velocity, axis))
        angular_velocity = angular_velocity + qd[joint_index] * axis
    return tuple(derivatives)


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _wrap_near(angle: float, reference: float) -> float:
    return float(reference) + _wrap_to_pi(float(angle) - float(reference))


# --- Mapper functions for API ---

def serial_inertial_from_robodimm(
    data: dict[str, Any], *, default_body: str, payload: bool = False
) -> SerialInertialSpec:
    frame = str(data.get("frame", "link" if payload else "cad"))
    if frame == "tcp":
        frame = "link"
    if frame not in {"cad", "link"}:
        raise ValueError(f"Unsupported inertial frame for {default_body}: {frame!r}")
    
    mass = float(data.get("massKg", 0.0))
    com = tuple(float(val) for val in data.get("comM", (0.0, 0.0, 0.0)))
    inertia = tuple(
        tuple(float(val) for val in row)
        for row in data.get(
            "inertiaKgM2",
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )
    )
    return SerialInertialSpec(
        mass_kg=mass,
        com_m=com,
        inertia_kg_m2=inertia,
        frame=frame
    )


def build_serial6_template_from_robot(robot: Dict[str, Any]) -> Serial6Template:
    geometry = robot["geometry"]
    limits_by_name = {limit["name"]: limit for limit in robot.get("limits", [])}
    joints = []
    for index, joint in enumerate(geometry["joints"]):
        link_name = f"LINK{index + 1}"
        limit = limits_by_name.get(joint["name"], {})
        joints.append(
            DHJointSpec(
                joint["name"],
                a_m=float(joint["a_m"]),
                alpha_rad=float(joint["alpha_rad"]),
                d_m=float(joint["d_m"]),
                theta_offset_rad=float(joint.get("theta_offset_rad", 0.0)),
                lower_limit_rad=float(limit.get("lowerLimitRad", -math.inf)),
                upper_limit_rad=float(limit.get("upperLimitRad", math.inf)),
                friction_coeff=float(limit.get("frictionCoeffNmSPerRad", 0.0)),
                inertial=serial_inertial_from_robodimm(
                    robot.get("inertials", {}).get(link_name, {}), default_body=link_name
                ),
            )
        )

    payload = serial_inertial_from_robodimm(
        robot.get("payload", {}), default_body="PAYLOAD", payload=True
    )
    
    # tool transform
    tool_raw = geometry.get("tool_transform", np.eye(4).tolist())
    tool_transform = np.array(tool_raw, dtype=float)
    
    return Serial6Template(
        joints=tuple(joints),
        tool_transform=tool_transform,
        gravity_m_s2=(0.0, 0.0, -9.80665),
        payload=payload
    )


def compute_cr6_serial_dynamics(
    robot_spec: Dict[str, Any],
    q: List[float],
    qd: List[float],
    qdd: List[float],
    options: Dict[str, Any] = None
) -> Tuple[List[float], List[float], List[str]]:
    """
    Computes inverse dynamics for a single sample of CR6 using serial open-chain.
    """
    template = build_serial6_template_from_robot(robot_spec)
    tau = template.inverse_dynamics(q, qd, qdd)
    power = np.array(tau) * np.array(qd)
    return tau.tolist(), power.tolist(), []


def compute_cr6_serial_batch(
    robot_spec: Dict[str, Any],
    samples: List[Dict[str, Any]],
    options: Dict[str, Any] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Computes inverse dynamics for a batch trajectory of CR6.
    """
    template = build_serial6_template_from_robot(robot_spec)
    
    out_samples = []
    for s in samples:
        tau = template.inverse_dynamics(s["q"], s["qd"], s["qdd"])
        power = np.array(tau) * np.array(s["qd"])
        out_samples.append({
            "time_s": s["time_s"],
            "q": s["q"],
            "velocity": s["qd"],
            "acceleration": s["qdd"],
            "tau": tau.tolist(),
            "power": power.tolist()
        })
    return out_samples, []
