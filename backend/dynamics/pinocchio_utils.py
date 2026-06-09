import math
import numpy as np
import pinocchio as pin
from typing import Any, Dict

def rigid_inertia(mass: float, com: np.ndarray, inertia_at_com: np.ndarray) -> pin.Inertia:
    """
    Creates a pin.Inertia instance from mass, center of mass, and inertia tensor at COM.
    """
    return pin.Inertia(float(mass), np.asarray(com, dtype=float), np.asarray(inertia_at_com, dtype=float))


def rod_x_inertia(mass: float, length: float, thickness: float = 0.01) -> pin.Inertia:
    """
    Creates a thin rod inertia aligned with the X-axis.
    """
    tensor = np.diag([
        mass * (thickness * thickness + thickness * thickness) / 12.0,
        mass * (length * length + thickness * thickness) / 12.0,
        mass * (length * length + thickness * thickness) / 12.0,
    ])
    return rigid_inertia(mass, np.array([0.5 * length, 0.0, 0.0]), tensor)


def cylinder_z_tensor(mass: float, radius: float, length: float) -> np.ndarray:
    """
    Calculates the inertia tensor of a cylinder aligned with the Z-axis.
    """
    return np.diag([
        mass * (3.0 * radius * radius + length * length) / 12.0,
        mass * (3.0 * radius * radius + length * length) / 12.0,
        0.5 * mass * radius * radius,
    ])


def triangle_plate_inertia_xz(mass: float, vertices_xz: np.ndarray, thickness: float = 0.01) -> pin.Inertia:
    """
    Calculates the inertia of a triangular plate in the XZ plane.
    """
    pts = np.asarray(vertices_xz, dtype=float)
    centroid = np.mean(pts, axis=0)
    rel = pts - centroid
    covariance = (rel.T @ rel) / 12.0
    var_x = covariance[0, 0]
    var_z = covariance[1, 1]
    cov_xz = covariance[0, 1]
    
    tensor = np.array([
        [mass * (var_z + thickness * thickness / 12.0), 0.0, -mass * cov_xz],
        [0.0, mass * (var_x + var_z), 0.0],
        [-mass * cov_xz, 0.0, mass * (var_x + thickness * thickness / 12.0)],
    ], dtype=float)
    
    return rigid_inertia(mass, np.array([centroid[0], 0.0, centroid[1]]), tensor)


def triangle_plate_inertia_xz_rotated_to_edge_frame(
    mass: float,
    *,
    vertices_xz: np.ndarray,
    canonical_from_edge: np.ndarray,
    thickness: float = 0.01,
) -> pin.Inertia:
    """
    Calculates the inertia of a triangular plate in the XZ plane, then transforms
    it into the edge frame of one of its sides.
    """
    canonical = triangle_plate_inertia_xz(mass, vertices_xz, thickness)
    rotation = np.asarray(canonical_from_edge, dtype=float)
    com_edge = rotation.T @ np.asarray(canonical.lever, dtype=float)
    inertia_edge = rotation.T @ np.asarray(canonical.inertia, dtype=float) @ rotation
    return rigid_inertia(mass, com_edge, inertia_edge)


def payload_inertia(mass: float, com: np.ndarray, inertia_kg_m2: Any) -> pin.Inertia:
    """
    Creates payload inertia. If custom inertia is not provided, defaults to a small cube.
    """
    if inertia_kg_m2 is not None:
        return pin.Inertia(mass, com, np.asarray(inertia_kg_m2, dtype=float))
    # Default to small cube
    side = 0.03
    inertia = mass * (side * side + side * side) / 12.0
    return pin.Inertia(mass, com, np.eye(3) * inertia)


def unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        raise ValueError("Zero-length vector")
    return v / norm


def distance(points: Dict[str, np.ndarray], first: str, second: str) -> float:
    return float(np.linalg.norm(points[second] - points[first]))


def frame_rotation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Returns rotation matrix for a frame where:
      - X points from first to second.
      - Y is parallel to world Y [0, 1, 0].
      - Z is perpendicular (right-hand rule).
    """
    x_axis = unit(second - first)
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = unit(np.cross(x_axis, y_axis))
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def local_point(points: Dict[str, np.ndarray], point: str, origin: str, x_target: str) -> np.ndarray:
    """
    Computes coordinates of `point` relative to frame defined at `origin` with X-axis pointing to `x_target`.
    """
    rot = frame_rotation(points[origin], points[x_target])
    return rot.T @ (points[point] - points[origin])


def se3(xyz) -> pin.SE3:
    return pin.SE3(np.eye(3), np.asarray(xyz, dtype=float))


def transform_inertial_frame(
    mass_kg: float,
    com_m: np.ndarray,
    inertia_kg_m2: np.ndarray,
    from_transform: np.ndarray,
    to_transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms center of mass and inertia tensor from one frame to another.
    """
    rotation_from = from_transform[:3, :3]
    rotation_to = to_transform[:3, :3]
    translation_from = from_transform[:3, 3]
    translation_to = to_transform[:3, 3]

    com_world = rotation_from @ com_m + translation_from
    com_to = rotation_to.T @ (com_world - translation_to)
    rotation_to_from = rotation_to.T @ rotation_from
    inertia_to = rotation_to_from @ inertia_kg_m2 @ rotation_to_from.T
    
    com_to[np.abs(com_to) < 1e-12] = 0.0
    inertia_to[np.abs(inertia_to) < 1e-12] = 0.0
    return com_to, inertia_to
