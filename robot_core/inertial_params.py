"""
Inertial Parameters for Robot Links
====================================

Standardized inertial parameters based on basic geometric shapes (prisms, cylinders)
referenced to DH frames. These parameters are designed to be:
- Easy to scale (all dimensions scale linearly)
- Consistent between DEMO and PRO modes
- Physically realistic (based on aluminum construction)
- Referenced to DH coordinate frames

Usage:
    from robot_core.inertial_params import get_cr4_inertial_params, get_cr6_inertial_params

    params = get_cr4_inertial_params(scale=1.0)
    # Returns dict with mass, com, inertia for each link
"""

import numpy as np
from typing import Dict, List, Tuple


# CR4 serial-5 reference inertial parameters validated against Simscape scripts.
CR4_SERIAL5_REF = {
    "masses": [8.0, 10.0, 6.0, 2.0, 1.5],
    "coms": [
        np.array([0.0, 0.20, 0.0], dtype=float),
        np.array([-0.27, -0.025, 0.0], dtype=float),
        np.array([-0.30, -0.025, 0.0], dtype=float),
        np.array([0.0625, -0.025, 0.0], dtype=float),
        np.array([-0.025, 0.0, -0.0675], dtype=float),
    ],
    "inertias": [
        np.diag([0.171667, 0.1300, 0.171667]),
        np.diag([0.03075, 0.2550, 0.26175]),
        np.diag([0.0122, 0.1850, 0.1872]),
        np.diag([0.0016, 0.003404, 0.003404]),
        np.diag([0.002616, 0.002616, 0.000675]),
    ],
}


def get_cr4_serial5_reference_params(
    scale: float = 1.0,
    structural_mass_scale_exp: float = 3.0,
    structural_inertia_scale_exp: float | None = None,
) -> Dict:
    """Return CR4 serial-5 inertial params with configurable structural scaling.

    Default behavior keeps Simscape-aligned similarity scaling (mass~s^3, inertia~s^5).
    """
    s = float(scale)
    if s <= 0.0:
        raise ValueError(f"scale must be > 0, received {scale}")

    m_exp = float(structural_mass_scale_exp)
    i_exp = float(structural_inertia_scale_exp) if structural_inertia_scale_exp is not None else (m_exp + 2.0)
    mass_scale = s**m_exp
    inertia_scale = s**i_exp
    return {
        "masses": [m * mass_scale for m in CR4_SERIAL5_REF["masses"]],
        "coms": [c * s for c in CR4_SERIAL5_REF["coms"]],
        "inertias": [I * inertia_scale for I in CR4_SERIAL5_REF["inertias"]],
        "scale": s,
    }


def get_cr4_closed_loop_joint_inertial_params(
    scale: float = 1.0,
    structural_mass_scale_exp: float = 3.0,
    structural_inertia_scale_exp: float | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Return CR4 inertial parameters mapped to real closed-loop joints.

    Mapping follows the validated serial-to-parallelogram transfer strategy:
    - Main links use serial-5 reference inertias.
    - Secondary parallelogram links are lightweight.
    """
    serial = get_cr4_serial5_reference_params(
        scale,
        structural_mass_scale_exp=structural_mass_scale_exp,
        structural_inertia_scale_exp=structural_inertia_scale_exp,
    )
    m_exp = float(structural_mass_scale_exp)
    i_exp = float(structural_inertia_scale_exp) if structural_inertia_scale_exp is not None else (m_exp + 2.0)
    tiny = np.diag([1e-5, 1e-5, 1e-5]) * (float(scale) ** i_exp)

    return {
        "J1": {
            "mass": serial["masses"][0],
            "com": serial["coms"][0],
            "inertia": serial["inertias"][0],
        },
        "J2": {
            "mass": serial["masses"][1],
            "com": serial["coms"][1],
            "inertia": serial["inertias"][1],
        },
        "J3": {
            "mass": serial["masses"][2],
            "com": serial["coms"][2],
            "inertia": serial["inertias"][2],
        },
        "J_aux": {
            "mass": serial["masses"][3],
            "com": serial["coms"][3],
            "inertia": serial["inertias"][3],
        },
        "J4": {
            "mass": serial["masses"][4],
            "com": serial["coms"][4],
            "inertia": serial["inertias"][4],
        },
        "J3real": {
            "mass": 0.001 * (float(scale) ** m_exp),
            "com": np.zeros(3),
            "inertia": tiny.copy(),
        },
        "J1p": {
            "mass": 0.001 * (float(scale) ** m_exp),
            "com": np.zeros(3),
            "inertia": tiny.copy(),
        },
        "J2p": {
            "mass": 0.02 * (float(scale) ** m_exp),
            "com": np.zeros(3),
            "inertia": tiny.copy(),
        },
    }


# Material density (kg/m³)
DENSITY_ALUMINUM = 2700.0  # Aluminum alloy
DENSITY_STEEL = 7850.0  # Steel for gears/motors


def box_inertia(mass: float, width: float, height: float, depth: float) -> np.ndarray:
    """
    Calculate inertia tensor for a rectangular box.

    I = [Ixx, Iyy, Izz] diagonal for box centered at origin

    Args:
        mass: Mass in kg
        width: X dimension (m)
        height: Y dimension (m)
        depth: Z dimension (m)

    Returns:
        3x3 inertia tensor (kg·m²)
    """
    Ixx = (mass / 12.0) * (height**2 + depth**2)
    Iyy = (mass / 12.0) * (width**2 + depth**2)
    Izz = (mass / 12.0) * (width**2 + height**2)

    return np.diag([Ixx, Iyy, Izz])


def cylinder_inertia(
    mass: float, radius: float, height: float, axis: str = "z"
) -> np.ndarray:
    """
    Calculate inertia tensor for a cylinder.

    Args:
        mass: Mass in kg
        radius: Cylinder radius (m)
        height: Cylinder height (m)
        axis: Cylinder axis ('x', 'y', or 'z')

    Returns:
        3x3 inertia tensor (kg·m²)
    """
    I_perp = (mass / 12.0) * (3 * radius**2 + height**2)
    I_axial = (mass / 2.0) * radius**2

    if axis == "z":
        return np.diag([I_perp, I_perp, I_axial])
    elif axis == "y":
        return np.diag([I_perp, I_axial, I_perp])
    else:  # axis == 'x'
        return np.diag([I_axial, I_perp, I_perp])


def hollow_cylinder_inertia(
    mass: float, r_outer: float, r_inner: float, height: float, axis: str = "z"
) -> np.ndarray:
    """
    Calculate inertia tensor for a hollow cylinder (tube).

    Args:
        mass: Mass in kg
        r_outer: Outer radius (m)
        r_inner: Inner radius (m)
        height: Cylinder height (m)
        axis: Cylinder axis ('x', 'y', or 'z')

    Returns:
        3x3 inertia tensor (kg·m²)
    """
    I_axial = (mass / 2.0) * (r_outer**2 + r_inner**2)
    I_perp = (mass / 12.0) * (3 * (r_outer**2 + r_inner**2) + height**2)

    if axis == "z":
        return np.diag([I_perp, I_perp, I_axial])
    elif axis == "y":
        return np.diag([I_perp, I_axial, I_perp])
    else:  # axis == 'x'
        return np.diag([I_axial, I_perp, I_perp])


def get_cr4_inertial_params(scale: float = 1.0) -> Dict:
    """
    Get CR4 inertial parameters based on geometric shapes.

    All parameters are referenced to DH frames and scale linearly.

    Link geometry (at scale=1.0):
    - Link 0 (Base): Hollow cylinder, H=0.4m, D=0.3m
    - Link 1 (Shoulder): Box, 0.3x0.2x0.25m
    - Link 2 (Upper arm): Box, 0.54x0.15x0.12m
    - Link 3 (Forearm): Box, 0.6x0.12x0.10m
    - Link 4 (Wrist aux): Cylinder, H=0.125m, D=0.08m
    - Link 5 (TCP): Cylinder, H=0.135m, D=0.06m

    Args:
        scale: Linear scale factor (default 1.0)

    Returns:
        Dictionary with keys: masses, coms, inertias
    """
    s = scale  # Shorthand

    # Link 0: Base (hollow cylinder)
    # DH: a=0, alpha=-π/2, d=0.4
    # Shape: Hollow cylinder, axis along Z
    mass_0 = 8.0 * s**3  # ~8kg at scale 1
    r_outer_0 = 0.15 * s
    r_inner_0 = 0.10 * s
    height_0 = 0.40 * s
    I_0 = hollow_cylinder_inertia(mass_0, r_outer_0, r_inner_0, height_0, axis="z")
    com_0 = np.array([0, 0, 0.20]) * s  # Center of cylinder

    # Link 1: Shoulder box
    # DH: a=0.54, alpha=0, d=0
    # Shape: Box, main dimension along X (a)
    mass_1 = 12.0 * s**3
    w_1, h_1, d_1 = 0.30 * s, 0.20 * s, 0.25 * s
    I_1 = box_inertia(mass_1, w_1, h_1, d_1)
    com_1 = np.array([0.0, 0, 0]) * s  # At joint

    # Link 2: Upper arm
    # DH: a=0.6, alpha=0, d=0
    # Shape: Box, long dimension along X (a)
    mass_2 = 10.0 * s**3
    w_2, h_2, d_2 = 0.54 * s, 0.15 * s, 0.12 * s
    I_2 = box_inertia(mass_2, w_2, h_2, d_2)
    com_2 = np.array([0.27, 0, 0]) * s  # Midpoint of a

    # Link 3: Forearm
    # DH: a=0.6, alpha=0 (J3_rel), plus a=-0.125
    # Shape: Box, long dimension along X
    mass_3 = 6.0 * s**3
    w_3, h_3, d_3 = 0.60 * s, 0.12 * s, 0.10 * s
    I_3 = box_inertia(mass_3, w_3, h_3, d_3)
    com_3 = np.array([0.30, 0, 0]) * s  # Midpoint of a

    # Link 4: Wrist auxiliary (parallelogram)
    # DH: a=-0.125, alpha=π/2, d=0
    # Shape: Short cylinder
    mass_4 = 2.0 * s**3
    r_4 = 0.04 * s
    h_4 = 0.125 * s
    I_4 = cylinder_inertia(mass_4, r_4, h_4, axis="x")
    com_4 = np.array([0, 0, 0]) * s

    # Link 5: TCP mount
    # DH: a=0, alpha=0, d=0.135
    # Shape: Cylinder, axis along Z
    mass_5 = 1.5 * s**3
    r_5 = 0.03 * s
    h_5 = 0.135 * s
    I_5 = cylinder_inertia(mass_5, r_5, h_5, axis="z")
    com_5 = np.array([0, 0, 0.0675]) * s  # Center

    return {
        "masses": [mass_0, mass_1, mass_2, mass_3, mass_4, mass_5],
        "coms": [com_0, com_1, com_2, com_3, com_4, com_5],
        "inertias": [I_0, I_1, I_2, I_3, I_4, I_5],
        "scale": scale,
    }


def get_cr6_inertial_params(scale: float = 1.0) -> Dict:
    """
    Get CR6 inertial parameters based on geometric shapes.

    Link geometry (at scale=1.0):
    - Link 0 (Base): Hollow cylinder, H=0.4m, D=0.3m
    - Link 1 (Shoulder): Box, 0.3x0.2x0.25m
    - Link 2 (Upper arm): Box, 0.54x0.15x0.12m
    - Link 3 (Elbow): Cylinder, H=0.1m, D=0.1m
    - Link 4 (Forearm): Box, 0.2x0.1x0.58m
    - Link 5 (Wrist): Cylinder, H=0.1m, D=0.08m
    - Link 6 (TCP): Cylinder, H=0.065m, D=0.06m

    Args:
        scale: Linear scale factor (default 1.0)

    Returns:
        Dictionary with keys: masses, coms, inertias
    """
    s = scale

    # Link 0: Base
    mass_0 = 8.0 * s**3
    r_outer_0 = 0.15 * s
    r_inner_0 = 0.10 * s
    height_0 = 0.40 * s
    I_0 = hollow_cylinder_inertia(mass_0, r_outer_0, r_inner_0, height_0, axis="z")
    com_0 = np.array([0, 0, 0.20]) * s

    # Link 1: Shoulder
    mass_1 = 12.0 * s**3
    w_1, h_1, d_1 = 0.30 * s, 0.20 * s, 0.25 * s
    I_1 = box_inertia(mass_1, w_1, h_1, d_1)
    com_1 = np.array([0.0, 0, 0]) * s

    # Link 2: Upper arm
    mass_2 = 10.0 * s**3
    w_2, h_2, d_2 = 0.54 * s, 0.15 * s, 0.12 * s
    I_2 = box_inertia(mass_2, w_2, h_2, d_2)
    com_2 = np.array([0.27, 0, 0]) * s

    # Link 3: Elbow
    mass_3 = 4.0 * s**3
    r_3 = 0.05 * s
    h_3 = 0.10 * s
    I_3 = cylinder_inertia(mass_3, r_3, h_3, axis="y")
    com_3 = np.array([0, 0, 0]) * s

    # Link 4: Forearm
    mass_4 = 5.0 * s**3
    w_4, h_4, d_4 = 0.20 * s, 0.10 * s, 0.58 * s
    I_4 = box_inertia(mass_4, w_4, h_4, d_4)
    com_4 = np.array([0, 0, 0.29]) * s

    # Link 5: Wrist
    mass_5 = 2.0 * s**3
    r_5 = 0.04 * s
    h_5 = 0.10 * s
    I_5 = cylinder_inertia(mass_5, r_5, h_5, axis="z")
    com_5 = np.array([0, 0, 0]) * s

    # Link 6: TCP
    mass_6 = 1.0 * s**3
    r_6 = 0.03 * s
    h_6 = 0.065 * s
    I_6 = cylinder_inertia(mass_6, r_6, h_6, axis="z")
    com_6 = np.array([0, 0, 0.0325]) * s

    return {
        "masses": [mass_0, mass_1, mass_2, mass_3, mass_4, mass_5, mass_6],
        "coms": [com_0, com_1, com_2, com_3, com_4, com_5, com_6],
        "inertias": [I_0, I_1, I_2, I_3, I_4, I_5, I_6],
        "scale": scale,
    }


def print_inertial_params(robot_type: str = "CR4", scale: float = 1.0):
    """Print inertial parameters for verification."""
    if robot_type == "CR4":
        params = get_cr4_inertial_params(scale)
    else:
        params = get_cr6_inertial_params(scale)

    print(f"\n{robot_type} Inertial Parameters (scale={scale})")
    print("=" * 60)

    for i, (m, com, I) in enumerate(
        zip(params["masses"], params["coms"], params["inertias"])
    ):
        print(f"\nLink {i}:")
        print(f"  Mass: {m:.3f} kg")
        print(f"  COM:  [{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}] m")
        print(
            f"  Inertia diagonal: [{I[0, 0]:.6f}, {I[1, 1]:.6f}, {I[2, 2]:.6f}] kg·m²"
        )

    total_mass = sum(params["masses"])
    print(f"\nTotal mass: {total_mass:.2f} kg")
    print("=" * 60)


if __name__ == "__main__":
    # Test
    print_inertial_params("CR4", scale=1.0)
    print_inertial_params("CR6", scale=1.0)
