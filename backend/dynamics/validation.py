import math
from typing import Dict, List, Tuple, Any

def dist_xz(p1: List[float], p2: List[float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[2] - p2[2])

def validate_cr4_geometry(geom: Dict[str, List[float]]) -> Tuple[List[str], List[str]]:
    """
    Validates the CR4 (Palletizer) geometry against strong physical invariants.
    Returns:
        errors: list of severe validation violations that prevent dynamics computation.
        warnings: list of non-fatal warnings or soft limits.
    """
    errors = []
    warnings = []

    # Required points
    required = {'O', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J4', 'EE', 'TCP'}
    missing = required - set(geom.keys())
    if missing:
        errors.append(f"Missing required geometry hardpoints: {sorted(list(missing))}")
        return errors, warnings

    # 1. Planar coordinate checks (Y coord should be 0.0)
    for name, pt in geom.items():
        if len(pt) < 3:
            errors.append(f"Point {name} does not have 3 coordinates (got {len(pt)})")
            continue
        if abs(pt[1]) > 1e-5:
            warnings.append(f"Point {name} has non-zero Y coordinate ({pt[1]:.5f}m). Y coordinates will be projected to 0.0 for planar math.")

    # Convert coordinates to local variables for convenience (ignoring Y for core invariant checks)
    points = {k: [pt[0], 0.0, pt[2]] for k, pt in geom.items()}

    # 2. Check crank horizontal: B.z == O.z
    if abs(points['B'][2] - points['O'][2]) > 1e-5:
        errors.append(f"Crank B must be horizontal with Pivot O: B.z={points['B'][2]:.5f} != O.z={points['O'][2]:.5f}")

    # 3. Check lower arm vertical: C.x == O.x
    if abs(points['C'][0] - points['O'][0]) > 1e-5:
        errors.append(f"Lower Arm C must be vertical with Pivot O: C.x={points['C'][0]:.5f} != O.x={points['O'][0]:.5f}")

    # 4. Check parallel linkage P == B + C - O
    expected_P_x = points['B'][0] + points['C'][0] - points['O'][0]
    expected_P_z = points['B'][2] + points['C'][2] - points['O'][2]
    if abs(points['P'][0] - expected_P_x) > 1e-5 or abs(points['P'][2] - expected_P_z) > 1e-5:
        errors.append(
            f"Parallel linkage constraint violated: Point P must equal B + C - O. "
            f"P=[{points['P'][0]:.5f}, 0, {points['P'][2]:.5f}], expected=[{expected_P_x:.5f}, 0, {expected_P_z:.5f}]"
        )

    # 5. Check lower parallelogram E == D + C - O
    expected_E_x = points['D'][0] + points['C'][0] - points['O'][0]
    expected_E_z = points['D'][2] + points['C'][2] - points['O'][2]
    if abs(points['E'][0] - expected_E_x) > 1e-5 or abs(points['E'][2] - expected_E_z) > 1e-5:
        errors.append(
            f"Lower parallelogram constraint violated: Point E must equal D + C - O. "
            f"E=[{points['E'][0]:.5f}, 0, {points['E'][2]:.5f}], expected=[{expected_E_x:.5f}, 0, {expected_E_z:.5f}]"
        )

    # 6. Check extension horizontal: H.z == C.z
    if abs(points['H'][2] - points['C'][2]) > 1e-5:
        errors.append(f"Extension H must be horizontal with Pivot C: H.z={points['H'][2]:.5f} != C.z={points['C'][2]:.5f}")

    # 7. Check tool mount horizontal: EE.z == H.z
    if abs(points['EE'][2] - points['H'][2]) > 1e-5:
        errors.append(f"Tool Mount EE must be horizontally aligned with Extension H: EE.z={points['EE'][2]:.5f} != H.z={points['H'][2]:.5f}")

    # 8. Check tool tip vertical: TCP.x == EE.x
    if abs(points['TCP'][0] - points['EE'][0]) > 1e-5:
        errors.append(f"Tool Tip TCP must be vertically aligned with Mount EE: TCP.x={points['TCP'][0]:.5f} != EE.x={points['EE'][0]:.5f}")

    # 9. Check tool tip points down: TCP.z < EE.z
    if points['TCP'][2] >= points['EE'][2] - 1e-5:
        errors.append(f"Tool Tip TCP must point downwards relative to Mount EE: TCP.z={points['TCP'][2]:.5f} >= EE.z={points['EE'][2]:.5f}")

    # 10. Check critical segment lengths are larger than 1e-4 meters
    critical_pairs = [
        ('O', 'C', "Lower Arm"),
        ('O', 'B', "Crank"),
        ('P', 'H', "Upper Arm Ext"),
        ('F', 'G', "Upper Link"),
        ('H', 'G', "Tilt Linkage"),
        ('J4', 'EE', "Disk offset"),
        ('EE', 'TCP', "Tool extension")
    ]
    for first, second, label in critical_pairs:
        l_val = dist_xz(points[first], points[second])
        if l_val <= 1e-4:
            errors.append(f"Critical segment {label} ({first} to {second}) is too small: {l_val:.6f}m (minimum 0.0001m)")

    # 11. Validate triangle inequality for assembly circle intersection
    dist_FG = dist_xz(points['F'], points['G'])
    dist_HG = dist_xz(points['H'], points['G'])
    dist_FH = dist_xz(points['F'], points['H'])

    if dist_FH > dist_FG + dist_HG + 1e-9:
        errors.append(f"Assembly linkage circles centered at F and H do not intersect: FH ({dist_FH:.4f}m) > FG + HG ({(dist_FG + dist_HG):.4f}m)")

    if dist_FH < abs(dist_FG - dist_HG) - 1e-9:
        errors.append(f"Assembly linkage circles centered at F and H do not intersect: FH ({dist_FH:.4f}m) < |FG - HG| ({abs(dist_FG - dist_HG):.4f}m)")

    return errors, warnings


def validate_cr6_geometry(geom: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Validates standard DH geometry parameters for CR6.
    """
    errors = []
    warnings = []

    joints = geom.get('joints', [])
    if not isinstance(joints, list) or len(joints) != 6:
        errors.append(f"CR6 geometry must specify a 'joints' list of exactly 6 elements (got {len(joints) if isinstance(joints, list) else type(joints)})")
        return errors, warnings

    for idx, joint in enumerate(joints):
        name = joint.get('name', f"Joint {idx + 1}")
        for field_name in ['a_m', 'alpha_rad', 'd_m']:
            if field_name not in joint:
                errors.append(f"Joint {name} (index {idx}) is missing required field: {field_name}")
            else:
                try:
                    float(joint[field_name])
                except (ValueError, TypeError):
                    errors.append(f"Joint {name} (index {idx}) field '{field_name}' must be a numeric float")

    return errors, warnings
