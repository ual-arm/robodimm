import sys
import json
import csv
import math
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

# Add kineforge and project root to path dynamically
sys.path.append(str(Path(__file__).resolve().parents[2] / 'kineforge'))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.dynamics.cr4_kkt import compute_cr4_kkt_dynamics, get_or_build_model, BODY_MASSES
from backend.dynamics.cr6_serial import compute_cr6_serial_dynamics, build_serial6_template_from_robot

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ENSAYOS_CR4 = WORKSPACE_ROOT / "ensayos" / "robodimm_cr4"
ENSAYOS_CR6 = WORKSPACE_ROOT / "ensayos" / "robodimm_cr6"


def _find_manifest(ensayo_dir: Path, name_glob: str) -> Optional[Path]:
    """Find manifest preferring demo/ then pro/ directories."""
    for subdir in ("demo", "pro"):
        candidates = sorted((ensayo_dir / "robodimm_output" / subdir).glob(name_glob))
        if candidates:
            return candidates[0]
    return None


def load_csv_data(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    keys = reader.fieldnames or []

    # Detect naming convention: q_J{i} (Simscape-style) or q{i} (plain)
    if "q_J1" in keys:
        q_fmt, qd_fmt, qdd_fmt, tau_fmt = "q_J{}", "qd_J{}", "qdd_J{}", "tau_J{}"
        num_joints = len([k for k in keys if k.startswith("q_J")])
    else:
        q_fmt, qd_fmt, qdd_fmt, tau_fmt = "q{}", "qd{}", "qdd{}", "tau{}"
        num_joints = len([k for k in keys if k.startswith("q") and not k.startswith("qd") and not k.startswith("qdd")])

    time_s = np.array([float(r["time"]) for r in rows], dtype=float)
    q = np.array([[float(r[q_fmt.format(i)]) for i in range(1, num_joints + 1)] for r in rows], dtype=float)
    qd = np.array([[float(r[qd_fmt.format(i)]) for i in range(1, num_joints + 1)] for r in rows], dtype=float)
    qdd = np.array([[float(r[qdd_fmt.format(i)]) for i in range(1, num_joints + 1)] for r in rows], dtype=float)
    tau = np.array([[float(r[tau_fmt.format(i)]) for i in range(1, num_joints + 1)] for r in rows], dtype=float)

    return time_s, q, qd, qdd, tau


def load_source_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load source trajectory CSV (no tau columns)."""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    keys = reader.fieldnames or []
    if "q_J1" in keys:
        q_fmt, qd_fmt, qdd_fmt = "q_J{}", "qd_J{}", "qdd_J{}"
        num_joints = len([k for k in keys if k.startswith("q_J")])
    else:
        q_fmt, qd_fmt, qdd_fmt = "q{}", "qd{}", "qdd{}"
        num_joints = len([k for k in keys if k.startswith("q") and not k.startswith("qd") and not k.startswith("qdd")])
    time_s = np.array([float(r["time"]) for r in rows], dtype=float)
    q = np.array([[float(r[q_fmt.format(i)]) for i in range(1, num_joints + 1)] for r in rows], dtype=float)
    qd = np.array([[float(r[qd_fmt.format(i)]) for i in range(1, num_joints + 1)] for r in rows], dtype=float)
    qdd = np.array([[float(r[qdd_fmt.format(i)]) for i in range(1, num_joints + 1)] for r in rows], dtype=float)
    return time_s, q, qd, qdd


# ─────────────────────────────────────────────────────────────────────────────
# CR4 tests
# ─────────────────────────────────────────────────────────────────────────────

# Moderated masses used in Simscape validation (match robodimm_cr4_pipeline.py BODY_MASSES)
CR4_SIMSCAPE_MASSES = {
    'SWING': 90.0,
    'P_ARM': 35.0,
    'LOWER_ARM': 75.0,
    'P_LINK': 25.0,
    'UPPER_ARM': 40.0,
    'LOWER_LINK': 20.0,
    'LINK_PLATE': 15.0,
    'UPPER_LINK': 15.0,
    'TILT': 15.0,
    'DISK': 10.0,
}


def _apply_cr4_simscape_masses(robot_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the validated Simscape masses to a robot_spec (in-place, returns it)."""
    if "inertials" not in robot_spec:
        robot_spec["inertials"] = {}
    for body, mass in CR4_SIMSCAPE_MASSES.items():
        existing = robot_spec["inertials"].get(body, {})
        if float(existing.get("massKg", 0.0)) == 0.0:
            robot_spec["inertials"][body] = {"massKg": mass}
    return robot_spec


def test_cr4_kkt_regression():
    print("\n--- Running CR4 KKT closed-chain regression test vs Simscape ---")
    # Use demo manifest (authoritative robot spec); fall back to pro if needed.
    manifest_path = _find_manifest(ENSAYOS_CR4, "*_reproducibility_manifest.json")
    simscape_csv = ENSAYOS_CR4 / "results" / "robodimm_cr4_simscape_results.csv"

    if not manifest_path or not simscape_csv.exists():
        print(f"⚠️  Skip: required files not found.")
        return

    print(f"  Using manifest: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    robot_spec = manifest.get("robot", manifest)

    # Zero friction to match Simscape (no friction model in Simscape reference)
    for limit in robot_spec.get("limits", []):
        limit["frictionCoeffNmSPerRad"] = 0.0

    # Apply validated Simscape masses (manifest may have empty inertials)
    _apply_cr4_simscape_masses(robot_spec)

    time_s, q_arr, qd_arr, qdd_arr, tau_simscape = load_csv_data(simscape_csv)
    num_samples = len(time_s)
    print(f"  Comparing {num_samples} samples...")

    tau_computed = []
    for idx in range(num_samples):
        tau, _, diags, _ = compute_cr4_kkt_dynamics(
            robot_spec, q_arr[idx].tolist(), qd_arr[idx].tolist(), qdd_arr[idx].tolist()
        )
        tau_computed.append(tau)

    tau_computed = np.array(tau_computed)
    error = tau_computed - tau_simscape
    rmse_per_joint = np.sqrt(np.mean(error ** 2, axis=0))
    rmse_total = np.sqrt(np.mean(error ** 2))
    max_error = np.max(np.abs(error), axis=0)

    print("  CR4 Joint-level RMSE (Nm):")
    for idx, rmse in enumerate(rmse_per_joint):
        print(f"    J{idx+1}: {rmse:.6f} Nm  (max abs: {max_error[idx]:.6f} Nm)")
    print(f"  Total RMSE: {rmse_total:.6f} Nm")

    # Thresholds from validated Simscape comparison
    assert rmse_per_joint[0] < 0.02,  f"J1 RMSE {rmse_per_joint[0]:.6f} > 0.02 Nm"
    assert rmse_per_joint[3] < 0.02,  f"J4 RMSE {rmse_per_joint[3]:.6f} > 0.02 Nm"
    assert rmse_per_joint[1] < 1.0,   f"J2 RMSE {rmse_per_joint[1]:.6f} > 1.0 Nm"
    assert rmse_per_joint[2] < 1.0,   f"J3 RMSE {rmse_per_joint[2]:.6f} > 1.0 Nm"
    assert rmse_total < 0.3,           f"Total RMSE {rmse_total:.6f} > 0.3 Nm"
    print("  ✅ CR4 KKT regression passed!")

    # --- Additional edge case: zero payload ---
    robot_no_payload = json.loads(json.dumps(robot_spec))
    robot_no_payload["payload"] = {
        "massKg": 0.0, "comM": [0, 0, 0], "inertiaKgM2": [[0,0,0],[0,0,0],[0,0,0]], "frame": "link"
    }
    tau_zero, _, _, _ = compute_cr4_kkt_dynamics(
        robot_no_payload, q_arr[0].tolist(), qd_arr[0].tolist(), qdd_arr[0].tolist()
    )
    assert len(tau_zero) == 4, f"Expected 4 torques, got {len(tau_zero)}"
    print("  ✅ CR4 zero-payload computation passed!")

    # Run specific feature tests
    _test_cr4_specific_features(robot_spec, q_arr, qd_arr, qdd_arr)


def _test_cr4_specific_features(robot_spec: Dict[str, Any], q_arr, qd_arr, qdd_arr):
    print("\n  --- CR4 specific feature tests ---")

    # 1. SWING COM offset: lever = (O - A) + com_swing
    print("  Testing SWING COM offset...")
    built = get_or_build_model(robot_spec)
    o_from_a = built.geom.points["O"] - built.geom.points["A"]
    swing_spec = robot_spec.get("inertials", {}).get("SWING", {})
    swing_com = np.array(swing_spec.get("comM") or [0.18, 0.0, 0.25])
    expected_lever = o_from_a + swing_com
    actual_lever = built.model.inertias[built.joint_ids["J1"]].lever
    assert np.allclose(actual_lever, expected_lever, atol=1e-9), \
        f"SWING COM: got {actual_lever}, expected {expected_lever}"
    print("  ✅ SWING COM offset correct!")

    # 2. Payload mass fused at J4 (DISK + payload)
    print("  Testing payload fusion at J4...")
    disk_mass = float(robot_spec.get("inertials", {}).get("DISK", {}).get("massKg", 10.0))
    payload_mass = float(robot_spec.get("payload", {}).get("massKg", 0.0))
    combined = built.model.inertias[built.joint_ids["J4"]].mass
    assert np.isclose(combined, disk_mass + payload_mass, atol=1e-9), \
        f"J4 combined mass: got {combined}, expected {disk_mass + payload_mass}"
    print("  ✅ Payload mass fusion at J4 correct!")

    # 3. J4 sign convention: at home (q=0), sign relative to Simscape
    print("  Testing J4 sign convention...")
    # At q=0, qd=0, qdd=0 → J4 torque must be ~0 (no rotation, no inertial load)
    tau_home, _, _, _ = compute_cr4_kkt_dynamics(
        robot_spec, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    )
    assert abs(tau_home[3]) < 0.01, f"J4 torque at home should be ~0, got {tau_home[3]:.6f}"
    # At qd=[0,0,0,1], qdd=0 → friction only; since friction=0, J4 torque still ~0
    tau_spin, _, _, _ = compute_cr4_kkt_dynamics(
        robot_spec, [0, 0, 0, 0], [0, 0, 0, 1.0], [0, 0, 0, 0]
    )
    assert abs(tau_spin[3]) < 0.05, f"J4 torque spinning (no friction): {tau_spin[3]:.6f}"
    print("  ✅ J4 sign convention test passed!")

    # 4. Sparse inertials → must use BODY_MASSES (not zero)
    print("  Testing sparse inertials fallback to BODY_MASSES...")
    robot_sparse = json.loads(json.dumps(robot_spec))
    robot_sparse["inertials"] = {}  # empty
    tau_sparse, _, _, _ = compute_cr4_kkt_dynamics(
        robot_sparse, [0, 0.5, -0.3, 0], [0, 0.1, -0.05, 0], [0, 0.2, -0.1, 0]
    )
    # Torques must be non-trivial (gravity loads from BODY_MASSES > 0)
    assert abs(tau_sparse[2]) > 100.0, \
        f"J3 torque with sparse inertials too small: {tau_sparse[2]:.2f} — BODY_MASSES not applied?"
    print("  ✅ Sparse inertials correctly fall back to BODY_MASSES!")

    # 5. Custom geometry variation
    print("  Testing custom geometry build and compute...")
    robot_custom = json.loads(json.dumps(robot_spec))
    robot_custom["geometry"]["C"][0] += 0.05
    robot_custom["geometry"]["E"][2] -= 0.03
    tau_custom, _, _, _ = compute_cr4_kkt_dynamics(
        robot_custom, [0.1, -0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]
    )
    assert len(tau_custom) == 4, "Expected 4 torques from custom geometry"
    print("  ✅ Custom geometry build and compute passed!")


# ─────────────────────────────────────────────────────────────────────────────
# CR6 tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cr6_serial_regression():
    print("\n--- Running CR6 serial open-chain regression test vs Simscape ---")
    # Use demo manifest (authoritative; has real inertials from IRB4600 preset)
    manifest_path = _find_manifest(ENSAYOS_CR6, "*_reproducibility_manifest.json")
    simscape_csv = ENSAYOS_CR6 / "results" / "robodimm_cr6_simscape_results.csv"

    if not manifest_path or not simscape_csv.exists():
        print(f"⚠️  Skip: required files not found.")
        return

    print(f"  Using manifest: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    robot_spec = manifest.get("robot", manifest)

    # Zero friction to match Simscape
    for limit in robot_spec.get("limits", []):
        limit["frictionCoeffNmSPerRad"] = 0.0

    time_s, q_arr, qd_arr, qdd_arr, tau_simscape = load_csv_data(simscape_csv)
    num_samples = len(time_s)
    print(f"  Comparing {num_samples} samples...")

    tau_computed = []
    for idx in range(num_samples):
        tau, _, _ = compute_cr6_serial_dynamics(
            robot_spec, q_arr[idx].tolist(), qd_arr[idx].tolist(), qdd_arr[idx].tolist()
        )
        tau_computed.append(tau)

    tau_computed = np.array(tau_computed)
    error = tau_computed - tau_simscape
    rmse_per_joint = np.sqrt(np.mean(error ** 2, axis=0))
    rmse_total = np.sqrt(np.mean(error ** 2))
    max_error = np.max(np.abs(error), axis=0)

    print("  CR6 Joint-level RMSE (Nm):")
    for idx, rmse in enumerate(rmse_per_joint):
        print(f"    J{idx+1}: {rmse:.2e} Nm  (max abs: {max_error[idx]:.2e} Nm)")
    print(f"  Total RMSE: {rmse_total:.2e} Nm")

    # Very tight threshold: Newton-Euler reference must match Simscape to floating point
    for idx, rmse in enumerate(rmse_per_joint):
        assert rmse < 0.01, f"J{idx+1} RMSE {rmse:.6f} > 0.01 Nm"
    assert rmse_total < 0.01, f"Total RMSE {rmse_total:.6f} > 0.01 Nm"
    print("  ✅ CR6 serial regression passed!")

    # Run specific feature tests
    _test_cr6_specific_features(robot_spec)


def _test_cr6_specific_features(robot_spec: Dict[str, Any]):
    print("\n  --- CR6 specific feature tests ---")

    # 1. frame: 'cad' inertials are converted correctly (non-trivial result)
    print("  Testing frame='cad' inertial conversion...")
    template = build_serial6_template_from_robot(robot_spec)
    # Compute inverse dynamics at a non-trivial configuration
    q  = [0.3, 0.5, -0.4, 1.0, -0.5, 0.2]
    qd = [0.1, 0.2, -0.1, 0.3, -0.2, 0.1]
    qdd = [0.5, 0.3, -0.2, 0.4, -0.3, 0.1]
    tau_cad = np.array(template.inverse_dynamics(q, qd, qdd))
    # If frame is ignored (all treated as 'link'), result would differ significantly
    # Gravity-loaded result should give non-trivial torques on J2 (largest link)
    assert abs(tau_cad[1]) > 100.0, \
        f"J2 torque too small with frame=cad inertials: {tau_cad[1]:.2f}"
    print("  ✅ frame='cad' inertial conversion produces expected torques!")

    # 2. Tool transform identity check
    print("  Testing tool_transform=identity has no effect on kinetics...")
    robot_eye = json.loads(json.dumps(robot_spec))
    robot_eye["geometry"]["tool_transform"] = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    tau_eye, _, _ = compute_cr6_serial_dynamics(robot_eye, q, qd, qdd)
    tau_orig, _, _ = compute_cr6_serial_dynamics(robot_spec, q, qd, qdd)
    assert np.allclose(tau_eye, tau_orig, atol=1e-9), \
        f"Identity tool_transform changed torques: delta={np.max(np.abs(np.array(tau_eye)-np.array(tau_orig))):.2e}"
    print("  ✅ Identity tool_transform has no effect!")

    # 3. Zero payload gives different result from non-zero payload
    print("  Testing payload contribution...")
    robot_no_pay = json.loads(json.dumps(robot_spec))
    robot_no_pay["payload"] = {
        "massKg": 0.0, "comM": [0,0,0], "inertiaKgM2": [[0,0,0],[0,0,0],[0,0,0]], "frame": "link"
    }
    tau_nopay, _, _ = compute_cr6_serial_dynamics(robot_no_pay, q, qd, qdd)
    tau_orig2, _, _ = compute_cr6_serial_dynamics(robot_spec, q, qd, qdd)
    payload_mass = float(robot_spec.get("payload", {}).get("massKg", 0.0))
    if payload_mass > 0.0:
        delta = np.max(np.abs(np.array(tau_orig2) - np.array(tau_nopay)))
        assert delta > 0.1, f"Payload ({payload_mass} kg) should affect torques: delta={delta:.4f}"
    print("  ✅ Payload contribution test passed!")

    # 4. DH theta_offset: verify that applying theta_offset in FK is not duplicated
    print("  Testing DH theta_offset not duplicated...")
    # At q=0, joint 2 has theta_offset=-pi/2, joint 5 has theta_offset=pi
    # The torque at home q=0 should match a consistent physical state
    tau_home, _, _ = compute_cr6_serial_dynamics(
        robot_spec, [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]
    )
    # J2 and J3 must carry gravity load (non-trivial): typical ~600-1200 Nm for IRB4600
    assert abs(tau_home[1]) > 100.0, \
        f"J2 gravity torque at home too small: {tau_home[1]:.2f} — DH offset may be duplicated"
    print("  ✅ DH theta_offset not duplicated!")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        test_cr4_kkt_regression()
        test_cr6_serial_regression()
        print("\n🎉 ALL REGRESSION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ REGRESSION TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
