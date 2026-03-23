"""CR4 validated dynamics runtime based on reference comparison script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pinocchio as pin


def _load_reference_module():
    ref_root = Path(__file__).resolve().parent / "reference"
    script_path = ref_root / "cr4_pinocchio_serial_vs_parallelogram.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("cr4_ref_compare", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load CR4 reference script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _smooth_array(arr: np.ndarray, window: int = 5) -> np.ndarray:
    if len(arr) < window:
        return arr
    out = np.copy(arr)
    half = window // 2
    for i in range(half, len(arr) - half):
        out[i] = np.mean(arr[i - half : i + half + 1], axis=0)
    return out


def _differentiate(q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n, nv = q.shape
    v = np.zeros((n, nv), dtype=float)
    for i in range(1, n - 1):
        v[i] = (q[i + 1] - q[i - 1]) / (2.0 * dt)
    if n > 1:
        v[0] = (q[1] - q[0]) / dt
        v[-1] = (q[-1] - q[-2]) / dt
    v = _smooth_array(v, window=7)

    a = np.zeros((n, nv), dtype=float)
    for i in range(1, n - 1):
        a[i] = (v[i + 1] - v[i - 1]) / (2.0 * dt)
    if n > 1:
        a[0] = (v[1] - v[0]) / dt
        a[-1] = (v[-1] - v[-2]) / dt
    a = _smooth_array(a, window=9)
    return v, a


def compute_cr4_validated_trajectory(
    trajectory_user5: List[np.ndarray],
    dt: float,
    scale: float,
    payload_kg: float,
    payload_inertia: Dict,
    reflected_inertia_full: List[float],
    friction_full: List[float],
    coulomb_full: List[float],
    motor_masses_axis4: List[float],
    parallel_motor_layout: str = "concentric_j2_j3act",
    torque_method: str = "hybrid_actuation",
    solver_method: str = "native_pinocchio",
    qd_user5: List[np.ndarray] | None = None,
    qdd_user5: List[np.ndarray] | None = None,
) -> Dict:
    ref = _load_reference_module()
    repo_root = Path(__file__).resolve().parents[2]
    ref_root = Path(__file__).resolve().parent / "reference"
    mat_file = str(ref_root / "cr4_params.mat")

    q_user = np.array(
        [np.array(q, dtype=float).reshape(5) for q in trajectory_user5], dtype=float
    )
    if qd_user5 is not None and qdd_user5 is not None:
        qd_user = np.array(
            [np.array(v, dtype=float).reshape(5) for v in qd_user5], dtype=float
        )
        qdd_user = np.array(
            [np.array(a, dtype=float).reshape(5) for a in qdd_user5], dtype=float
        )
    else:
        q_user = _smooth_array(q_user, window=5)
        qd_user, qdd_user = _differentiate(q_user, dt)
    q_theta = q_user + np.asarray(ref.CR4_DH_OFFSETS, dtype=float).reshape(1, 5)

    base_params = ref.load_params_from_mat(mat_file=mat_file, n_links=5)
    params = ref.ne.scale_rigid_body_params(base_params, float(scale))

    serial_model, serial_ee_frame_id, serial_joint_ids = ref.build_serial_model(params)
    serial_data = serial_model.createData()
    real_model, constraint_model, idx_v = ref.build_real_parallelogram_model(
        params, robodimm_path=str(repo_root), robot_scale=float(scale)
    )
    real_data = real_model.createData()
    tool0_id = real_model.getFrameId("tool0")

    ref.apply_inertia_mapping_serial_joint_transfer(
        serial_model,
        serial_data,
        serial_joint_ids,
        real_model,
        real_data,
        constraint_model,
    )

    m_axis4 = np.asarray(
        motor_masses_axis4 or [0.0, 0.0, 0.0, 0.0], dtype=float
    ).reshape(4)
    ref.apply_serial_motor_stator_masses(serial_model, serial_joint_ids, m_axis4)
    ref.apply_parallel_motor_stator_masses(
        real_model, m_axis4, layout=parallel_motor_layout
    )

    pI = payload_inertia or {}
    payload_case = {
        "mass": float(payload_kg or 0.0),
        "com_from_tcp": np.array(
            pI.get("com_from_tcp", [0.0, 0.0, 0.0]), dtype=float
        ).reshape(3),
        "inertia": np.diag(
            [
                float(pI.get("Ixx", 0.0)),
                float(pI.get("Iyy", 0.0)),
                float(pI.get("Izz", 0.0)),
            ]
        ),
    }
    ref.apply_payload_to_serial_model(
        serial_model, serial_joint_ids, params, payload_case
    )
    ref.apply_payload_to_real_model(real_model, payload_case, tool_frame_name="tool0")

    B = ref.build_actuator_projection_matrix(idx_v, real_model.nv)

    idx_act = [idx_v["J1"], idx_v["J2"], idx_v["J3real"], idx_v["J4"]]
    iref_axis4 = np.array(
        [
            float(reflected_inertia_full[idx_act[0]]),
            float(reflected_inertia_full[idx_act[1]]),
            float(reflected_inertia_full[idx_act[2]]),
            float(reflected_inertia_full[idx_act[3]]),
        ],
        dtype=float,
    )
    b_axis4 = np.array(
        [
            float(friction_full[idx_act[0]]),
            float(friction_full[idx_act[1]]),
            float(friction_full[idx_act[2]]),
            float(friction_full[idx_act[3]]),
        ],
        dtype=float,
    )
    c_axis4 = np.array(
        [
            float(coulomb_full[idx_act[0]]),
            float(coulomb_full[idx_act[1]]),
            float(coulomb_full[idx_act[2]]),
            float(coulomb_full[idx_act[3]]),
        ],
        dtype=float,
    )

    n = q_user.shape[0]
    q_real_hist = np.zeros((n, real_model.nq), dtype=float)
    v_real_hist = np.zeros((n, real_model.nv), dtype=float)
    a_real_hist = np.zeros((n, real_model.nv), dtype=float)
    tau_hist = np.zeros((n, real_model.nv), dtype=float)

    for i in range(n):
        q_s = q_theta[i]
        qd_s = qd_user[i]
        qdd_s = qdd_user[i]

        q_r, v_r, a_r = ref.build_real_state_from_user5(
            q_user[i], qd_user[i], qdd_user[i], idx_v, real_model.nq, real_model.nv
        )

        if torque_method in {"virtual_work", "hybrid_actuation"}:
            if solver_method == "native_pinocchio":
                tau_general = ref.compute_constrained_inverse_dynamics_native(
                    real_model, real_data, constraint_model, q_r, v_r, a_r
                )
            else:
                from .constrained import compute_constrained_inverse_dynamics

                tau_general = compute_constrained_inverse_dynamics(
                    real_model, real_data, constraint_model, q_r, v_r, a_r
                )
            tau_vw = B.T @ tau_general
            if torque_method == "virtual_work":
                tau_act = np.array(tau_vw, dtype=float)
            else:
                tau_act = np.array(
                    [tau_vw[0], tau_vw[1], tau_general[idx_v["J3"]], tau_vw[3]],
                    dtype=float,
                )
        else:
            from .constrained import compute_motor_inverse_dynamics

            tau_motor = compute_motor_inverse_dynamics(
                real_model,
                real_data,
                constraint_model,
                q_r,
                v_r,
                a_r,
                return_analysis=False,
                torque_method="legacy_motor_map",
            )
            tau_act = np.array(
                [
                    tau_motor[idx_v["J1"]],
                    tau_motor[idx_v["J2"]],
                    tau_motor[idx_v["J3real"]],
                    ref.J4_ACT_SIGN * tau_motor[idx_v["J4"]],
                ],
                dtype=float,
            )

        jdd3act = qdd_s[1] + qdd_s[2]
        qd3act = qd_s[1] + qd_s[2]
        sign_para = np.array(
            [
                np.sign(qd_s[0]) if abs(qd_s[0]) > 1e-9 else 0.0,
                np.sign(qd_s[1]) if abs(qd_s[1]) > 1e-9 else 0.0,
                np.sign(qd3act) if abs(qd3act) > 1e-9 else 0.0,
                np.sign(qd_s[4]) if abs(qd_s[4]) > 1e-9 else 0.0,
            ],
            dtype=float,
        )
        tau_act = tau_act + np.array(
            [
                iref_axis4[0] * qdd_s[0],
                iref_axis4[1] * qdd_s[1],
                iref_axis4[2] * jdd3act,
                iref_axis4[3] * qdd_s[4],
            ],
            dtype=float,
        )
        tau_act = tau_act + np.array(
            [
                b_axis4[0] * qd_s[0],
                b_axis4[1] * qd_s[1],
                b_axis4[2] * qd3act,
                b_axis4[3] * qd_s[4],
            ],
            dtype=float,
        )
        tau_act = tau_act + (c_axis4 * sign_para)

        tau_full = np.zeros((real_model.nv,), dtype=float)
        tau_full[idx_v["J1"]] = tau_act[0]
        tau_full[idx_v["J2"]] = tau_act[1]
        tau_full[idx_v["J3real"]] = tau_act[2]
        tau_full[idx_v["J4"]] = ref.J4_ACT_SIGN * tau_act[3]

        q_real_hist[i] = q_r
        v_real_hist[i] = v_r
        a_real_hist[i] = a_r
        tau_hist[i] = tau_full

        pin.forwardKinematics(serial_model, serial_data, q_s)
        pin.updateFramePlacements(serial_model, serial_data)
        pin.forwardKinematics(real_model, real_data, q_r)
        pin.updateFramePlacements(real_model, real_data)
        _ = (
            serial_data.oMf[serial_ee_frame_id].translation,
            real_data.oMf[tool0_id].translation,
        )

    return {
        "t": (np.arange(n, dtype=float) * float(dt)).tolist(),
        "q": q_real_hist.tolist(),
        "v": v_real_hist.tolist(),
        "a": a_real_hist.tolist(),
        "tau": tau_hist.tolist(),
        "method": "cr4_validated_runtime",
        "already_includes_motor_terms": True,
    }
