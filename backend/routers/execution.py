"""
execution.py - Program Execution & Dynamics Endpoints
================================================

Endpoints for executing programs, computing dynamics, and exporting data.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
import numpy as np
import pinocchio as pin
import pink
import io
import csv
import math
import subprocess
from pathlib import Path
import json
import tempfile
import os

from ..session import SimulationSession, get_session
from ..utils import q_pink_to_frontend, q_frontend_to_pink, get_frontend_nq
from robot_core import (
    q_pink_to_real,
    interpolate_joint,
    interpolate_cartesian,
    interpolate_circular,
    compute_inverse_dynamics_trajectory,
    compare_dynamics_methods,
)
from robot_core.dynamics.cr4_validated_runtime import compute_cr4_validated_trajectory
from pink.tasks import FrameTask


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class ExecuteProgramCommand(BaseModel):
    speed_factor: float = 1.0  # multiplier for animation speed


class CR4AlignmentStudyCommand(BaseModel):
    scenario: str = "scale3_payload20_bigmotors_concentric_j2_j3"
    regenerate_demo: bool = True


class WebCompareCommand(BaseModel):
    run_execute: bool = True
    include_data: bool = False


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter()


# =============================================================================
# EXECUTION ENDPOINTS (Program Execution & Dynamics)
# =============================================================================


@router.post("/execute_program")
def execute_program(
    cmd: ExecuteProgramCommand, session: SimulationSession = Depends(get_session)
):
    """
    Execute program and return full trajectory for animation.
    Also computes inverse dynamics.
    Speed parameter in instructions: higher value = faster motion = fewer interpolation steps.
    Supports: MoveJ, MoveL, MoveC (circular), Pause
    """
    if not session.program:
        return {"ok": False, "error": "Program is empty"}

    # Set animation flag to pause WebSocket updates
    session.is_animating = True

    full_trajectory = []
    pause_points = []  # Store pause information: [(trajectory_index, pause_time), ...]
    dt = 0.02 / cmd.speed_factor  # Base dt for trajectory points

    # Maximum joint velocity (rad/s) - typical for industrial robots
    # CR6: axis 1-3 ~150deg/s = 2.6 rad/s, axis 4-6 ~300deg/s = 5.2 rad/s
    max_joint_vel = 2.5  # Conservative max joint velocity [rad/s]

    # Start from current position
    current_q = session.configuration.q.copy()

    # Get first non-Pause instruction to check if we need initial motion
    first_motion_instr = next(
        (i for i in session.program if i["type"] != "Pause"), None
    )
    if first_motion_instr:
        first_target = next(
            (
                t
                for t in session.targets
                if t["name"] == first_motion_instr["target_name"]
            ),
            None,
        )

        if first_target:
            # Convert target q from frontend format to Pink format
            q_first_frontend = np.array(first_target["q"])
            q_first = q_frontend_to_pink(
                q_first_frontend, current_q, session.current_robot_type
            )
            joint_disp = np.abs(q_first - current_q).max()  # Max joint displacement

            if joint_disp > 0.05:
                # Calculate steps based on realistic approach velocity (half max speed)
                approach_vel = max_joint_vel * 0.5
                approach_time = joint_disp / approach_vel
                approach_steps = max(30, int(approach_time / dt))
                approach_traj = interpolate_joint(current_q, q_first, approach_steps)
                full_trajectory.extend(approach_traj)
                current_q = q_first.copy()

    for instr in session.program:
        # Handle Pause instruction
        if instr["type"] == "Pause":
            pause_time = instr.get("pause_time", 1.0)
            # Record where in trajectory pause should occur
            pause_points.append({"index": len(full_trajectory), "time": pause_time})
            # Add duplicate points to hold position during pause
            pause_steps = int(pause_time / dt)
            for _ in range(pause_steps):
                full_trajectory.append(current_q.copy())
            continue

        target = next(
            (t for t in session.targets if t["name"] == instr["target_name"]), None
        )
        if not target:
            continue

        # Calculate number of steps based on joint distance and realistic velocity limits
        # Speed parameter: 50-200 range, where 100 = nominal velocity (~120deg/s = 2.1 rad/s for axes 1-3)
        # CR6 specs: axis 1-3: 150deg/s max, axis 4-6: 300deg/s max
        speed_factor_instr = max(10, min(200, instr.get("speed", 100))) / 100.0
        effective_max_vel = (
            max_joint_vel * speed_factor_instr
        )  # Actual max velocity for this move [rad/s]

        if instr["type"] == "MoveJ":
            # Joint interpolation - convert target q from frontend to Pink format
            q_end_frontend = np.array(target["q"])
            q_end = q_frontend_to_pink(
                q_end_frontend, current_q, session.current_robot_type
            )

            # Calculate steps based on max joint displacement and velocity
            joint_disp = np.abs(q_end - current_q).max()  # Max joint displacement [rad]
            move_time = joint_disp / effective_max_vel  # Time for move [s]
            steps = max(20, int(move_time / dt))  # Number of trajectory points

            # Use trapezoidal velocity profile for smooth motion (matches DEMO mode)
            traj_segment = interpolate_joint(
                current_q, q_end, steps, profile="trapezoidal", accel_fraction=0.2
            )

        elif instr["type"] == "MoveC":
            # Circular interpolation through via point
            via_target = next(
                (
                    t
                    for t in session.targets
                    if t["name"] == instr.get("via_target_name", "")
                ),
                None,
            )
            if not via_target:
                # Fallback to MoveL if no via point
                via_target = target

            # Use Pink model for forward kinematics (current_q is Pink format)
            pin.forwardKinematics(session.model_pink, session.data_pink, current_q)
            pin.updateFramePlacements(session.model_pink, session.data_pink)
            fe = session.data_pink.oMf[session.model_pink.getFrameId("end_effector")]

            start_pose = {
                "position": fe.translation.tolist(),
                "rotation": fe.rotation.flatten().tolist(),
            }
            via_pose = {
                "position": via_target["position"],
                "rotation": via_target["rotation"],
            }
            end_pose = {"position": target["position"], "rotation": target["rotation"]}

            # Estimate arc length for circular motion (simplified)
            cart_dist = np.linalg.norm(
                np.array(end_pose["position"]) - np.array(start_pose["position"])
            )
            cart_vel = effective_max_vel * 0.3  # ~0.75 m/s at full speed (conservative)
            move_time = (
                max(cart_dist * 1.5, 0.2) / cart_vel
            )  # Arc is ~1.5x chord length
            steps = max(30, int(move_time / dt))

            session.configuration.q = current_q.copy()
            traj_segment = interpolate_circular(
                session.model,
                session.data,
                session.configuration,
                session.ee_task,
                session.post_task,
                start_pose,
                via_pose,
                end_pose,
                steps,
            )

        else:  # MoveL
            # Cartesian interpolation with IK (position + orientation)
            pin.forwardKinematics(session.model_pink, session.data_pink, current_q)
            pin.updateFramePlacements(session.model_pink, session.data_pink)
            fe = session.data_pink.oMf[session.model_pink.getFrameId("end_effector")]

            start_pose = {
                "position": fe.translation.tolist(),
                "rotation": fe.rotation.flatten().tolist(),
            }
            end_pose = {"position": target["position"], "rotation": target["rotation"]}

            # Calculate steps based on Cartesian distance
            cart_dist = np.linalg.norm(
                np.array(end_pose["position"]) - np.array(start_pose["position"])
            )
            cart_vel = effective_max_vel * 0.3  # ~0.75 m/s at full speed
            move_time = max(cart_dist, 0.1) / cart_vel
            steps = max(20, int(move_time / dt))

            session.configuration.q = current_q.copy()
            traj_segment = interpolate_cartesian(
                session.model,
                session.data,
                session.configuration,
                session.ee_task,
                session.post_task,
                start_pose,
                end_pose,
                steps,
                use_orientation=True,
            )

        full_trajectory.extend(traj_segment)
        if traj_segment:
            current_q = np.array(traj_segment[-1])

    # Convert trajectory to Real space for dynamics
    real_trajectory = []
    for q_p in full_trajectory:
        q_r = q_pink_to_real(
            session.current_robot_type,
            session.model,
            session.data,
            session.constraint_model,
            q_p,
        )
        real_trajectory.append(q_r)

    if session.current_robot_type == "CR4":
        user5_traj = [np.array(q, dtype=float).reshape(5) for q in full_trajectory]
        session.last_trajectory_data = compute_cr4_validated_trajectory(
            trajectory_user5=user5_traj,
            dt=dt,
            scale=float(session.current_scale),
            payload_kg=float(session.current_payload_kg),
            payload_inertia=session.current_payload_inertia or {},
            reflected_inertia_full=session.robot_reflected_inertia,
            friction_full=session.robot_friction_coeffs,
            coulomb_full=session.robot_coulomb_friction,
            motor_masses_axis4=(
                list(session.robot_motor_masses)
                if getattr(session, "robot_motor_masses", None) is not None
                else [0.0, 0.0, 0.0, 0.0]
            ),
            parallel_motor_layout=getattr(
                session, "current_motor_layout", "concentric_j2_j3act"
            ),
            torque_method="hybrid_actuation",
            solver_method="native_pinocchio",
        )
    else:
        session.last_trajectory_data = compute_inverse_dynamics_trajectory(
            session.model,
            session.data,
            real_trajectory,
            dt,
            friction_coeffs=session.robot_friction_coeffs,
            constraint_model=session.constraint_model,
            use_constrained_dynamics=True,
        )

    # Add reflected inertia and Coulomb friction (validated script model terms)
    if (
        session.last_trajectory_data
        and session.current_robot_type == "CR4"
        and not session.last_trajectory_data.get("already_includes_motor_terms", False)
    ):
        tau = np.array(session.last_trajectory_data.get("tau", []), dtype=float)
        v = np.array(session.last_trajectory_data.get("v", []), dtype=float)
        a = np.array(session.last_trajectory_data.get("a", []), dtype=float)
        if tau.size > 0 and v.size > 0 and a.size > 0:
            idx = [0, 1, 2, 7]  # CR4 actuated joints in real model
            iref = np.array(
                session.robot_reflected_inertia
                if session.robot_reflected_inertia is not None
                else np.zeros(session.model.nv),
                dtype=float,
            )
            coul = np.array(
                session.robot_coulomb_friction
                if session.robot_coulomb_friction is not None
                else np.zeros(session.model.nv),
                dtype=float,
            )
            iref_mode = str(
                getattr(session, "current_iref_model_mode", "diag") or "diag"
            ).lower()
            for j in idx:
                # q5_physical is embedded in model inertia at build time.
                if not (iref_mode in {"q5_physical", "q3_q5_physical"} and j == 7):
                    tau[:, j] += iref[j] * a[:, j]
                sign_v = np.where(np.abs(v[:, j]) > 1e-9, np.sign(v[:, j]), 0.0)
                tau[:, j] += coul[j] * sign_v
            session.last_trajectory_data["tau"] = tau.tolist()

    # Also store for actuator selection system

    # Update robot to final position
    if full_trajectory:
        session.configuration.q = np.array(full_trajectory[-1])
        session.q[:] = real_trajectory[-1]
        session.post_task.set_target(session.configuration.q)

    # Convert trajectory from Pink format to Frontend format for display
    frontend_trajectory = []
    for q_p in full_trajectory:
        q_frontend = q_pink_to_frontend(np.array(q_p), session.current_robot_type)
        frontend_trajectory.append(q_frontend.tolist())

    return {
        "ok": True,
        "trajectory": frontend_trajectory,
        "dt": dt,
        "num_points": len(frontend_trajectory),
        "pause_points": pause_points,
    }


@router.post("/animation_done")
def animation_done(session: SimulationSession = Depends(get_session)):
    """Called by frontend when animation finishes to resume WebSocket updates."""
    session.is_animating = False
    return {"ok": True}


@router.get("/trajectory_data")
def get_trajectory_data(session: SimulationSession = Depends(get_session)):
    """Get the last execution's dynamics data for plotting."""
    if session.last_trajectory_data is None:
        return {
            "ok": False,
            "error": "No trajectory data available. Execute a program first.",
        }
    data = session.last_trajectory_data
    if session.current_robot_type == "CR4":
        data = _compress_cr4_dynamics_for_frontend(data)
    return {"ok": True, "data": data}


def _compress_cr4_dynamics_for_frontend(data: dict):
    """Expose CR4 active joints only: [J1, J2, J3act, J4]."""
    out = dict(data)
    try:
        q = np.array(data.get("q", []), dtype=float)
        v = np.array(data.get("v", []), dtype=float)
        a = np.array(data.get("a", []), dtype=float)
        tau = np.array(data.get("tau", []), dtype=float)

        if q.ndim == 2 and q.shape[1] >= 8:
            out["q"] = np.column_stack([q[:, 0], q[:, 1], q[:, 2], -q[:, 7]]).tolist()
        if v.ndim == 2 and v.shape[1] >= 8:
            out["v"] = np.column_stack([v[:, 0], v[:, 1], v[:, 2], -v[:, 7]]).tolist()
        if a.ndim == 2 and a.shape[1] >= 8:
            out["a"] = np.column_stack([a[:, 0], a[:, 1], a[:, 2], -a[:, 7]]).tolist()
        if tau.ndim == 2 and tau.shape[1] >= 8:
            out["tau"] = np.column_stack(
                [tau[:, 0], tau[:, 1], tau[:, 2], -tau[:, 7]]
            ).tolist()
    except Exception:
        return data
    return out


def _load_csv_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = {}
            for k, v in row.items():
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = math.nan
            rows.append(out)
    return rows


def _load_kv_csv(path: Path):
    out = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("key") or "").strip()
            if not key:
                continue
            try:
                out[key] = float((row.get("value") or "0").strip())
            except Exception:
                pass
    return out


def _metrics(ref: np.ndarray, est: np.ndarray):
    err = est - ref
    corr = (
        float(np.corrcoef(ref, est)[0, 1])
        if np.std(ref) > 0 and np.std(est) > 0
        else float("nan")
    )
    return {
        "corr": corr,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


@router.post("/debug_web_compare")
def debug_web_compare(
    cmd: WebCompareCommand, session: SimulationSession = Depends(get_session)
):
    if cmd.run_execute:
        execute_program(ExecuteProgramCommand(speed_factor=1.0), session)

    if session.last_trajectory_data is None:
        return {"ok": False, "error": "No trajectory data. Execute program first."}

    pro = _compress_cr4_dynamics_for_frontend(session.last_trajectory_data)

    payload = {
        "robotType": session.current_robot_type,
        "program": session.program,
        "targets": session.targets,
        "currentQ": session.targets[0].get("q", [0, 0, 0, 0])
        if session.targets
        else [0, 0, 0, 0],
        "dt": 0.02,
        "config": {
            "robot_type": session.current_robot_type,
            "scale": session.current_scale,
            "payload_kg": session.current_payload_kg,
            "payload_inertia": session.current_payload_inertia or {},
            "friction_coeffs": [0.0, 0.0, 0.0, 0.0]
            if session.current_friction_coeffs is None
            else list(session.current_friction_coeffs),
            "reflected_inertia": [0.0, 0.0, 0.0, 0.0]
            if session.current_reflected_inertia is None
            else list(session.current_reflected_inertia),
            "coulomb_friction": [0.0, 0.0, 0.0, 0.0]
            if session.current_coulomb_friction is None
            else list(session.current_coulomb_friction),
            "motor_masses": [0.0, 0.0, 0.0, 0.0]
            if session.current_motor_masses is None
            else list(session.current_motor_masses),
        },
    }

    in_fd, in_path = tempfile.mkstemp(suffix=".json")
    os.close(in_fd)
    out_fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(out_fd)
    try:
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        root = Path(__file__).resolve().parent.parent.parent
        subprocess.run(
            [
                "node",
                str(root / "scripts" / "compute_demo_from_program.mjs"),
                in_path,
                out_path,
            ],
            cwd=str(root),
            check=True,
        )
        with open(out_path, "r", encoding="utf-8") as f:
            demo = json.load(f)
    except Exception as e:
        return {"ok": False, "error": f"Could not compute DEMO-side data: {e}"}
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)

    n = min(len(pro.get("t", [])), len(demo.get("t", [])))
    if n <= 0:
        return {"ok": False, "error": "Empty data"}

    names = ["J1", "J2", "J3", "J4"]
    out_metrics = {}
    for var in ["q", "v", "a", "tau"]:
        p = np.array(pro.get(var, [])[:n], dtype=float)
        d = np.array(demo.get(var, [])[:n], dtype=float)
        if p.ndim != 2 or d.ndim != 2:
            continue
        out_metrics[var] = {
            names[j]: _metrics(p[:, j], d[:, j])
            for j in range(min(4, p.shape[1], d.shape[1]))
        }

    response = {
        "ok": True,
        "num_points": n,
        "duration_pro": float(pro.get("t", [0])[:n][-1]),
        "duration_demo": float(demo.get("t", [0])[:n][-1]),
        "metrics_pro_vs_demo": out_metrics,
    }
    if cmd.include_data:
        response["pro"] = {
            "t": pro.get("t", [])[:n],
            "q": pro.get("q", [])[:n],
            "v": pro.get("v", [])[:n],
            "a": pro.get("a", [])[:n],
            "tau": pro.get("tau", [])[:n],
        }
        response["demo"] = {
            "t": demo.get("t", [])[:n],
            "q": demo.get("q", [])[:n],
            "v": demo.get("v", [])[:n],
            "a": demo.get("a", [])[:n],
            "tau": demo.get("tau", [])[:n],
        }
    return response


@router.post("/cr4_alignment_study")
def run_cr4_alignment_study(
    cmd: CR4AlignmentStudyCommand, session: SimulationSession = Depends(get_session)
):
    root = Path(__file__).resolve().parent.parent.parent
    scripts_root = root / "scripts"
    if not scripts_root.is_dir():
        return {
            "ok": False,
            "error": "CR4 alignment benchmark assets were moved to the companion repository 'robodimm_paper'.",
        }

    scenario = root / "scripts" / "results" / cmd.scenario
    traj_csv = scenario / "cr4_serial5_source_trajectory.csv"
    simscape_csv = scenario / "cr4_simscape_results.csv"
    sim_params_csv = scenario / "cr4_serial5_source_sim_params.csv"
    demo_csv = scenario / "cr4_demo_dynamics_results.csv"

    if (
        not traj_csv.is_file()
        or not simscape_csv.is_file()
        or not sim_params_csv.is_file()
    ):
        return {"ok": False, "error": f"Scenario files missing in {scenario}"}

    if cmd.regenerate_demo or not demo_csv.is_file():
        try:
            subprocess.run(
                [
                    "node",
                    str(scripts_root / "compute_demo_cr4_torques.mjs"),
                    str(traj_csv),
                    str(sim_params_csv),
                    str(demo_csv),
                ],
                check=True,
                cwd=str(root),
            )
        except Exception as e:
            return {"ok": False, "error": f"Could not generate DEMO csv: {e}"}

    traj = _load_csv_rows(traj_csv)
    demo = _load_csv_rows(demo_csv)
    simscape = _load_csv_rows(simscape_csv)
    params = _load_kv_csv(sim_params_csv)

    if len(traj) != len(demo):
        return {"ok": False, "error": "Trajectory and DEMO row counts do not match"}

    offsets = np.array([0.0, -np.pi / 2, np.pi / 2, np.pi, 0.0], dtype=float)
    q_user = []
    qd_user = []
    qdd_user = []
    for r in traj:
        q_theta = np.array([r["q1"], r["q2"], r["q3"], r["q4"], r["q5"]], dtype=float)
        q_user.append((q_theta - offsets).tolist())
        qd_user.append([r["qd1"], r["qd2"], r["qd3"], r["qd4"], r["qd5"]])
        qdd_user.append([r["qdd1"], r["qdd2"], r["qdd3"], r["qdd4"], r["qdd5"]])

    scale = float(params.get("robot_scale", 1.0))
    payload_kg = float(params.get("payload_mass_kg", 0.0))
    payload_inertia = {
        "Ixx": float(params.get("payload_Ixx_kgm2", 0.0)),
        "Iyy": float(params.get("payload_Iyy_kgm2", 0.0)),
        "Izz": float(params.get("payload_Izz_kgm2", 0.0)),
        "com_from_tcp": [
            float(params.get("payload_com_x_m", 0.0)),
            float(params.get("payload_com_y_m", 0.0)),
            float(params.get("payload_com_z_m", 0.0)),
        ],
    }
    friction4 = [
        float(params.get("friction_viscous_q1", 0.0)),
        float(params.get("friction_viscous_q2", 0.0)),
        float(params.get("friction_viscous_q3", 0.0)),
        float(params.get("friction_viscous_q5", 0.0)),
    ]
    iref4 = [
        float(params.get("rotor_inertia_reflected_q1", 0.0)),
        float(params.get("rotor_inertia_reflected_q2", 0.0)),
        float(params.get("rotor_inertia_reflected_q3", 0.0)),
        float(params.get("rotor_inertia_reflected_q5", 0.0)),
    ]
    coul4 = [
        float(params.get("friction_coulomb_q1", 0.0)),
        float(params.get("friction_coulomb_q2", 0.0)),
        float(params.get("friction_coulomb_q3", 0.0)),
        float(params.get("friction_coulomb_q5", 0.0)),
    ]
    motor_masses = [
        float(params.get("motor_mass_q1_kg", 0.0)),
        float(params.get("motor_mass_q2_kg", 0.0)),
        float(params.get("motor_mass_q3_kg", 0.0)),
        float(params.get("motor_mass_q5_kg", 0.0)),
    ]

    t_traj = np.array([r["time"] for r in traj], dtype=float)
    dt = float(np.mean(np.diff(t_traj))) if len(t_traj) > 1 else 0.02
    dyn = compute_cr4_validated_trajectory(
        trajectory_user5=q_user,
        qd_user5=qd_user,
        qdd_user5=qdd_user,
        dt=dt,
        scale=scale,
        payload_kg=payload_kg,
        payload_inertia=payload_inertia,
        reflected_inertia_full=[
            iref4[0],
            iref4[1],
            iref4[2],
            0.0,
            0.0,
            0.0,
            0.0,
            iref4[3],
        ],
        friction_full=[
            friction4[0],
            friction4[1],
            friction4[2],
            0.0,
            0.0,
            0.0,
            0.0,
            friction4[3],
        ],
        coulomb_full=[coul4[0], coul4[1], coul4[2], 0.0, 0.0, 0.0, 0.0, coul4[3]],
        motor_masses_axis4=motor_masses,
        parallel_motor_layout="concentric_j2_j3act",
        torque_method="hybrid_actuation",
        solver_method="native_pinocchio",
    )

    tau_all = np.array(dyn["tau"], dtype=float)
    tau_pro = np.column_stack(
        [tau_all[:, 0], tau_all[:, 1], tau_all[:, 2], -tau_all[:, 7]]
    )
    tau_demo = np.column_stack(
        [
            [r["tau1_demo"] for r in demo],
            [r["tau2_demo"] for r in demo],
            [r["tau3_demo"] for r in demo],
            [r["tau4_demo"] for r in demo],
        ]
    )
    tau_sc_raw = np.column_stack(
        [
            [r["tau1"] for r in simscape],
            [r["tau2"] - r["tau3"] for r in simscape],
            [r["tau3"] for r in simscape],
            [r["tau5"] for r in simscape],
        ]
    )
    t_sc = np.array([r["time"] for r in simscape], dtype=float)
    tau_sc = np.zeros_like(tau_demo)
    for i, t in enumerate(t_traj):
        tau_sc[i] = tau_sc_raw[int(np.argmin(np.abs(t_sc - t)))]

    names = ["J1", "J2", "J3", "J4"]
    demo_vs_pro = {names[j]: _metrics(tau_pro[:, j], tau_demo[:, j]) for j in range(4)}
    simscape_vs_pro = {
        names[j]: _metrics(tau_sc[:, j], tau_pro[:, j]) for j in range(4)
    }

    return {
        "ok": True,
        "scenario": cmd.scenario,
        "num_points": int(len(t_traj)),
        "demo_vs_pro": demo_vs_pro,
        "simscape_vs_pro": simscape_vs_pro,
        "method": "cr4_validated_runtime",
        "robot_type": session.current_robot_type,
    }


@router.get("/export_csv")
def export_csv(session: SimulationSession = Depends(get_session)):
    """Export trajectory data as CSV string."""
    if session.last_trajectory_data is None:
        return {"ok": False, "error": "No trajectory data available"}

    output = io.StringIO()
    writer = csv.writer(output)

    # Header (using 1-6 joint numbering as per robotics convention)
    nq = (
        len(session.last_trajectory_data["q"][0])
        if session.last_trajectory_data["q"]
        else 0
    )
    header = ["t"]
    header += [f"q{i + 1}" for i in range(nq)]
    header += [f"v{i + 1}" for i in range(nq)]
    header += [f"a{i + 1}" for i in range(nq)]
    header += [f"tau{i + 1}" for i in range(nq)]
    writer.writerow(header)

    # Data rows
    for i in range(len(session.last_trajectory_data["t"])):
        row = [session.last_trajectory_data["t"][i]]
        row += session.last_trajectory_data["q"][i]
        row += session.last_trajectory_data["v"][i]
        row += session.last_trajectory_data["a"][i]
        row += session.last_trajectory_data["tau"][i]
        writer.writerow(row)

    csv_content = output.getvalue()
    output.close()

    from fastapi.responses import Response

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trajectory.csv"},
    )


@router.get("/export_dynamics_comparison")
def export_dynamics_comparison(session: SimulationSession = Depends(get_session)):
    """
    Export comprehensive dynamics comparison data for validation with external tools.

    This endpoint provides data suitable for comparison with:
    - Simscape Multibody (MATLAB/Simulink)
    - Adams
    - Other multibody dynamics software

    The export includes:
    - Constrained dynamics (KKT with Lagrange multipliers) - physically accurate
    - Open-chain RNEA (for comparison) - ignores closed-loop effects
    - Lagrange multipliers (constraint forces in parallelogram)
    - Constraint violations (should be near zero)

    Returns JSON with all comparison data for offline analysis.
    """
    if session.last_trajectory_data is None:
        return {
            "ok": False,
            "error": "No trajectory data available. Execute a program first.",
        }

    # Get the real trajectory from stored data
    real_trajectory = [np.array(q) for q in session.last_trajectory_data["q"]]
    dt = (
        session.last_trajectory_data["t"][1] - session.last_trajectory_data["t"][0]
        if len(session.last_trajectory_data["t"]) > 1
        else 0.01
    )

    # Compute full comparison using compare_dynamics_methods
    try:
        comparison = compare_dynamics_methods(
            session.model,
            session.data,
            session.constraint_model,
            real_trajectory,
            dt,
            friction_coeffs=session.current_friction_coeffs,
        )

        # Add metadata for external tools
        comparison["metadata"] = {
            "robot_type": session.current_robot_type,
            "num_joints": session.model.nv,
            "num_constraints": session.constraint_model.size()
            if session.constraint_model
            else 0,
            "dt": dt,
            "num_points": len(real_trajectory),
            "joint_names": [
                session.model.names[i + 1] for i in range(session.model.njoints - 1)
            ],
            "units": {
                "time": "s",
                "position": "rad",
                "velocity": "rad/s",
                "acceleration": "rad/s^2",
                "torque": "Nm",
                "constraint_force": "N",
                "constraint_violation": "m",
            },
            "description": "Closed kinematic chain inverse dynamics using Lagrange multipliers",
            "method": "KKT formulation with Delassus matrix and Baumgarte stabilization",
        }

        return {"ok": True, "comparison": comparison}

    except Exception as e:
        return {"ok": False, "error": f"Error computing comparison: {str(e)}"}


@router.get("/export_dynamics_csv_full")
def export_dynamics_csv_full(session: SimulationSession = Depends(get_session)):
    """
    Export full dynamics comparison as CSV for Simscape Multibody validation.

    Columns include:
    - Time
    - Joint positions (q1..qN)
    - Joint velocities (v1..vN)
    - Joint accelerations (a1..aN)
    - Constrained torques (tau_constrained_1..N) - KKT method
    - RNEA torques (tau_rnea_1..N) - open chain
    - Torque correction (tau_diff_1..N) - difference due to constraint forces
    - Lagrange multipliers (lambda_1..3) - constraint forces
    - Constraint violation
    """
    if session.last_trajectory_data is None:
        return {
            "ok": False,
            "error": "No trajectory data available. Execute a program first.",
        }

    real_trajectory = [np.array(q) for q in session.last_trajectory_data["q"]]
    dt = (
        session.last_trajectory_data["t"][1] - session.last_trajectory_data["t"][0]
        if len(session.last_trajectory_data["t"]) > 1
        else 0.01
    )

    try:
        comparison = compare_dynamics_methods(
            session.model,
            session.data,
            session.constraint_model,
            real_trajectory,
            dt,
            friction_coeffs=session.current_friction_coeffs,
        )

        output = io.StringIO()
        writer = csv.writer(output)

        nq = (
            len(session.last_trajectory_data["q"][0])
            if session.last_trajectory_data["q"]
            else 0
        )
        nc = (
            len(comparison["lambda"][0])
            if comparison["lambda"] and len(comparison["lambda"]) > 0
            else 3
        )

        # Build header
        header = ["t"]
        header += [f"q{i + 1}" for i in range(nq)]
        header += [f"v{i + 1}" for i in range(nq)]
        header += [f"a{i + 1}" for i in range(nq)]
        header += [f"tau_constrained_{i + 1}" for i in range(nq)]
        header += [f"tau_rnea_{i + 1}" for i in range(nq)]
        header += [f"tau_diff_{i + 1}" for i in range(nq)]
        header += [f"lambda_{i + 1}" for i in range(nc)]
        header += ["constraint_violation"]
        writer.writerow(header)

        # Data rows
        tau_constrained = comparison["tau_constrained"]
        tau_rnea = comparison["tau_rnea_raw"]
        tau_diff = comparison["torque_diff_constrained_rnea"]
        lambdas = comparison["lambda"]
        violations = comparison["constraint_violation"]

        for i in range(len(comparison["t"])):
            row = [comparison["t"][i]]
            row += session.last_trajectory_data["q"][i]
            row += session.last_trajectory_data["v"][i]
            row += session.last_trajectory_data["a"][i]
            row += tau_constrained[i] if i < len(tau_constrained) else [0] * nq
            row += tau_rnea[i] if i < len(tau_rnea) else [0] * nq
            row += tau_diff[i] if i < len(tau_diff) else [0] * nq
            row += lambdas[i] if i < len(lambdas) else [0] * nc
            row += [violations[i] if i < len(violations) else 0]
            writer.writerow(row)

        csv_content = output.getvalue()
        output.close()

        from fastapi.responses import Response

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=dynamics_comparison_{session.current_robot_type}.csv"
            },
        )

    except Exception as e:
        import traceback

        return {
            "ok": False,
            "error": f"Error exporting: {str(e)}",
            "traceback": traceback.format_exc(),
        }


@router.get("/dynamics_summary")
def get_dynamics_summary(session: SimulationSession = Depends(get_session)):
    """
    Get a summary of dynamics analysis including constraint force statistics.

    Useful for quick verification that constrained dynamics is working correctly.
    """
    if session.last_trajectory_data is None:
        return {"ok": False, "error": "No trajectory data available"}

    # Check if we have RNEA comparison data
    has_comparison = "tau_rnea" in session.last_trajectory_data

    tau = np.array(session.last_trajectory_data["tau"])

    summary = {
        "robot_type": session.current_robot_type,
        "trajectory_duration_s": session.last_trajectory_data["t"][-1]
        if session.last_trajectory_data["t"]
        else 0,
        "num_points": len(session.last_trajectory_data["t"]),
        "num_joints": tau.shape[1] if len(tau.shape) > 1 else 0,
        "dynamics_method": session.last_trajectory_data.get("method", "unknown"),
        "has_constraint_comparison": has_comparison,
        "torque_statistics": {},
    }

    # Per-joint statistics
    joint_names = [
        session.model.names[i + 1]
        for i in range(min(session.model.njoints - 1, tau.shape[1]))
    ]
    for i, jname in enumerate(joint_names):
        if i < tau.shape[1]:
            summary["torque_statistics"][jname] = {
                "peak_Nm": float(np.max(np.abs(tau[:, i]))),
                "rms_Nm": float(np.sqrt(np.mean(tau[:, i] ** 2))),
                "mean_Nm": float(np.mean(tau[:, i])),
            }

    # If we have RNEA comparison, compute correction statistics
    if has_comparison:
        tau_rnea = np.array(session.last_trajectory_data["tau_rnea"])
        diff = tau - tau_rnea
        summary["constraint_correction"] = {
            "max_correction_Nm": float(np.max(np.abs(diff))),
            "mean_correction_Nm": float(np.mean(np.abs(diff))),
            "description": "Difference between constrained (KKT) and open-chain (RNEA) torques",
        }

    return {"ok": True, "summary": summary}
