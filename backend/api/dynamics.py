import json
import hashlib
import math
import time
import pinocchio as pin
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from datetime import datetime

from backend.dynamics.schemas import (
    InverseDynamicsRequest,
    DynamicsBatchRequest,
    RobotSpecModel,
    DynamicsResponse,
    DynamicsBatchResponse,
    BatchSampleResponse,
    DynamicsManifestModel,
    ValidationResponse,
    LegacyDynamicsRequest,
    CR4DiagnosticsModel
)
from backend.dynamics.validation import validate_cr4_geometry, validate_cr6_geometry
from backend.dynamics.cr4_kkt import compute_cr4_kkt_dynamics, compute_cr4_kkt_batch
from backend.dynamics.cr6_serial import compute_cr6_serial_dynamics, compute_cr6_serial_batch

router = APIRouter()

def get_robot_hash(robot: dict) -> str:
    # Serialize to deterministic JSON string
    serialized = json.dumps(robot, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

def get_trajectory_hash(samples: list) -> str:
    clean_samples = []
    for s in samples:
        if isinstance(s, dict):
            clean_samples.append({
                "time_s": float(s.get("time_s", 0.0)),
                "q": [float(x) for x in s.get("q", [])],
                "qd": [float(x) for x in s.get("qd", [])],
                "qdd": [float(x) for x in s.get("qdd", [])]
            })
        else:
            clean_samples.append({
                "time_s": float(s.time_s),
                "q": [float(x) for x in s.q],
                "qd": [float(x) for x in s.qd],
                "qdd": [float(x) for x in s.qdd]
            })
    serialized = json.dumps(clean_samples, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

def make_manifest(robot: dict, model_id: str, trajectory_hash: str = None) -> DynamicsManifestModel:
    return DynamicsManifestModel(
        model_id=model_id,
        backend_version="1.0.0",
        pinocchio_version=pin.__version__,
        robot_hash=get_robot_hash(robot),
        trajectory_hash=trajectory_hash,
        q_space_convention="standard_dh_with_offsets",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

@router.post("/dynamics/inverse", response_model=DynamicsResponse)
def calculate_inverse_dynamics(req: InverseDynamicsRequest):
    robot_dict = req.robot.model_dump()
    q = req.q
    qd = req.qd
    qdd = req.qdd
    options = req.options or {}

    # Check joint counts
    expected_joints = 4 if req.robot.kind == 'CR4' else 6
    if len(q) != expected_joints or len(qd) != expected_joints or len(qdd) != expected_joints:
        raise HTTPException(
            status_code=400,
            detail=f"Joint coordinate mismatch: kind {req.robot.kind} expects {expected_joints} elements, got q={len(q)}, qd={len(qd)}, qdd={len(qdd)}"
        )

    try:
        if req.robot.kind == 'CR4':
            errors, warnings = validate_cr4_geometry(req.robot.geometry)
            if errors:
                raise HTTPException(status_code=400, detail=f"CR4 geometry validation failed: {errors}")
            
            tau, power, _, cr4_warnings = compute_cr4_kkt_dynamics(robot_dict, q, qd, qdd, options)
            all_warnings = warnings + cr4_warnings
            return DynamicsResponse(
                tauNm=tau,
                powerW=power,
                engine_used="pro_cr4_kkt",
                model_id="cr4_pinocchio_kkt.v1",
                manifest=make_manifest(robot_dict, "cr4_pinocchio_kkt.v1"),
                warnings=all_warnings
            )
        else:
            errors, warnings = validate_cr6_geometry(req.robot.geometry)
            if errors:
                raise HTTPException(status_code=400, detail=f"CR6 geometry validation failed: {errors}")
            
            tau, power, cr6_warnings = compute_cr6_serial_dynamics(robot_dict, q, qd, qdd, options)
            all_warnings = warnings + cr6_warnings
            return DynamicsResponse(
                tauNm=tau,
                powerW=power,
                engine_used="pro_cr6_serial",
                model_id="cr6_serial6_template.v1",
                manifest=make_manifest(robot_dict, "cr6_serial6_template.v1"),
                warnings=all_warnings
            )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


@router.post("/dynamics/batch", response_model=DynamicsBatchResponse)
def calculate_batch_dynamics(req: DynamicsBatchRequest):
    robot_dict = req.robot.model_dump()
    samples = req.samples
    options = req.options or {}
    
    # 1. Enforce batch size limit (Pydantic does it, but we can do extra checks)
    if not samples:
        raise HTTPException(status_code=400, detail="Trajectory batch must contain at least 1 sample")

    # 2. Check joint counts on all samples
    expected_joints = 4 if req.robot.kind == 'CR4' else 6
    for idx, s in enumerate(samples):
        if len(s.q) != expected_joints or len(s.qd) != expected_joints or len(s.qdd) != expected_joints:
            raise HTTPException(
                status_code=400,
                detail=f"Joint coordinate mismatch in sample index {idx}: kind {req.robot.kind} expects {expected_joints} elements"
            )

    # 3. Geometric Invariant Checks
    try:
        if req.robot.kind == 'CR4':
            errors, warnings = validate_cr4_geometry(req.robot.geometry)
            if errors:
                raise HTTPException(status_code=400, detail=f"CR4 geometry validation failed: {errors}")
            
            # Map samples to list of dicts for the solver
            samples_input = [s.model_dump() for s in samples]
            
            # Evaluate CR4 Batch calculation
            out_samples_dict, out_diags_dict, cr4_warnings = compute_cr4_kkt_batch(robot_dict, samples_input, options)
            all_warnings = warnings + cr4_warnings
            
            samples_res = [BatchSampleResponse(**s) for s in out_samples_dict]
            diags_res = [CR4DiagnosticsModel(**d) for d in out_diags_dict]
            
            # Calculate dt_s dynamically from first two samples if possible
            dt_s = 0.005
            if len(samples) > 1:
                dt_s = samples[1].time_s - samples[0].time_s

            return DynamicsBatchResponse(
                joint_names=[lim.name for lim in req.robot.limits],
                samples=samples_res,
                dt_s=dt_s,
                engine_used="pro_cr4_kkt",
                model_id="cr4_pinocchio_kkt.v1",
                manifest=make_manifest(robot_dict, "cr4_pinocchio_kkt.v1", get_trajectory_hash(samples)),
                diagnostics=diags_res,
                warnings=all_warnings
            )
        else:
            errors, warnings = validate_cr6_geometry(req.robot.geometry)
            if errors:
                raise HTTPException(status_code=400, detail=f"CR6 geometry validation failed: {errors}")
            
            # Map samples
            samples_input = [s.model_dump() for s in samples]
            
            # Evaluate CR6 Batch calculation
            out_samples_dict, cr6_warnings = compute_cr6_serial_batch(robot_dict, samples_input, options)
            all_warnings = warnings + cr6_warnings
            
            samples_res = [BatchSampleResponse(**s) for s in out_samples_dict]
            
            # Calculate dt_s dynamically
            dt_s = 0.005
            if len(samples) > 1:
                dt_s = samples[1].time_s - samples[0].time_s
 
            return DynamicsBatchResponse(
                joint_names=[lim.name for lim in req.robot.limits],
                samples=samples_res,
                dt_s=dt_s,
                engine_used="pro_cr6_serial",
                model_id="cr6_serial6_template.v1",
                manifest=make_manifest(robot_dict, "cr6_serial6_template.v1", get_trajectory_hash(samples)),
                warnings=all_warnings
            )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


@router.post("/dynamics/validate", response_model=ValidationResponse)
def validate_robot(robot: RobotSpecModel):
    if robot.kind == 'CR4':
        errors, warnings = validate_cr4_geometry(robot.geometry)
    else:
        errors, warnings = validate_cr6_geometry(robot.geometry)
        
    return ValidationResponse(
        valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings
    )


# --- Legacy compatibility support ---
# Helper code for legacy trajectory scaling
def append_quintic_segment(times, qs, qds, qdds, q0, q1, duration_s, dt_s):
    steps = max(int(math.ceil(duration_s / dt_s)), 1)
    n = len(q0)
    delta = [q1[i] - q0[i] for i in range(n)]
    last_time = times[-1]
    step_dt = duration_s / steps

    for step in range(1, steps + 1):
        ti = step * step_dt
        u = ti / duration_s

        s = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)
        sd = (30.0 * (u**2) - 60.0 * (u**3) + 30.0 * (u**4)) / duration_s
        sdd = (60.0 * u - 180.0 * (u**2) + 120.0 * (u**3)) / (duration_s * duration_s)

        q = [q0[i] + s * delta[i] for i in range(n)]
        qd = [sd * delta[i] for i in range(n)]
        qdd = [sdd * delta[i] for i in range(n)]

        times.append(last_time + ti)
        qs.append(q)
        qds.append(qd)
        qdds.append(qdd)


def build_program_trajectory_py(start_q, program, active_robot, dt_s=0.005):
    current_q = list(start_q)
    times = [0.0]
    qs = [current_q.copy()]
    qds = [[0.0] * len(start_q)]
    qdds = [[0.0] * len(start_q)]

    # Mock tool flange distance checks
    def get_tcp_position(q_vals):
        # Simplification for legacy path distance calculations
        return [0.0, 0.0, 0.0]

    for inst in program['instructions']:
        itype = inst['type']
        if itype == 'MoveJ':
            target = next((t for t in program['targets'] if t['name'] == inst['target_name']), None)
            if not target:
                continue
            target_q = target['q']
            speed = max(float(inst.get('speed_rad_s', 1.0)), 1e-3)
            max_delta = 0.0
            for i in range(len(current_q)):
                max_delta = max(max_delta, abs(target_q[i] - current_q[i]))
            duration_s = max((1.875 * max_delta) / speed, dt_s)
            append_quintic_segment(times, qs, qds, qdds, current_q, target_q, duration_s, dt_s)
            current_q = list(target_q)

        elif itype == 'MoveL':
            target = next((t for t in program['targets'] if t['name'] == inst['target_name']), None)
            if not target:
                continue
            target_q = target['q']
            start_pos = get_tcp_position(current_q)
            target_pos = get_tcp_position(target_q)
            distance = math.hypot(
                target_pos[0] - start_pos[0],
                target_pos[1] - start_pos[1],
                target_pos[2] - start_pos[2]
            )
            tcp_speed = max(float(inst.get('tcp_speed_m_s', 0.1)), 1e-4)
            duration_s = max(distance / tcp_speed, dt_s)

            max_delta = 0.0
            for i in range(len(current_q)):
                max_delta = max(max_delta, abs(target_q[i] - current_q[i]))
            duration_s = max(duration_s, (1.875 * max_delta) / tcp_speed)

            append_quintic_segment(times, qs, qds, qdds, current_q, target_q, duration_s, dt_s)
            current_q = list(target_q)

        elif itype == 'Pause':
            duration = max(float(inst.get('duration_s', 1.0)), dt_s)
            steps = max(int(math.ceil(duration / dt_s)), 1)
            for _ in range(steps):
                times.append(times[-1] + dt_s)
                qs.append(current_q.copy())
                qds.append([0.0] * len(start_q))
                qdds.append([0.0] * len(start_q))

    return times, qs, qds, qdds


@router.post("/dynamics")
def calculate_legacy_dynamics(req: LegacyDynamicsRequest):
    try:
        robot = req.robot.model_dump()
        program = req.program.model_dump()
        
        start_q = [0.0] * 6 if robot['kind'] == 'CR6' else [0.0] * 4
        
        dt_s = 0.005
        times, qs, qds, qdds = build_program_trajectory_py(start_q, program, robot, dt_s)
        
        samples = []
        
        if robot['kind'] == 'CR6':
            samples_input = []
            for idx in range(len(times)):
                samples_input.append({
                    "time_s": times[idx],
                    "q": qs[idx],
                    "qd": qds[idx],
                    "qdd": qdds[idx]
                })
            out_samples, _ = compute_cr6_serial_batch(robot, samples_input)
            
            for idx in range(len(times)):
                samples.append({
                    "time_s": times[idx],
                    "q": qs[idx],
                    "velocity": qds[idx],
                    "acceleration": qdds[idx],
                    "joint_velocity": qds[idx],
                    "joint_acceleration": qdds[idx],
                    "tau": out_samples[idx]["tau"]
                })
        else:
            samples_input = []
            for idx in range(len(times)):
                samples_input.append({
                    "time_s": times[idx],
                    "q": qs[idx],
                    "qd": qds[idx],
                    "qdd": qdds[idx]
                })
            out_samples, _, _ = compute_cr4_kkt_batch(robot, samples_input)
            
            for idx in range(len(times)):
                samples.append({
                    "time_s": times[idx],
                    "q": qs[idx],
                    "velocity": qds[idx],
                    "acceleration": qdds[idx],
                    "joint_velocity": qds[idx],
                    "joint_acceleration": qdds[idx],
                    "tau": out_samples[idx]["tau"]
                })
                
        joint_names = [limit['name'] for limit in robot['limits']]
        
        return {
            "joint_names": joint_names,
            "samples": samples,
            "dt_s": dt_s
        }
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
