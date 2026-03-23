"""
jog.py - Jog Control Endpoints
==============================

Manual robot control endpoints (joint, Cartesian, orientation) and WebSocket for state streaming.
"""

from fastapi import APIRouter, Depends, WebSocket
from pydantic import BaseModel
import numpy as np
import asyncio
import logging
import pinocchio as pin

from ..session import SimulationSession, get_session
from ..utils import get_frontend_nq, q_pink_to_frontend, q_frontend_to_pink, map_joint_index_frontend_to_pink
from robot_core import (
    run_ik_linear, run_ik_pose, get_end_effector_position,
    q_pink_to_real
)
from pink.tasks import FrameTask


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class JointCommand(BaseModel):
    """Set robot joint configuration directly."""
    q: list

class MoveLinearCommand(BaseModel):
    target: list
    dt: float = 0.005
    max_iter: int = 1000
    tol: float = 5e-4

class JogJointCommand(BaseModel):
    index: int
    delta: float

class JogCartesianCommand(BaseModel):
    delta: list  # [dx, dy, dz]
    frame: str = "base"  # "base" or "ee"

class JogOrientationCommand(BaseModel):
    delta: list  # [droll, dpitch, dyaw] in radians
    frame: str = "ee"  # "base" or "ee"


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter()


# =============================================================================
# JOG ENDPOINTS (Manual Robot Control)
# =============================================================================

@router.post("/set_joint")
def set_joint(cmd: JointCommand, session: SimulationSession = Depends(get_session)):
    arr = np.array(cmd.q, dtype=float)
    
    expected_nq = get_frontend_nq(session.current_robot_type, session.model_pink)
    if arr.size != expected_nq:
        logging.warning(f"set_joint: q length mismatch for {session.current_robot_type} (got {arr.size}, expected {expected_nq})")
        return {"ok": False, "error": f"q length mismatch (expected {expected_nq})"}
    
    q_pink_new = q_frontend_to_pink(arr, session.configuration.q, session.current_robot_type)
    session.configuration.q = q_pink_new
    session.q[:] = q_pink_to_real(session.current_robot_type, session.model, session.data, session.constraint_model, q_pink_new)
    session.post_task.set_target(session.configuration.q)
    
    # Compute EE pose for response
    pin.forwardKinematics(session.model, session.data, session.q)
    pin.updateFramePlacements(session.model, session.data)
    fe = session.data.oMf[session.model.getFrameId('end_effector')]
    ee_pos = np.array(fe.translation)
    ee_rot = np.array(fe.rotation)
    
    # Return updated q and EE pose for frontend confirmation
    q_frontend = q_pink_to_frontend(session.configuration.q, session.current_robot_type)
    return {
        "ok": True, 
        "q": q_frontend.tolist(),
        "ee_pos": ee_pos.tolist(),
        "ee_rot": ee_rot.flatten().tolist()
    }


@router.post("/move_linear")
def move_linear(cmd: MoveLinearCommand, session: SimulationSession = Depends(get_session)):
    target = np.array(cmd.target, dtype=float)
    # Using session objects instead of globals
    ok, iters, err = run_ik_linear(session.configuration, session.ee_task, session.post_task, target, dt=cmd.dt, max_iter=cmd.max_iter, tol=cmd.tol)
    # Sync q with updated configuration
    session.q[:] = q_pink_to_real(session.current_robot_type, session.model, session.data, session.constraint_model, session.configuration.q)
    session.post_task.set_target(session.configuration.q)
    return {"ok": ok, "iters": iters, "error_m": float(err)}


@router.post("/jog_joint")
def jog_joint(cmd: JogJointCommand, session: SimulationSession = Depends(get_session)):
    """Apply a small delta to joint `index` and update visualization."""
    idx = int(cmd.index)
    
    pink_idx = map_joint_index_frontend_to_pink(idx, session.current_robot_type)
    if pink_idx < 0 or pink_idx >= session.model_pink.nq:
        return {"ok": False, "error": f"index out of range or invalid (idx={idx})"}
    
    session.configuration.q[pink_idx] += float(cmd.delta)
    
    # If 4-DOF, enforce constraint on J_aux (index 3) if we moved J2(1) or J3(2)
    if session.current_robot_type == "CR4":
        session.configuration.q[3] = -(session.configuration.q[1] + session.configuration.q[2])
    
    # Sync q with updated configuration
    session.q[:] = q_pink_to_real(session.current_robot_type, session.model, session.data, session.constraint_model, session.configuration.q)
    
    # Keep posture regularization around the latest pose
    session.post_task.set_target(session.configuration.q)
    
    ee_pos = get_end_effector_position(session.model, session.data, session.q)
    return {"ok": True, "q": q_pink_to_frontend(session.configuration.q, session.current_robot_type).tolist(), "ee_pos": ee_pos.tolist()}


@router.post("/jog_cartesian")
def jog_cartesian(cmd: JogCartesianCommand, session: SimulationSession = Depends(get_session)):
    """Move end-effector by a small Cartesian delta using IK."""
    d = np.array(cmd.delta, dtype=float)
    frame = (cmd.frame or "base").lower()

    # Access session objects
    configuration = session.configuration
    model_pink = session.model_pink
    data_pink = session.data_pink
    ee_task = session.ee_task
    post_task = session.post_task

    # Current pose (Pink model)
    pin.forwardKinematics(model_pink, data_pink, configuration.q)
    pin.updateFramePlacements(model_pink, data_pink)
    fe = data_pink.oMf[model_pink.getFrameId('end_effector')]
    ee_pos = np.array(fe.translation)

    # If EE frame requested, rotate delta to world
    if frame == "ee":
        d = fe.rotation.dot(d)

    target = ee_pos + d
    # NOTE: If robot is in a straight-up configuration (x≈y≈0), pure +Y is locally impossible.
    # In that case, help the solver by allowing a small waist yaw step to create lateral leverage.
    if abs(d[1]) > 1e-12 and abs(ee_pos[0]) < 1e-6 and abs(ee_pos[1]) < 1e-6:
        q0_step = float(np.clip(d[1] / 0.25, -0.15, 0.15))
        configuration.q[0] += q0_step

    # Cartesian jog (position task). Posture is regularized around current pose (updated after moves).
    ok, iters, err = run_ik_linear(configuration, ee_task, post_task, target, dt=0.005, max_iter=800, tol=5e-4)
    # Sync q with updated configuration
    session.q[:] = q_pink_to_real(session.current_robot_type, session.model, session.data, session.constraint_model, configuration.q)
    
    pin.forwardKinematics(session.model, session.data, session.q)
    pin.updateFramePlacements(session.model, session.data)
    fe_new = session.data.oMf[session.model.getFrameId('end_effector')]
    ee_pos_new = np.array(fe_new.translation)
    ee_rot_new = np.array(fe_new.rotation)
    post_task.set_target(configuration.q)
    return {
        "ok": ok,
        "iters": iters,
        "error_m": float(err),
        "q": q_pink_to_frontend(configuration.q, session.current_robot_type).tolist(),
        "ee_pos": ee_pos_new.tolist(),
        "ee_rot": ee_rot_new.flatten().tolist()
    }


@router.post("/jog_orientation")
def jog_orientation(cmd: JogOrientationCommand, session: SimulationSession = Depends(get_session)):
    """Reorient end-effector by small RPY delta (keeps position fixed)."""
    d = np.array(cmd.delta, dtype=float)
    if d.shape[0] != 3:
        return {"ok": False, "error": "delta must be [droll, dpitch, dyaw]"}
    frame = (cmd.frame or "ee").lower()

    configuration = session.configuration
    model = session.model
    data = session.data
    model_pink = session.model_pink
    data_pink = session.data_pink

    # For 4-DOF palletizer: only Z rotation (yaw) is allowed
    # The tool always points down, so roll/pitch are not possible
    if session.current_robot_type == "CR4":
        if abs(d[0]) > 1e-9 or abs(d[1]) > 1e-9:
            # Roll or pitch requested - not possible for palletizer
            return {
                "ok": False, 
                "error": "4-DOF palletizer can only rotate around Z (yaw)",
                "q": q_pink_to_frontend(configuration.q, session.current_robot_type).tolist(),
                "ee_pos": get_end_effector_position(model, data, session.q).tolist(),
                "ee_rot": data.oMf[model.getFrameId('end_effector')].rotation.flatten().tolist()
            }
        
        # For Z rotation, directly adjust J4 (which controls TCP rotation)
        # d[2] is the yaw delta in radians
        # In Pink model for 4-DOF: q[4] is J4 (but inverted for Z-down convention)
        new_q = configuration.q.copy()
        new_q[4] -= d[2]  # Subtract because J4 is inverted
        
        # Enforce J_aux constraint
        new_q[3] = -(new_q[1] + new_q[2])
        
        configuration.q = new_q
        
        # Sync to real model
        session.q[:] = q_pink_to_real(session.current_robot_type, model, data, session.constraint_model, configuration.q)
        session.post_task.set_target(configuration.q)
        
        pin.forwardKinematics(model, data, session.q)
        pin.updateFramePlacements(model, data)
        fe = data.oMf[model.getFrameId('end_effector')]
        
        return {
            "ok": True,
            "iters": 0,
            "error": 0.0,
            "q": q_pink_to_frontend(configuration.q, session.current_robot_type).tolist(),
            "ee_pos": fe.translation.tolist(),
            "ee_rot": fe.rotation.flatten().tolist()
        }

    # 6-DOF: Use IK with very high position cost to keep TCP position fixed
    pin.forwardKinematics(model_pink, data_pink, configuration.q)
    pin.updateFramePlacements(model_pink, data_pink)
    fe = data_pink.oMf[model_pink.getFrameId('end_effector')]
    
    # Store original position
    original_pos = fe.translation.copy()

    # Target orientation = current * delta_rpy
    R_delta = pin.rpy.rpyToMatrix(d)
    if frame == "base":
        # Apply delta in world frame: pre-multiply
        R_target = R_delta.dot(fe.rotation)
    else:
        # Default: apply in EE frame (post-multiply)
        R_target = fe.rotation.dot(R_delta)
    
    # Target SE3 with original position (to keep it fixed)
    target_se3 = pin.SE3(R_target, original_pos)

    # Use extremely high position cost to ensure position stays fixed
    # Use small orientation cost to allow orientation to change easily
    ori_task = FrameTask('end_effector', position_cost=1e4, orientation_cost=1.0)

    ok, iters, err = run_ik_pose(configuration, ori_task, session.post_task, target_se3, dt=0.002, max_iter=1000, tol=1e-4)
    
    # After IK, verify position drift and correct if needed
    pin.forwardKinematics(model_pink, data_pink, configuration.q)
    pin.updateFramePlacements(model_pink, data_pink)
    fe_after = data_pink.oMf[model_pink.getFrameId('end_effector')]
    pos_drift = np.linalg.norm(fe_after.translation - original_pos)
    
    # If position drifted more than 0.1mm, do a correction step
    if pos_drift > 1e-4:
        # Force target position to be exactly original
        target_se3_corrected = pin.SE3(fe_after.rotation, original_pos)
        correction_task = FrameTask('end_effector', position_cost=1e6, orientation_cost=1e-3)
        run_ik_pose(configuration, correction_task, session.post_task, target_se3_corrected, dt=0.001, max_iter=500, tol=1e-5)
    
    # Sync q
    session.q[:] = q_pink_to_real(session.current_robot_type, model, data, session.constraint_model, configuration.q)
    
    pin.forwardKinematics(model, data, session.q)
    pin.updateFramePlacements(model, data)
    fe = data.oMf[model.getFrameId('end_effector')]
    ee_pos = np.array(fe.translation)
    ee_rot = np.array(fe.rotation)
    session.post_task.set_target(configuration.q)
    return {
        "ok": ok,
        "iters": iters,
        "error": err,
        "q": q_pink_to_frontend(configuration.q, session.current_robot_type).tolist(),
        "ee_pos": ee_pos.tolist(),
        "ee_rot": ee_rot.flatten().tolist()
    }


# WebSocket to stream q periodically
@router.websocket("/ws/state")
async def websocket_state(ws: WebSocket):
    await ws.accept()
    
    # Retrieve session ID from headers or query params
    session_id = ws.headers.get("x-session-id") or ws.query_params.get("session_id") or "default"
    
    # Get or create session (lazy)
    from ..session import sessions
    if session_id not in sessions:
        sessions[session_id] = SimulationSession(session_id)
        logging.info(f"WebSocket: created new session {session_id}")
    session = sessions[session_id]
    
    logging.info(f"WebSocket client connected: {session_id}")
    
    try:
        while True:
            try:
                # Skip sending state during animation to prevent picotazos
                if session.is_animating:
                    await asyncio.sleep(0.05)
                    continue
                
                # Compute frontend q
                q_frontend = q_pink_to_frontend(session.configuration.q, session.current_robot_type)
                
                # Send joints and EE pose (position + rotation) for frontend convenience
                pin.forwardKinematics(session.model, session.data, session.q)
                pin.updateFramePlacements(session.model, session.data)
                fe = session.data.oMf[session.model.getFrameId('end_effector')]
                ee_pos = np.array(fe.translation)
                ee_rot = np.array(fe.rotation)
                msg = {
                    "q": q_frontend.tolist(),
                    "ee_pos": ee_pos.tolist(),
                    "ee_rot": ee_rot.flatten().tolist()
                }
                await ws.send_json(msg)
                await asyncio.sleep(0.05)
            except Exception as e:
                logging.info(f"WebSocket send error: {e}")
                break
    except Exception as e:
        logging.exception(f"WebSocket error: {e}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logging.info(f"WebSocket client disconnected: {session_id}")
