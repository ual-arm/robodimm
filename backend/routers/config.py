"""
config.py - Robot Configuration Endpoints
========================================

Endpoints for robot type, scale, payload and configuration management.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json
import threading

from ..session import SimulationSession, get_session
from ..utils import get_frontend_nq, create_demo_targets_and_program
from robot_core import normalize_payload_inertia


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class RobotConfigCommand(BaseModel):
    robot_type: str = "CR6"  # "CR6" or "CR4"
    scale: float = 1.0  # scale factor for robot reach
    payload_kg: float = 0.0  # payload mass at TCP in kg
    payload_inertia: Optional[dict] = (
        None  # payload inertia tensor or box model {Ixx, Iyy, Izz, com_from_tcp} or {box_size_xyz_m, com_from_tcp}
    )
    # For palletizers (4DOF), Izz is critical for J4!
    # Example: {"Ixx": 0.01, "Iyy": 0.01, "Izz": 0.05}
    friction_coeffs: Optional[list] = None  # viscous friction [Nm/(rad/s)] per joint
    reflected_inertia: Optional[list] = (
        None  # rotor inertia reflected to joints [kg*m^2]
    )
    coulomb_friction: Optional[list] = None  # coulomb friction [Nm] per joint
    motor_masses: Optional[list] = None  # stator masses per actuated joint [kg]
    motor_layout: Optional[str] = None  # CR4: concentric_j2_j3act | serial_like
    iref_model_mode: Optional[str] = None  # diag | q5_physical | q3_q5_physical
    structural_mass_scale_exp: Optional[float] = None  # default 3.0
    structural_inertia_scale_exp: Optional[float] = None  # default mass_exp + 2


class SaveRobotConfigRequest(BaseModel):
    """Request to save robot configuration to file."""

    filename: str
    description: Optional[str] = None


class LoadRobotConfigRequest(BaseModel):
    """Request to load robot configuration from file."""

    filename: str


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter()

# Lock for thread-safe config changes
_config_lock = threading.Lock()

# Directory for saved robot configurations
_configs_dir = Path(__file__).resolve().parent.parent / "saved_configs"
_configs_dir.mkdir(exist_ok=True)


# =============================================================================
# CONFIGURATION ENDPOINTS (Robot Type & Scale)
# =============================================================================


@router.post("/robot_config")
def set_robot_config(
    cmd: RobotConfigCommand, session: SimulationSession = Depends(get_session)
):
    """Reconfigure robot type, scale, payload (with inertia) and friction. Rebuilds the robot model."""
    import logging

    logging.info(
        f"set_robot_config: request robot_type={cmd.robot_type}, scale={cmd.scale}"
    )
    logging.info(
        f"set_robot_config: current session robot_type={session.current_robot_type}"
    )

    with _config_lock:
        payload_inertia = normalize_payload_inertia(
            payload_kg=cmd.payload_kg,
            payload_inertia=cmd.payload_inertia,
            robot_type=cmd.robot_type,
        )
        # Rebuild robot with new configuration
        session.current_robot_type = cmd.robot_type
        session.current_scale = cmd.scale
        session.current_payload_kg = cmd.payload_kg
        session.current_payload_inertia = payload_inertia
        session.current_friction_coeffs = cmd.friction_coeffs
        session.current_reflected_inertia = cmd.reflected_inertia
        session.current_coulomb_friction = cmd.coulomb_friction
        if cmd.motor_masses is not None:
            session.current_motor_masses = cmd.motor_masses
        if cmd.motor_layout is not None:
            session.current_motor_layout = cmd.motor_layout
        if cmd.iref_model_mode is not None:
            session.current_iref_model_mode = cmd.iref_model_mode
        if cmd.structural_mass_scale_exp is not None:
            session.current_structural_mass_scale_exp = cmd.structural_mass_scale_exp
        if cmd.structural_inertia_scale_exp is not None:
            session.current_structural_inertia_scale_exp = (
                cmd.structural_inertia_scale_exp
            )

        session.rebuild_robot()
        model_pink_nq = session.model_pink.nq if session.model_pink is not None else 0
        logging.info(
            f"set_robot_config: after rebuild robot_type={session.current_robot_type}, model_pink.nq={model_pink_nq}"
        )

        # Always reload demo targets and program when robot type changes
        # This ensures both targets AND program are consistent for the new robot
        session.targets, session.program = create_demo_targets_and_program(
            session.current_robot_type
        )

    return {
        "ok": True,
        "robot_type": session.current_robot_type,
        "scale": session.current_scale,
        "payload_kg": session.current_payload_kg,
        "payload_inertia": session.current_payload_inertia,
        "friction_coeffs": list(session.robot_friction_coeffs)
        if session.robot_friction_coeffs is not None
        else None,
        "reflected_inertia": list(session.robot_reflected_inertia)
        if session.robot_reflected_inertia is not None
        else None,
        "coulomb_friction": list(session.robot_coulomb_friction)
        if session.robot_coulomb_friction is not None
        else None,
        "motor_masses": list(session.robot_motor_masses)
        if getattr(session, "robot_motor_masses", None) is not None
        else None,
        "motor_layout": session.current_motor_layout,
        "iref_model_mode": session.current_iref_model_mode,
        "structural_mass_scale_exp": session.current_structural_mass_scale_exp,
        "structural_inertia_scale_exp": session.current_structural_inertia_scale_exp,
        "nq": get_frontend_nq(session.current_robot_type, session.model_pink)
        if session.model_pink is not None
        else 0,
        "ee_always_down": session.ee_always_down,
        "dimensions": (session.core or {}).get("dimensions", {}),
    }


@router.get("/robot_config")
def get_robot_config(session: SimulationSession = Depends(get_session)):
    """Get current robot configuration."""
    return {
        "ok": True,
        "robot_type": session.current_robot_type,
        "scale": session.current_scale,
        "payload_kg": session.current_payload_kg,
        "payload_inertia": session.current_payload_inertia,
        "friction_coeffs": list(session.current_friction_coeffs)
        if session.current_friction_coeffs is not None
        else None,
        "reflected_inertia": list(session.current_reflected_inertia)
        if session.current_reflected_inertia is not None
        else None,
        "coulomb_friction": list(session.current_coulomb_friction)
        if session.current_coulomb_friction is not None
        else None,
        "motor_masses": list(session.current_motor_masses)
        if session.current_motor_masses is not None
        else None,
        "motor_layout": session.current_motor_layout,
        "iref_model_mode": session.current_iref_model_mode,
        "structural_mass_scale_exp": session.current_structural_mass_scale_exp,
        "structural_inertia_scale_exp": session.current_structural_inertia_scale_exp,
        "nq": get_frontend_nq(session.current_robot_type, session.model_pink)
        if session.model_pink is not None
        else 0,
        "ee_always_down": session.ee_always_down,
        "dimensions": (session.core or {}).get("dimensions", {}),
    }


@router.get("/list_configs")
def list_saved_configs():
    """List all saved robot configuration files."""
    configs_list = []
    for f in _configs_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                config_data = json.load(fp)
                configs_list.append(
                    {
                        "filename": f.name,
                        "description": config_data.get("description", ""),
                        "robot_type": config_data.get("robot_type", "unknown"),
                        "scale": config_data.get("scale", 1.0),
                        "payload_kg": config_data.get("payload_kg", 0),
                        "saved_at": config_data.get("saved_at", ""),
                    }
                )
        except Exception as e:
            import logging

            logging.warning(f"Could not read config file {f}: {e}")
    return {"ok": True, "configs": configs_list}


@router.post("/save_robot_config")
def save_robot_config(req: SaveRobotConfigRequest):
    """Save current robot configuration to a file (without program/targets)."""
    from datetime import datetime

    from ..session import sessions

    # Get the first session (any session will do for config save)
    if not sessions:
        return {"ok": False, "error": "No active session"}

    session = next(iter(sessions.values()))

    filename = req.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not filename.endswith(".json"):
        filename += ".json"

    config_data = {
        "description": req.description or "",
        "robot_type": session.current_robot_type,
        "scale": session.current_scale,
        "payload_kg": session.current_payload_kg,
        "payload_inertia": session.current_payload_inertia,
        "friction_coeffs": list(session.current_friction_coeffs)
        if session.current_friction_coeffs is not None
        else None,
        "reflected_inertia": list(session.current_reflected_inertia)
        if session.current_reflected_inertia is not None
        else None,
        "coulomb_friction": list(session.current_coulomb_friction)
        if session.current_coulomb_friction is not None
        else None,
        "motor_masses": list(session.current_motor_masses)
        if session.current_motor_masses is not None
        else None,
        "motor_layout": session.current_motor_layout,
        "iref_model_mode": session.current_iref_model_mode,
        "structural_mass_scale_exp": session.current_structural_mass_scale_exp,
        "structural_inertia_scale_exp": session.current_structural_inertia_scale_exp,
        "saved_at": datetime.now().isoformat(),
    }

    filepath = _configs_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    return {"ok": True, "filename": filename, "path": str(filepath)}


@router.post("/load_robot_config")
def load_robot_config(
    req: LoadRobotConfigRequest, session: SimulationSession = Depends(get_session)
):
    """Load a robot configuration from file (applies to current robot)."""

    filename = req.filename
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = _configs_dir / filename
    if not filepath.exists():
        return {"ok": False, "error": f"Config file not found: {filename}"}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as e:
        return {"ok": False, "error": f"Could not read config file: {e}"}

    # Apply configuration
    session.current_robot_type = config_data.get("robot_type", "CR6")
    session.current_scale = config_data.get("scale", 1.0)
    session.current_payload_kg = config_data.get("payload_kg", 0.0)
    session.current_payload_inertia = normalize_payload_inertia(
        payload_kg=session.current_payload_kg,
        payload_inertia=config_data.get("payload_inertia"),
        robot_type=session.current_robot_type,
    )
    session.current_friction_coeffs = config_data.get("friction_coeffs")
    session.current_reflected_inertia = config_data.get("reflected_inertia")
    session.current_coulomb_friction = config_data.get("coulomb_friction")
    session.current_motor_masses = config_data.get("motor_masses")
    session.current_motor_layout = config_data.get(
        "motor_layout", "concentric_j2_j3act"
    )
    session.current_iref_model_mode = config_data.get("iref_model_mode", "diag")
    session.current_structural_mass_scale_exp = config_data.get(
        "structural_mass_scale_exp", 3.0
    )
    session.current_structural_inertia_scale_exp = config_data.get(
        "structural_inertia_scale_exp"
    )

    session.rebuild_robot()

    return {
        "ok": True,
        "filename": filename,
        "description": config_data.get("description", ""),
        "robot_type": session.current_robot_type,
        "scale": session.current_scale,
        "payload_kg": session.current_payload_kg,
        "payload_inertia": session.current_payload_inertia,
        "friction_coeffs": session.current_friction_coeffs,
        "reflected_inertia": session.current_reflected_inertia,
        "coulomb_friction": session.current_coulomb_friction,
        "motor_masses": session.current_motor_masses,
        "motor_layout": session.current_motor_layout,
        "iref_model_mode": session.current_iref_model_mode,
        "structural_mass_scale_exp": session.current_structural_mass_scale_exp,
        "structural_inertia_scale_exp": session.current_structural_inertia_scale_exp,
    }


@router.delete("/delete_robot_config/{filename}")
def delete_robot_config(filename: str):
    """Delete a saved robot configuration file."""
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = _configs_dir / filename
    if not filepath.exists():
        return {"ok": False, "error": f"Config file not found: {filename}"}

    try:
        filepath.unlink()
        return {"ok": True, "deleted": filename}
    except Exception as e:
        return {"ok": False, "error": f"Could not delete: {e}"}
