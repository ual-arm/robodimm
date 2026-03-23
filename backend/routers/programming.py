"""
programming.py - Programming Endpoints
=====================================

Endpoints for saving targets and managing motion programs.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pinocchio as pin
import pink
from pathlib import Path
import json

from ..session import SimulationSession, get_session
from ..utils import q_pink_to_frontend, q_frontend_to_pink
from robot_core import q_pink_to_real, normalize_payload_inertia


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class SaveTargetCommand(BaseModel):
    name: str


class AddInstructionCommand(BaseModel):
    type: str  # "MoveJ", "MoveL", "MoveC", or "Pause"
    target_name: str = ""  # Not required for Pause
    via_target_name: str = ""  # For MoveC - intermediate point
    speed: float = 100.0  # mm/s or deg/s depending on type
    zone: float = 50.0  # precision zone in mm (default 50)
    pause_time: float = 1.0  # seconds (for Pause instruction)


class DeleteTargetCommand(BaseModel):
    name: str


class DeleteInstructionCommand(BaseModel):
    index: int


class SaveProgramRequest(BaseModel):
    """Request to save a program to file."""

    filename: str
    description: Optional[str] = None


class LoadProgramRequest(BaseModel):
    """Request to load a program from file."""

    filename: str


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter()

# Directory for saved programs
_programs_dir = Path(__file__).resolve().parent.parent / "saved_programs"
_programs_dir.mkdir(exist_ok=True)


# =============================================================================
# PROGRAMMING ENDPOINTS (Targets & Instructions)
# =============================================================================


@router.post("/save_target")
def save_target(
    cmd: SaveTargetCommand, session: SimulationSession = Depends(get_session)
):
    """Save current EE pose as a named target."""
    pin.forwardKinematics(
        session.model_pink, session.data_pink, session.configuration.q
    )
    pin.updateFramePlacements(session.model_pink, session.data_pink)
    fe = session.data_pink.oMf[session.model_pink.getFrameId("end_effector")]

    # Store q in FRONTEND format
    q_frontend = q_pink_to_frontend(session.configuration.q, session.current_robot_type)

    target_data = {
        "name": cmd.name,
        "position": fe.translation.tolist(),
        "rotation": fe.rotation.flatten().tolist(),
        "q": q_frontend.tolist(),
    }

    # Replace if name exists
    for i, t in enumerate(session.targets):
        if t["name"] == cmd.name:
            session.targets[i] = target_data
            return {"ok": True, "replaced": True, "target": target_data}

    session.targets.append(target_data)
    return {"ok": True, "replaced": False, "target": target_data}


@router.get("/targets")
def get_targets(session: SimulationSession = Depends(get_session)):
    """Get all saved targets."""
    return {"targets": session.targets}


@router.post("/delete_target")
def delete_target(
    cmd: DeleteTargetCommand, session: SimulationSession = Depends(get_session)
):
    """Delete a target by name."""
    # Remove from targets
    session.targets = [t for t in session.targets if t["name"] != cmd.name]
    # Also remove any program instructions referencing this target
    session.program = [p for p in session.program if p["target_name"] != cmd.name]
    return {"ok": True}


@router.post("/add_instruction")
def add_instruction(
    cmd: AddInstructionCommand, session: SimulationSession = Depends(get_session)
):
    """Add a motion instruction to the program."""

    # Handle Pause instruction (no target needed)
    if cmd.type == "Pause":
        instruction = {
            "type": "Pause",
            "target_name": "",
            "speed": 0,
            "zone": 0,
            "pause_time": cmd.pause_time,
        }
        session.program.append(instruction)
        return {
            "ok": True,
            "instruction": instruction,
            "index": len(session.program) - 1,
        }

    # Verify target exists for motion instructions
    target_exists = any(t["name"] == cmd.target_name for t in session.targets)
    if not target_exists:
        return {"ok": False, "error": f"Target '{cmd.target_name}' not found"}

    instruction = {
        "type": cmd.type,
        "target_name": cmd.target_name,
        "speed": cmd.speed,
        "zone": cmd.zone,
    }

    # For MoveC, also store the via_target
    if cmd.type == "MoveC":
        via_exists = any(t["name"] == cmd.via_target_name for t in session.targets)
        if not via_exists:
            return {
                "ok": False,
                "error": f"Via target '{cmd.via_target_name}' not found",
            }
        instruction["via_target_name"] = cmd.via_target_name

    session.program.append(instruction)
    return {"ok": True, "instruction": instruction, "index": len(session.program) - 1}


@router.get("/program")
def get_program(session: SimulationSession = Depends(get_session)):
    """Get the current program."""
    return {"program": session.program}


@router.post("/delete_instruction")
def delete_instruction(
    cmd: DeleteInstructionCommand, session: SimulationSession = Depends(get_session)
):
    """Delete an instruction by index."""
    if 0 <= cmd.index < len(session.program):
        session.program.pop(cmd.index)
        return {"ok": True}
    return {"ok": False, "error": "Index out of range"}


@router.post("/clear_program")
def clear_program(session: SimulationSession = Depends(get_session)):
    """Clear the entire program."""
    session.program = []
    return {"ok": True}


@router.get("/list_programs")
def list_programs():
    """List all saved programs."""
    programs_list = []
    for f in _programs_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                programs_list.append(
                    {
                        "filename": f.stem,
                        "description": data.get("description", ""),
                        "robot_type": data.get("robot_type", "unknown"),
                        "num_instructions": len(data.get("program", [])),
                        "num_targets": len(data.get("targets", [])),
                        "saved_at": data.get("saved_at", ""),
                    }
                )
        except Exception as e:
            import logging

            logging.warning(f"Could not read program file {f}: {e}")
    return {"ok": True, "programs": programs_list}


@router.post("/save_program")
def save_program(
    req: SaveProgramRequest, session: SimulationSession = Depends(get_session)
):
    """Save current program and targets to a file."""

    from datetime import datetime

    filename = req.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not filename.endswith(".json"):
        filename += ".json"

    program_data = {
        "description": req.description or "",
        "robot_type": session.current_robot_type,
        "scale": session.current_scale,
        "payload_kg": session.current_payload_kg,
        "payload_inertia": session.current_payload_inertia,
        "friction_coeffs": list(session.current_friction_coeffs)
        if session.current_friction_coeffs
        else None,
        "targets": session.targets,
        "program": session.program,
        "saved_at": datetime.now().isoformat(),
    }

    filepath = _programs_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(program_data, f, indent=2, ensure_ascii=False)

    return {"ok": True, "filename": filename, "path": str(filepath)}


@router.post("/load_program")
def load_program(
    req: LoadProgramRequest, session: SimulationSession = Depends(get_session)
):
    """Load a program from file. Optionally restores robot config."""

    filename = req.filename
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = _programs_dir / filename
    if not filepath.exists():
        return {"ok": False, "error": f"Program file not found: {filename}"}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            program_data = json.load(f)
    except Exception as e:
        return {"ok": False, "error": f"Could not read program file: {e}"}

    # Check if robot config needs to change
    saved_robot_type = program_data.get("robot_type", session.current_robot_type)
    saved_scale = program_data.get("scale", session.current_scale)
    saved_payload = program_data.get("payload_kg", 0.0)
    saved_payload_inertia = program_data.get("payload_inertia")
    saved_friction = program_data.get("friction_coeffs")
    saved_payload_inertia = normalize_payload_inertia(
        payload_kg=saved_payload,
        payload_inertia=saved_payload_inertia,
        robot_type=saved_robot_type,
    )

    # Rebuild robot if type changed or config differs significantly
    config_changed = (
        saved_robot_type != session.current_robot_type
        or saved_scale != session.current_scale
        or saved_payload != session.current_payload_kg
        or saved_payload_inertia != session.current_payload_inertia
    )

    if config_changed:
        session.current_robot_type = saved_robot_type
        session.current_scale = saved_scale
        session.current_payload_kg = saved_payload
        session.current_payload_inertia = saved_payload_inertia
        session.current_friction_coeffs = saved_friction

        # This will rebuild core, model, data using new params
        session.rebuild_robot()

    # Load targets and program
    session.targets = program_data.get("targets", [])
    session.program = program_data.get("program", [])

    return {
        "ok": True,
        "filename": filename,
        "description": program_data.get("description", ""),
        "robot_type": saved_robot_type,
        "scale": saved_scale,
        "payload_kg": saved_payload,
        "payload_inertia": saved_payload_inertia,
        "num_targets": len(session.targets),
        "num_instructions": len(session.program),
        "targets": session.targets,
        "program": session.program,
    }


@router.delete("/delete_program/{filename}")
def delete_program(filename: str):
    """Delete a saved program file."""
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = _programs_dir / filename
    if not filepath.exists():
        return {"ok": False, "error": f"Program file not found: {filename}"}

    try:
        filepath.unlink()
        return {"ok": True, "deleted": filename}
    except Exception as e:
        return {"ok": False, "error": f"Could not delete: {e}"}
