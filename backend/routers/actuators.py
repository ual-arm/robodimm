"""
backend/routers/actuators.py - Actuator Library and Selection Endpoints
====================================================================

This module provides endpoints for:
- Managing the actuator library (motors, gearboxes, compatibility matrix)
- Selecting optimal motor+gearbox combinations based on trajectory requirements
- Validating actuator selections with mass effects
- Saving/loading actuator library files
- Exporting complete projects

Dependencies:
- robot_core.actuators for selection algorithms
- FastAPI for REST API
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path
import json
from datetime import datetime

from ..session import SimulationSession, get_session
from robot_core import (
    analyze_trajectory_requirements,
    select_actuators,
    get_actuator_masses,
)

# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter()

# Path to actuators library JSON file
_actuators_library_path = (
    Path(__file__).resolve().parent.parent.parent / "actuators_library.json"
)

# Directory for saved actuator libraries
_libraries_dir = Path(__file__).resolve().parent.parent.parent / "saved_libraries"
_libraries_dir.mkdir(exist_ok=True)

# Directory for saved programs (for export_full_project)
_programs_dir = Path(__file__).resolve().parent.parent.parent / "saved_programs"
_programs_dir.mkdir(exist_ok=True)

# Cache for actuator library
_actuators_cache = None


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class MotorSpec(BaseModel):
    """Motor specification."""

    id: str
    description: str = ""
    flange_mm: Optional[float] = None
    nominal_torque_Nm: float
    peak_torque_Nm: Optional[float] = None
    nominal_speed_rpm: float
    max_speed_rpm: Optional[float] = None
    rotor_inertia_kgcm2: Optional[float] = None
    mass_kg: Optional[float] = None
    length_mm: Optional[float] = None
    compatible_gearboxes: List[str] = []


class GearboxSpec(BaseModel):
    """Gearbox specification."""

    id: str
    description: str = ""
    for_servo_mm: Optional[float] = None
    ratios: List[int] = []
    rated_torque_Nm: Optional[float] = None
    peak_torque_Nm: Optional[float] = None
    max_input_speed_rpm: Optional[float] = None
    backlash_arcmin: Optional[float] = None
    efficiency: Optional[float] = None
    mass_kg: Optional[float] = None
    length_mm: Optional[float] = None
    output_inertia_kgcm2: Optional[float] = None


class ActuatorLibraryUpdate(BaseModel):
    """Update actuator library (motors and/or gearboxes)."""

    motors: Optional[List[dict]] = None
    gearboxes: Optional[List[dict]] = None
    compatibility_matrix: Optional[dict] = None


class ActuatorSelectionRequest(BaseModel):
    """Request actuator selection for trajectory requirements."""

    safety_factor_torque: float = 1.5
    safety_factor_speed: float = 1.2
    apply_to_robot_config: bool = False


class SaveActuatorLibraryRequest(BaseModel):
    """Request to save actuator library to file."""

    filename: str
    description: Optional[str] = None


class LoadActuatorLibraryRequest(BaseModel):
    """Request to load actuator library from file."""

    filename: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _load_actuators_library(force_reload=False):
    """Load actuators library from JSON file."""
    global _actuators_cache
    if _actuators_cache is None or force_reload:
        if _actuators_library_path.exists():
            with open(_actuators_library_path, "r", encoding="utf-8") as f:
                _actuators_cache = json.load(f)
        else:
            _actuators_cache = {
                "motors": [],
                "gearboxes": [],
                "compatibility_matrix": {},
                "metadata": {},
            }
    return _actuators_cache


def _save_actuators_library(library_data: dict):
    """Save actuators library to JSON file."""
    global _actuators_cache
    _actuators_cache = library_data
    with open(_actuators_library_path, "w", encoding="utf-8") as f:
        json.dump(library_data, f, indent=2, ensure_ascii=False)


def _build_cr4_dynamic_case_from_selection(selection: dict, motors_by_id: dict) -> dict:
    """Build CR4 dynamics arrays from selected actuators (script-compatible mapping)."""
    # Serial-5 arrays q1..q5 where q4 is passive in CR4 palletizer models.
    motor_mass_q = [0.0, 0.0, 0.0, 0.0, 0.0]
    iref_q = [0.0, 0.0, 0.0, 0.0, 0.0]

    joint_to_serial_idx = {
        "J1": 0,
        "J2": 1,
        "J3": 2,
        "J4": 4,
    }

    for joint_name, data in selection.items():
        rec = data.get("recommended")
        if not rec or joint_name not in joint_to_serial_idx:
            continue

        motor = motors_by_id.get(rec.get("motor_id"), {})
        ratio = float(rec.get("ratio") or 0.0)
        serial_idx = joint_to_serial_idx[joint_name]

        m_motor = float(motor.get("mass_kg") or 0.0)
        rotor_kgcm2 = float(motor.get("rotor_inertia_kgcm2") or 0.0)
        rotor_kgm2 = rotor_kgcm2 * 1e-7
        iref = rotor_kgm2 * (ratio**2)

        motor_mass_q[serial_idx] = m_motor
        iref_q[serial_idx] = iref

    return {
        "motor_mass_q": motor_mass_q,
        "rotor_inertia_reflected_q": iref_q,
        "iref_model_mode": "q5_physical",
        "motor_masses_axis4": [
            motor_mass_q[0],
            motor_mass_q[1],
            motor_mass_q[2],
            motor_mass_q[4],
        ],
        "reflected_inertia_axis4": [iref_q[0], iref_q[1], iref_q[2], iref_q[4]],
    }


# =============================================================================
# ACTUATOR LIBRARY ENDPOINTS
# =============================================================================


@router.get("/actuators_library")
def get_actuators_library(reload: bool = False):
    """Get complete actuators library (motors, gearboxes, compatibility)."""
    library = _load_actuators_library(force_reload=reload)
    return {
        "ok": True,
        "motors": library.get("motors", []),
        "gearboxes": library.get("gearboxes", []),
        "compatibility_matrix": library.get("compatibility_matrix", {}),
        "metadata": library.get("metadata", {}),
    }


@router.post("/reload_actuator_library")
def reload_actuator_library():
    """Force reload the actuator library from file."""
    global _actuators_cache
    _actuators_cache = None
    library = _load_actuators_library(force_reload=True)
    return {
        "ok": True,
        "message": "Actuator library reloaded",
        "num_motors": len(library.get("motors", [])),
        "num_gearboxes": len(library.get("gearboxes", [])),
    }


@router.post("/actuators_library")
def update_actuators_library(update: ActuatorLibraryUpdate):
    """Update actuators library (partial or full update)."""
    library = _load_actuators_library()

    if update.motors is not None:
        library["motors"] = update.motors
    if update.gearboxes is not None:
        library["gearboxes"] = update.gearboxes
    if update.compatibility_matrix is not None:
        library["compatibility_matrix"] = update.compatibility_matrix

    # Update metadata
    library["metadata"] = library.get("metadata", {})
    library["metadata"]["last_updated"] = datetime.now().isoformat()

    _save_actuators_library(library)
    return {"ok": True, "message": "Library updated successfully"}


@router.post("/add_motor")
def add_motor(motor: MotorSpec):
    """Add or update a motor in the library."""
    library = _load_actuators_library()
    motors = library.get("motors", [])

    # Check if motor with same ID exists
    existing_idx = None
    for i, m in enumerate(motors):
        if m["id"] == motor.id:
            existing_idx = i
            break

    motor_dict = motor.dict()
    if existing_idx is not None:
        motors[existing_idx] = motor_dict
    else:
        motors.append(motor_dict)

    library["motors"] = motors
    _save_actuators_library(library)
    return {"ok": True, "motor": motor_dict}


@router.post("/add_gearbox")
def add_gearbox(gearbox: GearboxSpec):
    """Add or update a gearbox in the library."""
    library = _load_actuators_library()
    gearboxes = library.get("gearboxes", [])

    # Check if gearbox with same ID exists
    existing_idx = None
    for i, g in enumerate(gearboxes):
        if g["id"] == gearbox.id:
            existing_idx = i
            break

    gearbox_dict = gearbox.dict()
    if existing_idx is not None:
        gearboxes[existing_idx] = gearbox_dict
    else:
        gearboxes.append(gearbox_dict)

    library["gearboxes"] = gearboxes
    _save_actuators_library(library)
    return {"ok": True, "gearbox": gearbox_dict}


@router.delete("/motor/{motor_id}")
def delete_motor(motor_id: str):
    """Delete a motor from the library."""
    library = _load_actuators_library()
    motors = library.get("motors", [])

    new_motors = [m for m in motors if m["id"] != motor_id]
    if len(new_motors) == len(motors):
        return {"ok": False, "error": "Motor not found"}

    library["motors"] = new_motors
    # Also remove from compatibility matrix
    if motor_id in library.get("compatibility_matrix", {}):
        del library["compatibility_matrix"][motor_id]

    _save_actuators_library(library)
    return {"ok": True}


@router.delete("/gearbox/{gearbox_id}")
def delete_gearbox(gearbox_id: str):
    """Delete a gearbox from the library."""
    library = _load_actuators_library()
    gearboxes = library.get("gearboxes", [])

    new_gearboxes = [g for g in gearboxes if g["id"] != gearbox_id]
    if len(new_gearboxes) == len(gearboxes):
        return {"ok": False, "error": "Gearbox not found"}

    library["gearboxes"] = new_gearboxes
    _save_actuators_library(library)
    return {"ok": True}


# =============================================================================
# ACTUATOR SELECTION ENDPOINTS
# =============================================================================


@router.get("/trajectory_requirements")
def get_trajectory_requirements(session: SimulationSession = Depends(get_session)):
    """
    Get torque/speed requirements from the last executed trajectory.
    Must run execute_program first to have dynamics data.
    """

    if session.last_trajectory_data is None:
        return {
            "ok": False,
            "error": "No trajectory data available. Execute a program first.",
        }

    requirements = analyze_trajectory_requirements(
        session.last_trajectory_data, robot_type=session.current_robot_type
    )

    return {
        "ok": True,
        "requirements": requirements,
        "num_joints": len(requirements),
        "robot_type": session.current_robot_type,
    }


@router.post("/select_actuators")
def select_actuators_endpoint(
    request: ActuatorSelectionRequest, session: SimulationSession = Depends(get_session)
):
    """
    Select optimal motor+gearbox combinations based on trajectory requirements.

    Must run execute_program first to have dynamics data.
    Returns recommendations for each joint with margin analysis.
    """

    if session.last_trajectory_data is None:
        return {
            "ok": False,
            "error": "No trajectory data available. Execute a program first.",
        }

    # Get requirements (only for active joints)
    requirements = analyze_trajectory_requirements(
        session.last_trajectory_data, robot_type=session.current_robot_type
    )

    if not requirements:
        return {"ok": False, "error": "Could not analyze trajectory requirements."}

    # Load actuator library
    library = _load_actuators_library()
    motors = library.get("motors", [])
    gearboxes = library.get("gearboxes", [])
    compatibility = library.get("compatibility_matrix", {})

    if not motors or not gearboxes:
        return {
            "ok": False,
            "error": "Actuator library is empty. Add motors and gearboxes first.",
        }

    # Run selection algorithm
    selection = select_actuators(
        requirements=requirements,
        motors=motors,
        gearboxes=gearboxes,
        compatibility_matrix=compatibility,
        safety_factor_torque=request.safety_factor_torque,
        safety_factor_speed=request.safety_factor_speed,
    )

    # Get estimated masses for iteration
    masses = get_actuator_masses(selection, motors, gearboxes)

    dynamic_case = None
    if session.current_robot_type == "CR4":
        motors_by_id = {m.get("id"): m for m in motors}
        dynamic_case = _build_cr4_dynamic_case_from_selection(selection, motors_by_id)

    if request.apply_to_robot_config and dynamic_case is not None:
        session.current_motor_masses = dynamic_case["motor_masses_axis4"]
        session.current_reflected_inertia = dynamic_case["reflected_inertia_axis4"]
        session.current_motor_layout = "concentric_j2_j3act"
        session.current_iref_model_mode = dynamic_case.get(
            "iref_model_mode", "q5_physical"
        )
        session.rebuild_robot()

    return {
        "ok": True,
        "selection": selection,
        "estimated_masses": masses,
        "safety_factors": {
            "torque": request.safety_factor_torque,
            "speed": request.safety_factor_speed,
        },
        "dynamic_case": dynamic_case,
        "applied_to_robot_config": bool(
            request.apply_to_robot_config and dynamic_case is not None
        ),
        "note": "Run 'validate_selection' to re-simulate with actual actuator masses for verification.",
    }


@router.post("/validate_selection")
def validate_selection_endpoint(
    request: ActuatorSelectionRequest, session: SimulationSession = Depends(get_session)
):
    """
    Second round validation: re-run dynamics with actuator masses added to joints.

    Motor placement considerations:
    - Joint 1 (base): motor is fixed to ground, mass doesn't affect dynamics
    - Joints 2-3: motors are near base, minimal effect on J1
    - Joints 4-6: motor masses significantly affect J1-J3 torque requirements

    This endpoint:
    1. Takes the current selection (from first round)
    2. Adds motor+gearbox masses to each joint's payload
    3. Re-runs inverse dynamics
    4. Re-validates that the selection still meets requirements
    """

    if session.last_trajectory_data is None:
        return {"ok": False, "error": "No trajectory data. Execute program first."}

    # Load library and get first-round selection
    library = _load_actuators_library()
    motors = library.get("motors", [])
    gearboxes = library.get("gearboxes", [])
    compatibility = library.get("compatibility_matrix", {})

    motor_dict = {m["id"]: m for m in motors}
    gearbox_dict = {g["id"]: g for g in gearboxes}

    # Get first-round requirements (only active joints)
    req_round1 = analyze_trajectory_requirements(
        session.last_trajectory_data, robot_type=session.current_robot_type
    )

    # First-round selection
    selection_round1 = select_actuators(
        requirements=req_round1,
        motors=motors,
        gearboxes=gearboxes,
        compatibility_matrix=compatibility,
        safety_factor_torque=request.safety_factor_torque,
        safety_factor_speed=request.safety_factor_speed,
    )

    # Calculate additional mass per joint from selected actuators
    # Joints are named J1-J6 (or J1-J4). Motor N drives joint N.
    # Mass of motor N affects joints 1 to N-1 (proximal joints must support it)

    # Helper to extract joint number from key (handles both "J1" and "joint_1" formats)
    def get_joint_idx(key):
        if key.startswith("J"):
            return int(key[1:]) - 1  # "J1" -> 0
        else:
            return int(key.split("_")[1]) - 1  # "joint_1" -> 0

    num_joints = len(req_round1)
    additional_masses = [
        0.0
    ] * num_joints  # extra payload per joint from downstream actuators

    for joint_key, sel_data in selection_round1.items():
        rec = sel_data.get("recommended")
        if not rec:
            continue

        motor = motor_dict.get(rec["motor_id"], {})
        gearbox = gearbox_dict.get(rec["gearbox_id"], {})

        motor_mass = motor.get("mass_kg") or motor.get("mass", 0)
        gearbox_mass = gearbox.get("mass_kg") or gearbox.get("mass", 0)
        total_mass = motor_mass + gearbox_mass

        # Reasoned mass accumulation
        if session.current_robot_type == "CR4":
            # J1 motor is on base
            if joint_key == "J1":
                pass
            # J2 motor is at the shoulder -> loads J1
            elif joint_key == "J2":
                additional_masses[0] += total_mass
            # J3 motor (J3real) is also at the shoulder -> loads J1 ONLY
            elif joint_key == "J3":
                additional_masses[0] += total_mass
            # J4 motor is at the elbow/wrist end -> loads J3, J2, J1
            elif joint_key == "J4":
                for i in range(3):
                    additional_masses[i] += total_mass
        else:
            # Serial chain (CR6): motor N affects all joints 0 to N-1
            joint_idx = get_joint_idx(joint_key)
            for i in range(joint_idx):
                additional_masses[i] += total_mass

    # Calculate total extra load effect
    extra_payload_at_base = additional_masses[0] if additional_masses else 0

    # Re-run dynamics with new payload (estimated)
    # Simple heuristic: torque scales with (LinkMass + Payload + ActuatorMass)
    # Since we don't want to re-run full simulation inside this endpoint (for speed),
    # we use a reasoned multiplier.

    req_round2 = {}
    for joint_key, req in req_round1.items():
        joint_idx = get_joint_idx(joint_key)
        extra_for_this_joint = (
            additional_masses[joint_idx] if joint_idx < len(additional_masses) else 0
        )

        # Estimate torque increase:
        # We assume a base 'robot mass' that this joint supports.
        # CR4 J2 supports ~20kg of structure. If extra_for_this_joint is 5kg, torque increases ~25%.
        base_mass_supported = 15.0 if session.current_robot_type == "CR4" else 8.0
        torque_multiplier = 1.0 + (
            extra_for_this_joint
            / max(base_mass_supported + session.current_payload_kg, 1.0)
        )

        req_round2[joint_key] = {
            "peak_torque_Nm": req["peak_torque_Nm"] * torque_multiplier,
            "rms_torque_Nm": req["rms_torque_Nm"] * torque_multiplier,
            "peak_velocity_rad_s": req["peak_velocity_rad_s"],
            "peak_velocity_rpm": req["peak_velocity_rpm"],
            "original_peak_torque_Nm": req.get(
                "original_peak_torque_Nm", req["peak_torque_Nm"]
            ),
            "original_peak_speed_rpm": req.get(
                "original_peak_speed_rpm", req["peak_velocity_rpm"]
            ),
            "mean_velocity_rpm": req.get("mean_velocity_rpm", 0),
            # Pass through for selection algorithm's internal calc
            "torque_Nm": req["peak_torque_Nm"] * torque_multiplier,
            "speed_rpm": req["peak_velocity_rpm"],
        }

    # Second-round selection with adjusted requirements
    try:
        selection_round2 = select_actuators(
            requirements=req_round2,
            motors=motors,
            gearboxes=gearboxes,
            compatibility_matrix=compatibility,
            safety_factor_torque=request.safety_factor_torque,
            safety_factor_speed=request.safety_factor_speed,
        )
    except Exception as e:
        return {"ok": False, "error": f"Selection algorithm failed: {str(e)}"}

    # Compare selections
    changes = []
    for joint_key in selection_round1:
        rec1 = selection_round1[joint_key].get("recommended")
        rec2 = selection_round2[joint_key].get("recommended")

        if rec1 and rec2:
            if (
                rec1["motor_id"] != rec2["motor_id"]
                or rec1["gearbox_id"] != rec2["gearbox_id"]
            ):
                changes.append(
                    {
                        "joint": joint_key,
                        "round1": f"{rec1['motor_id']} + {rec1['gearbox_id']}",
                        "round2": f"{rec2['motor_id']} + {rec2['gearbox_id']}",
                        "reason": "Actuator mass effect",
                    }
                )
        elif rec1 and not rec2:
            changes.append(
                {
                    "joint": joint_key,
                    "round1": f"{rec1['motor_id']} + {rec1['gearbox_id']}",
                    "round2": "NO SOLUTION",
                    "reason": "Actuator masses exceed capacity",
                }
            )

    return {
        "ok": True,
        "round1_selection": selection_round1,
        "round2_selection": selection_round2,
        "additional_masses_kg": {
            f"joint_{i + 1}": round(m, 2) for i, m in enumerate(additional_masses)
        },
        "total_actuator_contribution_kg": round(extra_payload_at_base, 2),
        "changes_needed": changes,
        "validated": len(changes) == 0,
        "note": "Round 2 accounts for actuator masses. If changes_needed is empty, selection is validated.",
    }


# =============================================================================
# ACTUATOR LIBRARY FILE MANAGEMENT ENDPOINTS
# =============================================================================


@router.get("/list_actuator_libraries")
def list_actuator_libraries():
    """List all saved actuator library files."""
    libs_list = []
    for f in _libraries_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                lib_data = json.load(fp)
                libs_list.append(
                    {
                        "filename": f.name,
                        "description": lib_data.get("description", ""),
                        "num_motors": len(lib_data.get("motors", [])),
                        "num_gearboxes": len(lib_data.get("gearboxes", [])),
                        "saved_at": lib_data.get("saved_at", ""),
                    }
                )
        except:
            libs_list.append({"filename": f.name, "description": "Error reading file"})
    return {"ok": True, "libraries": libs_list}


@router.post("/save_actuator_library")
def save_actuator_library_to_file(req: SaveActuatorLibraryRequest):
    """Save current actuator library to a separate file (backup/share)."""

    filename = req.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not filename.endswith(".json"):
        filename += ".json"

    library = _load_actuators_library()
    library_data = {
        "description": req.description or "",
        "motors": library.get("motors", []),
        "gearboxes": library.get("gearboxes", []),
        "compatibility_matrix": library.get("compatibility_matrix", {}),
        "metadata": library.get("metadata", {}),
        "saved_at": datetime.now().isoformat(),
    }

    filepath = _libraries_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(library_data, f, indent=2, ensure_ascii=False)

    return {"ok": True, "filename": filename, "path": str(filepath)}


@router.post("/load_actuator_library")
def load_actuator_library_from_file(req: LoadActuatorLibraryRequest):
    """Load an actuator library from file (replaces current library)."""
    filename = req.filename
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = _libraries_dir / filename
    if not filepath.exists():
        return {"ok": False, "error": f"Library file not found: {filename}"}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            library_data = json.load(f)
    except Exception as e:
        return {"ok": False, "error": f"Could not read library file: {e}"}

    # Update the main actuator library
    _save_actuators_library(
        {
            "motors": library_data.get("motors", []),
            "gearboxes": library_data.get("gearboxes", []),
            "compatibility_matrix": library_data.get("compatibility_matrix", {}),
            "metadata": library_data.get("metadata", {}),
        }
    )

    return {
        "ok": True,
        "filename": filename,
        "description": library_data.get("description", ""),
        "num_motors": len(library_data.get("motors", [])),
        "num_gearboxes": len(library_data.get("gearboxes", [])),
    }


@router.delete("/delete_actuator_library/{filename}")
def delete_actuator_library(filename: str):
    """Delete a saved actuator library file."""
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = _libraries_dir / filename
    if not filepath.exists():
        return {"ok": False, "error": f"Library file not found: {filename}"}

    filepath.unlink()
    return {"ok": True, "deleted": filename}


# =============================================================================
# PROJECT EXPORT ENDPOINT
# =============================================================================


@router.post("/export_full_project")
def export_full_project(
    filename: str,
    description: str = "",
    session: SimulationSession = Depends(get_session),
):
    """
    Export complete project: robot config + program + actuator library.
    Useful for sharing complete setups with others.
    """

    filename = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not filename.endswith(".json"):
        filename += ".json"

    library = _load_actuators_library()

    project_data = {
        "description": description,
        "version": "1.0",
        "robot_config": {
            "robot_type": session.current_robot_type,
            "scale": session.current_scale,
            "payload_kg": session.current_payload_kg,
            "payload_inertia": session.current_payload_inertia,
            "friction_coeffs": list(session.current_friction_coeffs)
            if session.current_friction_coeffs
            else None,
            "reflected_inertia": list(session.current_reflected_inertia)
            if session.current_reflected_inertia
            else None,
            "coulomb_friction": list(session.current_coulomb_friction)
            if session.current_coulomb_friction
            else None,
            "motor_masses": list(session.current_motor_masses)
            if session.current_motor_masses
            else None,
            "motor_layout": session.current_motor_layout,
        },
        "program": {"targets": session.targets, "instructions": session.program},
        "actuator_library": {
            "motors": library.get("motors", []),
            "gearboxes": library.get("gearboxes", []),
            "compatibility_matrix": library.get("compatibility_matrix", {}),
        },
        "station_geometries": session.station_geometries,
        "exported_at": datetime.now().isoformat(),
    }

    filepath = _programs_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(project_data, f, indent=2, ensure_ascii=False)

    return {"ok": True, "filename": filename, "path": str(filepath)}
