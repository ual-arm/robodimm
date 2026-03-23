"""
Pydantic models for request/response schemas.
"""

from pydantic import BaseModel
from typing import Optional, List


class LoginRequest(BaseModel):
    username: str
    password: str


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


class ExecuteProgramCommand(BaseModel):
    speed_factor: float = 1.0  # multiplier for animation speed


class RobotConfigCommand(BaseModel):
    robot_type: str = "CR6"  # "CR6" or "CR4"
    scale: float = 1.0  # scale factor for robot reach
    payload_kg: float = 0.0  # payload mass at TCP in kg
    payload_inertia: Optional[dict] = (
        None  # payload inertia tensor or box model {Ixx, Iyy, Izz, com_from_tcp} or {box_size_xyz_m, com_from_tcp}
    )
    friction_coeffs: Optional[List[float]] = (
        None  # viscous friction [Nm/(rad/s)] per joint
    )
    reflected_inertia: Optional[List[float]] = None  # reflected rotor inertia [kg*m^2]
    coulomb_friction: Optional[List[float]] = None  # coulomb friction [Nm]
    structural_mass_scale_exp: Optional[float] = None
    structural_inertia_scale_exp: Optional[float] = None


class StationGeometryUpdate(BaseModel):
    id: str
    position: List[float]  # [x, y, z]
    rotation: List[float]  # [rx, ry, rz] in radians
    scale: float = 1.0


class DeleteStationGeometry(BaseModel):
    """Delete a station geometry by ID."""

    id: str


class SaveProgramRequest(BaseModel):
    """Request to save a program to file."""

    filename: str
    description: Optional[str] = None


class LoadProgramRequest(BaseModel):
    """Request to load a program from file."""

    filename: str


# -- Actuator Library Models --


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
