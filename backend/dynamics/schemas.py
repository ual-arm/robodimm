from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator

# Supported robot types
RobotKind = Literal['CR4', 'CR6']

class InertialSpecModel(BaseModel):
    massKg: float = Field(ge=0.0, description="Mass in kilograms")
    comM: Optional[List[float]] = Field(default=None, description="Center of mass translation [x, y, z] in meters")
    inertiaKgM2: Optional[List[List[float]]] = Field(default=None, description="Inertia tensor (3x3 matrix) in kg*m^2")
    frame: Optional[str] = Field(default=None, description="Frame of reference ('cad' or 'link')")

    @field_validator('comM')
    @classmethod
    def validate_com(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None and len(v) != 3:
            raise ValueError("comM must be a 3-dimensional vector")
        return v

    @field_validator('inertiaKgM2')
    @classmethod
    def validate_inertia_matrix(cls, v: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
        if v is not None:
            if len(v) != 3 or any(len(row) != 3 for row in v):
                raise ValueError("inertiaKgM2 must be a 3x3 matrix")
            # Verify diagonal components are non-negative
            if v[0][0] < 0 or v[1][1] < 0 or v[2][2] < 0:
                raise ValueError("Diagonal elements of the inertia tensor (Ixx, Iyy, Izz) must be non-negative")
        return v


class LimitSpecModel(BaseModel):
    name: str
    lowerLimitRad: float
    upperLimitRad: float
    frictionCoeffNmSPerRad: Optional[float] = Field(default=0.0, description="Viscous friction coefficient in N*m/(rad/s)")


class RobotSpecModel(BaseModel):
    kind: RobotKind
    name: str
    geometry: Dict[str, Any]
    inertials: Dict[str, InertialSpecModel]
    payload: InertialSpecModel
    limits: List[LimitSpecModel]

    @field_validator('limits')
    @classmethod
    def validate_limits_count(cls, v: List[LimitSpecModel], info) -> List[LimitSpecModel]:
        # We check the count of limits based on robot kind if possible (requires access to kind)
        # But we can do general validation
        return v


class InverseDynamicsRequest(BaseModel):
    robot: RobotSpecModel
    q: List[float]
    qd: List[float]
    qdd: List[float]
    schema_version: str = Field(description="Schema version, e.g., 'robodimm.dynamics.v1'")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('q', 'qd', 'qdd')
    @classmethod
    def validate_joint_vectors(cls, v: List[float]) -> List[float]:
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Joint vectors must contain only numeric values")
        return v


class ProgramSpecModel(BaseModel):
    name: str
    targets: List[Dict[str, Any]]
    instructions: List[Dict[str, Any]]


class LegacyDynamicsRequest(BaseModel):
    robot: RobotSpecModel
    program: ProgramSpecModel



class TrajectorySampleModel(BaseModel):
    time_s: float
    q: List[float]
    qd: List[float]  # velocity
    qdd: List[float]  # acceleration


class DynamicsBatchRequest(BaseModel):
    robot: RobotSpecModel
    samples: List[TrajectorySampleModel] = Field(description="List of trajectory samples")
    schema_version: str = Field(description="Schema version, e.g., 'robodimm.dynamics.v1'")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('samples')
    @classmethod
    def validate_batch_size(cls, v: List[TrajectorySampleModel]) -> List[TrajectorySampleModel]:
        if len(v) == 0:
            raise ValueError("Trajectory batch must contain at least 1 sample")
        if len(v) > 10000:
            raise ValueError(f"Trajectory batch size exceeds limit of 10,000 samples (got {len(v)})")
        return v


class DynamicsManifestModel(BaseModel):
    model_id: str
    backend_version: str
    pinocchio_version: str
    robot_hash: str
    trajectory_hash: Optional[str] = None
    q_space_convention: str
    timestamp: str


class DynamicsResponse(BaseModel):
    tauNm: List[float]
    powerW: List[float]
    engine_used: str
    model_id: str
    manifest: DynamicsManifestModel
    warnings: List[str] = Field(default_factory=list)


class CR4DiagnosticsModel(BaseModel):
    constraint_residual_norm: float
    passive_torque_residual_norm: float
    condition_number: float


class BatchSampleResponse(BaseModel):
    time_s: float
    q: List[float]
    velocity: List[float]
    acceleration: List[float]
    tau: List[float]
    power: List[float]


class DynamicsBatchResponse(BaseModel):
    joint_names: List[str]
    samples: List[BatchSampleResponse]
    dt_s: float
    engine_used: str
    model_id: str
    manifest: DynamicsManifestModel
    diagnostics: Optional[List[CR4DiagnosticsModel]] = None
    warnings: List[str] = Field(default_factory=list)


class ValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
