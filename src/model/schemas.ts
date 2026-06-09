export interface InertialSpec {
  body: string;
  massKg: number;
  comM?: [number, number, number];
  inertiaKgM2?: [
    [number, number, number],
    [number, number, number],
    [number, number, number]
  ];
  frame?: 'cad' | 'link' | 'tcp';
}

export interface PrimitiveSpec {
  type: 'cylinder' | 'box' | 'sphere';
  dimensions: number[]; // [radius, length] or [x, y, z] or [radius]
}

export interface VisualSpec {
  body: string;
  frameName: string;
  kind: 'primitive' | 'mesh';
  meshUrl?: string;
  primitive?: PrimitiveSpec;
  originM: [number, number, number];
  rpyRad: [number, number, number];
  scale: [number, number, number];
  visible: boolean;
  axesVisible?: boolean;
}

export interface StationObject {
  id: string;
  name: string;
  meshUrl?: string;
  primitive?: PrimitiveSpec;
  positionM: [number, number, number];
  rotationRpyRad: [number, number, number];
  scale: [number, number, number];
  visible: boolean;
}

export interface JointLimit {
  name: string;
  lowerLimitRad: number;
  upperLimitRad: number;
  maxVelocityRadS: number;
  maxAccelerationRadS2: number;
  frictionCoeffNmSPerRad?: number;
}

export interface Cr4Geometry {
  A: [number, number, number];
  O: [number, number, number];
  B: [number, number, number];
  C: [number, number, number];
  D: [number, number, number];
  E: [number, number, number];
  F: [number, number, number];
  G: [number, number, number];
  H: [number, number, number];
  P: [number, number, number];
  J4: [number, number, number];
  EE: [number, number, number];
  TCP: [number, number, number];
}

export interface DHJointSpec {
  name: string;
  a_m: number;
  alpha_rad: number;
  d_m: number;
  theta_offset_rad: number;
}

export interface Cr6Geometry {
  joints: DHJointSpec[];
  tool_transform: number[][]; // 4x4 matrix
}

export interface RobotSpec {
  schema: 'robodimm.robot.v1';
  kind: 'CR4' | 'CR6';
  name: string;
  units: 'SI';
  geometry: Cr4Geometry | Cr6Geometry;
  inertials: Record<string, InertialSpec>;
  payload: InertialSpec;
  visuals: VisualSpec[];
  station: StationObject[];
  limits: JointLimit[];
}

export interface ProgramTarget {
  name: string;
  q: number[]; // joint values in radians
  tcpPose?: number[][]; // 4x4 matrix
}

export interface MoveJInstruction {
  type: 'MoveJ';
  target_name: string;
  speed_rad_s: number;
  zone_m: number;
}

export interface MoveLInstruction {
  type: 'MoveL';
  target_name: string;
  tcp_speed_m_s: number;
  zone_m: number;
}

export interface PauseInstruction {
  type: 'Pause';
  duration_s: number;
}

export type ProgramInstruction = MoveJInstruction | MoveLInstruction | PauseInstruction;

export interface ProgramSpec {
  schema: 'robodimm.program.v1';
  name: string;
  targets: ProgramTarget[];
  instructions: ProgramInstruction[];
}

export interface TorqueSample {
  time_s: number;
  q: number[];
  velocity: number[];
  acceleration: number[];
  joint_velocity: number[];
  joint_acceleration: number[];
  tau: number[];
}

export interface TorqueLog {
  joint_names: string[];
  samples: TorqueSample[];
  dt_s: number;
  engine_used?: string;
  model_id?: string;
  manifest?: any;
}

export type GearboxType = 'harmonic' | 'cycloidal';

export type SizingObjective = 'min_mass' | 'min_power' | 'min_gearbox' | 'max_margin';

export interface SizingMargins {
  continuous: number;
  peak: number;
  speed: number;
  power: number;
  motorPeakFactor: number;
  enforcePowerLimit?: boolean;
  sizingObjective?: SizingObjective;
}

export interface JointDemand {
  joint_name: string;
  tau_rms_Nm: number;
  tau_peak_Nm: number;
  speed_rms_rad_s: number;
  speed_peak_rad_s: number;
  power_rms_W: number;
  power_peak_W: number;
  regen_peak_W: number;
  cycle_time_s: number;
}

export interface MotorSpec {
  id: string;
  name: string;
  manufacturer: string;
  series: string;
  rated_power_W: number;
  rated_torque_Nm: number;
  rated_speed_rpm: number;
  stall_torque_Nm: number;
  no_load_speed_rpm: number;
  max_continuous_current_A: number;
  terminal_resistance_ohm: number;
  terminal_inductance_mH: number;
  torque_constant_Nm_A: number;
  speed_constant_rpm_V: number;
  rotor_inertia_gcm2: number;
  mass_kg: number;
  voltage_V: number;
  flange_mm: number;
  length_mm: number;
}

export interface GearboxSpec {
  id: string;
  name: string;
  manufacturer: string;
  type: string;
  series: string;
  ratio: number;
  stages: number;
  max_continuous_torque_Nm: number;
  max_intermittent_torque_Nm: number;
  efficiency: number;
  max_input_speed_rpm: number;
  backlash_arcmin: number;
  mass_kg: number;
  inertia_gcm2: number;
  for_servo_mm: number;
  length_mm: number;
}

export interface ActuatorLibrary {
  motors: MotorSpec[];
  gearboxes: GearboxSpec[];
  compatibility_matrix: Record<string, string[]>; // motor_id -> gearbox_ids[]
  metadata: {
    version: string;
    last_updated: string;
    description: string;
  };
}

export interface ActuatorCandidate {
  motor_id: string;
  gearbox_id: string;
  gearbox_type: GearboxType;
  ratio: number;
  passes: boolean;
  failure_reasons: string[];
  continuous_margin: number;
  peak_margin: number;
  speed_margin: number;
  gearbox_continuous_margin: number;
  gearbox_peak_margin: number;
  power_margin: number;
  min_margin: number;
  total_mass_kg: number;
  motor: MotorSpec;
  gearbox: GearboxSpec;
}

export interface JointActuatorSelection {
  demand: JointDemand;
  candidates: ActuatorCandidate[];
  best?: ActuatorCandidate;
}

export interface RobotActuatorSelection {
  joints: JointActuatorSelection[];
  complete: boolean;
}

export interface ActuatorSizingReport {
  schema: "robodimm.actuator_sizing_report.v1";
  robot_kind: string;
  robot_name: string;
  dynamics_source: string;
  torque_log_hash: string;
  catalog_version: string;
  catalog_anonymized: boolean;
  motor_peak_policy: string;
  margins: SizingMargins;
  joints: JointActuatorSelection[];
  complete: boolean;
}

export function cloneRobotSpec(spec: RobotSpec): RobotSpec {
  return JSON.parse(JSON.stringify(spec)) as RobotSpec;
}

export interface RobotPackageAssetMesh {
  body: string;
  frameName: string;
  path: string;
  format: 'glb' | 'gltf' | 'stl';
  units: 'm' | 'mm';
  frame: 'cad' | 'link' | 'tcp';
}

export interface RobotPackageSpec {
  schema: 'robodimm.package.v1';
  robot: RobotSpec;
  assets: {
    meshes: RobotPackageAssetMesh[];
  };
}
