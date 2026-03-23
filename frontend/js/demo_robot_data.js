/**
 * demo_robot_data.js - Default Robot Data for DEMO Mode
 * ======================================================
 *
 * Robot models for DEMO mode (serial chain, simplified like Pink).
 * Dimensions match robot_core builders (CR4, CR6).
 *
 * Enhanced with declarative joint configuration:
 * - offsets: DH joint offsets applied during FK
 * - relativeJoints: joints whose angle is relative to another (e.g., J3_rel = J3 - J2)
 * - dependentJoints: computed from other joints (e.g., J4_aux = -(J2 + J3))
 * - sliderToJoint: mapping from user slider index to DH joint index
 */

export const DEMO_ROBOT_DATA = {
  CR4: {
    name: "CR4",
    type: "CR4",
    scale: 1.0,

    // Joint counts
    nq: 5, // Total DH joints: J1, J2, J3, J4_aux, J5_TCP
    nq_user: 4, // Joints visible to user (sliders): J1, J2, J3, J4_TCP

    joints: [
      { id: 0, name: "J1", type: "revolute", axis: "Z" },
      { id: 1, name: "J2", type: "revolute", axis: "Y" },
      { id: 2, name: "J3", type: "revolute", axis: "Y" },
      { id: 3, name: "J4_aux", type: "revolute", axis: "Y", dependent: true },
      { id: 4, name: "J4_TCP", type: "revolute", axis: "Z" },
    ],

    // DH parameters: T = RotZ(theta) * TransZ(d) * TransX(a) * RotX(alpha)
    dh: [
      { a: 0, alpha: -Math.PI / 2, d: 0.4, theta: 0 }, // J1
      { a: 0.54, alpha: 0, d: 0, theta: 0 }, // J2
      { a: 0.6, alpha: 0, d: 0, theta: 0 }, // J3
      { a: -0.125, alpha: Math.PI / 2, d: 0, theta: 0 }, // J4_aux
      { a: 0, alpha: 0, d: 0.135, theta: 0 }, // J5_TCP
    ],

    // DH joint offsets (theta_DH = theta_user + offset)
    offsets: [0, -Math.PI / 2, Math.PI / 2, Math.PI, 0],

    // Relative joints: q_dh[joint] = q_user[joint] - q_user[relativeTo]
    // J3_DH is relative to J2 (elbow angle is relative)
    relativeJoints: [{ joint: 2, relativeTo: 1 }],

    // Dependent joints: computed from other joints after relative conversion
    // J4_aux = -(J2 + J3_rel) to keep TCP horizontal
    dependentJoints: [{ joint: 3, compute: (q_dh) => -(q_dh[1] + q_dh[2]) }],

    // Mapping from user slider index to DH joint index
    // User has 4 sliders: [J1, J2, J3, J4_TCP]
    // DH has 5 joints:     [J1, J2, J3, J4_aux, J5_TCP]
    // slider[0] -> dh[0] (J1)
    // slider[1] -> dh[1] (J2)
    // slider[2] -> dh[2] (J3)
    // slider[3] -> dh[4] (J4_TCP, skipping J4_aux at index 3)
    sliderToJoint: [0, 1, 2, 4],

    // DEMO mode uses simplified cylinder links (no GLTF meshes)
    // GLTF meshes are only used in PRO mode

    q_home: [0, 0, 0, 0],
    q_min: [-3.14, -1.57, -2.0, -6.28],
    q_max: [3.14, 1.57, 2.0, 6.28],
  },

  CR6: {
    name: "CR6",
    type: "CR6",
    scale: 1.0,

    nq: 6,
    nq_user: 6,

    joints: [
      { id: 0, name: "J1", type: "revolute", axis: "Z" },
      { id: 1, name: "J2", type: "revolute", axis: "Y" },
      { id: 2, name: "J3", type: "revolute", axis: "Y" },
      { id: 3, name: "J4", type: "revolute", axis: "Y" },
      { id: 4, name: "J5", type: "revolute", axis: "Z" },
      { id: 5, name: "J6", type: "revolute", axis: "X" },
    ],

    dh: [
      { a: 0.0, alpha: -Math.PI / 2, d: 0.4, theta: 0 },
      { a: 0.54, alpha: 0, d: 0, theta: 0 },
      { a: 0.1, alpha: -Math.PI / 2, d: 0.0, theta: 0 },
      { a: 0.0, alpha: Math.PI / 2, d: 0.58, theta: 0 },
      { a: 0.0, alpha: Math.PI / 2, d: 0, theta: 0 },
      { a: 0.0, alpha: 0, d: 0.065, theta: 0 },
    ],

    // No offsets for CR6 (all zeros)
    offsets: [0, -Math.PI / 2, 0, 0, Math.PI, 0],

    // No relative joints (each joint is independent)
    relativeJoints: [],

    // No dependent joints
    dependentJoints: [],

    // Direct 1:1 mapping
    sliderToJoint: [0, 1, 2, 3, 4, 5],

    // DEMO mode uses simplified cylinder links (no GLTF meshes)
    // GLTF meshes are only used in PRO mode

    q_home: [0, 0, 0, 0, 0, 0],
    q_min: [-3.14, -1.57, -2.0, -3.14, -2.0, -6.28],
    q_max: [3.14, 1.57, 2.0, 3.14, 2.0, 6.28],
  },
};

export function getDemoRobotData(robotType = "CR4") {
  return DEMO_ROBOT_DATA[robotType] || DEMO_ROBOT_DATA.CR4;
}

export function getRobotList() {
  return Object.keys(DEMO_ROBOT_DATA);
}

// =============================================================================
// STANDARD BENCHMARK PROGRAM (for DEMO/PRO comparison)
// =============================================================================
// This program is designed to exercise all joints with significant motion
// for comparing dynamics results between DEMO and PRO modes.

export const BENCHMARK_TARGETS_CR4 = [
  { name: 'PALLET_HOME', q: [0.0, 0.0, 0.0, 0.0] },
  { name: 'PICK_APPROACH_LEFT', q: [0.9, 1.1, 1.2, 0.0] },
  { name: 'PICK_LEFT_FLOOR', q: [0.9, 1.15, 1.25, 0.0] },
  { name: 'PICK_LIFT_LEFT', q: [0.9, 1.0, 1.1, 0.0] },
  { name: 'PLACE_APPROACH_FRONT', q: [0.0, 1.1, 1.2, -1.571] },
  { name: 'PLACE_FRONT_BASE', q: [0.0, 1.15, 1.25, -1.571] },
  { name: 'PLACE_LIFT_FRONT', q: [0.0, 1.0, 1.1, -1.571] }
];

export const BENCHMARK_PROGRAM_CR4 = [
  { type: 'MoveJ', target_name: 'PALLET_HOME', speed: 45, zone: 0 },
  { type: 'MoveJ', target_name: 'PICK_APPROACH_LEFT', speed: 45, zone: 0 },
  { type: 'MoveJ', target_name: 'PICK_LEFT_FLOOR', speed: 25, zone: 0 },
  { type: 'MoveJ', target_name: 'PICK_LIFT_LEFT', speed: 30, zone: 0 },
  { type: 'MoveJ', target_name: 'PLACE_APPROACH_FRONT', speed: 45, zone: 0 },
  { type: 'MoveJ', target_name: 'PLACE_FRONT_BASE', speed: 25, zone: 0 },
  { type: 'MoveJ', target_name: 'PLACE_LIFT_FRONT', speed: 30, zone: 0 },
  { type: 'MoveJ', target_name: 'PALLET_HOME', speed: 45, zone: 0 }
];

export const BENCHMARK_TARGETS_CR6 = [
  { name: 'BENCH_HOME', q: [0, 0, 0, 0, 0, 0] },
  { name: 'BENCH_EXTEND', q: [-0.785, 0.785, -0.785, 0.785, 0.785, 0] },
  { name: 'BENCH_MID', q: [0.524, -0.524, 0.524, 0.524, -0.524, 1.571] },
  { name: 'BENCH_BACK', q: [0.262, -0.785, 0.262, -0.262, 0.785, -0.785] }
];

export const BENCHMARK_PROGRAM_CR6 = [
  { type: 'MoveJ', target_name: 'BENCH_HOME', speed: 50, zone: 0 },
  { type: 'MoveJ', target_name: 'BENCH_EXTEND', speed: 50, zone: 0 },
  { type: 'MoveJ', target_name: 'BENCH_MID', speed: 50, zone: 0 },
  { type: 'MoveJ', target_name: 'BENCH_BACK', speed: 50, zone: 0 },
  { type: 'MoveJ', target_name: 'BENCH_HOME', speed: 50, zone: 0 }
];

/**
 * Get benchmark data for a robot type
 */
export function getBenchmarkData(robotType = 'CR4') {
  if (robotType === 'CR6') {
    return {
      targets: BENCHMARK_TARGETS_CR6,
      program: BENCHMARK_PROGRAM_CR6
    };
  }
  return {
    targets: BENCHMARK_TARGETS_CR4,
    program: BENCHMARK_PROGRAM_CR4
  };
}

export default { 
  DEMO_ROBOT_DATA, 
  getDemoRobotData, 
  getRobotList,
  getBenchmarkData,
  BENCHMARK_TARGETS_CR4,
  BENCHMARK_PROGRAM_CR4,
  BENCHMARK_TARGETS_CR6,
  BENCHMARK_PROGRAM_CR6
};
