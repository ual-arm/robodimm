/**
 * kinematics_lite.js - Forward Kinematics using DH Parameters
 * =============================================================
 * 
 * Lightweight kinematics for DEMO mode (no backend required).
 * Uses standard Denavit-Hartenberg parameters for forward kinematics.
 * 
 * Supported robots: CR4 (4-DOF), CR6 (6-DOF)
 * 
 * DH Convention (Standard):
 * - a: link length (distance along X axis of previous frame)
 * - alpha: link twist (rotation about X axis of previous frame)
 * - d: link offset (distance along Z axis of current frame)
 * - theta: joint angle (rotation about Z axis)
 * 
 * Transformation: T = RotZ(theta) * TransZ(d) * TransX(a) * RotX(alpha)
 */

import { DEMO_ROBOT_DATA } from './demo_robot_data.js';

// Re-export robot data for backwards compatibility
export const ROBOTS = {
  CR4: {
    name: 'CR4',
    nq: DEMO_ROBOT_DATA.CR4.nq_user,
    description: DEMO_ROBOT_DATA.CR4.name,
    dh: DEMO_ROBOT_DATA.CR4.dh
  },
  CR6: {
    name: 'CR6',
    nq: DEMO_ROBOT_DATA.CR6.nq_user,
    description: DEMO_ROBOT_DATA.CR6.name,
    dh: DEMO_ROBOT_DATA.CR6.dh
  }
};

/**
 * Legacy CR4 offsets export (for backwards compatibility)
 * These are now defined in demo_robot_data.js
 */
export const CR4_OFFSETS = DEMO_ROBOT_DATA.CR4.offsets;

/**
 * Compute DH transformation matrix
 * Standard DH: T = RotZ(theta) * TransZ(d) * TransX(a) * RotX(alpha)
 * 
 * @param {number} a - link length
 * @param {number} alpha - link twist
 * @param {number} d - link offset
 * @param {number} theta - joint angle
 * @returns {number[][]} 4x4 transformation matrix
 */
export function dhTransform(a, alpha, d, theta) {
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const ca = Math.cos(alpha);
  const sa = Math.sin(alpha);
  
  return [
    [ct, -st * ca,  st * sa, a * ct],
    [st,  ct * ca, -ct * sa, a * st],
    [0,   sa,       ca,      d],
    [0,   0,        0,       1]
  ];
}

/**
 * Multiply two 4x4 matrices
 * @param {number[][]} A - 4x4 matrix
 * @param {number[][]} B - 4x4 matrix
 * @returns {number[][]} 4x4 result
 */
export function multiplyMatrices(A, B) {
  const result = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

/**
 * Identity 4x4 matrix
 * @returns {number[][]} identity matrix
 */
function identity4() {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
}

/**
 * Convert user joint values to DH joint values using robot model rules.
 * 
 * Process:
 * 1. Map user sliders to DH joints using sliderToJoint
 * 2. Apply relative joint conversions (e.g., J3_rel = J3 - J2)
 * 3. Compute dependent joints (e.g., J4_aux = -(J2 + J3))
 * 
 * @param {string} robotType - 'CR4' or 'CR6'
 * @param {number[]} q_user - user joint values (length = nq_user)
 * @returns {number[]} q_dh - DH joint values (length = nq)
 */
export function userToDH(robotType, q_user) {
  const robot = DEMO_ROBOT_DATA[robotType];
  if (!robot) {
    throw new Error(`Unknown robot: ${robotType}`);
  }
  
  // Initialize DH joint array
  const q_dh = new Array(robot.nq).fill(0);
  
  // 1. Map user sliders to DH joints
  for (let i = 0; i < robot.sliderToJoint.length; i++) {
    const dhIdx = robot.sliderToJoint[i];
    q_dh[dhIdx] = q_user[i] || 0;
  }
  
  // 2. Apply relative joint conversions
  // J_rel = J_absolute - J_parent
  for (const rel of (robot.relativeJoints || [])) {
    q_dh[rel.joint] = q_dh[rel.joint] - q_dh[rel.relativeTo];
  }
  
  // 3. Compute dependent joints
  for (const dep of (robot.dependentJoints || [])) {
    q_dh[dep.joint] = dep.compute(q_dh);
  }
  
  return q_dh;
}

/**
 * Compute Forward Kinematics
 * 
 * @param {string} robotType - 'CR4' or 'CR6'
 * @param {number[]} q_user - user joint angles array (length = nq_user)
 * @param {number} scale - scale factor (applied to a and d parameters)
 * @returns {Object} - frame positions and end effector transform
 */
export function computeFK(robotType, q_user, scale = 1.0) {
  const robot = DEMO_ROBOT_DATA[robotType];
  if (!robot) {
    throw new Error(`Unknown robot: ${robotType}`);
  }
  
  // Convert user joints to DH joints
  const q_dh = userToDH(robotType, q_user);
  
  // Get DH parameters with scaling
  const dh = robot.dh.map((joint, i) => ({
    a: joint.a * scale,
    d: joint.d * scale,
    alpha: joint.alpha,
    offset: (robot.offsets || [])[i] || 0
  }));
  
  // Compute FK chain
  let T = identity4();
  const frames = [];
  
  // Add initial frame (before first joint) - base reference frame
  frames.push({
    joint: -1,
    x: 0,
    y: 0,
    z: 0,
    matrix: identity4().map(row => [...row])
  });
  
  for (let i = 0; i < dh.length; i++) {
    // theta = q_dh + offset
    const theta = q_dh[i] + dh[i].offset;
    
    const Ti = dhTransform(dh[i].a, dh[i].alpha, dh[i].d, theta);
    T = multiplyMatrices(T, Ti);
    
    frames.push({
      joint: i,
      x: T[0][3],
      y: T[1][3],
      z: T[2][3],
      matrix: T.map(row => [...row])
    });
  }
  
  return {
    frames: frames,
    ee: frames[frames.length - 1],
    transform: T,
    q_dh: q_dh  // Return computed DH joints for debugging
  };
}

/**
 * Get DH parameters for a robot (with scaling)
 * 
 * @param {string} robotType - 'CR4' or 'CR6'
 * @param {number} scale - scale factor
 * @returns {Object[]} DH parameters array
 */
export function getRobotDH(robotType, scale = 1.0) {
  const robot = DEMO_ROBOT_DATA[robotType];
  if (!robot) {
    throw new Error(`Unknown robot: ${robotType}`);
  }
  
  return robot.dh.map(joint => ({
    a: joint.a * scale,
    alpha: joint.alpha,
    d: joint.d * scale,
    theta: joint.theta,
    offset: (robot.offsets || [])[robot.dh.indexOf(joint)] || 0
  }));
}

/**
 * Get robot info
 * 
 * @param {string} robotType - 'CR4' or 'CR6'
 * @param {number} scale - scale factor
 * @returns {Object} robot info object
 */
export function getRobotInfo(robotType, scale = 1.0) {
  const robot = DEMO_ROBOT_DATA[robotType];
  if (!robot) return null;
  
  return {
    name: robot.name,
    type: robot.type,
    nq: robot.nq_user,
    nq_internal: robot.nq,
    description: robot.name,
    dh: getRobotDH(robotType, scale),
    q_home: robot.q_home,
    q_min: robot.q_min,
    q_max: robot.q_max,
    hasDependentJoints: (robot.dependentJoints || []).length > 0,
    hasRelativeJoints: (robot.relativeJoints || []).length > 0
  };
}

/**
 * Get number of user-controlled joints
 * @param {string} robotType - 'CR4' or 'CR6'
 * @returns {number} number of user joints
 */
export function getUserJointCount(robotType) {
  const robot = DEMO_ROBOT_DATA[robotType];
  return robot ? robot.nq_user : 0;
}

/**
 * Get number of DH joints (internal)
 * @param {string} robotType - 'CR4' or 'CR6'
 * @returns {number} number of DH joints
 */
export function getDHJointCount(robotType) {
  const robot = DEMO_ROBOT_DATA[robotType];
  return robot ? robot.nq : 0;
}

export default {
  ROBOTS,
  CR4_OFFSETS,
  dhTransform,
  multiplyMatrices,
  computeFK,
  userToDH,
  getRobotDH,
  getRobotInfo,
  getUserJointCount,
  getDHJointCount
};
