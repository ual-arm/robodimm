/**
 * interpolation.js - Motion interpolation and program execution
 * ==============================================================
 * 
 * Provides trapezoidal velocity profile interpolation for smooth
 * robot motion in DEMO mode. Matches industrial robot controller behavior.
 */

/**
 * Trapezoidal velocity profile interpolation.
 * 
 * Acceleration phase → Constant velocity cruise → Deceleration phase
 * 
 * @param {number} t - normalized time [0, 1]
 * @param {number} accelFraction - fraction of time for accel/decel (default 0.2)
 * @returns {number} normalized position [0, 1]
 */
export function trapezoidalProfile(t, accelFraction = 0.2) {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  
  const ta = accelFraction;
  const td = 1.0 - accelFraction;
  // Max velocity during cruise (area under trapezoid must equal 1.0)
  const vmax = 1.0 / (1.0 - accelFraction);
  
  if (t < ta) {
    // Acceleration phase: s = 0.5 * a * t², where a = vmax/ta
    return 0.5 * vmax * t * t / ta;
  } else if (t < td) {
    // Constant velocity phase
    return 0.5 * vmax * ta + vmax * (t - ta);
  } else {
    // Deceleration phase
    const dt = 1.0 - t;
    return 1.0 - 0.5 * vmax * dt * dt / ta;
  }
}

/**
 * Linear interpolation (for comparison/debugging).
 * @param {number} t - normalized time [0, 1]
 * @returns {number} normalized position [0, 1]
 */
export function linearProfile(t) {
  return Math.max(0, Math.min(1, t));
}

/**
 * Interpolate between two joint configurations.
 * 
 * @param {number[]} q_start - starting joint angles
 * @param {number[]} q_end - target joint angles
 * @param {number} t - normalized time [0, 1]
 * @param {string} profile - 'trapezoidal' or 'linear'
 * @returns {number[]} interpolated joint angles
 */
export function interpolateJoints(q_start, q_end, t, profile = 'trapezoidal') {
  const prof = profile === 'trapezoidal' ? trapezoidalProfile(t) : linearProfile(t);
  return q_start.map((q, i) => q + (q_end[i] - q) * prof);
}

/**
 * Animate a joint motion from q_start to q_end.
 * 
 * @param {number[]} q_start - starting joint angles
 * @param {number[]} q_end - target joint angles
 * @param {number} durationMs - total duration in milliseconds
 * @param {function} onFrame - callback(currentQ) for each frame
 * @param {Object} control - { cancelled: boolean } for early stop
 * @returns {Promise} resolves when animation completes or is cancelled
 */
export function animateMoveJ(q_start, q_end, durationMs, onFrame, control = { cancelled: false }) {
  return new Promise((resolve) => {
    const startTime = performance.now();
    
    function step(currentTime) {
      if (control.cancelled) {
        resolve();
        return;
      }
      
      const elapsed = currentTime - startTime;
      const t = Math.min(elapsed / durationMs, 1.0);
      const s = trapezoidalProfile(t);
      
      const q_current = q_start.map((q, i) => q + (q_end[i] - q) * s);
      onFrame(q_current);
      
      if (t < 1.0) {
        requestAnimationFrame(step);
      } else {
        resolve();
      }
    }
    
    requestAnimationFrame(step);
  });
}

/**
 * Calculate MoveJ duration based on largest joint displacement.
 * 
 * Matches PRO mode calculation using industrial robot velocity limits.
 * 
 * @param {number[]} q_start - starting joints
 * @param {number[]} q_end - target joints
 * @param {number} speedPercent - speed percentage (100 = default)
 * @returns {number} duration in milliseconds
 */
export function calculateMoveJDuration(q_start, q_end, speedPercent = 100) {
  const maxDelta = Math.max(...q_end.map((q, i) => Math.abs(q - q_start[i])));
  const speedFactor = Math.max(10, Math.min(200, speedPercent)) / 100;
  
  // Match PRO mode: max_joint_vel = 2.5 rad/s at 100% speed
  // This is typical for industrial robot axes 1-3
  const maxJointVel = 2.5; // rad/s
  const effectiveVel = maxJointVel * speedFactor;
  
  // Duration = displacement / velocity
  const durationSec = maxDelta / effectiveVel;
  const durationMs = durationSec * 1000;
  
  return Math.max(200, durationMs); // Min 200ms for very small moves
}

/**
 * Execute a program instruction by instruction.
 * 
 * @param {Object[]} program - list of instructions
 * @param {Object[]} targets - available targets
 * @param {number[]} q_current - current joint angles
 * @param {function} onFrame - called each frame with current q
 * @param {function} onInstruction - called when instruction changes (index, instr)
 * @param {Object} control - { cancelled: boolean, speedFactor: number }
 * @returns {Promise}
 */
export async function executeProgram(program, targets, q_current, onFrame, onInstruction, control = { cancelled: false, speedFactor: 1.0 }) {
  let q = [...q_current];
  
  for (let i = 0; i < program.length; i++) {
    if (control.cancelled) break;
    
    const instr = program[i];
    onInstruction(i, instr);
    
    if (instr.type === 'Pause') {
      await sleep((instr.pause_time || 1) * 1000);
      continue;
    }
    
    if (instr.type === 'MoveJ') {
      const target = targets.find(t => t.name === instr.target_name);
      if (!target) {
        console.warn(`[executeProgram] Target ${instr.target_name} not found`);
        continue;
      }
      
      const speed = (instr.speed || 100) * control.speedFactor;
      const duration = calculateMoveJDuration(q, target.q, speed);
      
      await animateMoveJ(q, target.q, duration, (q_frame) => {
        q = q_frame;
        onFrame(q_frame);
      }, control);
    }
    
    // MoveL and MoveC are not supported in DEMO mode
    if (instr.type === 'MoveL' || instr.type === 'MoveC') {
      console.warn(`[executeProgram] ${instr.type} not supported in DEMO mode, using MoveJ`);
      const target = targets.find(t => t.name === instr.target_name);
      if (target) {
        const speed = (instr.speed || 100) * control.speedFactor;
        const duration = calculateMoveJDuration(q, target.q, speed);
        await animateMoveJ(q, target.q, duration, (q_frame) => {
          q = q_frame;
          onFrame(q_frame);
        }, control);
      }
    }
  }
  
  onInstruction(-1, null); // Signal completion
  return q;
}

/**
 * Generate complete trajectory with dynamics for a program.
 * Used for plotting in DEMO mode.
 * 
 * @param {Object[]} program - list of instructions
 * @param {Object[]} targets - available targets
 * @param {number[]} q_current - current joint angles
 * @param {number} dt - time step in seconds (default 0.02 = 50Hz)
 * @param {number} speedFactor - speed multiplier
 * @returns {Object} trajectory data {time, q, qd, qdd, tau}
 */
export function generateProgramTrajectory(program, targets, q_current, dt = 0.02, speedFactor = 1.0) {
  let q = [...q_current];
  let time = 0;
  const maxJointVel = 2.5;
  
  const trajectory = {
    time: [],
    q: [],
    qd: [],
    qdd: [],
    tau: []
  };
  
  // Helper to add point to trajectory
  function addPoint(t, q_val, qd_val, qdd_val) {
    trajectory.time.push(t);
    trajectory.q.push([...q_val]);
    trajectory.qd.push([...qd_val]);
    trajectory.qdd.push([...qdd_val]);
  }
  
  // Initial point (zero velocity/acceleration)
  addPoint(time, q, [0,0,0,0], [0,0,0,0]);

  // Match backend behavior: optional approach to first motion target.
  const firstMotion = program.find(instr => instr.type !== 'Pause');
  if (firstMotion) {
    const firstTarget = targets.find(t => t.name === firstMotion.target_name);
    if (firstTarget) {
      const jointDisp = Math.max(...q.map((qi, j) => Math.abs(firstTarget.q[j] - qi)));
      if (jointDisp > 0.05) {
        const approachVel = maxJointVel * 0.5;
        const approachTime = jointDisp / approachVel;
        const approachSteps = Math.max(30, Math.ceil(approachTime / dt));
        let qPrev = [...q];
        let qdPrev = [0,0,0,0];
        for (let s = 1; s <= approachSteps; s++) {
          const tNorm = s / approachSteps;
          const sProf = trapezoidalProfile(tNorm);
          const qNew = q.map((qi, j) => qi + (firstTarget.q[j] - qi) * sProf);
          const qdNew = qNew.map((v, j) => (v - qPrev[j]) / dt);
          const qddNew = qdNew.map((v, j) => (v - qdPrev[j]) / dt);
          time += dt;
          addPoint(time, qNew, qdNew, qddNew);
          qPrev = qNew;
          qdPrev = qdNew;
        }
        q = [...firstTarget.q];
      }
    }
  }
  
  for (let i = 0; i < program.length; i++) {
    const instr = program[i];
    
    if (instr.type === 'Pause') {
      const pauseTime = (instr.pause_time || 1);
      const pauseSteps = Math.ceil(pauseTime / dt);
      
      for (let s = 1; s <= pauseSteps; s++) {
        time += dt;
        addPoint(time, q, [0,0,0,0], [0,0,0,0]);
      }
      continue;
    }
    
    if (instr.type === 'MoveJ' || instr.type === 'MoveL' || instr.type === 'MoveC') {
      const target = targets.find(t => t.name === instr.target_name);
      if (!target) continue;
      
      const speedRaw = (instr.speed || 100) * speedFactor;
      const speedInstr = Math.max(10, Math.min(200, speedRaw)) / 100.0;
      const effectiveMaxVel = maxJointVel * speedInstr;
      const jointDisp = Math.max(...q.map((qi, j) => Math.abs(target.q[j] - qi)));
      const moveTime = jointDisp / effectiveMaxVel;
      const steps = Math.max(20, Math.ceil(moveTime / dt));
      
      let q_prev = [...q];
      let qd_prev = [0,0,0,0];
      
      for (let s = 1; s <= steps; s++) {
        const t_norm = s / steps;
        const s_prof = trapezoidalProfile(t_norm);
        
        // Position
        const q_new = q.map((qi, j) => qi + (target.q[j] - qi) * s_prof);
        
        // Velocity (numerical derivative)
        const qd_new = q_new.map((qji, j) => (qji - q_prev[j]) / dt);
        
        // Acceleration (numerical derivative)
        const qdd_new = qd_new.map((qdji, j) => (qdji - qd_prev[j]) / dt);
        
        time += dt;
        addPoint(time, q_new, qd_new, qdd_new);
        
        q_prev = q_new;
        qd_prev = qd_new;
      }
      
      q = [...target.q];
    }
  }

  // Recompute derivatives globally with central differences and smoothing
  // to match backend dynamics pipeline and avoid local segment spikes.
  const n = trajectory.q.length;
  const nq = trajectory.q[0]?.length || 0;
  const qArr = trajectory.q.map(v => [...v]);
  const vArr = Array.from({ length: n }, () => Array(nq).fill(0));
  const aArr = Array.from({ length: n }, () => Array(nq).fill(0));

  for (let i = 1; i < n - 1; i++) {
    for (let j = 0; j < nq; j++) {
      vArr[i][j] = (qArr[i + 1][j] - qArr[i - 1][j]) / (2 * dt);
    }
  }
  if (n > 1) {
    for (let j = 0; j < nq; j++) {
      vArr[0][j] = (qArr[1][j] - qArr[0][j]) / dt;
      vArr[n - 1][j] = (qArr[n - 1][j] - qArr[n - 2][j]) / dt;
    }
  }

  const smooth2D = (arr, window) => {
    if (arr.length < window) return arr;
    const out = arr.map(v => [...v]);
    const half = Math.floor(window / 2);
    for (let i = half; i < arr.length - half; i++) {
      for (let j = 0; j < nq; j++) {
        let sum = 0;
        for (let k = i - half; k <= i + half; k++) sum += arr[k][j];
        out[i][j] = sum / window;
      }
    }
    return out;
  };

  const vSmooth = smooth2D(vArr, 7);
  for (let i = 1; i < n - 1; i++) {
    for (let j = 0; j < nq; j++) {
      aArr[i][j] = (vSmooth[i + 1][j] - vSmooth[i - 1][j]) / (2 * dt);
    }
  }
  if (n > 1) {
    for (let j = 0; j < nq; j++) {
      aArr[0][j] = (vSmooth[1][j] - vSmooth[0][j]) / dt;
      aArr[n - 1][j] = (vSmooth[n - 1][j] - vSmooth[n - 2][j]) / dt;
    }
  }

  trajectory.qd = vSmooth;
  trajectory.qdd = smooth2D(aArr, 9);
  
  return trajectory;
}

/**
 * Simple sleep utility.
 * @param {number} ms - milliseconds
 * @returns {Promise}
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Generate trajectory points for a MoveJ (for pre-computation).
 * 
 * @param {number[]} q_start - starting joints
 * @param {number[]} q_end - target joints
 * @param {number} durationMs - total duration
 * @param {number} dt - time step (default 50ms for 20Hz)
 * @returns {number[][]} array of joint configurations
 */
export function generateTrajectory(q_start, q_end, durationMs, dt = 50) {
  const points = [];
  const steps = Math.ceil(durationMs / dt);
  
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const q = interpolateJoints(q_start, q_end, t);
    points.push(q);
  }
  
  return points;
}

export default {
  trapezoidalProfile,
  linearProfile,
  interpolateJoints,
  animateMoveJ,
  calculateMoveJDuration,
  executeProgram,
  generateTrajectory
};
