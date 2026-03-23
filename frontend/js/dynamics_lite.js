/**
 * dynamics_lite.js - Inverse Dynamics for DEMO Mode (CR4 & CR6)
 * ==============================================================
 * 
 * Recursive Newton-Euler algorithm for CR4 (4-DOF) and CR6 (6-DOF) robots.
 * Simplified model: treats robot as open chain.
 * Results are approximate compared to PRO mode with Pinocchio.
 * 
 * Based on: dinamicaInversaNEmain.m (MATLAB)
 * Algorithm: Recursive Newton-Euler (forward + backward recursion)
 */

import { getDemoRobotData } from './demo_robot_data.js';
import { userToDH } from './kinematics_lite.js';

// =============================================================================
// CR4 SERIAL-5 PARAMETERS (aligned with validation scripts)
// =============================================================================

const CR4_SERIAL_BASE = {
  dhA: [0.0, 0.540, 0.600, -0.125, 0.0],
  dhAlpha: [-Math.PI / 2, 0.0, 0.0, Math.PI / 2, 0.0],
  dhD: [0.40, 0.0, 0.0, 0.0, 0.135],
  masses: [8.0, 10.0, 6.0, 2.0, 1.5],
  coms: [
    [0.0, 0.20, 0.0],
    [-0.27, -0.025, 0.0],
    [-0.30, -0.025, 0.0],
    [0.0625, -0.025, 0.0],
    [-0.025, 0.0, -0.0675]
  ],
  inertias: [
    [[0.171667, 0, 0], [0, 0.1300, 0], [0, 0, 0.171667]],
    [[0.03075, 0, 0], [0, 0.2550, 0], [0, 0, 0.26175]],
    [[0.0122, 0, 0], [0, 0.1850, 0], [0, 0, 0.1872]],
    [[0.0016, 0, 0], [0, 0.003404, 0], [0, 0, 0.003404]],
    [[0.002616, 0, 0], [0, 0.002616, 0], [0, 0, 0.000675]]
  ]
};

const CR4_DYNAMIC_DEFAULTS = {
  reflectedInertia: [0.0, 0.0, 0.0, 0.0],
  viscousFriction: [0.0, 0.0, 0.0, 0.0],
  coulombFriction: [0.0, 0.0, 0.0, 0.0],
  payload: {
    mass: 0.0,
    comFromTcp: [0.0, 0.0, 0.0],
    inertiaDiag: [0.0, 0.0, 0.0]
  }
};

// =============================================================================
// CR6 INERTIAL PARAMETERS (Estimated from CAD/URDF data)
// =============================================================================

const CR6_INERTIA = {
  // Link 1 (Base) - Rotates around Z
  link1: { 
    m: 18.0, 
    com: [0.0, 0.0, 0.25], 
    I: [[0.25,0,0],[0,0.25,0],[0,0,0.08]] 
  },
  // Link 2 (Shoulder) - Rotates around Y  
  link2: { 
    m: 15.0, 
    com: [0.0, 0.0, 0.27], 
    I: [[0.15,0,0],[0,0.35,0],[0,0,0.30]] 
  },
  // Link 3 (Elbow) - Rotates around Y
  link3: { 
    m: 10.0, 
    com: [0.20, 0.0, 0.10], 
    I: [[0.10,0,0],[0,0.25,0],[0,0,0.20]] 
  },
  // Link 4 (Wrist Roll) - Rotates around X
  link4: { 
    m: 3.0, 
    com: [0.19, 0.0, 0.0], 
    I: [[0.02,0,0],[0,0.02,0],[0,0,0.02]] 
  },
  // Link 5 (Wrist Pitch) - Rotates around Y
  link5: { 
    m: 2.0, 
    com: [0.03, 0.0, 0.0], 
    I: [[0.01,0,0],[0,0.01,0],[0,0,0.01]] 
  },
  // Link 6 (Wrist Yaw + TCP) - Rotates around X
  link6: { 
    m: 1.0, 
    com: [0.0, 0.0, 0.04], 
    I: [[0.005,0,0],[0,0.005,0],[0,0,0.005]] 
  }
};

// Gravity vector [m/s²]
const GRAVITY = [0, 0, -9.81];
const G = 9.81;

// =============================================================================
// MATH UTILITIES
// =============================================================================

function mat3Transpose(M) {
  return [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]]
  ];
}

function mat3Multiply(A, B) {
  const result = [[0,0,0], [0,0,0], [0,0,0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
}

function mat3MultiplyVec(M, v) {
  return [
    M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
    M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
    M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]
  ];
}

function vecCross(a, b) {
  return [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0]
  ];
}

function vecAdd(a, b) {
  return [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
}

function vecSub(a, b) {
  return [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
}

function vecScale(v, s) {
  return [v[0]*s, v[1]*s, v[2]*s];
}

function vecDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function add3x3(A, B) {
  return [
    [A[0][0] + B[0][0], A[0][1] + B[0][1], A[0][2] + B[0][2]],
    [A[1][0] + B[1][0], A[1][1] + B[1][1], A[1][2] + B[1][2]],
    [A[2][0] + B[2][0], A[2][1] + B[2][1], A[2][2] + B[2][2]]
  ];
}

function pointMassInertiaAboutCom(m, d) {
  const dd = vecDot(d, d);
  return [
    [m * (dd - d[0] * d[0]), -m * d[0] * d[1], -m * d[0] * d[2]],
    [-m * d[1] * d[0], m * (dd - d[1] * d[1]), -m * d[1] * d[2]],
    [-m * d[2] * d[0], -m * d[2] * d[1], m * (dd - d[2] * d[2])]
  ];
}

function combineTwoBodies(m1, c1, I1, m2, c2, I2) {
  const m = m1 + m2;
  if (m <= 0) {
    return { m: 0, c: [0, 0, 0], I: [[0, 0, 0], [0, 0, 0], [0, 0, 0]] };
  }
  const c = [
    (m1 * c1[0] + m2 * c2[0]) / m,
    (m1 * c1[1] + m2 * c2[1]) / m,
    (m1 * c1[2] + m2 * c2[2]) / m
  ];
  const d1 = vecSub(c1, c);
  const d2 = vecSub(c2, c);
  const I = add3x3(
    add3x3(I1, pointMassInertiaAboutCom(m1, d1)),
    add3x3(I2, pointMassInertiaAboutCom(m2, d2))
  );
  return { m, c, I };
}

function scale3x3(I, s) {
  return [
    [I[0][0] * s, I[0][1] * s, I[0][2] * s],
    [I[1][0] * s, I[1][1] * s, I[1][2] * s],
    [I[2][0] * s, I[2][1] * s, I[2][2] * s]
  ];
}

function mapActuatedToUserDynamics(cfg = {}) {
  const dyn = cfg.dynamic_case || {};

  const toNumArray = (src, fallback, len) => {
    const arr = Array.isArray(src) ? src : Array.isArray(fallback) ? fallback : [];
    const out = new Array(len).fill(0);
    for (let i = 0; i < len; i++) {
      out[i] = Number(arr[i] || 0);
    }
    return out;
  };

  const axis4ToSerial5 = (axis4) => [axis4[0], axis4[1], axis4[2], 0.0, axis4[3]];
  const fromAnyToSerial5 = (src, fallback, fallbackAxis4) => {
    const arr = Array.isArray(src) ? src : Array.isArray(fallback) ? fallback : null;
    if (arr && arr.length >= 5) {
      return [Number(arr[0] || 0), Number(arr[1] || 0), Number(arr[2] || 0), Number(arr[3] || 0), Number(arr[4] || 0)];
    }
    const axis4 = toNumArray(arr || fallbackAxis4, fallbackAxis4, 4);
    return axis4ToSerial5(axis4);
  };

  const reflectedAxis4 = toNumArray(
    dyn.reflected_inertia || cfg.reflected_inertia || cfg.rotor_inertia_reflected,
    CR4_DYNAMIC_DEFAULTS.reflectedInertia,
    4
  );
  const viscousAxis4 = toNumArray(
    dyn.viscous_friction || cfg.friction_coeffs,
    CR4_DYNAMIC_DEFAULTS.viscousFriction,
    4
  );
  const coulombAxis4 = toNumArray(
    dyn.coulomb_friction || cfg.coulomb_friction,
    CR4_DYNAMIC_DEFAULTS.coulombFriction,
    4
  );

  const reflectedSerial5 = fromAnyToSerial5(
    dyn.rotor_inertia_reflected_q || dyn.reflected_inertia_q || cfg.rotor_inertia_reflected_q,
    cfg.rotor_inertia_reflected,
    reflectedAxis4
  );
  const viscousSerial5 = fromAnyToSerial5(
    dyn.friction_viscous_q || cfg.friction_viscous_q,
    cfg.friction_coeffs,
    viscousAxis4
  );
  const coulombSerial5 = fromAnyToSerial5(
    dyn.friction_coulomb_q || cfg.friction_coulomb_q,
    cfg.coulomb_friction,
    coulombAxis4
  );

  const motorMassSerial5 = fromAnyToSerial5(
    dyn.motor_mass_q || cfg.motor_mass_q,
    cfg.motor_masses,
    toNumArray(dyn.motor_masses_axis4 || cfg.motor_masses_axis4, [0, 0, 0, 0], 4)
  );

  const payloadMass = Number((dyn.payload_kg ?? cfg.payload_kg ?? CR4_DYNAMIC_DEFAULTS.payload.mass) || 0);
  const pI = cfg.payload_inertia || dyn.payload_inertia || {};
  const payloadInertiaDiag = [Number(pI.Ixx || 0), Number(pI.Iyy || 0), Number(pI.Izz || 0)];
  const payloadComFromTcp = Array.isArray(pI.com_from_tcp) ? pI.com_from_tcp.map(v => Number(v || 0)).slice(0, 3) : [0, 0, 0];
  const irefMode = String(dyn.iref_model_mode || cfg.iref_model_mode || 'diag').toLowerCase();

  return {
    reflectedSerial5,
    viscousSerial5,
    coulombSerial5,
    motorMassSerial5,
    payloadMass,
    payloadInertiaDiag,
    payloadComFromTcp,
    irefMode,
  };
}

function buildCR4SerialParams(scale = 1.0, cfg = {}) {
  const s = Number(scale) > 0 ? Number(scale) : 1.0;
  const massExpRaw = cfg.structural_mass_scale_exp ?? cfg.mass_scale_exponent ?? 3.0;
  const massExp = Number.isFinite(Number(massExpRaw)) ? Number(massExpRaw) : 3.0;
  const inertiaExpRaw = cfg.structural_inertia_scale_exp ?? (massExp + 2.0);
  const inertiaExp = Number.isFinite(Number(inertiaExpRaw)) ? Number(inertiaExpRaw) : (massExp + 2.0);
  const massScale = s ** massExp;
  const inertiaScale = s ** inertiaExp;

  const params = {
    dhA: CR4_SERIAL_BASE.dhA.map((v) => v * s),
    dhAlpha: CR4_SERIAL_BASE.dhAlpha.slice(),
    dhD: CR4_SERIAL_BASE.dhD.map((v) => v * s),
    masses: CR4_SERIAL_BASE.masses.map((m) => m * massScale),
    coms: CR4_SERIAL_BASE.coms.map((c) => c.map((x) => x * s)),
    inertias: CR4_SERIAL_BASE.inertias.map((I) => scale3x3(I, inertiaScale))
  };

  const dyn = mapActuatedToUserDynamics(cfg);

  // Motor stator masses mapped as in validated scripts:
  // q1 on static base (ignored), q2->Link1, q3->Link2, q4(passive)->Link3, q5->Link4.
  const statorLinkMasses = [dyn.motorMassSerial5[1], dyn.motorMassSerial5[2], dyn.motorMassSerial5[3], dyn.motorMassSerial5[4], 0.0];
  for (let i = 0; i < 5; i++) {
    if (statorLinkMasses[i] > 0) {
      const combined = combineTwoBodies(
        params.masses[i],
        params.coms[i],
        params.inertias[i],
        statorLinkMasses[i],
        [0, 0, 0],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
      );
      params.masses[i] = combined.m;
      params.coms[i] = combined.c;
      params.inertias[i] = combined.I;
    }
  }

  const irefDiag = dyn.reflectedSerial5.slice();
  if (dyn.irefMode === 'q5_physical' || dyn.irefMode === 'q3_q5_physical') {
    params.inertias[4][2][2] += irefDiag[4];
    irefDiag[4] = 0.0;
  }
  if (dyn.irefMode === 'q3_q5_physical') {
    params.inertias[2][2][2] += irefDiag[2];
    irefDiag[2] = 0.0;
  }

  if (dyn.payloadMass > 0) {
    const payloadInertia = [
      [dyn.payloadInertiaDiag[0], 0, 0],
      [0, dyn.payloadInertiaDiag[1], 0],
      [0, 0, dyn.payloadInertiaDiag[2]]
    ];
    const combined = combineTwoBodies(
      params.masses[4],
      params.coms[4],
      params.inertias[4],
      dyn.payloadMass,
      dyn.payloadComFromTcp,
      payloadInertia
    );
    params.masses[4] = combined.m;
    params.coms[4] = combined.c;
    params.inertias[4] = combined.I;
  }

  return { params, dyn, irefDiag };
}

function newtonEulerSerial5(q5, qd5, qdd5, params) {
  const n = 5;
  const p = [];
  const R = [];
  for (let i = 0; i < n; i++) {
    const ct = Math.cos(q5[i]);
    const st = Math.sin(q5[i]);
    const ca = Math.cos(params.dhAlpha[i]);
    const sa = Math.sin(params.dhAlpha[i]);
    R.push([
      [ct, -ca * st, sa * st],
      [st, ca * ct, -sa * ct],
      [0, sa, ca]
    ]);
    p.push([
      params.dhA[i],
      params.dhD[i] * Math.sin(params.dhAlpha[i]),
      params.dhD[i] * Math.cos(params.dhAlpha[i])
    ]);
  }

  const z0 = [0, 0, 1];
  const w = [[0, 0, 0]];
  const dw = [[0, 0, 0]];
  const dv = [[0, 0, G]];

  for (let i = 0; i < n; i++) {
    const w_i = mat3MultiplyVec(mat3Transpose(R[i]), vecAdd(w[i], vecScale(z0, qd5[i])));
    const dw_i = mat3MultiplyVec(
      mat3Transpose(R[i]),
      vecAdd(vecAdd(dw[i], vecScale(z0, qdd5[i])), vecCross(w[i], vecScale(z0, qd5[i])))
    );
    const dv_i = vecAdd(
      vecAdd(vecCross(dw_i, p[i]), vecCross(w_i, vecCross(w_i, p[i]))),
      mat3MultiplyVec(mat3Transpose(R[i]), dv[i])
    );
    w.push(w_i);
    dw.push(dw_i);
    dv.push(dv_i);
  }

  const aCom = [];
  for (let i = 0; i < n; i++) {
    aCom.push(
      vecAdd(
        vecAdd(vecCross(dw[i + 1], params.coms[i]), vecCross(w[i + 1], vecCross(w[i + 1], params.coms[i]))),
        dv[i + 1]
      )
    );
  }

  const f = new Array(n + 1).fill(null).map(() => [0, 0, 0]);
  const nMoment = new Array(n + 1).fill(null).map(() => [0, 0, 0]);

  for (let i = n - 1; i >= 0; i--) {
    if (i < n - 1) {
      f[i] = vecAdd(mat3MultiplyVec(R[i + 1], f[i + 1]), vecScale(aCom[i], params.masses[i]));
    } else {
      f[i] = vecScale(aCom[i], params.masses[i]);
    }

    const N_i = vecAdd(
      mat3MultiplyVec(params.inertias[i], dw[i + 1]),
      vecCross(w[i + 1], mat3MultiplyVec(params.inertias[i], w[i + 1]))
    );

    if (i < n - 1) {
      const rec = vecCross(mat3MultiplyVec(mat3Transpose(R[i + 1]), p[i]), f[i + 1]);
      nMoment[i] = vecAdd(
        vecAdd(
          mat3MultiplyVec(R[i + 1], vecAdd(nMoment[i + 1], rec)),
          vecCross(vecAdd(p[i], params.coms[i]), vecScale(aCom[i], params.masses[i]))
        ),
        N_i
      );
    } else {
      nMoment[i] = vecAdd(vecCross(vecAdd(p[i], params.coms[i]), vecScale(aCom[i], params.masses[i])), N_i);
    }
  }

  const tau = [0, 0, 0, 0, 0];
  for (let i = 0; i < n; i++) {
    tau[i] = vecDot(nMoment[i], mat3MultiplyVec(mat3Transpose(R[i]), z0));
  }
  return tau;
}

// =============================================================================
// DH TRANSFORMATION
// =============================================================================

/**
 * Compute rotation matrix from DH parameters (standard convention)
 * T = RotZ(theta) * TransZ(d) * TransX(a) * RotX(alpha)
 */
function dhTransform(a, alpha, d, theta) {
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const ca = Math.cos(alpha);
  const sa = Math.sin(alpha);
  
  return {
    R: [
      [ct, -ca*st, sa*st],
      [st, ca*ct, -sa*ct],
      [0, sa, ca]
    ],
    p: [a*ct, a*st, d]
  };
}

// =============================================================================
// RECURSIVE NEWTON-EULER ALGORITHM FOR CR4
// =============================================================================

/**
 * Compute inverse dynamics for CR4 using recursive Newton-Euler
 * 
 * @param {Array} q - Joint positions [q1, q2, q3, q4] in radians
 * @param {Array} qd - Joint velocities [qd1, qd2, qd3, qd4] in rad/s
 * @param {Array} qdd - Joint accelerations [qdd1, qdd2, qdd3, qdd4] in rad/s²
 * @param {Object} robotData - Robot data from demo_robot_data.js
 * @returns {Object} {tau: Array, forces: Array, torques: Array}
 */
export function computeInverseDynamicsCR4(q, qd, qdd, robotData = null, config = null) {
  if (!robotData) {
    robotData = getDemoRobotData('CR4');
  }

  const scale = Number((config && config.scale) || robotData.scale || 1.0);
  const { params, dyn, irefDiag } = buildCR4SerialParams(scale, config || {});

  const q5 = userToDH('CR4', q);
  const offsets = Array.isArray(robotData.offsets) ? robotData.offsets : [0, 0, 0, 0, 0];
  for (let i = 0; i < 5; i++) {
    q5[i] = (q5[i] || 0) + Number(offsets[i] || 0);
  }
  const qd5 = userToDH('CR4', qd);
  const qdd5 = userToDH('CR4', qdd);

  const tau5 = newtonEulerSerial5(q5, qd5, qdd5, params);
  const tau5Total = tau5.slice();
  for (let i = 0; i < 5; i++) {
    const sign = Math.abs(qd5[i]) > 1e-9 ? Math.sign(qd5[i]) : 0;
    tau5Total[i] += irefDiag[i] * qdd5[i];
    tau5Total[i] += dyn.viscousSerial5[i] * qd5[i];
    tau5Total[i] += dyn.coulombSerial5[i] * sign;
  }

  const tauAct = [
    tau5Total[0],
    tau5Total[1] - tau5Total[2],
    tau5Total[2],
    tau5Total[4]
  ];

  return {
    tau: tauAct,
    tau_serial5: tau5Total,
    q_dh: q5
  };
}

// =============================================================================
// TRAJECTORY DYNAMICS
// =============================================================================

/**
 * Compute dynamics along a trajectory
 * 
 * @param {Array} trajectory - Array of {q, qd, qdd} objects
 * @returns {Array} Array of torque arrays
 */
export function computeTrajectoryDynamics(trajectory) {
  return trajectory.map(point => {
    const result = computeInverseDynamicsCR4(point.q, point.qd, point.qdd, null, point.config || null);
    return {
      tau: result.tau,
      time: point.time || 0
    };
  });
}

// =============================================================================
// RECURSIVE NEWTON-EULER ALGORITHM FOR CR6
// =============================================================================

/**
 * Compute inverse dynamics for CR6 using recursive Newton-Euler
 * 
 * @param {Array} q - Joint positions [q1, q2, q3, q4, q5, q6] in radians
 * @param {Array} qd - Joint velocities [qd1..qd6] in rad/s
 * @param {Array} qdd - Joint accelerations [qdd1..qdd6] in rad/s²
 * @param {Object} robotData - Robot data from demo_robot_data.js
 * @returns {Object} {tau: Array, forces: Array, torques: Array}
 */
export function computeInverseDynamicsCR6(q, qd, qdd, robotData = null) {
  if (!robotData) {
    robotData = getDemoRobotData('CR6');
  }
  
  // Get DH parameters
  const dh = robotData.dh;
  
  // Link inertial parameters
  const links = [
    CR6_INERTIA.link1, CR6_INERTIA.link2, CR6_INERTIA.link3,
    CR6_INERTIA.link4, CR6_INERTIA.link5, CR6_INERTIA.link6
  ];
  
  // Initial conditions (base is stationary)
  const w0 = [0, 0, 0];
  const dw0 = [0, 0, 0];
  const dv0 = [0, 0, 0];
  const z0 = [0, 0, 1];
  
  // External forces at TCP
  const f_ext = [0, 0, 0];
  const n_ext = [0, 0, 0];
  
  // Arrays to store computed values
  const w = [];
  const dw = [];
  const dv = [];
  const a = [];
  const R = [];
  const p = [];
  
  // ==========================================================================
  // FORWARD RECURSION
  // ==========================================================================
  
  for (let i = 0; i < 6; i++) {
    const dhp = dh[i];
    const T = dhTransform(dhp.a, dhp.alpha, dhp.d, q[i] + dhp.theta);
    R[i] = T.R;
    p[i] = T.p;
    
    const w_prev = i === 0 ? w0 : w[i-1];
    const qd_i = qd[i];
    
    if (i === 0) {
      w[i] = vecAdd(w_prev, vecScale(z0, qd_i));
    } else {
      w[i] = mat3MultiplyVec(mat3Transpose(R[i]), vecAdd(w_prev, vecScale(z0, qd_i)));
    }
    
    const dw_prev = i === 0 ? dw0 : dw[i-1];
    const qdd_i = qdd[i];
    const term = vecAdd(vecAdd(dw_prev, vecScale(z0, qdd_i)), vecCross(w_prev, vecScale(z0, qd_i)));
    
    if (i === 0) {
      dw[i] = term;
    } else {
      dw[i] = mat3MultiplyVec(mat3Transpose(R[i]), term);
    }
    
    const dv_prev = i === 0 ? dv0 : dv[i-1];
    const term1 = vecCross(dw[i], p[i]);
    const term2 = vecCross(w[i], vecCross(w[i], p[i]));
    
    if (i === 0) {
      dv[i] = vecAdd(vecAdd(dv_prev, term1), term2);
    } else {
      const dv_trans = mat3MultiplyVec(mat3Transpose(R[i]), dv_prev);
      dv[i] = vecAdd(vecAdd(dv_trans, term1), term2);
    }
    
    const s = links[i].com;
    a[i] = vecAdd(vecAdd(dv[i], vecCross(dw[i], s)), vecCross(w[i], vecCross(w[i], s)));
  }
  
  // ==========================================================================
  // BACKWARD RECURSION
  // ==========================================================================
  
  const f = [];
  const n = [];
  const tau = [0, 0, 0, 0, 0, 0];
  const g = [];
  
  let g_curr = GRAVITY;
  for (let i = 5; i >= 0; i--) {
    g[i] = mat3MultiplyVec(mat3Transpose(R[i]), g_curr);
    g_curr = g[i];
  }
  
  // Link 6 (last)
  f[5] = vecAdd(vecSub(f_ext, vecScale(g[5], links[5].m)), vecScale(a[5], links[5].m));
  
  const s6 = links[5].com;
  const p6_plus_s6 = vecAdd(p[5], s6);
  const I6_dw6 = mat3MultiplyVec(links[5].I, dw[5]);
  const I6_w6 = mat3MultiplyVec(links[5].I, w[5]);
  
  n[5] = vecAdd(vecAdd(mat3MultiplyVec(mat3Transpose(R[5]), n_ext),
                       vecCross(p6_plus_s6, vecScale(a[5], links[5].m))),
                vecAdd(I6_dw6, vecCross(w[5], I6_w6)));
  n[5] = vecSub(n[5], vecCross(p6_plus_s6, vecScale(g[5], links[5].m)));
  
  tau[5] = n[5][0];  // X axis for J6
  
  // Backward recursion for links 5 to 1
  for (let i = 4; i >= 0; i--) {
    const s = links[i].com;
    const p_plus_s = vecAdd(p[i], s);
    
    f[i] = vecAdd(mat3MultiplyVec(R[i+1], f[i+1]), 
                  vecSub(vecScale(a[i], links[i].m), vecScale(g[i], links[i].m)));
    
    const I_dw = mat3MultiplyVec(links[i].I, dw[i]);
    const I_w = mat3MultiplyVec(links[i].I, w[i]);
    
    n[i] = vecAdd(vecAdd(mat3MultiplyVec(R[i+1], vecAdd(n[i+1], vecCross(vecScale(p[i+1], -1), f[i+1]))),
                         vecCross(p_plus_s, vecScale(a[i], links[i].m))),
                  vecAdd(I_dw, vecCross(w[i], I_w)));
    n[i] = vecSub(n[i], vecCross(p_plus_s, vecScale(g[i], links[i].m)));
    
    // Select axis based on joint type
    if (i === 0) tau[i] = n[i][2];      // J1: Z axis
    else if (i === 1) tau[i] = n[i][1]; // J2: Y axis
    else if (i === 2) tau[i] = n[i][1]; // J3: Y axis
    else if (i === 3) tau[i] = n[i][0]; // J4: X axis
    else if (i === 4) tau[i] = n[i][1]; // J5: Y axis
  }
  
  return {
    tau: tau,
    forces: f,
    torques: n,
    w: w,
    dw: dw,
    a: a
  };
}

// =============================================================================
// GENERIC DYNAMICS FUNCTION
// =============================================================================

/**
 * Compute inverse dynamics for any supported robot
 * @param {string} robotType - 'CR4' or 'CR6'
 * @param {Array} q - Joint positions
 * @param {Array} qd - Joint velocities
 * @param {Array} qdd - Joint accelerations
 * @returns {Object} Dynamics result
 */
export function computeInverseDynamics(robotType, q, qd, qdd, config = null) {
  if (robotType === 'CR6') {
    return computeInverseDynamicsCR6(q, qd, qdd);
  }
  return computeInverseDynamicsCR4(q, qd, qdd, null, config);
}

// =============================================================================
// GET DYNAMICS INFO
// =============================================================================

export function getDynamicsInfo() {
  return {
    algorithm: 'Recursive Newton-Euler',
    model: 'Serial-5 CR4 + closed-loop torque mapping',
    accuracy: 'Aligned with validated CR4 script assumptions',
    supportedRobots: ['CR4', 'CR6'],
    gravity: GRAVITY,
    units: {
      torque: 'Nm',
      force: 'N',
      mass: 'kg',
      inertia: 'kg·m²'
    }
  };
}

export default {
  computeInverseDynamics,
  computeInverseDynamicsCR4,
  computeInverseDynamicsCR6,
  computeTrajectoryDynamics,
  getDynamicsInfo,
  CR4_SERIAL_BASE,
  CR4_DYNAMIC_DEFAULTS,
  CR6_INERTIA
};
