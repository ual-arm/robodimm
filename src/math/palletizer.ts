import {
  Matrix4,
  Vector3,
  Matrix3,
  createIdentity4,
  rotz,
  roty,
  getTranslation,
  getMatrix3,
  crossProduct,
  unitVector,
  multiply4x4,
  multiply3x3AndVector,
  multiply3x3,
  transpose3x3,
  subtractVectors,
  addVectors,
  scaleVector,
  norm3,
  wrapToPi,
  translation
} from './matrix';
import { RobotSpec, Cr4Geometry, InertialSpec, VisualSpec } from '../model/schemas';
import { getDefaultPalletizerBodyCOM } from './palletizerGeometry';

export interface PalletizerFK {
  q: number[];
  points: Record<string, Vector3>;
  transforms: Record<string, Matrix4>;
  link_segments: [string, string, string][];
}

export interface PalletizerIKResult {
  success: boolean;
  q: number[];
  position_error: number;
  orientation_error: number;
  iterations: number;
  message?: string;
}

export class PalletizerEngine {
  spec: RobotSpec;

  constructor(spec: RobotSpec) {
    if (spec.kind !== 'CR4') {
      throw new Error(`PalletizerEngine requires CR4 robot, got ${spec.kind}`);
    }
    this.spec = spec;
  }

  get geometry(): Cr4Geometry {
    return this.spec.geometry as Cr4Geometry;
  }

  get limits(): Record<string, [number, number]> {
    const limits: Record<string, [number, number]> = {};
    for (const lim of this.spec.limits) {
      limits[lim.name] = [lim.lowerLimitRad, lim.upperLimitRad];
    }
    return limits;
  }

  get joint_names(): string[] {
    return ['J1', 'J2', 'J3', 'J4'];
  }

  get gravity(): Vector3 {
    return [0.0, 0.0, -9.80665];
  }

  clampConfiguration(q: number[]): number[] {
    const qClamped = [...q];
    const names = this.joint_names;
    const limits = this.limits;
    for (let i = 0; i < 4; i++) {
      const name = names[i];
      if (limits[name]) {
        const [lower, upper] = limits[name];
        qClamped[i] = Math.max(lower, Math.min(upper, qClamped[i]));
      }
    }
    return qClamped;
  }

  planarPoints(j2: number, j3: number): Record<string, Vector3> {
    const home = this.geometry;
    const points: Record<string, Vector3> = {};
    
    // Copy all home hardpoints
    for (const key of Object.keys(home)) {
      points[key] = [...(home as any)[key]] as Vector3;
    }

    const o = points['O'];
    
    // Ry(j2) and Ry(j3) rotations
    const Ry_j2 = getMatrix3(roty(j2));
    const Ry_j3 = getMatrix3(roty(j3));

    // points["C"] = o + Ry(j2) @ (home["C"] - home["O"])
    const OC_home = subtractVectors(points['C'], o);
    points['C'] = addVectors(o, multiply3x3AndVector(Ry_j2, OC_home));

    // points["B"] = o + Ry(j3) @ (home["B"] - home["O"])
    const OB_home = subtractVectors(points['B'], o);
    points['B'] = addVectors(o, multiply3x3AndVector(Ry_j3, OB_home));

    // points["P"] = points["B"] + (points["C"] - points["O"])
    points['P'] = addVectors(points['B'], subtractVectors(points['C'], o));

    // points["E"] = points["D"] + (points["C"] - points["O"])
    points['E'] = addVectors(points['D'], subtractVectors(points['C'], o));

    // points["H"] = points["P"] + unit(points["C"] - points["P"]) * dist(home, "P", "H")
    const CP_diff = subtractVectors(points['C'], points['P']);
    const dist_PH = norm3(subtractVectors(home.H, home.P));
    points['H'] = addVectors(points['P'], scaleVector(unitVector(CP_diff), dist_PH));

    // points["F"] = points["C"] + frame_rotation(points["C"], points["E"]) @ point_local(home, "F", "C", "E")
    const rot_CE = this._frameRotation(points['C'], points['E']);
    const local_F_CE = this._pointLocal(home, 'F', 'C', 'E');
    points['F'] = addVectors(points['C'], multiply3x3AndVector(rot_CE, local_F_CE));

    // points["G"] = circle_intersection_xz(...)
    const dist_FG = norm3(subtractVectors(home.G, home.F));
    const dist_HG = norm3(subtractVectors(home.G, home.H));
    points['G'] = this._circleIntersectionXZ(
      points['F'],
      dist_FG,
      points['H'],
      dist_HG,
      home.G,
      this._orientedAreaXZ(home.F, home.H, home.G)
    );

    // hgee_rotation = frame_rotation(points["H"], points["G"])
    const rot_HG = this._frameRotation(points['H'], points['G']);
    const j4_local = this._pointLocal(home, 'J4', 'H', 'G');
    const ee_local = this._pointLocal(home, 'EE', 'H', 'G');
    const tcp_local = this._pointLocal(home, 'TCP', 'H', 'G');

    // points["J4"] = points["H"] + hgee_rotation @ j4_local
    points['J4'] = addVectors(points['H'], multiply3x3AndVector(rot_HG, j4_local));

    // points["EE"] = points["H"] + hgee_rotation @ ee_local
    points['EE'] = addVectors(points['H'], multiply3x3AndVector(rot_HG, ee_local));

    // points["TCP"] = points["H"] + hgee_rotation @ tcp_local
    points['TCP'] = addVectors(points['H'], multiply3x3AndVector(rot_HG, tcp_local));

    return points;
  }

  forwardKinematics(q: number[]): PalletizerFK {
    const qClamped = this.clampConfiguration(q);
    const j1 = qClamped[0];
    const j2 = qClamped[1];
    const j3 = qClamped[2];
    const j4 = qClamped[3];

    const planarPoints = this.planarPoints(j2, j3);
    const points: Record<string, Vector3> = {};
    for (const key of Object.keys(planarPoints)) {
      points[key] = this._yawApply(planarPoints[key], j1);
    }

    const transforms: Record<string, Matrix4> = {
      base_link: createIdentity4(),
      swing_link: this._yawTransform(j1),
    };

    // Frame per points
    for (const key of Object.keys(points)) {
      transforms[`${key}_frame`] = this._translationMatrix(points[key]);
    }

    // Body segments
    const segments: [string, string, string][] = [
      ['seg_A_O', 'A', 'O'],
      ['seg_O_B', 'O', 'B'],
      ['seg_O_C', 'O', 'C'],
      ['seg_B_P', 'B', 'P'],
      ['seg_P_C', 'P', 'C'],
      ['seg_C_H', 'C', 'H'],
      ['seg_P_H', 'P', 'H'],
      ['seg_D_E', 'D', 'E'],
      ['seg_C_E', 'C', 'E'],
      ['seg_C_F', 'C', 'F'],
      ['seg_E_F', 'E', 'F'],
      ['seg_F_G', 'F', 'G'],
      ['seg_H_G', 'H', 'G'],
      ['seg_H_EE', 'H', 'EE'],
      ['seg_G_EE', 'G', 'EE'],
      ['seg_EE_TCP', 'EE', 'TCP'],
    ];

    for (const [name, first, second] of segments) {
      transforms[name] = this._segmentTransform(points[first], points[second], j1);
    }

    // Orientation matrix (yaw of tool disk: j1 + j4)
    const j4_transform = this._translationMatrix(points['J4']);
    const tcpRot = this._tcpRotation(j1, j4);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        j4_transform[r][c] = tcpRot[r][c];
      }
    }

    const ee_transform = multiply4x4(j4_transform, createIdentity4());
    ee_transform[0][3] = points['EE'][0];
    ee_transform[1][3] = points['EE'][1];
    ee_transform[2][3] = points['EE'][2];

    const tcp_transform = multiply4x4(j4_transform, createIdentity4());
    tcp_transform[0][3] = points['TCP'][0];
    tcp_transform[1][3] = points['TCP'][1];
    tcp_transform[2][3] = points['TCP'][2];

    transforms['J4_link'] = j4_transform;
    transforms['EE_frame'] = ee_transform;
    transforms['TCP_frame'] = tcp_transform;
    transforms['payload_link'] = tcp_transform;

    // Alias frame mappings for UI Inspector
    transforms['FOOT'] = transforms['base_link'];
    transforms['DISK'] = transforms['EE_frame'];
    transforms['SWING'] = transforms['swing_link'];
    transforms['UPPER_ARM'] = transforms['seg_C_H']; // CAD origin is at point C (J3 axis)
    
    // TILT CAD origin is at point G (parallel linkage tip)
    const T_tilt = this._segmentTransform(points['H'], points['G'], j1);
    T_tilt[0][3] = points['G'][0];
    T_tilt[1][3] = points['G'][1];
    T_tilt[2][3] = points['G'][2];
    transforms['TILT'] = T_tilt;

    transforms['UPPER_LINK'] = transforms['seg_F_G'];
    transforms['P_LINK'] = transforms['seg_B_P'];
    transforms['P_ARM'] = transforms['seg_O_B'];
    transforms['LOWER_LINK'] = transforms['seg_D_E'];
    transforms['LOWER_ARM'] = transforms['seg_O_C'];
    transforms['LINK_PLATE'] = transforms['seg_C_E'];
    transforms['PAYLOAD'] = transforms['payload_link'];

    return {
      q: qClamped,
      points,
      transforms,
      link_segments: segments
    };
  }

  solveIK(
    targetXYZ: Vector3,
    seedQ?: number[],
    targetYaw?: number,
    pointLabel = 'TCP',
    tolerance = 1e-6,
    maxIterations = 80
  ): PalletizerIKResult {
    const q = seedQ ? [...seedQ] : Array(4).fill(0);
    const radial = Math.hypot(targetXYZ[0], targetXYZ[1]);
    if (radial > 1e-12) {
      q[0] = Math.atan2(targetXYZ[1], targetXYZ[0]);
    }
    let qClamped = this.clampConfiguration(q);

    const targetPlanar: [number, number] = [radial, targetXYZ[2]];
    let iterations = 0;
    
    const step = 1e-6;

    for (iterations = 1; iterations <= maxIterations; iterations++) {
      const current = this._planarPoint(qClamped, pointLabel);
      const error: [number, number] = [targetPlanar[0] - current[0], targetPlanar[1] - current[1]];
      
      if (Math.hypot(error[0], error[1]) <= tolerance) {
        break;
      }

      // Compute Jacobian numerically
      const jacobian = Array.from({ length: 2 }, () => Array(2).fill(0));
      const qIndices = [1, 2];
      for (let localIdx = 0; localIdx < 2; localIdx++) {
        const qIdx = qIndices[localIdx];
        const plus = [...qClamped];
        const minus = [...qClamped];
        plus[qIdx] += step;
        minus[qIdx] -= step;

        const plusClamped = this.clampConfiguration(plus);
        const minusClamped = this.clampConfiguration(minus);
        const denom = plusClamped[qIdx] - minusClamped[qIdx];
        if (Math.abs(denom) > 1e-12) {
          const pPlus = this._planarPoint(plusClamped, pointLabel);
          const pMinus = this._planarPoint(minusClamped, pointLabel);
          jacobian[0][localIdx] = (pPlus[0] - pMinus[0]) / denom;
          jacobian[1][localIdx] = (pPlus[1] - pMinus[1]) / denom;
        }
      }

      // Solve J.T @ J + 1e-8 I
      const JtJ = [
        [
          jacobian[0][0] * jacobian[0][0] + jacobian[1][0] * jacobian[1][0] + 1e-8,
          jacobian[0][0] * jacobian[0][1] + jacobian[1][0] * jacobian[1][1]
        ],
        [
          jacobian[0][1] * jacobian[0][0] + jacobian[1][1] * jacobian[1][0],
          jacobian[0][1] * jacobian[0][1] + jacobian[1][1] * jacobian[1][1] + 1e-8
        ]
      ];
      const Jte = [
        jacobian[0][0] * error[0] + jacobian[1][0] * error[1],
        jacobian[0][1] * error[0] + jacobian[1][1] * error[1]
      ];

      // Solve 2x2 linear system
      const det = JtJ[0][0] * JtJ[1][1] - JtJ[0][1] * JtJ[1][0];
      if (Math.abs(det) < 1e-15) {
        break;
      }
      const dq1 = (Jte[0] * JtJ[1][1] - Jte[1] * JtJ[0][1]) / det;
      const dq2 = (JtJ[0][0] * Jte[1] - JtJ[1][0] * Jte[0]) / det;

      qClamped[1] += Math.max(-0.2, Math.min(0.2, dq1));
      qClamped[2] += Math.max(-0.2, Math.min(0.2, dq2));
      qClamped = this.clampConfiguration(qClamped);
    }

    if (targetYaw !== undefined) {
      qClamped[3] = wrapToPi(targetYaw - qClamped[0] - Math.PI);
      qClamped = this.clampConfiguration(qClamped);
    }

    const fk = this.forwardKinematics(qClamped);
    const actual = fk.points[pointLabel];
    const posError = norm3(subtractVectors(actual, targetXYZ));
    
    let orientError = 0.0;
    if (targetYaw !== undefined) {
      orientError = Math.abs(wrapToPi(qClamped[0] + qClamped[3] + Math.PI - targetYaw));
    }

    let isSuccess = posError <= Math.max(tolerance * 10, 1e-5);
    if (targetYaw !== undefined) {
      isSuccess = isSuccess && orientError <= 1e-5;
    }

    return {
      success: isSuccess,
      q: qClamped,
      position_error: posError,
      orientation_error: orientError,
      iterations
    };
  }

  approximateDynamics(q: number[], qd: number[], qdd: number[]): number[] {
    const qClamped = this.clampConfiguration(q);
    const tau = Array(4).fill(0);

    const fk = this.forwardKinematics(qClamped);
    const gravity = this.gravity;

    // Define all body items for palletizer
    const bodyNames = [
      'SWING', 'P_ARM', 'LOWER_ARM', 'P_LINK',
      'UPPER_ARM', 'LOWER_LINK', 'LINK_PLATE',
      'UPPER_LINK', 'TILT', 'DISK'
    ];

    const bodyInertials: InertialSpec[] = [];
    const bodyTransforms: Matrix4[] = [];

    for (const body of bodyNames) {
      let inertial = this.spec.inertials[body];
      if (!inertial) {
        // default template visual shapes mass defaults
        inertial = {
          body,
          massKg: this._getDefaultBodyMass(body),
          comM: getDefaultPalletizerBodyCOM(this.geometry, body),
          inertiaKgM2: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
          frame: 'cad'
        };
      } else {
        inertial = {
          ...inertial,
          comM: inertial.comM ?? getDefaultPalletizerBodyCOM(this.geometry, body),
          inertiaKgM2: inertial.inertiaKgM2 ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        };
      }
      bodyInertials.push(inertial);
      bodyTransforms.push(fk.transforms[body]);
    }

    // Add Payload
    if (this.spec.payload && this.spec.payload.massKg > 0) {
      const p = this.spec.payload;
      bodyInertials.push({
        ...p,
        comM: p.comM ?? [0, 0, 0],
        inertiaKgM2: p.inertiaKgM2 ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
      });
      bodyTransforms.push(fk.transforms['PAYLOAD']);
    }

    const step = 1e-5;

    for (let bodyIdx = 0; bodyIdx < bodyInertials.length; bodyIdx++) {
      const inertial = bodyInertials[bodyIdx];
      if (inertial.massKg === 0) continue;

      const T = bodyTransforms[bodyIdx];
      const com_local = inertial.comM!;

      // Transform local COM to world space
      const com_world: Vector3 = [
        T[0][0] * com_local[0] + T[0][1] * com_local[1] + T[0][2] * com_local[2] + T[0][3],
        T[1][0] * com_local[0] + T[1][1] * com_local[1] + T[1][2] * com_local[2] + T[1][3],
        T[2][0] * com_local[0] + T[2][1] * com_local[1] + T[2][2] * com_local[2] + T[2][3]
      ];

      // Numerical Jacobian for COM
      const Jv = Array.from({ length: 3 }, () => Array(4).fill(0));
      for (let j = 0; j < 4; j++) {
        const plus = [...qClamped];
        const minus = [...qClamped];
        plus[j] += step;
        minus[j] -= step;

        const fkPlus = this.forwardKinematics(plus);
        const fkMinus = this.forwardKinematics(minus);

        const TPlus = fkPlus.transforms[inertial.body === 'PAYLOAD' ? 'PAYLOAD' : inertial.body];
        const TMinus = fkMinus.transforms[inertial.body === 'PAYLOAD' ? 'PAYLOAD' : inertial.body];

        const comPlus: Vector3 = [
          TPlus[0][0] * com_local[0] + TPlus[0][1] * com_local[1] + TPlus[0][2] * com_local[2] + TPlus[0][3],
          TPlus[1][0] * com_local[0] + TPlus[1][1] * com_local[1] + TPlus[1][2] * com_local[2] + TPlus[1][3],
          TPlus[2][0] * com_local[0] + TPlus[2][1] * com_local[1] + TPlus[2][2] * com_local[2] + TPlus[2][3]
        ];
        const comMinus: Vector3 = [
          TMinus[0][0] * com_local[0] + TMinus[0][1] * com_local[1] + TMinus[0][2] * com_local[2] + TMinus[0][3],
          TMinus[1][0] * com_local[0] + TMinus[1][1] * com_local[1] + TMinus[1][2] * com_local[2] + TMinus[1][3],
          TMinus[2][0] * com_local[0] + TMinus[2][1] * com_local[1] + TMinus[2][2] * com_local[2] + TMinus[2][3]
        ];

        Jv[0][j] = (comPlus[0] - comMinus[0]) / (2 * step);
        Jv[1][j] = (comPlus[1] - comMinus[1]) / (2 * step);
        Jv[2][j] = (comPlus[2] - comMinus[2]) / (2 * step);
      }

      // Gravity Force: - mass * gravity
      // Force due to gravity: Fg = mass * gravity
      const Fg = scaleVector(gravity, inertial.massKg);

      // Inertial force: mass * (Jv @ qdd)
      let com_accel: Vector3 = [0, 0, 0];
      for (let j = 0; j < 4; j++) {
        com_accel[0] += Jv[0][j] * qdd[j];
        com_accel[1] += Jv[1][j] * qdd[j];
        com_accel[2] += Jv[2][j] * qdd[j];
      }
      const F_inertia = scaleVector(com_accel, inertial.massKg);

      // Total force = F_inertia - Fg
      const F_total = subtractVectors(F_inertia, Fg);

      // Sum to torques: Jv.T @ F_total
      for (let j = 0; j < 4; j++) {
        tau[j] += Jv[0][j] * F_total[0] + Jv[1][j] * F_total[1] + Jv[2][j] * F_total[2];
      }
    }

    // Add joint viscous friction
    for (let i = 0; i < 4; i++) {
      const limit = this.spec.limits.find(l => l.name === this.joint_names[i]);
      const frictionCoeff = limit?.frictionCoeffNmSPerRad ?? 0.0;
      tau[i] += frictionCoeff * qd[i];
    }

    return tau;
  }

  _getDefaultBodyMass(body: string): number {
    // Estimated masses of MOVING bodies only (downstream of each joint).
    // The IRB460 total shipping weight (~925 kg) includes fixed base, foot,
    // covers, ballast, gearboxes and motors mounted near the base — none of
    // which load J2/J3 gravitationally. Only moving mass downstream of each
    // joint should be used for dynamics. Do NOT scale these to match the
    // datasheet total without a proper fixed-vs-moving mass breakdown.
    switch (body) {
      case 'SWING':      return  90.0;  // Column + J1 rotor; only loads J1
      case 'P_ARM':      return  35.0;  // Proximal arm link
      case 'LOWER_ARM':  return  75.0;  // Main lower arm beam
      case 'P_LINK':     return  25.0;  // Proximal parallel link
      case 'UPPER_ARM':  return  40.0;  // Upper arm / forearm
      case 'LOWER_LINK': return  20.0;  // Lower parallel link
      case 'LINK_PLATE': return  15.0;  // Link plate / coupler
      case 'UPPER_LINK': return  15.0;  // Upper parallel link
      case 'TILT':       return  15.0;  // Tilt / wrist body
      case 'DISK':       return  10.0;  // J4 disk / output flange
      default:           return   0.0;
    }
  }

  _planarPoint(q: number[], pointLabel: string): [number, number] {
    const pt = this.forwardKinematics(q).points[pointLabel];
    return [Math.hypot(pt[0], pt[1]), pt[2]];
  }

  _pointLocal(points: any, point: string, origin: string, x_target: string): Vector3 {
    const rot = transpose3x3(this._frameRotation(points[origin], points[x_target]));
    const diff = subtractVectors(points[point], points[origin]);
    return multiply3x3AndVector(rot, diff);
  }

  _frameRotation(first: Vector3, second: Vector3): Matrix3 {
    const x_axis = unitVector(subtractVectors(second, first));
    const y_axis: Vector3 = [0.0, 1.0, 0.0];
    const z_axis = unitVector(crossProduct(x_axis, y_axis));
    const y_axis_ortho = crossProduct(z_axis, x_axis);
    
    // Matrix with columns: x_axis, y_axis_ortho, z_axis
    return [
      [x_axis[0], y_axis_ortho[0], z_axis[0]],
      [x_axis[1], y_axis_ortho[1], z_axis[1]],
      [x_axis[2], y_axis_ortho[2], z_axis[2]]
    ];
  }

  _circleIntersectionXZ(
    center_a: Vector3,
    radius_a: number,
    center_b: Vector3,
    radius_b: number,
    prefer: Vector3,
    prefer_side: number
  ): Vector3 {
    const delta = subtractVectors(center_b, center_a);
    const dxz: [number, number] = [delta[0], delta[2]];
    const dist = Math.hypot(dxz[0], dxz[1]);
    if (dist <= 1e-12) {
      throw new Error('Circle centers are coincident in the XZ plane');
    }
    const a = (radius_a * radius_a - radius_b * radius_b + dist * dist) / (2.0 * dist);
    const h = Math.sqrt(Math.max(radius_a * radius_a - a * a, 0.0));
    
    const ex: [number, number] = [dxz[0] / dist, dxz[1] / dist];
    const base: [number, number] = [center_a[0] + a * ex[0], center_a[2] + a * ex[1]];
    const perp: [number, number] = [-ex[1], ex[0]];
    
    const candidates: [number, number][] = [
      [base[0] + h * perp[0], base[1] + h * perp[1]],
      [base[0] - h * perp[0], base[1] - h * perp[1]]
    ];

    let finalCandidates = [...candidates];
    if (Math.abs(prefer_side) > 1e-12) {
      const side_sign = Math.sign(prefer_side);
      const matching = candidates.filter(cand => {
        const area = dxz[0] * (cand[1] - center_a[2]) - dxz[1] * (cand[0] - center_a[0]);
        return Math.sign(area) === side_sign;
      });
      if (matching.length > 0) {
        finalCandidates = matching;
      }
    }

    const preferred: [number, number] = [prefer[0], prefer[2]];
    let best = finalCandidates[0];
    let bestDist = Math.hypot(best[0] - preferred[0], best[1] - preferred[1]);
    for (let i = 1; i < finalCandidates.length; i++) {
      const d = Math.hypot(finalCandidates[i][0] - preferred[0], finalCandidates[i][1] - preferred[1]);
      if (d < bestDist) {
        bestDist = d;
        best = finalCandidates[i];
      }
    }

    return [best[0], 0.0, best[1]];
  }

  _orientedAreaXZ(first: Vector3, second: Vector3, third: Vector3): number {
    return (second[0] - first[0]) * (third[2] - first[2]) - (second[2] - first[2]) * (third[0] - first[0]);
  }

  _yawApply(point: Vector3, yaw: number): Vector3 {
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    return [
      c * point[0] - s * point[1],
      s * point[0] + c * point[1],
      point[2]
    ];
  }

  _yawTransform(yaw: number): Matrix4 {
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    const T = createIdentity4();
    T[0][0] = c; T[0][1] = -s;
    T[1][0] = s; T[1][1] = c;
    return T;
  }

  _tcpRotation(base_yaw: number, tool_yaw: number): Matrix3 {
    const yaw = base_yaw + tool_yaw;
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    // Yaw matrix * diag([-1, 1, -1])
    return [
      [-c, -s, 0],
      [-s, c, 0],
      [0, 0, -1]
    ];
  }

  _segmentTransform(first: Vector3, second: Vector3, yaw: number): Matrix4 {
    const x_axis = unitVector(subtractVectors(second, first));
    const y_axis: Vector3 = [-Math.sin(yaw), Math.cos(yaw), 0.0];
    const z_axis = unitVector(crossProduct(x_axis, y_axis));
    const y_axis_ortho = crossProduct(z_axis, x_axis);

    const T = this._translationMatrix(first);
    T[0][0] = x_axis[0]; T[0][1] = y_axis_ortho[0]; T[0][2] = z_axis[0];
    T[1][0] = x_axis[1]; T[1][1] = y_axis_ortho[1]; T[1][2] = z_axis[1];
    T[2][0] = x_axis[2]; T[2][1] = y_axis_ortho[2]; T[2][2] = z_axis[2];
    return T;
  }

  _translationMatrix(point: Vector3): Matrix4 {
    return translation(point[0], point[1], point[2]);
  }
}

export function irb460PalletizerSpec(): RobotSpec {
  const geom: Cr4Geometry = {
    A: [0.0, 0.0, 0.0],
    O: [0.260, 0.0, 0.7425],
    B: [-0.140, 0.0, 0.7425],
    C: [0.260, 0.0, 1.6875],
    D: [-0.00488, 0.0, 0.88334],
    E: [-0.00488, 0.0, 1.82834],
    F: [0.48991, 0.0, 1.88034],
    G: [1.51481, 0.0, 1.88034],
    H: [1.285, 0.0, 1.6875],
    P: [-0.140, 0.0, 1.6875],
    J4: [1.505, 0.0, 1.476],
    EE: [1.505, 0.0, 1.6875],
    TCP: [1.505, 0.0, 1.436],
  };

  const limits = [
    { name: 'J1', lowerLimitRad: -2.87979, upperLimitRad: 2.87979, maxVelocityRadS: 2.53073, maxAccelerationRadS2: 12.0, frictionCoeffNmSPerRad: 0.5 },
    { name: 'J2', lowerLimitRad: -0.69813, upperLimitRad: 1.48353, maxVelocityRadS: 1.91986, maxAccelerationRadS2: 10.0, frictionCoeffNmSPerRad: 0.5 },
    { name: 'J3', lowerLimitRad: -0.69813, upperLimitRad: 2.09440, maxVelocityRadS: 2.09440, maxAccelerationRadS2: 10.0, frictionCoeffNmSPerRad: 0.5 },
    { name: 'J4', lowerLimitRad: -5.23599, upperLimitRad: 5.23599, maxVelocityRadS: 6.98132, maxAccelerationRadS2: 35.0, frictionCoeffNmSPerRad: 0.5 },
  ];

  const payloadMass = 50.0;
  const side = 0.03;
  const pInertia = (payloadMass * (side * side + side * side)) / 12.0;

  return {
    schema: 'robodimm.robot.v1',
    kind: 'CR4',
    name: 'CR4 (IRB460 preset)',
    units: 'SI',
    geometry: geom,
    inertials: {},
    payload: {
      body: 'PAYLOAD',
      massKg: payloadMass,
      comM: [0.0, 0.0, 0.0],
      inertiaKgM2: [
        [pInertia, 0.0, 0.0],
        [0.0, pInertia, 0.0],
        [0.0, 0.0, pInertia]
      ],
      frame: 'link',
    },
    visuals: [
      { body: 'SWING', frameName: 'SWING', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.38, 0.08] }, originM: [0, 0, 0.04], rpyRad: [0, 0, 0], scale: [1, 1, 1], visible: true },
      { body: 'SWING', frameName: 'SWING', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.16, 0.7425] }, originM: [0.260, 0, 0.37125], rpyRad: [0, 0, 0], scale: [1, 1, 1], visible: true },
      
      { body: 'P_ARM', frameName: 'seg_O_B', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.4] }, originM: [0.2, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      
      { body: 'LOWER_ARM', frameName: 'seg_O_C', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.945] }, originM: [0.4725, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      
      { body: 'P_LINK', frameName: 'seg_B_P', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.945] }, originM: [0.4725, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      
      // UPPER_ARM (3 cylinders)
      { body: 'UPPER_ARM', frameName: 'seg_P_H', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 1.425] }, originM: [0.7125, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      { body: 'UPPER_ARM', frameName: 'seg_P_C', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.4] }, originM: [0.2, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      { body: 'UPPER_ARM', frameName: 'seg_C_H', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 1.025] }, originM: [0.9125, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      
      { body: 'LOWER_LINK', frameName: 'seg_D_E', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.945] }, originM: [0.4725, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      
      // LINK_PLATE (3 cylinders)
      { body: 'LINK_PLATE', frameName: 'seg_C_E', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.299995533] }, originM: [0.149997767, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      { body: 'LINK_PLATE', frameName: 'seg_C_F', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.300076446] }, originM: [-0.0562324626, 0, -0.139102044], rpyRad: [0, -2.75742501018, 0], scale: [1, 1, 1], visible: true },
      { body: 'LINK_PLATE', frameName: 'seg_E_F', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.497514969] }, originM: [0.0937653041, 0, -0.139102044], rpyRad: [0, -2.16420159942, 0], scale: [1, 1, 1], visible: true },
      
      { body: 'UPPER_LINK', frameName: 'seg_F_G', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 1.0249] }, originM: [0.51245, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      
      // TILT/HGEE structural links. J4 is a wrist frame; the tool shaft belongs to DISK.
      { body: 'TILT', frameName: 'seg_H_G', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.299999836] }, originM: [0.149999918, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      { body: 'TILT', frameName: 'seg_H_EE', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.22] }, originM: [0, 0, 0], rpyRad: [0, 0, 0], scale: [1, 1, 1], visible: true },
      { body: 'TILT', frameName: 'seg_G_EE', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.316638] }, originM: [0, 0, 0], rpyRad: [0, 0, 0], scale: [1, 1, 1], visible: true },
      
      { body: 'DISK', frameName: 'DISK', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.07, 0.04] }, originM: [0, 0, 0], rpyRad: [0, 0, 0], scale: [1, 1, 1], visible: true },
      { body: 'DISK', frameName: 'seg_EE_TCP', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.025, 0.2515] }, originM: [0, 0, 0], rpyRad: [0, 0, 0], scale: [1, 1, 1], visible: true }
    ],
    station: [],
    limits
  };
}
