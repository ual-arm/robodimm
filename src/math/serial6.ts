import {
  Matrix4,
  Vector3,
  Matrix3,
  createIdentity4,
  standardDH,
  rotz,
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
  wrapNear,
  rotationError,
  angleDifference,
  norm3,
  norm,
  wrapToPi
} from './matrix';
import { RobotSpec, DHJointSpec, InertialSpec } from '../model/schemas';

export interface Serial6FK {
  q: number[];
  joint_origins: Vector3[];
  joint_axes: Vector3[];
  joint_body_transforms: Matrix4[];
  link_transforms: Matrix4[];
  tcp_transform: Matrix4;
}

export interface Serial6IKResult {
  success: boolean;
  q: number[];
  position_error: number;
  orientation_error: number;
  branch: string;
  message?: string;
}

export class Serial6Engine {
  spec: RobotSpec;

  constructor(spec: RobotSpec) {
    if (spec.kind !== 'CR6') {
      throw new Error(`Serial6Engine requires CR6 robot, got ${spec.kind}`);
    }
    this.spec = spec;
  }

  get joints(): DHJointSpec[] {
    return (this.spec.geometry as any).joints;
  }

  get tool_transform(): Matrix4 {
    return (this.spec.geometry as any).tool_transform || createIdentity4();
  }

  get gravity(): Vector3 {
    return [0.0, 0.0, -9.80665];
  }

  clampConfiguration(q: number[]): number[] {
    const qClamped = [...q];
    for (let i = 0; i < 6; i++) {
      const limit = this.spec.limits[i];
      if (limit) {
        qClamped[i] = Math.max(limit.lowerLimitRad, Math.min(limit.upperLimitRad, qClamped[i]));
      }
    }
    return qClamped;
  }

  forwardKinematics(q: number[]): Serial6FK {
    const qClamped = this.clampConfiguration(q);
    let transform = createIdentity4();
    const origins: Vector3[] = [];
    const axes: Vector3[] = [];
    const joint_body_transforms: Matrix4[] = [];
    const link_transforms: Matrix4[] = [];

    const joints = this.joints;
    for (let i = 0; i < 6; i++) {
      const joint = joints[i];
      origins.push(getTranslation(transform));
      // Axis is the Z axis of the current joint transform (column 2)
      axes.push([transform[0][2], transform[1][2], transform[2][2]]);

      const theta = qClamped[i] + joint.theta_offset_rad;
      joint_body_transforms.push(multiply4x4(transform, rotz(theta)));

      transform = multiply4x4(
        transform,
        standardDH(joint.a_m, joint.alpha_rad, joint.d_m, theta)
      );
      link_transforms.push(transform);
    }

    return {
      q: qClamped,
      joint_origins: origins,
      joint_axes: axes,
      joint_body_transforms,
      link_transforms,
      tcp_transform: multiply4x4(transform, this.tool_transform),
    };
  }

  geometricJacobian(q: number[], point?: Vector3): number[][] {
    const fk = this.forwardKinematics(q);
    const tip = point || getTranslation(fk.tcp_transform);
    const jacobian = Array.from({ length: 6 }, () => Array(6).fill(0));

    for (let i = 0; i < 6; i++) {
      const origin = fk.joint_origins[i];
      const axis = fk.joint_axes[i];
      const radius = subtractVectors(tip, origin);
      const jv = crossProduct(axis, radius);
      
      jacobian[0][i] = jv[0];
      jacobian[1][i] = jv[1];
      jacobian[2][i] = jv[2];
      jacobian[3][i] = axis[0];
      jacobian[4][i] = axis[1];
      jacobian[5][i] = axis[2];
    }
    return jacobian;
  }

  solveSphericalWristIK(
    targetTransform: Matrix4,
    qSeed?: number[],
    positionToleranceM = 1e-5,
    orientationToleranceRad = 1e-5
  ): Serial6IKResult {
    // Validate spherical wrist geometry
    this._validateSupportedSphericalWristGeometry();

    const target = targetTransform;
    const seed = qSeed ? [...qSeed] : Array(6).fill(0);
    const joints = this.joints;

    const a1 = joints[0].a_m;
    const d1 = joints[0].d_m;
    const a2 = joints[1].a_m;
    const a3 = joints[2].a_m;
    const d4 = joints[3].d_m;
    const d6 = joints[5].d_m;

    const link_23 = Math.hypot(a3, d4);
    const phi = Math.atan2(d4, a3);

    // Calculate wrist center
    const targetZAxis: Vector3 = [target[0][2], target[1][2], target[2][2]];
    const targetPosition: Vector3 = [target[0][3], target[1][3], target[2][3]];
    const wrist_center = subtractVectors(targetPosition, scaleVector(targetZAxis, d6));

    const base_angle = Math.atan2(wrist_center[1], wrist_center[0]);
    const radial_distance = Math.hypot(wrist_center[0], wrist_center[1]);

    const candidates: { q: number[]; branch: string }[] = [];

    const shoulderOptions = [
      { q1_base: base_angle, signed_radius: radial_distance },
      { q1_base: base_angle + Math.PI, signed_radius: -radial_distance }
    ];

    for (let sIndex = 0; sIndex < 2; sIndex++) {
      const { q1_base, signed_radius } = shoulderOptions[sIndex];
      const planar_x = signed_radius - a1;
      const planar_y = d1 - wrist_center[2];
      const distance_sq = planar_x * planar_x + planar_y * planar_y;

      const cos_elbow = (distance_sq - a2 * a2 - link_23 * link_23) / (2.0 * a2 * link_23);
      if (cos_elbow < -1.0 - 1e-9 || cos_elbow > 1.0 + 1e-9) {
        continue;
      }
      const cos_elbow_clamped = Math.max(-1.0, Math.min(1.0, cos_elbow));

      for (const elbow_sign of [-1.0, 1.0]) {
        const sin_elbow = elbow_sign * Math.sqrt(Math.max(0.0, 1.0 - cos_elbow_clamped * cos_elbow_clamped));
        const q2_plane = Math.atan2(planar_y, planar_x) - Math.atan2(
          link_23 * sin_elbow,
          a2 + link_23 * cos_elbow_clamped
        );
        const second_link_angle = q2_plane + Math.atan2(sin_elbow, cos_elbow_clamped);

        const q1 = wrapNear(q1_base - joints[0].theta_offset_rad, seed[0]);
        const q2 = wrapNear(q2_plane - joints[1].theta_offset_rad, seed[1]);
        const q3 = wrapNear(
          second_link_angle - phi - q2_plane - joints[2].theta_offset_rad,
          seed[2]
        );

        // Calculate wrist orientation
        const q123 = [q1, q2, q3, 0.0, 0.0, 0.0];
        const fk_3 = this.forwardKinematics(q123);
        const rotation_03 = getMatrix3(fk_3.link_transforms[2]);
        const targetRotation = getMatrix3(target);
        
        // rotation_36 = rotation_03.T @ targetRotation
        const rotation_36 = multiply3x3(transpose3x3(rotation_03), targetRotation);

        const wristSolutions = this._extractSphericalWrist(rotation_36, seed.slice(3) as [number, number, number]);
        for (let wIndex = 0; wIndex < wristSolutions.length; wIndex++) {
          const wrist = wristSolutions[wIndex];
          let q = [q1, q2, q3, wrist[0], wrist[1], wrist[2]];
          q = q.map((val, idx) => wrapNear(val, seed[idx]));

          if (!this._isWithinLimits(q)) {
            continue;
          }

          candidates.push({
            q,
            branch: `shoulder=${sIndex},elbow=${elbow_sign},wrist=${wIndex}`
          });
        }
      }
    }

    let best: Serial6IKResult | null = null;
    for (const cand of candidates) {
      const fk = this.forwardKinematics(cand.q);
      const posError = norm3(subtractVectors(getTranslation(fk.tcp_transform), targetPosition));
      const rotError = rotationError(getMatrix3(fk.tcp_transform), getMatrix3(target));

      const isSuccess = posError <= positionToleranceM && rotError <= orientationToleranceRad;
      const score = norm(angleDifference(cand.q, seed));

      const result: Serial6IKResult = {
        success: isSuccess,
        q: cand.q,
        position_error: posError,
        orientation_error: rotError,
        branch: cand.branch
      };

      if (!best) {
        best = result;
      } else {
        const bestScore = norm(angleDifference(best.q, seed));
        // Prioritize success, then smaller joint change
        if ((result.success && !best.success) || (result.success === best.success && score < bestScore)) {
          best = result;
        }
      }
    }

    if (!best) {
      return {
        success: false,
        q: [...seed],
        position_error: Infinity,
        orientation_error: Infinity,
        branch: '',
        message: 'Target unreachable or outside joint limits'
      };
    }
    return best;
  }

  inverseDynamics(q: number[], qd: number[], qdd: number[]): number[] {
    const q_clamped = this.clampConfiguration(q);
    const tau = Array(6).fill(0);

    const fk = this.forwardKinematics(q_clamped);
    const gravity = this.gravity;
    const origins = fk.joint_origins;
    const axes = fk.joint_axes;

    const origin_velocities = this._serialOriginVelocities(origins, axes, qd);
    const axis_derivatives = this._serialAxisDerivatives(axes, qd);

    const body_inertials: InertialSpec[] = [];
    const body_transforms: Matrix4[] = [];

    const joints = this.joints;
    for (let i = 0; i < 6; i++) {
      const linkName = `LINK${i + 1}`;
      let inertial = this.spec.inertials[linkName] || this.spec.inertials[joints[i].name];
      if (inertial) {
        body_inertials.push(inertial);
      } else {
        body_inertials.push({
          body: linkName,
          massKg: 0.0,
          comM: [0.0, 0.0, 0.0],
          inertiaKgM2: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
          frame: 'link'
        });
      }
      body_transforms.push(fk.link_transforms[i]);
    }

    if (this.spec.payload && this.spec.payload.massKg > 0) {
      body_inertials.push(this.spec.payload);
      body_transforms.push(fk.tcp_transform);
    }

    for (let bodyIdx = 0; bodyIdx < body_inertials.length; bodyIdx++) {
      let inertial: InertialSpec = {
        ...body_inertials[bodyIdx],
        comM: body_inertials[bodyIdx].comM ?? [0.0, 0.0, 0.0],
        inertiaKgM2: body_inertials[bodyIdx].inertiaKgM2 ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        frame: body_inertials[bodyIdx].frame ?? 'link'
      };
      if (inertial.massKg === 0 && !this._hasInertia(inertial)) {
        continue;
      }
      const linkIdx = Math.min(bodyIdx, 5);

      if (bodyIdx < 6 && inertial.frame === 'cad') {
        inertial = this._cadInertialToLinkInertial(bodyIdx, inertial);
      }

      const com_local = inertial.comM!;
      const rotation_world_link = getMatrix3(body_transforms[bodyIdx]);
      
      // Transform local COM to world space: T @ [*com, 1]
      const T = body_transforms[bodyIdx];
      const com_world: Vector3 = [
        T[0][0] * com_local[0] + T[0][1] * com_local[1] + T[0][2] * com_local[2] + T[0][3],
        T[1][0] * com_local[0] + T[1][1] * com_local[1] + T[1][2] * com_local[2] + T[1][3],
        T[2][0] * com_local[0] + T[2][1] * com_local[1] + T[2][2] * com_local[2] + T[2][3]
      ];

      // Transform local inertia matrix to world space: R @ I_local @ R.T
      const I_local = inertial.inertiaKgM2!;
      const inertia_world = multiply3x3(
        multiply3x3(rotation_world_link, I_local),
        transpose3x3(rotation_world_link)
      );

      const jacobian_v = Array.from({ length: 3 }, () => Array(6).fill(0));
      const jacobian_w = Array.from({ length: 3 }, () => Array(6).fill(0));
      let jacobian_v_dot_qd: Vector3 = [0, 0, 0];
      let jacobian_w_dot_qd: Vector3 = [0, 0, 0];

      for (let j = 0; j <= linkIdx; j++) {
        const axis = axes[j];
        const origin = origins[j];
        const radius = subtractVectors(com_world, origin);
        const jv = crossProduct(axis, radius);
        
        jacobian_v[0][j] = jv[0];
        jacobian_v[1][j] = jv[1];
        jacobian_v[2][j] = jv[2];
        jacobian_w[0][j] = axis[0];
        jacobian_w[1][j] = axis[1];
        jacobian_w[2][j] = axis[2];
      }

      // Calculate COM velocity
      let com_velocity: Vector3 = [0, 0, 0];
      for (let j = 0; j <= linkIdx; j++) {
        com_velocity[0] += jacobian_v[0][j] * qd[j];
        com_velocity[1] += jacobian_v[1][j] * qd[j];
        com_velocity[2] += jacobian_v[2][j] * qd[j];
      }

      for (let j = 0; j <= linkIdx; j++) {
        const axis_dot = axis_derivatives[j];
        const radius = subtractVectors(com_world, origins[j]);
        const radius_dot = subtractVectors(com_velocity, origin_velocities[j]);
        
        // term = qd[j] * (axis_dot x radius + axis x radius_dot)
        const term = scaleVector(
          addVectors(crossProduct(axis_dot, radius), crossProduct(axes[j], radius_dot)),
          qd[j]
        );
        jacobian_v_dot_qd = addVectors(jacobian_v_dot_qd, term);
        jacobian_w_dot_qd = addVectors(jacobian_w_dot_qd, scaleVector(axis_dot, qd[j]));
      }

      // Accelerations
      let jacobian_v_qdd: Vector3 = [0, 0, 0];
      let jacobian_w_qdd: Vector3 = [0, 0, 0];
      for (let j = 0; j <= linkIdx; j++) {
        jacobian_v_qdd[0] += jacobian_v[0][j] * qdd[j];
        jacobian_v_qdd[1] += jacobian_v[1][j] * qdd[j];
        jacobian_v_qdd[2] += jacobian_v[2][j] * qdd[j];
        jacobian_w_qdd[0] += jacobian_w[0][j] * qdd[j];
        jacobian_w_qdd[1] += jacobian_w[1][j] * qdd[j];
        jacobian_w_qdd[2] += jacobian_w[2][j] * qdd[j];
      }

      const com_acceleration = addVectors(jacobian_v_qdd, jacobian_v_dot_qd);
      let angular_velocity: Vector3 = [0, 0, 0];
      for (let j = 0; j <= linkIdx; j++) {
        angular_velocity[0] += jacobian_w[0][j] * qd[j];
        angular_velocity[1] += jacobian_w[1][j] * qd[j];
        angular_velocity[2] += jacobian_w[2][j] * qd[j];
      }
      const angular_acceleration = addVectors(jacobian_w_qdd, jacobian_w_dot_qd);

      // Force = mass * (com_acceleration - gravity)
      const force = scaleVector(subtractVectors(com_acceleration, gravity), inertial.massKg);
      // Moment = inertia_world * angular_acceleration + angular_velocity x (inertia_world * angular_velocity)
      const Iw_w = multiply3x3AndVector(inertia_world, angular_velocity);
      const moment = addVectors(
        multiply3x3AndVector(inertia_world, angular_acceleration),
        crossProduct(angular_velocity, Iw_w)
      );

      // Add to joint torques: Jv.T @ force + Jw.T @ moment
      for (let j = 0; j <= linkIdx; j++) {
        tau[j] += (
          jacobian_v[0][j] * force[0] + jacobian_v[1][j] * force[1] + jacobian_v[2][j] * force[2] +
          jacobian_w[0][j] * moment[0] + jacobian_w[1][j] * moment[1] + jacobian_w[2][j] * moment[2]
        );
      }
    }

    // Add joint viscous friction
    for (let i = 0; i < 6; i++) {
      const limit = this.spec.limits[i];
      const frictionCoeff = limit?.frictionCoeffNmSPerRad ?? 0.0;
      tau[i] += frictionCoeff * qd[i];
    }

    return tau;
  }

  _extractSphericalWrist(rotation_36: Matrix3, wristSeed: [number, number, number]): [number, number, number][] {
    const r = rotation_36;
    const cos_b = Math.max(-1.0, Math.min(1.0, -r[2][2]));
    const sin_abs = Math.sqrt(Math.max(0.0, 1.0 - cos_b * cos_b));

    const j4_offset = this.joints[3].theta_offset_rad;
    const j5_offset = this.joints[4].theta_offset_rad;
    const j6_offset = this.joints[5].theta_offset_rad;

    if (sin_abs < 1e-9) {
      const q4 = wristSeed[0];
      const combined = Math.atan2(r[1][0], r[0][0]);
      const q6 = wrapNear(combined - q4, wristSeed[2]);
      const q5 = wrapNear(Math.atan2(0.0, cos_b) - j5_offset, wristSeed[1]);
      return [[q4, q5, q6]];
    }

    const solutions: [number, number, number][] = [];
    for (const sin_b of [sin_abs, -sin_abs]) {
      const b = Math.atan2(sin_b, cos_b);
      const q4 = Math.atan2(r[1][2] / sin_b, r[0][2] / sin_b);
      const q5 = b;
      const q6 = Math.atan2(-r[2][1] / sin_b, r[2][0] / sin_b);

      solutions.push([
        wrapNear(q4 - j4_offset, wristSeed[0]),
        wrapNear(q5 - j5_offset, wristSeed[1]),
        wrapNear(q6 - j6_offset, wristSeed[2])
      ]);
    }
    return solutions;
  }

  _validateSupportedSphericalWristGeometry(): void {
    const expectedAlpha = [-Math.PI / 2.0, 0.0, -Math.PI / 2.0, Math.PI / 2.0, Math.PI / 2.0, 0.0];
    const joints = this.joints;
    for (let i = 0; i < 6; i++) {
      if (Math.abs(joints[i].alpha_rad - expectedAlpha[i]) > 1e-9) {
        throw new Error('Spherical wrist IK requires IRB4600-like DH topology');
      }
    }
    if (Math.abs(joints[3].a_m) > 1e-12 || Math.abs(joints[4].a_m) > 1e-12 || Math.abs(joints[5].a_m) > 1e-12) {
      throw new Error('Spherical wrist IK requires a4=a5=a6=0');
    }
    if (Math.abs(joints[4].d_m) > 1e-12) {
      throw new Error('Spherical wrist IK requires d5=0');
    }
  }

  _isWithinLimits(q: number[]): boolean {
    for (let i = 0; i < 6; i++) {
      const limit = this.spec.limits[i];
      if (limit) {
        if (q[i] < limit.lowerLimitRad - 1e-9 || q[i] > limit.upperLimitRad + 1e-9) {
          return false;
        }
      }
    }
    return true;
  }

  _hasInertia(inertial: InertialSpec): boolean {
    const I = inertial.inertiaKgM2;
    if (!I) return false;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        if (Math.abs(I[r][c]) > 1e-12) return true;
      }
    }
    return false;
  }

  _cadInertialToLinkInertial(linkIndex: number, inertial: InertialSpec): InertialSpec {
    // Port of _cad_inertial_to_link_inertial
    const home_fk = this.forwardKinematics(Array(6).fill(0));
    const cad_transform = multiply4x4(home_fk.joint_body_transforms[linkIndex], createIdentity4());
    
    // Set rotation part to identity as done in python: cad_transform[:3, :3] = np.eye(3)
    cad_transform[0][0] = 1; cad_transform[0][1] = 0; cad_transform[0][2] = 0;
    cad_transform[1][0] = 0; cad_transform[1][1] = 1; cad_transform[1][2] = 0;
    cad_transform[2][0] = 0; cad_transform[2][1] = 0; cad_transform[2][2] = 1;

    const link_transform = home_fk.link_transforms[linkIndex];
    return this._transformInertialFrame(inertial, cad_transform, link_transform, 'link');
  }

  _linkInertialToCadInertial(linkIndex: number, inertial: InertialSpec): InertialSpec {
    const home_fk = this.forwardKinematics(Array(6).fill(0));
    const cad_transform = multiply4x4(home_fk.joint_body_transforms[linkIndex], createIdentity4());
    
    // Set rotation part to identity as done in python: cad_transform[:3, :3] = np.eye(3)
    cad_transform[0][0] = 1; cad_transform[0][1] = 0; cad_transform[0][2] = 0;
    cad_transform[1][0] = 0; cad_transform[1][1] = 1; cad_transform[1][2] = 0;
    cad_transform[2][0] = 0; cad_transform[2][1] = 0; cad_transform[2][2] = 1;

    const link_transform = home_fk.link_transforms[linkIndex];
    return this._transformInertialFrame(inertial, link_transform, cad_transform, 'cad');
  }

  _transformInertialFrame(
    inertial: InertialSpec,
    fromTransform: Matrix4,
    toTransform: Matrix4,
    frame: 'link' | 'cad' | 'tcp'
  ): InertialSpec {
    const com_from = inertial.comM ?? [0, 0, 0];
    const inertia_from = inertial.inertiaKgM2 ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]];

    const rotation_from = getMatrix3(fromTransform);
    const rotation_to = getMatrix3(toTransform);
    const translation_from = getTranslation(fromTransform);
    const translation_to = getTranslation(toTransform);

    // com_world = R_from @ com_from + t_from
    const com_world = addVectors(multiply3x3AndVector(rotation_from, com_from), translation_from);
    // com_to = R_to.T @ (com_world - t_to)
    const com_to = multiply3x3AndVector(transpose3x3(rotation_to), subtractVectors(com_world, translation_to));

    // rotation_to_from = R_to.T @ R_from
    const rotation_to_from = multiply3x3(transpose3x3(rotation_to), rotation_from);
    // inertia_to = rotation_to_from @ inertia_from @ rotation_to_from.T
    const inertia_to = multiply3x3(
      multiply3x3(rotation_to_from, inertia_from),
      transpose3x3(rotation_to_from)
    );

    // Clean up small numbers
    const cleanCom = com_to.map(val => Math.abs(val) < 1e-12 ? 0 : val) as Vector3;
    const cleanInertia = inertia_to.map(row => row.map(val => Math.abs(val) < 1e-12 ? 0 : val)) as Matrix3;

    return {
      body: inertial.body,
      massKg: inertial.massKg,
      comM: cleanCom,
      inertiaKgM2: cleanInertia,
      frame
    };
  }

  _serialOriginVelocities(origins: Vector3[], axes: Vector3[], qd: number[]): Vector3[] {
    const velocities: Vector3[] = [];
    for (let i = 0; i < origins.length; i++) {
      let velocity: Vector3 = [0, 0, 0];
      for (let j = 0; j < i; j++) {
        const radius = subtractVectors(origins[i], origins[j]);
        const term = scaleVector(crossProduct(axes[j], radius), qd[j]);
        velocity = addVectors(velocity, term);
      }
      velocities.push(velocity);
    }
    return velocities;
  }

  _serialAxisDerivatives(axes: Vector3[], qd: number[]): Vector3[] {
    const derivatives: Vector3[] = [];
    let angular_velocity: Vector3 = [0, 0, 0];
    for (let i = 0; i < axes.length; i++) {
      derivatives.push(crossProduct(angular_velocity, axes[i]));
      angular_velocity = addVectors(angular_velocity, scaleVector(axes[i], qd[i]));
    }
    return derivatives;
  }
}

export function irb4600Serial6Spec(): RobotSpec {
  const jointsSpec: DHJointSpec[] = [
    { name: 'J1', a_m: 0.175, alpha_rad: -Math.PI / 2.0, d_m: 0.495, theta_offset_rad: 0.0 },
    { name: 'J2', a_m: 1.095, alpha_rad: 0.0, d_m: 0.0, theta_offset_rad: -Math.PI / 2.0 },
    { name: 'J3', a_m: 0.175, alpha_rad: -Math.PI / 2.0, d_m: 0.0, theta_offset_rad: 0.0 },
    { name: 'J4', a_m: 0.0, alpha_rad: Math.PI / 2.0, d_m: 1.2305, theta_offset_rad: 0.0 },
    { name: 'J5', a_m: 0.0, alpha_rad: Math.PI / 2.0, d_m: 0.0, theta_offset_rad: Math.PI },
    { name: 'J6', a_m: 0.0, alpha_rad: 0.0, d_m: 0.085, theta_offset_rad: 0.0 },
  ];

  // Estimated masses of MOVING bodies only (downstream of each joint).
  // The IRB 4600-45/2.05 shipping weight is 425 kg, but that includes the
  // fixed base casting, J1 motor+reducer housing, covers and cabling.
  // Only downstream moving mass of each joint is used for dynamics.
  // Do NOT inflate these values to match the datasheet total weight without
  // a proper fixed-vs-moving mass breakdown.
  //
  //   LINK1 45 kg  – column + lower structure rotating on J1 (J1 housing excluded)
  //   LINK2 65 kg  – upper arm beam (main contributor to J2 gravity torque)
  //   LINK3 40 kg  – forearm with J4 motor housing
  //   LINK4 28 kg  – wrist roll body
  //   LINK5 17 kg  – wrist pitch body
  //   LINK6  5 kg  – wrist flange
  //   Total moving: 200 kg + payload
  const masses = [45.0, 65.0, 40.0, 28.0, 17.0, 5.0];
  const coms: Vector3[] = [
    [0.175, 0.0, 0.495],
    [0.0, 0.0, 0.5475],
    [0.0, 0.0, 0.0875],
    [0.61525, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0425, 0.0, 0.0],
  ];
  // Inertia tensors scaled proportionally to mass changes (I ∝ m, same geometry):
  //   LINK1: ×(45/55)=0.818,  LINK2: ×(65/95)=0.684,  LINK3: ×(40/70)=0.571
  //   LINK4–6 unchanged
  const inertias: Matrix3[] = [
    [[0.920, 0.0, 0.0], [0.0, 0.920, 0.0], [0.0, 0.0, 0.004593]],   // LINK1 ×0.818
    [[6.498, 0.0, 0.0], [0.0, 6.498, 0.0], [0.0, 0.0, 0.006636]],   // LINK2 ×0.684
    [[0.104, 0.0, 0.0], [0.0, 0.104, 0.0], [0.0, 0.0, 0.004083]],   // LINK3 ×0.571
    [[0.002858, 0.0, 0.0], [0.0, 3.534399, 0.0], [0.0, 0.0, 3.534399]],
    [[0.002499, 0.0, 0.0], [0.0, 0.002499, 0.0], [0.0, 0.0, 0.002499]],
    [[0.000510, 0.0, 0.0], [0.0, 0.003265, 0.0], [0.0, 0.0, 0.003265]],
  ];

  const inertials: Record<string, InertialSpec> = {};
  for (let i = 0; i < 6; i++) {
    const linkName = `LINK${i + 1}`;
    inertials[linkName] = {
      body: linkName,
      massKg: masses[i],
      comM: coms[i],
      inertiaKgM2: inertias[i],
      frame: 'cad',
    };
  }

  // Default payload set to a representative application load (not the robot\'s max rated capacity).
  // Users can change this in the Inertial Parameters editor before running sizing.
  const payloadMass = 15.0;
  const side = 0.03;
  const pInertia = (payloadMass * (side * side + side * side)) / 12.0;

  return {
    schema: 'robodimm.robot.v1',
    kind: 'CR6',
    name: 'CR6 (IRB4600 preset)',
    units: 'SI',
    geometry: {
      joints: jointsSpec,
      tool_transform: createIdentity4(),
    },
    inertials,
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
      { body: 'BASE', frameName: 'base_link', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.32, 0.08] }, originM: [0, 0, 0.04], rpyRad: [0, 0, 0], scale: [1, 1, 1], visible: true },
      { body: 'LINK1', frameName: 'LINK1', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.14, 0.495] }, originM: [0.175, 0, 0.2475], rpyRad: [0, 0, 0], scale: [1, 1, 1], visible: true },
      { body: 'LINK2', frameName: 'LINK2', kind: 'primitive', primitive: { type: 'box', dimensions: [0.15, 0.15, 1.095] }, originM: [0, 0, 0.5475], rpyRad: [0, 0, 0], scale: [1, 1, 1], visible: true },
      { body: 'LINK3', frameName: 'LINK3', kind: 'primitive', primitive: { type: 'box', dimensions: [0.12, 0.12, 0.175] }, originM: [0.0875, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      { body: 'LINK4', frameName: 'LINK4', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.09, 1.2305] }, originM: [0.61525, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true },
      { body: 'LINK5', frameName: 'LINK5', kind: 'primitive', primitive: { type: 'box', dimensions: [0.0297, 0.0297, 0.0297] }, originM: [0, 0, 0], rpyRad: [0, 0, 0], scale: [1, 1, 1], visible: true },
      { body: 'LINK6', frameName: 'LINK6', kind: 'primitive', primitive: { type: 'cylinder', dimensions: [0.07, 0.085] }, originM: [0.0425, 0, 0], rpyRad: [0, Math.PI / 2.0, 0], scale: [1, 1, 1], visible: true }
    ],
    station: [],
    limits: [
      { name: 'J1', lowerLimitRad: -3.141592653589793, upperLimitRad: 3.141592653589793, maxVelocityRadS: 3.054, maxAccelerationRadS2: 12.0, frictionCoeffNmSPerRad: 0.5 },
      { name: 'J2', lowerLimitRad: -1.5707963267948966, upperLimitRad: 2.617993877991494, maxVelocityRadS: 3.054, maxAccelerationRadS2: 12.0, frictionCoeffNmSPerRad: 0.5 },
      { name: 'J3', lowerLimitRad: -3.141592653589793, upperLimitRad: 1.3089969389957472, maxVelocityRadS: 3.054, maxAccelerationRadS2: 12.0, frictionCoeffNmSPerRad: 0.5 },
      { name: 'J4', lowerLimitRad: -6.981317007977318, upperLimitRad: 6.981317007977318, maxVelocityRadS: 4.363, maxAccelerationRadS2: 25.0, frictionCoeffNmSPerRad: 0.5 },
      { name: 'J5', lowerLimitRad: -2.181661564992912, upperLimitRad: 2.0943951023931953, maxVelocityRadS: 4.363, maxAccelerationRadS2: 25.0, frictionCoeffNmSPerRad: 0.5 },
      { name: 'J6', lowerLimitRad: -6.981317007977318, upperLimitRad: 6.981317007977318, maxVelocityRadS: 6.283, maxAccelerationRadS2: 25.0, frictionCoeffNmSPerRad: 0.5 }
    ],
  };
}
