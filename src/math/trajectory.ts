import { RobotSpec, ProgramSpec, TorqueSample, TorqueLog } from '../model/schemas';
import { Serial6Engine } from './serial6';
import { PalletizerEngine } from './palletizer';
import { getTranslation } from './matrix';

export interface TrajectoryPoint {
  time_s: number;
  q: number[];
  qd: number[];
  qdd: number[];
  instruction_index: number;
}

export function buildProgramDynamicsTrajectory(
  start_q: number[],
  program: ProgramSpec,
  robotSpec: RobotSpec,
  dt_s = 0.005
): TrajectoryPoint[] {
  if (dt_s <= 0) {
    throw new Error('dt_s must be positive');
  }

  let current_q = [...start_q];
  const points: TrajectoryPoint[] = [];

  // Initialize engines for FK calculations
  let engine6: Serial6Engine | null = null;
  let engine4: PalletizerEngine | null = null;
  if (robotSpec.kind === 'CR6') {
    engine6 = new Serial6Engine(robotSpec);
  } else {
    engine4 = new PalletizerEngine(robotSpec);
  }

  const getTCPPosition = (q: number[]): [number, number, number] => {
    if (engine6) {
      const fk = engine6.forwardKinematics(q);
      return getTranslation(fk.tcp_transform);
    } else if (engine4) {
      const fk = engine4.forwardKinematics(q);
      return fk.points['TCP'];
    }
    return [0, 0, 0];
  };

  // Add initial point
  points.push({
    time_s: 0.0,
    q: [...current_q],
    qd: Array(current_q.length).fill(0),
    qdd: Array(current_q.length).fill(0),
    instruction_index: -1,
  });

  for (let instIdx = 0; instIdx < program.instructions.length; instIdx++) {
    const instruction = program.instructions[instIdx];

    if (instruction.type === 'MoveJ') {
      const target = program.targets.find(t => t.name === instruction.target_name);
      if (!target) continue;
      const target_q = target.q;

      const speed = Math.max(instruction.speed_rad_s, 1e-3);
      let max_delta = 0.0;
      for (let i = 0; i < current_q.length; i++) {
        max_delta = Math.max(max_delta, Math.abs(target_q[i] - current_q[i]));
      }

      let duration_s = (1.875 * max_delta) / speed;

      // Limit duration by physical velocity and acceleration limits per joint
      for (let i = 0; i < current_q.length; i++) {
        const delta_qi = Math.abs(target_q[i] - current_q[i]);
        if (delta_qi > 1e-6) {
          const limit = robotSpec.limits[i];
          const v_lim = limit ? limit.maxVelocityRadS : 3.0;
          const a_lim = limit ? limit.maxAccelerationRadS2 : 15.0;
          
          const t_vel = (1.875 * delta_qi) / v_lim;
          const t_acc = Math.sqrt((5.7735 * delta_qi) / a_lim);
          duration_s = Math.max(duration_s, t_vel, t_acc);
        }
      }

      duration_s = Math.max(duration_s, dt_s);
      appendQuinticSegment(points, current_q, target_q, duration_s, dt_s, instIdx);
      current_q = [...target_q];

    } else if (instruction.type === 'MoveL') {
      const target = program.targets.find(t => t.name === instruction.target_name);
      if (!target) continue;
      const target_q = target.q;

      const start_pos = getTCPPosition(current_q);
      const target_pos = getTCPPosition(target_q);

      const dx = target_pos[0] - start_pos[0];
      const dy = target_pos[1] - start_pos[1];
      const dz = target_pos[2] - start_pos[2];
      const distance = Math.hypot(dx, dy, dz);

      const tcp_speed = Math.max(instruction.tcp_speed_m_s, 1e-4);
      let duration_s = distance / tcp_speed;

      // Limit duration by physical velocity and acceleration limits per joint
      for (let i = 0; i < current_q.length; i++) {
        const delta_qi = Math.abs(target_q[i] - current_q[i]);
        if (delta_qi > 1e-6) {
          const limit = robotSpec.limits[i];
          const v_lim = limit ? limit.maxVelocityRadS : 3.0;
          const a_lim = limit ? limit.maxAccelerationRadS2 : 15.0;
          
          const t_vel = (1.875 * delta_qi) / v_lim;
          const t_acc = Math.sqrt((5.7735 * delta_qi) / a_lim);
          duration_s = Math.max(duration_s, t_vel, t_acc);
        }
      }

      duration_s = Math.max(duration_s, dt_s);
      appendQuinticSegment(points, current_q, target_q, duration_s, dt_s, instIdx);
      current_q = [...target_q];

    } else if (instruction.type === 'Pause') {
      const duration = Math.max(instruction.duration_s, dt_s);
      const steps = Math.max(Math.ceil(duration / dt_s), 1);
      for (let step = 0; step < steps; step++) {
        const last = points[points.length - 1];
        points.push({
          time_s: last.time_s + dt_s,
          q: [...current_q],
          qd: Array(current_q.length).fill(0),
          qdd: Array(current_q.length).fill(0),
          instruction_index: instIdx,
        });
      }
    }
  }

  return points;
}

function appendQuinticSegment(
  points: TrajectoryPoint[],
  q0: number[],
  q1: number[],
  duration_s: number,
  dt_s: number,
  instruction_index: number
): void {
  const steps = Math.max(Math.ceil(duration_s / dt_s), 1);
  const n = q0.length;
  const delta = q1.map((val, i) => val - q0[i]);

  const lastTime = points[points.length - 1].time_s;
  const step_dt = duration_s / steps;

  for (let step = 1; step <= steps; step++) {
    const ti = step * step_dt;
    const u = ti / duration_s;

    // Quintic polynomial scaling profile
    const s = 10.0 * Math.pow(u, 3) - 15.0 * Math.pow(u, 4) + 6.0 * Math.pow(u, 5);
    const sd = (30.0 * Math.pow(u, 2) - 60.0 * Math.pow(u, 3) + 30.0 * Math.pow(u, 4)) / duration_s;
    const sdd = (60.0 * u - 180.0 * Math.pow(u, 2) + 120.0 * Math.pow(u, 3)) / (duration_s * duration_s);

    const q = q0.map((val, i) => val + s * delta[i]);
    const qd = delta.map(d => sd * d);
    const qdd = delta.map(d => sdd * d);

    points.push({
      time_s: lastTime + ti,
      q,
      qd,
      qdd,
      instruction_index,
    });
  }
}
