import { RobotSpec, ProgramSpec, TorqueLog, ActuatorLibrary, RobotActuatorSelection, cloneRobotSpec } from '../model/schemas';
import { Serial6Engine } from './serial6';
import { PalletizerEngine } from './palletizer';
import { buildProgramDynamicsTrajectory, TrajectoryPoint } from './trajectory';
import { selectActuators, getCr4LinkForJoint } from './actuators';

export interface SimulationResult {
  torqueLog: TorqueLog;
  playbackPoints: TrajectoryPoint[];
}

// Generate trajectory dynamics and torque timeseries in frontend
export function generateSimulationTorques(
  robot: RobotSpec,
  program: ProgramSpec,
  dt = 0.005
): SimulationResult {
  const startQ = robot.kind === 'CR6' ? Array(6).fill(0) : Array(4).fill(0);
  const trajectory = buildProgramDynamicsTrajectory(startQ, program, robot, dt);

  let engine6: Serial6Engine | null = null;
  let engine4: PalletizerEngine | null = null;
  if (robot.kind === 'CR6') {
    engine6 = new Serial6Engine(robot);
  } else {
    engine4 = new PalletizerEngine(robot);
  }

  const samples = trajectory.map(pt => {
    let tau = Array(startQ.length).fill(0);
    if (engine6) {
      tau = engine6.inverseDynamics(pt.q, pt.qd, pt.qdd);
    } else if (engine4) {
      tau = engine4.approximateDynamics(pt.q, pt.qd, pt.qdd);
    }
    return {
      time_s: pt.time_s,
      q: pt.q,
      velocity: pt.qd,
      acceleration: pt.qdd,
      joint_velocity: pt.qd,
      joint_acceleration: pt.qdd,
      tau
    };
  });

  const jointNames = robot.limits.map(l => l.name);

  return {
    torqueLog: {
      joint_names: jointNames,
      samples,
      dt_s: dt,
      engine_used: 'demo_frontend',
      model_id: robot.kind === 'CR6' ? 'cr6_demo_frontend' : 'cr4_demo_frontend'
    },
    playbackPoints: trajectory
  };
}

/**
 * @deprecated This iterative sizing algorithm is deprecated in favor of passive, deterministic actuator sizing.
 * It is preserved only for backwards compatibility with legacy tests or scripts.
 */
export async function runIterativeSizingAlgorithm(
  active: RobotSpec,
  lib: ActuatorLibrary,
  margins: { continuous: number; peak: number; speed: number },
  computeDynamics: (robot: RobotSpec) => Promise<TorqueLog>
): Promise<{ sizingResults: RobotActuatorSelection; updatedRobot: RobotSpec }> {
  let currentRobotSpec = cloneRobotSpec(active);
  const jointNames = active.limits.map(l => l.name);

  let selection: RobotActuatorSelection | null = null;
  let hasConverged = false;
  let iteration = 0;
  const maxIterations = 5;

  const baseInertials = { ...active.inertials };

  while (!hasConverged && iteration < maxIterations) {
    const currentLog = await computeDynamics(currentRobotSpec);
    const nextSelection = selectActuators(lib, currentLog, margins);
    
    if (selection) {
      let changed = false;
      for (let i = 0; i < jointNames.length; i++) {
        const prevBest = selection.joints[i].best?.motor.id;
        const nextBest = nextSelection.joints[i].best?.motor.id;
        if (prevBest !== nextBest) {
          changed = true;
          break;
        }
      }
      if (!changed) {
        hasConverged = true;
      }
    }
    
    selection = nextSelection;

    if (!hasConverged) {
      for (let i = 0; i < jointNames.length; i++) {
        const best = selection!.joints[i].best;
        if (best) {
          const addedWeight = best.total_mass_kg;
          const linkName = active.kind === 'CR6' ? `LINK${i + 1}` : getCr4LinkForJoint(i);
          
          const currentInertial = currentRobotSpec.inertials[linkName] || {
            body: linkName,
            massKg: 0.0,
            comM: [0, 0, 0],
            inertiaKgM2: [[0,0,0],[0,0,0],[0,0,0]],
            frame: 'cad'
          };

          const baseInertial = baseInertials[linkName] || {
            body: linkName,
            massKg: 0.0,
            comM: [0, 0, 0],
            inertiaKgM2: [[0,0,0],[0,0,0],[0,0,0]],
            frame: 'cad'
          };

          currentRobotSpec.inertials[linkName] = {
            ...currentInertial,
            massKg: baseInertial.massKg + addedWeight
          };
        }
      }
    }

    iteration++;
  }

  return {
    sizingResults: selection!,
    updatedRobot: currentRobotSpec
  };
}
