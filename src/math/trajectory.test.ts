import { describe, expect, it } from 'vitest';
import { ProgramSpec } from '../model/schemas';
import { irb460PalletizerSpec, PalletizerEngine } from './palletizer';
import { buildProgramDynamicsTrajectory } from './trajectory';

describe('program dynamics trajectory semantics', () => {
  it('treats legacy MoveL as joint interpolation with a TCP-distance duration floor', () => {
    const robot = irb460PalletizerSpec();
    const start = [
      1.541553735733068,
      0.9609910004498775,
      1.039232676381714,
      -1.1129717281121999e-7
    ];
    const target = [
      1.5474723577497647,
      1.2991925307033179,
      0.6203535464305907,
      0.005918496578899646
    ];
    const program: ProgramSpec = {
      schema: 'robodimm.program.v1',
      name: 'MoveL semantics test',
      targets: [{ name: 'target', q: target }],
      instructions: [{ type: 'MoveL', target_name: 'target', tcp_speed_m_s: 1, zone_m: 0 }]
    };

    const trajectory = buildProgramDynamicsTrajectory(start, program, robot, 0.005);
    const duration = trajectory[trajectory.length - 1].time_s;
    const midpoint = trajectory.reduce((best, point) =>
      Math.abs(point.time_s / duration - 0.5) < Math.abs(best.time_s / duration - 0.5)
        ? point
        : best
    );
    const u = midpoint.time_s / duration;
    const quinticScale = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5;
    midpoint.q.forEach((value, index) => {
      expect(value).toBeCloseTo(start[index] + quinticScale * (target[index] - start[index]), 12);
    });

    const engine = new PalletizerEngine(robot);
    const startTcp = engine.forwardKinematics(start).points.TCP;
    const targetTcp = engine.forwardKinematics(target).points.TCP;
    const midpointTcp = engine.forwardKinematics(midpoint.q).points.TCP;
    const chordPoint = startTcp.map((value, index) => value + quinticScale * (targetTcp[index] - value));
    const cartesianDeviation = Math.hypot(...midpointTcp.map((value, index) => value - chordPoint[index]));

    // This guards the documented limitation: the source does not implement a
    // Cartesian straight-line primitive despite retaining the legacy MoveL label.
    expect(cartesianDeviation).toBeGreaterThan(1e-4);
  });
});
