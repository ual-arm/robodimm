import { describe, it, expect } from 'vitest';
import referenceTests from './reference_tests.json';
import { Serial6Engine, irb4600Serial6Spec } from './serial6';
import { PalletizerEngine, irb460PalletizerSpec } from './palletizer';
import { Matrix4, Vector3 } from './matrix';

describe('Serial6 (CR6) Kinematics & Dynamics Port', () => {
  const spec = irb4600Serial6Spec();
  const engine = new Serial6Engine(spec);

  const dynamicsSpec = {
    ...spec,
    limits: spec.limits.map(l => ({ ...l, frictionCoeffNmSPerRad: 0.0 })),
    inertials: (() => {
      const inertials: any = {};
      const masses = [120.0, 180.0, 120.0, 55.0, 35.0, 15.0];
      const coms: Vector3[] = [
        [0.175, 0.0, 0.495],
        [0.0, 0.0, 0.5475],
        [0.0, 0.0, 0.0875],
        [0.61525, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0425, 0.0, 0.0],
      ];
      const inertias = [
        [[2.456375625, 0.0, 0.0], [0.0, 2.456375625, 0.0], [0.0, 0.0, 0.01225125]],
        [[17.9945634375, 0.0, 0.0], [0.0, 17.9945634375, 0.0], [0.0, 0.0, 0.018376875]],
        [[0.312375625, 0.0, 0.0], [0.0, 0.312375625, 0.0], [0.0, 0.0, 0.01225125]],
        [[0.00561515625, 0.0, 0.0], [0.0, 6.942571223958332, 0.0], [0.0, 0.0, 6.942571223958332]],
        [[0.005145525, 0.0, 0.0], [0.0, 0.005145525, 0.0], [0.0, 0.0, 0.005145525]],
        [[0.00153140625, 0.0, 0.0], [0.0, 0.009796953125, 0.0], [0.0, 0.0, 0.009796953125]],
      ];
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
      return inertials;
    })(),
    payload: (() => {
      const payloadMass = 15.0;
      const side = 0.03;
      const pInertia = (payloadMass * (side * side + side * side)) / 12.0;
      return {
        body: 'PAYLOAD',
        massKg: payloadMass,
        comM: [0.0, 0.0, 0.0],
        inertiaKgM2: [
          [pInertia, 0.0, 0.0],
          [0.0, pInertia, 0.0],
          [0.0, 0.0, pInertia]
        ],
        frame: 'link',
      };
    })()
  };
  const dynamicsEngine = new Serial6Engine(dynamicsSpec as any);

  it('should match forward kinematics against python backend', () => {
    for (const testCase of referenceTests.cr6) {
      const fk = engine.forwardKinematics(testCase.q);
      
      // Compare TCP transforms
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          expect(fk.tcp_transform[r][c]).toBeCloseTo(testCase.tcp_transform[r][c], 5);
        }
      }

      // Compare joint origins
      for (let i = 0; i < 6; i++) {
        const actualOrigin = fk.joint_origins[i];
        const expectedOrigin = testCase.joint_origins[i];
        expect(actualOrigin[0]).toBeCloseTo(expectedOrigin[0], 5);
        expect(actualOrigin[1]).toBeCloseTo(expectedOrigin[1], 5);
        expect(actualOrigin[2]).toBeCloseTo(expectedOrigin[2], 5);
      }

      // Compare joint axes
      for (let i = 0; i < 6; i++) {
        const actualAxis = fk.joint_axes[i];
        const expectedAxis = testCase.joint_axes[i];
        expect(actualAxis[0]).toBeCloseTo(expectedAxis[0], 5);
        expect(actualAxis[1]).toBeCloseTo(expectedAxis[1], 5);
        expect(actualAxis[2]).toBeCloseTo(expectedAxis[2], 5);
      }
    }
  });

  it('should match geometric jacobian against python backend', () => {
    for (const testCase of referenceTests.cr6) {
      const jac = engine.geometricJacobian(testCase.q);
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) {
          expect(jac[r][c]).toBeCloseTo(testCase.jacobian[r][c], 5);
        }
      }
    }
  });

  it('should solve spherical wrist IK correctly', () => {
    for (let idx = 0; idx < referenceTests.cr6.length; idx++) {
      const testCase = referenceTests.cr6[idx];
      const ik = engine.solveSphericalWristIK(testCase.tcp_transform as Matrix4, testCase.q);
      expect(ik.success).toBe(true);
      for (let i = 0; i < 6; i++) {
        expect(ik.q[i]).toBeCloseTo(testCase.ik_q[i], 4);
      }
    }
  });

  it('should match inverse dynamics (gravity only) against python backend', () => {
    for (const testCase of referenceTests.cr6) {
      const tau = dynamicsEngine.inverseDynamics(testCase.q, Array(6).fill(0), Array(6).fill(0));
      for (let i = 0; i < 6; i++) {
        expect(tau[i]).toBeCloseTo(testCase.tau_gravity[i], 4);
      }
    }
  });

  it('should match inverse dynamics (full dynamic) against python backend', () => {
    for (const testCase of referenceTests.cr6) {
      const tau = dynamicsEngine.inverseDynamics(testCase.q, testCase.qd, testCase.qdd);
      for (let i = 0; i < 6; i++) {
        expect(tau[i]).toBeCloseTo(testCase.tau_dynamic[i], 4);
      }
    }
  });

  it('should round-trip inertial transforms between CAD and Link frames', () => {
    for (let i = 0; i < 6; i++) {
      const linkName = `LINK${i + 1}`;
      const originalCadSpec = spec.inertials[linkName];
      expect(originalCadSpec).toBeDefined();

      const linkSpec = engine._cadInertialToLinkInertial(i, originalCadSpec);
      expect(linkSpec.frame).toBe('link');

      const restoredCadSpec = engine._linkInertialToCadInertial(i, linkSpec);
      expect(restoredCadSpec.frame).toBe('cad');

      expect(restoredCadSpec.massKg).toBeCloseTo(originalCadSpec.massKg, 6);

      for (let j = 0; j < 3; j++) {
        expect(restoredCadSpec.comM![j]).toBeCloseTo(originalCadSpec.comM![j], 6);
      }

      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          expect(restoredCadSpec.inertiaKgM2![r][c]).toBeCloseTo(originalCadSpec.inertiaKgM2![r][c], 6);
        }
      }
    }
  });
});

describe('Palletizer (CR4) Kinematics Port', () => {
  const spec = irb460PalletizerSpec();
  const engine = new PalletizerEngine(spec);

  it('should match forward kinematics points against python template', () => {
    for (const testCase of referenceTests.cr4) {
      const fk = engine.forwardKinematics(testCase.q);
      
      for (const [key, expectedPt] of Object.entries(testCase.points)) {
        const actualPt = fk.points[key === 'EE' ? 'TCP' : key];
        expect(actualPt).toBeDefined();
        expect(actualPt[0]).toBeCloseTo(expectedPt[0], 5);
        expect(actualPt[1]).toBeCloseTo(expectedPt[1], 5);
        expect(actualPt[2]).toBeCloseTo(expectedPt[2], 5);
      }
    }
  });

  it('should solve planar numeric IK correctly', () => {
    for (const testCase of referenceTests.cr4) {
      const ik = engine.solveIK(testCase.target_xyz as Vector3, testCase.q, testCase.target_yaw);
      expect(ik.success).toBe(true);
      for (let i = 0; i < 4; i++) {
        expect(ik.q[i]).toBeCloseTo(testCase.q[i], 4);
      }
    }
  });

  it('should keep TCP as a rigid local offset from EE on the tilt link', () => {
    const q = [0.0, 0.25, -0.15, 0.0];
    const fk = engine.forwardKinematics(q);
    const home = spec.geometry as any;
    const expectedLength = Math.hypot(
      home.TCP[0] - home.EE[0],
      home.TCP[1] - home.EE[1],
      home.TCP[2] - home.EE[2]
    );
    const offset = [
      fk.points.TCP[0] - fk.points.EE[0],
      fk.points.TCP[1] - fk.points.EE[1],
      fk.points.TCP[2] - fk.points.EE[2]
    ];
    const actualLength = Math.hypot(offset[0], offset[1], offset[2]);

    expect(actualLength).toBeCloseTo(expectedLength, 6);
    expect(actualLength).toBeGreaterThan(1e-4);
  });
});
