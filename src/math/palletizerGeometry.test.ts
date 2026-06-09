import { describe, it, expect, beforeEach } from 'vitest';
import { useRobodimmStore } from '../model/state';
import { irb460PalletizerSpec } from './palletizer';
import { irb4600Serial6Spec } from './serial6';
import {
  sanitizeCr4Geometry,
  validateCr4Geometry,
  cr4SegmentLength,
  designFromCr4Geometry,
  cr4GeometryFromDesign,
  validateCr4DesignParameters,
  getDefaultPalletizerBodyCOM
} from './palletizerGeometry';
import { Cr4Geometry } from '../model/schemas';

describe('CR4 Geometry & Validation Tests', () => {
  it('should sanitize geometries by forcing Y to 0.0 and deriving P and E', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    
    const distorted: Cr4Geometry = {
      ...defaultGeom,
      A: [0.1, 0.2, 0.3],
      P: [999.0, 999.0, 999.0],
      E: [-999.0, -999.0, -999.0]
    };
    
    const sanitized = sanitizeCr4Geometry(distorted);
    
    // All Y coordinates must be 0.0
    for (const key of Object.keys(sanitized) as (keyof Cr4Geometry)[]) {
      expect(sanitized[key][1]).toBe(0.0);
    }
    
    // Verify derivation of P: B + C - O
    const expectedP = [
      sanitized.B[0] + sanitized.C[0] - sanitized.O[0],
      0.0,
      sanitized.B[2] + sanitized.C[2] - sanitized.O[2]
    ];
    expect(sanitized.P).toEqual(expectedP);
    
    // Verify derivation of E: D + C - O
    const expectedE = [
      sanitized.D[0] + sanitized.C[0] - sanitized.O[0],
      0.0,
      sanitized.D[2] + sanitized.C[2] - sanitized.O[2]
    ];
    expect(sanitized.E).toEqual(expectedE);
  });

  it('should validate the default preset geometry without errors', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const res = validateCr4Geometry(defaultGeom);
    expect(res.ok).toBe(true);
    expect(res.issues.length).toBe(0);
  });

  it('should warn on non-zero Y input but still validate successfully', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const geomWithY: Cr4Geometry = {
      ...defaultGeom,
      A: [0.0, 0.05, 0.0]
    };
    const res = validateCr4Geometry(geomWithY);
    expect(res.ok).toBe(true);
    expect(res.issues.length).toBe(1);
    expect(res.issues[0].severity).toBe('warning');
    expect(res.issues[0].id).toBe('y_nonzero_A');
  });

  it('should fail validation on invalid critical lengths', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const invalidGeom: Cr4Geometry = {
      ...defaultGeom,
      C: [...defaultGeom.O] as [number, number, number]
    };
    const res = validateCr4Geometry(invalidGeom);
    expect(res.ok).toBe(false);
    expect(res.issues.some(i => i.severity === 'error' && i.id.includes('len_O_C'))).toBe(true);
  });

  it('should fail validation on non-finite coordinates', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const invalidGeom: Cr4Geometry = {
      ...defaultGeom,
      F: [NaN, 0.0, 1.88]
    };
    const res = validateCr4Geometry(invalidGeom);
    expect(res.ok).toBe(false);
    expect(res.issues.some(i => i.severity === 'error' && i.id.includes('finite_F'))).toBe(true);
  });

  it('should error on assembly branch flip when oriented area is non-positive', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const flipGeom: Cr4Geometry = {
      ...defaultGeom,
      G: [0.2, 0.0, 1.88]
    };
    const res = validateCr4Geometry(flipGeom);
    expect(res.issues.some(i => i.severity === 'error' && i.id === 'branch_flip_error')).toBe(true);
  });
});

describe('Store State Integration & Visual Segments', () => {
  beforeEach(() => {
    useRobodimmStore.setState({
      editRobot: irb460PalletizerSpec(),
      activeRobot: irb460PalletizerSpec(),
      isSet: false,
      cr4ValidationIssues: [],
      q: [0, 0, 0, 0]
    });
  });

  it('should verify updateCr4HardpointXZ recalculates P and E and does not mutate activeRobot', () => {
    const store = useRobodimmStore.getState();
    const originalActive = JSON.parse(JSON.stringify(store.activeRobot));

    // Update O
    store.updateCr4HardpointXZ('O', 0.5, 0.8);

    const updatedEdit = useRobodimmStore.getState().editRobot;
    const updatedActive = useRobodimmStore.getState().activeRobot;

    expect((updatedEdit.geometry as Cr4Geometry).O).toEqual([0.5, 0.0, 0.8]);

    const geom = updatedEdit.geometry as Cr4Geometry;
    const expectedP_X = geom.B[0] + geom.C[0] - 0.5;
    const expectedP_Z = geom.B[2] + geom.C[2] - 0.8;
    expect(geom.P[0]).toBeCloseTo(expectedP_X, 5);
    expect(geom.P[2]).toBeCloseTo(expectedP_Z, 5);

    expect(updatedActive).toEqual(originalActive);
  });

  it('should clear cr4ValidationIssues when changing robot kind, loading programs, or loading specs', () => {
    useRobodimmStore.setState({
      cr4ValidationIssues: [{ id: 'test_err', severity: 'error', message: 'Test error' }]
    });
    expect(useRobodimmStore.getState().cr4ValidationIssues.length).toBe(1);

    useRobodimmStore.getState().changeRobotKind('CR6');
    expect(useRobodimmStore.getState().cr4ValidationIssues.length).toBe(0);

    useRobodimmStore.setState({
      cr4ValidationIssues: [{ id: 'test_err', severity: 'error', message: 'Test error' }]
    });
    useRobodimmStore.getState().loadProgram({
      schema: 'robodimm.program.v1',
      name: 'prog',
      targets: [],
      instructions: []
    });
    expect(useRobodimmStore.getState().cr4ValidationIssues.length).toBe(0);

    useRobodimmStore.setState({
      cr4ValidationIssues: [{ id: 'test_err', severity: 'error', message: 'Test error' }]
    });
    useRobodimmStore.getState().loadRobotSpec(irb460PalletizerSpec());
    expect(useRobodimmStore.getState().cr4ValidationIssues.length).toBe(0);
  });

  it('should verify every visual with frameName of form seg_X_Y has valid points X and Y in Cr4Geometry', () => {
    const spec = irb460PalletizerSpec();
    const geomKeys = Object.keys(spec.geometry) as string[];

    spec.visuals.forEach((vis) => {
      const frameName = vis.frameName;
      if (frameName && frameName.startsWith('seg_')) {
        const parts = frameName.split('_');
        expect(parts.length).toBe(3);
        
        const ptX = parts[1];
        const ptY = parts[2];
        
        expect(geomKeys).toContain(ptX);
        expect(geomKeys).toContain(ptY);
      }
    });
  });

  it('should verify parameter round-trip for default IRB460 preset', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const params = designFromCr4Geometry(defaultGeom);
    const roundTripGeom = cr4GeometryFromDesign(params);
    
    for (const key of Object.keys(defaultGeom) as (keyof Cr4Geometry)[]) {
      expect(roundTripGeom[key][0]).toBeCloseTo(defaultGeom[key][0], 4);
      expect(roundTripGeom[key][1]).toBeCloseTo(defaultGeom[key][1], 4);
      expect(roundTripGeom[key][2]).toBeCloseTo(defaultGeom[key][2], 4);
    }
  });

  it('should verify that setting O = [0,0,0] via design parameters maintains horizontal crank and vertical lower arm', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const params = designFromCr4Geometry(defaultGeom);
    params.Ox = 0.0;
    params.Oz = 0.0;
    const derivedGeom = cr4GeometryFromDesign(params);
    
    expect(derivedGeom.O).toEqual([0.0, 0.0, 0.0]);
    expect(derivedGeom.B[2]).toBe(0.0);
    expect(derivedGeom.B[0]).toBe(-params.L_OB);
    expect(derivedGeom.C[0]).toBe(0.0);
    expect(derivedGeom.C[2]).toBe(params.L_OC);
  });

  it('should trigger validation errors for invalid design parameters', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const params = designFromCr4Geometry(defaultGeom);
    
    const badParamsNeg = { ...params, L_OB: -0.5 };
    const resNeg = validateCr4DesignParameters(badParamsNeg);
    expect(resNeg.ok).toBe(false);
    expect(resNeg.issues.some(i => i.severity === 'error' && i.id.includes('neg_L_OB'))).toBe(true);

    const badParamsSmall = { ...params, L_OB: 1e-5 };
    const resSmall = validateCr4DesignParameters(badParamsSmall);
    expect(resSmall.ok).toBe(false);
    expect(resSmall.issues.some(i => i.severity === 'error' && i.id.includes('len_L_OB'))).toBe(true);

    const badParamsCircles = { ...params, L_FG: 10.0 };
    const resCircles = validateCr4DesignParameters(badParamsCircles);
    expect(resCircles.ok).toBe(false);
    expect(resCircles.issues.some(i => i.severity === 'error' && (i.id.includes('triangle_ineq') || i.id.includes('branch_flip')))).toBe(true);
  });

  it('should verify store action updateCr4DesignParameter updates params, geometry, and issues', () => {
    const store = useRobodimmStore.getState();
    store.changeRobotKind('CR4');
    expect(useRobodimmStore.getState().editRobot.kind).toBe('CR4');
    expect(useRobodimmStore.getState().cr4DesignParams).not.toBeNull();
    
    const initialParams = useRobodimmStore.getState().cr4DesignParams!;
    const newL_OB = initialParams.L_OB + 0.1;
    
    store.updateCr4DesignParameter('L_OB', newL_OB);
    
    const updatedParams = useRobodimmStore.getState().cr4DesignParams!;
    const updatedGeom = useRobodimmStore.getState().editRobot.geometry as Cr4Geometry;
    
    expect(updatedParams.L_OB).toBe(newL_OB);
    expect(updatedGeom.B[0]).toBeCloseTo(updatedParams.Ox - newL_OB, 4);
    expect(useRobodimmStore.getState().isSet).toBe(false);
  });

  it('should verify that P and E are correctly derived after changing L_OB, L_OC, Ox, or Oz', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const params = designFromCr4Geometry(defaultGeom);
    
    params.L_OB = 0.3;
    params.L_OC = 1.2;
    params.Ox = 0.4;
    params.Oz = 0.9;
    
    const derivedGeom = cr4GeometryFromDesign(params);
    
    const expectedP = [
      derivedGeom.B[0] + derivedGeom.C[0] - derivedGeom.O[0],
      0.0,
      derivedGeom.B[2] + derivedGeom.C[2] - derivedGeom.O[2]
    ];
    expect(derivedGeom.P[0]).toBeCloseTo(expectedP[0], 5);
    expect(derivedGeom.P[2]).toBeCloseTo(expectedP[2], 5);
    
    const expectedE = [
      derivedGeom.D[0] + derivedGeom.C[0] - derivedGeom.O[0],
      0.0,
      derivedGeom.D[2] + derivedGeom.C[2] - derivedGeom.O[2]
    ];
    expect(derivedGeom.E[0]).toBeCloseTo(expectedE[0], 5);
    expect(derivedGeom.E[2]).toBeCloseTo(expectedE[2], 5);
  });

  it('should verify that G is solved at distance L_FG from F and L_HG from H', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const params = designFromCr4Geometry(defaultGeom);
    const derivedGeom = cr4GeometryFromDesign(params);
    
    const dist_FG = Math.hypot(derivedGeom.G[0] - derivedGeom.F[0], derivedGeom.G[2] - derivedGeom.F[2]);
    const dist_HG = Math.hypot(derivedGeom.G[0] - derivedGeom.H[0], derivedGeom.G[2] - derivedGeom.H[2]);
    
    expect(dist_FG).toBeCloseTo(params.L_FG, 4);
    expect(dist_HG).toBeCloseTo(params.L_HG, 4);
  });

  it('should verify tool mount and TCP coordinates align with L_HEE and L_TCP', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const params = designFromCr4Geometry(defaultGeom);
    const derivedGeom = cr4GeometryFromDesign(params);
    
    expect(derivedGeom.EE[0]).toBeCloseTo(derivedGeom.H[0] + params.L_HEE, 5);
    expect(derivedGeom.EE[2]).toBeCloseTo(derivedGeom.H[2], 5);
    expect(derivedGeom.TCP[0]).toBeCloseTo(derivedGeom.EE[0], 5);
    expect(derivedGeom.TCP[2]).toBeCloseTo(derivedGeom.EE[2] - params.L_TCP, 5);
  });

  it('should verify store action updateCr4DesignParameter does not mutate activeRobot and does not reset joint positions q', () => {
    const store = useRobodimmStore.getState();
    store.changeRobotKind('CR4');
    
    useRobodimmStore.setState({ q: [0.1, 0.2, 0.3, 0.4] });
    
    const originalActive = JSON.parse(JSON.stringify(useRobodimmStore.getState().activeRobot));
    const initialQ = [...useRobodimmStore.getState().q];
    
    store.updateCr4DesignParameter('L_OB', 0.25);
    
    const updatedActive = useRobodimmStore.getState().activeRobot;
    const updatedQ = useRobodimmStore.getState().q;
    
    expect(updatedActive).toEqual(originalActive);
    expect(updatedQ).toEqual(initialQ);
  });

  it('should trigger errors in validateCr4Geometry if strong canonical invariants are violated', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    
    const badCrank = { ...defaultGeom, B: [defaultGeom.B[0], 0.0, defaultGeom.B[2] + 0.1] as [number, number, number] };
    const resCrank = validateCr4Geometry(badCrank);
    expect(resCrank.ok).toBe(false);
    expect(resCrank.issues.some(i => i.id === 'invariant_crank_horizontal')).toBe(true);

    const badArm = { ...defaultGeom, C: [defaultGeom.C[0] + 0.1, 0.0, defaultGeom.C[2]] as [number, number, number] };
    const resArm = validateCr4Geometry(badArm);
    expect(resArm.ok).toBe(false);
    expect(resArm.issues.some(i => i.id === 'invariant_lower_arm_vertical')).toBe(true);
    
    const badP = { ...defaultGeom, P: [0.0, 0.0, 0.0] as [number, number, number] };
    const resP = validateCr4Geometry(badP);
    expect(resP.ok).toBe(false);
    expect(resP.issues.some(i => i.id === 'invariant_parallel_linkage')).toBe(true);

    const badEE = { ...defaultGeom, EE: [defaultGeom.EE[0], 0.0, defaultGeom.EE[2] + 0.1] as [number, number, number] };
    const resEE = validateCr4Geometry(badEE);
    expect(resEE.ok).toBe(false);
    expect(resEE.issues.some(i => i.id === 'invariant_tool_mount_horizontal')).toBe(true);

    const badTCP = { ...defaultGeom, TCP: [defaultGeom.TCP[0] + 0.1, 0.0, defaultGeom.TCP[2]] as [number, number, number] };
    const resTCP = validateCr4Geometry(badTCP);
    expect(resTCP.ok).toBe(false);
    expect(resTCP.issues.some(i => i.id === 'invariant_tool_tip_vertical')).toBe(true);
  });

  it('should verify validateCr4DesignParameters allows L_HEE and L_TCP to be 0', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const params = designFromCr4Geometry(defaultGeom);
    
    // Set L_HEE and L_TCP to 0
    params.L_HEE = 0.0;
    params.L_TCP = 0.0;
    
    const res = validateCr4DesignParameters(params);
    expect(res.ok).toBe(true);
    expect(res.issues.length).toBe(0);

    const derivedGeom = cr4GeometryFromDesign(params);
    const geomRes = validateCr4Geometry(derivedGeom);
    expect(geomRes.ok).toBe(true);
  });
});

describe('CR4 Default Body COM Fallbacks', () => {
  it('should resolve default COM coordinates for preset geometry correctly', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    
    // 1. Preset defaults check:
    // P_ARM = 0.20
    const pArmCOM = getDefaultPalletizerBodyCOM(defaultGeom, 'P_ARM');
    expect(pArmCOM[0]).toBeCloseTo(0.20, 5);
    expect(pArmCOM[1]).toBe(0.0);
    expect(pArmCOM[2]).toBe(0.0);

    // LOWER_ARM = 0.4725
    const lowerArmCOM = getDefaultPalletizerBodyCOM(defaultGeom, 'LOWER_ARM');
    expect(lowerArmCOM[0]).toBeCloseTo(0.4725, 5);
    expect(lowerArmCOM[1]).toBe(0.0);
    expect(lowerArmCOM[2]).toBe(0.0);

    // UPPER_ARM = 0.3125
    const upperArmCOM = getDefaultPalletizerBodyCOM(defaultGeom, 'UPPER_ARM');
    expect(upperArmCOM[0]).toBeCloseTo(0.3125, 5);
    expect(upperArmCOM[1]).toBe(0.0);
    expect(upperArmCOM[2]).toBe(0.0);

    // LINK_PLATE = centroid of C-E-F triangle in seg_C_E frame
    const linkPlateCOM = getDefaultPalletizerBodyCOM(defaultGeom, 'LINK_PLATE');
    expect(linkPlateCOM[0]).toBeCloseTo(0.0625102, 5);
    expect(linkPlateCOM[1]).toBe(0.0);
    expect(linkPlateCOM[2]).toBeCloseTo(-0.0927347, 5);

    // TILT = centroid of H-G-J4 triangle in seg_H_G frame with origin at G
    const tiltCOM = getDefaultPalletizerBodyCOM(defaultGeom, 'TILT');
    expect(tiltCOM[0]).toBeCloseTo(-0.1891415, 5);
    expect(tiltCOM[1]).toBe(0.0);
    expect(tiltCOM[2]).toBeCloseTo(-0.1011441, 5);

    // SWING = [0.18, 0, 0.25]
    const swingCOM = getDefaultPalletizerBodyCOM(defaultGeom, 'SWING');
    expect(swingCOM).toEqual([0.18, 0, 0.25]);

    // DISK = [0, 0, 0]
    const diskCOM = getDefaultPalletizerBodyCOM(defaultGeom, 'DISK');
    expect(diskCOM).toEqual([0, 0, 0]);
  });

  it('should derive UPPER_ARM expected COM coordinates algebraically from active geometry segment lengths', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;

    // L_CH = |H - C|
    const L_CH = Math.hypot(defaultGeom.H[0] - defaultGeom.C[0], defaultGeom.H[2] - defaultGeom.C[2]);
    // L_PC = |C - P|
    const L_PC = Math.hypot(defaultGeom.C[0] - defaultGeom.P[0], defaultGeom.C[2] - defaultGeom.P[2]);
    // Expected midpoint: (L_CH - L_PC) / 2
    const expectedX = (L_CH - L_PC) / 2;

    const upperArmCOM = getDefaultPalletizerBodyCOM(defaultGeom, 'UPPER_ARM');
    expect(upperArmCOM[0]).toBeCloseTo(expectedX, 5);
    expect(expectedX).toBeCloseTo(0.3125, 5); // Verify actual preset value matches 0.3125
  });

  it('should dynamically shift effective COM values when geometry is modified', () => {
    const defaultGeom = irb460PalletizerSpec().geometry as Cr4Geometry;
    const params = designFromCr4Geometry(defaultGeom);

    // Modify lower arm and extension length parameters
    params.L_OC = 1.5; // changes LOWER_ARM length
    params.L_CH = 1.2; // changes UPPER_ARM extension length

    const modifiedGeom = cr4GeometryFromDesign(params);

    // 1. LOWER_ARM midpoint should now reflect L_OC / 2 = 0.75
    const lowerArmCOM = getDefaultPalletizerBodyCOM(modifiedGeom, 'LOWER_ARM');
    expect(lowerArmCOM[0]).toBeCloseTo(0.75, 5);

    // 2. UPPER_ARM should reflect modified (L_CH - L_PC) / 2
    // Note: L_PC = L_OB (which is 0.40)
    const expectedUpperArmX = (1.2 - 0.40) / 2; // 0.40
    const upperArmCOM = getDefaultPalletizerBodyCOM(modifiedGeom, 'UPPER_ARM');
    expect(upperArmCOM[0]).toBeCloseTo(expectedUpperArmX, 5);

    // 3. TILT should reflect modified H-G-J4 wrist triangle centroid
    params.L_HG = 0.5;
    const modifiedGeom2 = cr4GeometryFromDesign(params);
    const tiltCOM = getDefaultPalletizerBodyCOM(modifiedGeom2, 'TILT');
    const hgX = modifiedGeom2.G[0] - modifiedGeom2.H[0];
    const hgZ = modifiedGeom2.G[2] - modifiedGeom2.H[2];
    const hgLen = Math.hypot(hgX, hgZ);
    const xAxis: [number, number] = [hgX / hgLen, hgZ / hgLen];
    const zAxis: [number, number] = [-xAxis[1], xAxis[0]];
    const j4Diff: [number, number] = [modifiedGeom2.J4[0] - modifiedGeom2.H[0], modifiedGeom2.J4[2] - modifiedGeom2.H[2]];
    const j4LocalX = j4Diff[0] * xAxis[0] + j4Diff[1] * xAxis[1];
    const j4LocalZ = j4Diff[0] * zAxis[0] + j4Diff[1] * zAxis[1];
    expect(tiltCOM[0]).toBeCloseTo((-hgLen + j4LocalX - hgLen) / 3, 5);
    expect(tiltCOM[2]).toBeCloseTo(j4LocalZ / 3, 5);
  });
});
