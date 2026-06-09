import { Cr4Geometry } from '../model/schemas';

export interface Cr4GeometryValidationIssue {
  id: string;
  severity: 'error' | 'warning';
  message: string;
  field?: string;
}

export interface Cr4GeometryValidationResult {
  ok: boolean;
  issues: Cr4GeometryValidationIssue[];
}

export interface Cr4DesignParameters {
  Ox: number;
  Oz: number;
  L_OB: number;
  L_OC: number;
  L_CH: number;
  L_HEE: number;
  L_TCP: number;
  // Advanced parameters (Linkage details)
  D_offset_x: number;
  D_offset_z: number;
  F_offset_x: number;
  F_offset_z: number;
  L_FG: number;
  L_HG: number;
  J4_offset_x: number;
  J4_offset_z: number;
}

export function sanitizeCr4Geometry(input: Cr4Geometry): Cr4Geometry {
  // Deep clone
  const sanitized = JSON.parse(JSON.stringify(input)) as Cr4Geometry;

  // Force all Y to 0.0, and clean near-zero values to 0.0
  const keys = Object.keys(sanitized) as (keyof Cr4Geometry)[];
  for (const key of keys) {
    sanitized[key][1] = 0.0;
    for (let i = 0; i < 3; i++) {
      if (Math.abs(sanitized[key][i]) < 1e-12) {
        sanitized[key][i] = 0.0;
      }
    }
  }

  // Recalculate dependent points P and E to preserve parallel linkages:
  // P = B + C - O
  sanitized.P[0] = sanitized.B[0] + sanitized.C[0] - sanitized.O[0];
  sanitized.P[1] = 0.0;
  sanitized.P[2] = sanitized.B[2] + sanitized.C[2] - sanitized.O[2];

  // E = D + C - O
  sanitized.E[0] = sanitized.D[0] + sanitized.C[0] - sanitized.O[0];
  sanitized.E[1] = 0.0;
  sanitized.E[2] = sanitized.D[2] + sanitized.C[2] - sanitized.O[2];

  // Clean near-zero again on recalculated coordinates
  for (const key of ['P', 'E'] as (keyof Cr4Geometry)[]) {
    for (let i = 0; i < 3; i++) {
      if (Math.abs(sanitized[key][i]) < 1e-12) {
        sanitized[key][i] = 0.0;
      }
    }
  }

  return sanitized;
}

export function cr4SegmentLength(geom: Cr4Geometry, first: keyof Cr4Geometry, second: keyof Cr4Geometry): number {
  const p1 = geom[first];
  const p2 = geom[second];
  return Math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
}

/**
 * Solve for point G in plane based on F, H and lengths L_FG, L_HG
 * choosing the branch with positive oriented area area(F, H, G) > 0.
 */
export function solveGFromFH(
  F: [number, number, number],
  r_F: number,
  H: [number, number, number],
  r_H: number
): { point: [number, number, number]; orientedArea: number } {
  const dx = H[0] - F[0];
  const dz = H[2] - F[2];
  const dist = Math.hypot(dx, dz);
  if (dist <= 1e-12) {
    return { point: [F[0] + r_F, 0.0, F[2]], orientedArea: 0.0 };
  }
  const a = (r_F * r_F - r_H * r_H + dist * dist) / (2.0 * dist);
  const hSq = r_F * r_F - a * a;
  const h = Math.sqrt(Math.max(hSq, 0.0));

  const ex = dx / dist;
  const ez = dz / dist;
  const base_x = F[0] + a * ex;
  const base_z = F[2] + a * ez;

  // Perp direction [-ez, ex]
  const cand1_x = base_x + h * (-ez);
  const cand1_z = base_z + h * ex;
  const cand2_x = base_x - h * (-ez);
  const cand2_z = base_z - h * ex;

  const area1 = dx * (cand1_z - F[2]) - dz * (cand1_x - F[0]);
  if (area1 >= 0.0) {
    return { point: [cand1_x, 0.0, cand1_z], orientedArea: area1 };
  } else {
    const area2 = dx * (cand2_z - F[2]) - dz * (cand2_x - F[0]);
    return { point: [cand2_x, 0.0, cand2_z], orientedArea: area2 };
  }
}

export function designFromCr4Geometry(geom: Cr4Geometry): Cr4DesignParameters {
  const Ox = geom.O[0];
  const Oz = geom.O[2];
  const L_OB = geom.O[0] - geom.B[0];
  const L_OC = geom.C[2] - geom.O[2];
  const L_CH = geom.H[0] - geom.C[0];
  const L_HEE = geom.EE[0] - geom.H[0];
  const L_TCP = geom.EE[2] - geom.TCP[2];

  const D_offset_x = geom.D[0] - geom.O[0];
  const D_offset_z = geom.D[2] - geom.O[2];
  const F_offset_x = geom.F[0] - geom.C[0];
  const F_offset_z = geom.F[2] - geom.C[2];

  const L_FG = Math.hypot(geom.G[0] - geom.F[0], geom.G[2] - geom.F[2]);
  const L_HG = Math.hypot(geom.G[0] - geom.H[0], geom.G[2] - geom.H[2]);

  const J4_offset_x = geom.J4[0] - geom.H[0];
  const J4_offset_z = geom.J4[2] - geom.H[2];

  return {
    Ox,
    Oz,
    L_OB,
    L_OC,
    L_CH,
    L_HEE,
    L_TCP,
    D_offset_x,
    D_offset_z,
    F_offset_x,
    F_offset_z,
    L_FG,
    L_HG,
    J4_offset_x,
    J4_offset_z
  };
}

export function cr4GeometryFromDesign(params: Cr4DesignParameters): Cr4Geometry {
  const {
    Ox,
    Oz,
    L_OB,
    L_OC,
    L_CH,
    L_HEE,
    L_TCP,
    D_offset_x,
    D_offset_z,
    F_offset_x,
    F_offset_z,
    L_FG,
    L_HG,
    J4_offset_x,
    J4_offset_z
  } = params;

  const A: [number, number, number] = [0.0, 0.0, 0.0];
  const O: [number, number, number] = [Ox, 0.0, Oz];
  const B: [number, number, number] = [Ox - L_OB, 0.0, Oz];
  const C: [number, number, number] = [Ox, 0.0, Oz + L_OC];

  // P = B + C - O
  const P: [number, number, number] = [
    B[0] + C[0] - O[0],
    0.0,
    B[2] + C[2] - O[2]
  ];

  const D: [number, number, number] = [Ox + D_offset_x, 0.0, Oz + D_offset_z];
  
  // E = D + C - O
  const E: [number, number, number] = [
    D[0] + C[0] - O[0],
    0.0,
    D[2] + C[2] - O[2]
  ];

  // H = C + [L_CH, 0, 0]
  const H: [number, number, number] = [C[0] + L_CH, 0.0, C[2]];

  // F = C + F_offset
  const F: [number, number, number] = [C[0] + F_offset_x, 0.0, C[2] + F_offset_z];

  // G solved by F, H, L_FG, L_HG circle intersection
  const solveG = solveGFromFH(F, L_FG, H, L_HG);
  const G = solveG.point;

  // J4 = H + J4_offset
  const J4: [number, number, number] = [H[0] + J4_offset_x, 0.0, H[2] + J4_offset_z];

  // EE = H + [L_HEE, 0, 0] (horizontal tool mount)
  const EE: [number, number, number] = [H[0] + L_HEE, 0.0, H[2]];

  // TCP = EE - [0, 0, L_TCP] (vertical tool tip)
  const TCP: [number, number, number] = [EE[0], 0.0, EE[2] - L_TCP];

  return { A, O, B, C, D, E, F, G, H, P, J4, EE, TCP };
}

export function validateCr4DesignParameters(params: Cr4DesignParameters): Cr4GeometryValidationResult {
  const issues: Cr4GeometryValidationIssue[] = [];

  // 1. Check finiteness
  const keys = Object.keys(params) as (keyof Cr4DesignParameters)[];
  let hasInfinities = false;
  for (const key of keys) {
    const val = params[key];
    if (!Number.isFinite(val)) {
      hasInfinities = true;
      issues.push({
        id: `finite_${key}`,
        severity: 'error',
        message: `Design parameter ${key} is non-finite or invalid.`,
        field: key
      });
    }
  }

  if (hasInfinities) {
    return { ok: false, issues };
  }

  // 2. Validate lengths are strictly positive (> 1e-4)
  const lengthParams: (keyof Cr4DesignParameters)[] = ['L_OB', 'L_OC', 'L_CH', 'L_FG', 'L_HG'];
  for (const key of lengthParams) {
    const val = params[key];
    if (val <= 0.0) {
      issues.push({
        id: `neg_${key}`,
        severity: 'error',
        message: `Parameter ${key} must be strictly positive (is ${val.toFixed(4)}m).`,
        field: key
      });
    } else if (val <= 1e-4) {
      issues.push({
        id: `len_${key}`,
        severity: 'error',
        message: `Parameter ${key} is too small: ${val.toFixed(6)}m (minimum 0.0001m).`,
        field: key
      });
    }
  }

  // 3. Compute derived positions to check assembly linkage circle intersection
  const Ox = params.Ox;
  const Oz = params.Oz;
  const C: [number, number, number] = [Ox, 0.0, Oz + params.L_OC];
  const H: [number, number, number] = [C[0] + params.L_CH, 0.0, C[2]];
  const F: [number, number, number] = [C[0] + params.F_offset_x, 0.0, C[2] + params.F_offset_z];

  const dist_FH = Math.hypot(H[0] - F[0], H[2] - F[2]);
  const L_FG = params.L_FG;
  const L_HG = params.L_HG;

  if (dist_FH > L_FG + L_HG + 1e-9) {
    issues.push({
      id: `triangle_ineq_sum`,
      severity: 'error',
      message: `Assembly linkage circles do not intersect: distance between F and H (${dist_FH.toFixed(4)}m) is greater than sum of radii FG + HG (${(L_FG + L_HG).toFixed(4)}m).`
    });
  }

  if (dist_FH < Math.abs(L_FG - L_HG) - 1e-9) {
    issues.push({
      id: `triangle_ineq_diff`,
      severity: 'error',
      message: `Assembly linkage circles do not intersect: distance between F and H (${dist_FH.toFixed(4)}m) is less than absolute difference of radii |FG - HG| (${Math.abs(L_FG - L_HG).toFixed(4)}m).`
    });
  }

  // Check branch flip / tangent singularity
  if (dist_FH > 0.0 && dist_FH <= L_FG + L_HG) {
    const solveG = solveGFromFH(F, L_FG, H, L_HG);
    const orientedArea = solveG.orientedArea;
    const h = Math.abs(orientedArea) / dist_FH;
    if (orientedArea <= 1e-12) {
      issues.push({
        id: `branch_flip_error`,
        severity: 'error',
        message: `Mechanism assembly branch flip detected (oriented area is non-positive or zero).`
      });
    } else if (h <= 1e-3) {
      issues.push({
        id: `tangent_singularity_error`,
        severity: 'error',
        message: `Mechanism tangent singularity detected (height G-to-FH is too small: ${h.toFixed(6)}m, minimum 0.001m).`
      });
    }
  }

  // L_HEE and L_TCP direction checks
  if (params.L_HEE < 0) {
    issues.push({
      id: `lhee_direction_error`,
      severity: 'error',
      message: `Parameter L_HEE must be non-negative (can be 0 if the flange is directly at wrist H).`,
      field: 'L_HEE'
    });
  }
  if (params.L_TCP < 0) {
    issues.push({
      id: `ltcp_direction_error`,
      severity: 'error',
      message: `Parameter L_TCP must be non-negative (can be 0 if TCP is at the flange face).`,
      field: 'L_TCP'
    });
  }

  const ok = !issues.some(issue => issue.severity === 'error');
  return { ok, issues };
}

export function validateCr4Geometry(originalInput: Cr4Geometry): Cr4GeometryValidationResult {
  const issues: Cr4GeometryValidationIssue[] = [];

  // 1. Check finiteness of all coordinates in original input
  const keys = Object.keys(originalInput) as (keyof Cr4Geometry)[];
  let hasInfinities = false;
  for (const key of keys) {
    const pt = originalInput[key];
    if (!Array.isArray(pt) || pt.length !== 3 || !Number.isFinite(pt[0]) || !Number.isFinite(pt[1]) || !Number.isFinite(pt[2])) {
      hasInfinities = true;
      issues.push({
        id: `finite_${key}`,
        severity: 'error',
        message: `Hardpoint ${key} has non-finite or invalid coordinates.`,
        field: key
      });
    }
  }

  if (hasInfinities) {
    return { ok: false, issues };
  }

  // 2. Warn if any Y coordinate in originalInput is non-zero
  for (const key of keys) {
    const originalY = originalInput[key][1];
    if (Math.abs(originalY) > 1e-12) {
      issues.push({
        id: `y_nonzero_${key}`,
        severity: 'warning',
        message: `Hardpoint ${key} has non-zero Y coordinate (${originalY}m). It will be normalized to 0.0 in plane.`,
        field: key
      });
    }
  }

  // Sanitize for the geometric checks
  const sanitized = sanitizeCr4Geometry(originalInput);

  // 3. Validate strong canonical invariants
  if (Math.abs(originalInput.B[2] - originalInput.O[2]) > 1e-5) {
    issues.push({
      id: 'invariant_crank_horizontal',
      severity: 'error',
      message: `Crank B must be horizontal relative to Pivot O (B.z is ${originalInput.B[2].toFixed(4)}m, expected ${originalInput.O[2].toFixed(4)}m).`,
      field: 'B'
    });
  }
  if (Math.abs(originalInput.C[0] - originalInput.O[0]) > 1e-5) {
    issues.push({
      id: 'invariant_lower_arm_vertical',
      severity: 'error',
      message: `Lower Arm C must be vertical relative to Pivot O (C.x is ${originalInput.C[0].toFixed(4)}m, expected ${originalInput.O[0].toFixed(4)}m).`,
      field: 'C'
    });
  }
  const expectedP_x = originalInput.B[0] + originalInput.C[0] - originalInput.O[0];
  const expectedP_z = originalInput.B[2] + originalInput.C[2] - originalInput.O[2];
  if (Math.abs(originalInput.P[0] - expectedP_x) > 1e-5 || Math.abs(originalInput.P[2] - expectedP_z) > 1e-5) {
    issues.push({
      id: 'invariant_parallel_linkage',
      severity: 'error',
      message: `Point P must preserve parallel linkage P = B + C - O (P is [${originalInput.P[0].toFixed(4)}, ${originalInput.P[2].toFixed(4)}], expected [${expectedP_x.toFixed(4)}, ${expectedP_z.toFixed(4)}]).`,
      field: 'P'
    });
  }
  const expectedE_x = originalInput.D[0] + originalInput.C[0] - originalInput.O[0];
  const expectedE_z = originalInput.D[2] + originalInput.C[2] - originalInput.O[2];
  if (Math.abs(originalInput.E[0] - expectedE_x) > 1e-5 || Math.abs(originalInput.E[2] - expectedE_z) > 1e-5) {
    issues.push({
      id: 'invariant_lower_parallelogram',
      severity: 'error',
      message: `Point E must preserve parallel linkage E = D + C - O (E is [${originalInput.E[0].toFixed(4)}, ${originalInput.E[2].toFixed(4)}], expected [${expectedE_x.toFixed(4)}, ${expectedE_z.toFixed(4)}]).`,
      field: 'E'
    });
  }
  if (Math.abs(originalInput.H[2] - originalInput.C[2]) > 1e-5) {
    issues.push({
      id: 'invariant_extension_horizontal',
      severity: 'error',
      message: `Extension H must be horizontal relative to C (H.z is ${originalInput.H[2].toFixed(4)}m, expected ${originalInput.C[2].toFixed(4)}m).`,
      field: 'H'
    });
  }
  if (Math.abs(originalInput.EE[2] - originalInput.H[2]) > 1e-5) {
    issues.push({
      id: 'invariant_tool_mount_horizontal',
      severity: 'error',
      message: `Tool Mount EE must be horizontally aligned with extension H (EE.z is ${originalInput.EE[2].toFixed(4)}m, expected ${originalInput.H[2].toFixed(4)}m).`,
      field: 'EE'
    });
  }
  if (Math.abs(originalInput.TCP[0] - originalInput.EE[0]) > 1e-5) {
    issues.push({
      id: 'invariant_tool_tip_vertical',
      severity: 'error',
      message: `Tool Tip TCP must be vertically aligned with mount EE (TCP.x is ${originalInput.TCP[0].toFixed(4)}m, expected ${originalInput.EE[0].toFixed(4)}m).`,
      field: 'TCP'
    });
  }
  if (originalInput.TCP[2] > originalInput.EE[2] + 1e-5) {
    issues.push({
      id: 'invariant_tool_tip_pointing_down',
      severity: 'error',
      message: `Tool Tip TCP must point downwards relative to mount EE (TCP.z is ${originalInput.TCP[2].toFixed(4)}m, must be less than or equal to EE.z ${originalInput.EE[2].toFixed(4)}m).`,
      field: 'TCP'
    });
  }
  if (Math.abs(originalInput.EE[1]) > 1e-5 || Math.abs(originalInput.TCP[1]) > 1e-5) {
    issues.push({
      id: 'invariant_planar_y_zero',
      severity: 'error',
      message: `Tool mount and tip must reside in XZ plane (EE.y is ${originalInput.EE[1]}m, TCP.y is ${originalInput.TCP[1]}m, must be 0.0).`
    });
  }

  // 4. Asserts that critical loop linkage lengths are larger than epsilon = 1e-4 meters.
  const criticalPairs: [keyof Cr4Geometry, keyof Cr4Geometry][] = [
    ['O', 'C'],
    ['O', 'B'],
    ['P', 'H'],
    ['F', 'G'],
    ['H', 'G']
  ];

  for (const [first, second] of criticalPairs) {
    const len = cr4SegmentLength(sanitized, first, second);
    if (len <= 1e-4) {
      issues.push({
        id: `len_${first}_${second}`,
        severity: 'error',
        message: `Critical segment length between ${first} and ${second} is too small: ${len.toFixed(6)}m (minimum 0.0001m).`
      });
    }
  }

  // 4. Validate that the triangle inequality for assembly circle intersection holds:
  // |FG| + |HG| >= |FH| and ||FG| - |HG|| <= |FH|
  const dist_FG = cr4SegmentLength(sanitized, 'F', 'G');
  const dist_HG = cr4SegmentLength(sanitized, 'H', 'G');
  const dist_FH = cr4SegmentLength(sanitized, 'F', 'H');

  if (dist_FH > dist_FG + dist_HG + 1e-9) {
    issues.push({
      id: `triangle_ineq_sum`,
      severity: 'error',
      message: `Assembly linkage circles do not intersect: distance between F and H (${dist_FH.toFixed(4)}m) is greater than sum of radii FG + HG (${(dist_FG + dist_HG).toFixed(4)}m).`
    });
  }

  if (dist_FH < Math.abs(dist_FG - dist_HG) - 1e-9) {
    issues.push({
      id: `triangle_ineq_diff`,
      severity: 'error',
      message: `Assembly linkage circles do not intersect: distance between F and H (${dist_FH.toFixed(4)}m) is less than absolute difference of radii |FG - HG| (${Math.abs(dist_FG - dist_HG).toFixed(4)}m).`
    });
  }

  // 5. Detect and error about mechanism assembly branch flips and tangent singularities
  const orientedArea = (sanitized.H[0] - sanitized.F[0]) * (sanitized.G[2] - sanitized.F[2]) - (sanitized.H[2] - sanitized.F[2]) * (sanitized.G[0] - sanitized.F[0]);
  const h = dist_FH > 1e-12 ? Math.abs(orientedArea) / dist_FH : 0.0;

  if (orientedArea <= 1e-12) {
    issues.push({
      id: `branch_flip_error`,
      severity: 'error',
      message: `Mechanism assembly branch flip detected. Oriented area of F, H, G is non-positive (${orientedArea.toFixed(6)}).`
    });
  } else if (h <= 1e-3) {
    issues.push({
      id: `tangent_singularity_error`,
      severity: 'error',
      message: `Mechanism tangent singularity detected. Height of G relative to FH line is too small (${h.toFixed(6)}m, minimum 0.001m).`
    });
  }

  const ok = !issues.some(issue => issue.severity === 'error');

  return { ok, issues };
}

export type Cr4BodyName =
  | 'SWING'
  | 'P_ARM'
  | 'LOWER_ARM'
  | 'P_LINK'
  | 'UPPER_ARM'
  | 'LOWER_LINK'
  | 'LINK_PLATE'
  | 'UPPER_LINK'
  | 'TILT'
  | 'DISK';

export function getDefaultPalletizerBodyCOM(geom: Cr4Geometry, body: Cr4BodyName | string): [number, number, number] {
  if (!geom || !geom.O || !geom.C) return [0, 0, 0];

  switch (body) {
    case 'SWING':
      return [0.18, 0, 0.25];
    case 'P_ARM': {
      const L = Math.hypot(geom.B[0] - geom.O[0], geom.B[2] - geom.O[2]);
      return [L / 2, 0, 0];
    }
    case 'LOWER_ARM': {
      const L = Math.hypot(geom.C[0] - geom.O[0], geom.C[2] - geom.O[2]);
      return [L / 2, 0, 0];
    }
    case 'P_LINK': {
      const L = Math.hypot(geom.P[0] - geom.B[0], geom.P[2] - geom.B[2]);
      return [L / 2, 0, 0];
    }
    case 'UPPER_ARM': {
      // UPPER_ARM frame is seg_C_H (origin at C, X points to H)
      // The physical arm runs from P (distal-back) to H (proximal-front)
      // Distance from P to C is |C - P|, which is |O - B| (due to parallel constraint).
      // Total length from P to H is |H - P|
      // Center of mass of the segment PH relative to C is:
      // COM_x = (L_CH - L_PC) / 2
      const L_CH = Math.hypot(geom.H[0] - geom.C[0], geom.H[2] - geom.C[2]);
      const L_PC = Math.hypot(geom.C[0] - geom.P[0], geom.C[2] - geom.P[2]);
      return [(L_CH - L_PC) / 2, 0, 0];
    }
    case 'LOWER_LINK': {
      const L = Math.hypot(geom.E[0] - geom.D[0], geom.E[2] - geom.D[2]);
      return [L / 2, 0, 0];
    }
    case 'LINK_PLATE': {
      const localE = localPointInSegmentFrame(geom, 'E', 'C', 'E');
      const localF = localPointInSegmentFrame(geom, 'F', 'C', 'E');
      return [
        (localE[0] + localF[0]) / 3,
        (localE[1] + localF[1]) / 3,
        (localE[2] + localF[2]) / 3
      ];
    }
    case 'UPPER_LINK': {
      const L = Math.hypot(geom.G[0] - geom.F[0], geom.G[2] - geom.F[2]);
      return [L / 2, 0, 0];
    }
    case 'TILT': {
      // TILT frame is aligned with seg_H_G but origin is at G. Use the
      // triangular wrist plate centroid H-G-J4, not the H-G edge midpoint.
      const localH = localPointInSegmentFrame(geom, 'H', 'H', 'G');
      const localG = localPointInSegmentFrame(geom, 'G', 'H', 'G');
      const localJ4 = localPointInSegmentFrame(geom, 'J4', 'H', 'G');
      return [
        (localH[0] + localG[0] + localJ4[0]) / 3 - localG[0],
        (localH[1] + localG[1] + localJ4[1]) / 3 - localG[1],
        (localH[2] + localG[2] + localJ4[2]) / 3 - localG[2]
      ];
    }
    case 'DISK':
    default:
      return [0, 0, 0];
  }
}

function localPointInSegmentFrame(
  geom: Cr4Geometry,
  pointName: keyof Cr4Geometry,
  originName: keyof Cr4Geometry,
  xTargetName: keyof Cr4Geometry
): [number, number, number] {
  const point = geom[pointName];
  const origin = geom[originName];
  const target = geom[xTargetName];
  const dx = target[0] - origin[0];
  const dy = target[1] - origin[1];
  const dz = target[2] - origin[2];
  const len = Math.hypot(dx, dy, dz);
  const xAxis: [number, number, number] = len > 1e-12 ? [dx / len, dy / len, dz / len] : [1, 0, 0];
  const yAxis: [number, number, number] = [0, 1, 0];
  const zAxis = unit(cross(xAxis, yAxis));
  const yOrtho = cross(zAxis, xAxis);
  const diff: [number, number, number] = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]];
  return [dot(diff, xAxis), dot(diff, yOrtho), dot(diff, zAxis)];
}

function dot(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

function unit(v: [number, number, number]): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]);
  return len > 1e-12 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, 1];
}
