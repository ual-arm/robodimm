export type Vector3 = [number, number, number];
export type Matrix3 = [Vector3, Vector3, Vector3];
export type Matrix4 = number[][]; // 4x4 array

export function createIdentity4(): Matrix4 {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

export function createZeroMatrix(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

export function multiply4x4(A: Matrix4, B: Matrix4): Matrix4 {
  const C = createZeroMatrix(4, 4);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
  return C;
}

export function multiplyMatrixAndVector(A: Matrix4, v: number[]): number[] {
  const result = new Array(A.length).fill(0);
  for (let i = 0; i < A.length; i++) {
    let sum = 0;
    for (let j = 0; j < v.length; j++) {
      sum += A[i][j] * v[j];
    }
    result[i] = sum;
  }
  return result;
}

export function multiply3x3(A: Matrix3, B: Matrix3): Matrix3 {
  const C: Matrix3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
  return C;
}

export function multiply3x3AndVector(A: Matrix3, v: Vector3): Vector3 {
  return [
    A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
    A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
    A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]
  ];
}

export function transpose3x3(A: Matrix3): Matrix3 {
  return [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]]
  ];
}

export function crossProduct(u: Vector3, v: Vector3): Vector3 {
  return [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0]
  ];
}

export function norm3(v: Vector3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

export function norm(v: number[]): number {
  return Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
}

export function unitVector(v: Vector3): Vector3 {
  const len = norm3(v);
  return len > 1e-12 ? [v[0] / len, v[1] / len, v[2] / len] : [1, 0, 0];
}

export function rotz(theta: number): Matrix4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [c, -s, 0, 0],
    [s, c, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

export function roty(theta: number): Matrix4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [c, 0, s, 0],
    [0, 1, 0, 0],
    [-s, 0, c, 0],
    [0, 0, 0, 1],
  ];
}

export function rotx(theta: number): Matrix4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [1, 0, 0, 0],
    [0, c, -s, 0],
    [0, s, c, 0],
    [0, 0, 0, 1]
  ];
}

export function translation(x: number, y: number, z: number): Matrix4 {
  return [
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1],
  ];
}

export function standardDH(
  a: number,
  alpha: number,
  d: number,
  theta: number
): Matrix4 {
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const ca = Math.cos(alpha);
  const sa = Math.sin(alpha);
  return [
    [ct, -st * ca, st * sa, a * ct],
    [st, ct * ca, -ct * sa, a * st],
    [0, sa, ca, d],
    [0, 0, 0, 1],
  ];
}

export function wrapToPi(angle: number): number {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

export function wrapNear(angle: number, reference: number): number {
  return reference + wrapToPi(angle - reference);
}

export function angleDifference(first: number[], second: number[]): number[] {
  return first.map((a, i) => wrapToPi(a - second[i]));
}

export function rotationError(first: Matrix3, second: Matrix3): number {
  let trace = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      trace += first[j][i] * second[j][i]; // trace of first.T @ second
    }
  }
  const val = (trace - 1.0) * 0.5;
  return Math.acos(Math.max(-1.0, Math.min(1.0, val)));
}

export function getMatrix3(T: Matrix4): Matrix3 {
  return [
    [T[0][0], T[0][1], T[0][2]],
    [T[1][0], T[1][1], T[1][2]],
    [T[2][0], T[2][1], T[2][2]]
  ];
}

export function getTranslation(T: Matrix4): Vector3 {
  return [T[0][3], T[1][3], T[2][3]];
}

export function addVectors(u: Vector3, v: Vector3): Vector3 {
  return [u[0] + v[0], u[1] + v[1], u[2] + v[2]];
}

export function subtractVectors(u: Vector3, v: Vector3): Vector3 {
  return [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
}

export function scaleVector(v: Vector3, scale: number): Vector3 {
  return [v[0] * scale, v[1] * scale, v[2] * scale];
}
