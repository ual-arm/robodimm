import * as THREE from 'three';
import { Matrix4 } from '../math/matrix';

export function getCadAlignedTransform(T: Matrix4, T_home: Matrix4): Matrix4 {
  const C = T.map(row => [...row]);
  for (let r = 0; r < 3; r++) {
    const row = [T[r][0], T[r][1], T[r][2]];
    for (let c = 0; c < 3; c++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += row[k] * T_home[c][k];
      }
      C[r][c] = sum;
    }
  }
  return C;
}

export function updateFrameHelper(
  helper: THREE.AxesHelper,
  pos: THREE.Vector3,
  rot: THREE.Quaternion,
  visible: boolean
): void {
  helper.visible = visible;
  helper.position.copy(pos);
  helper.quaternion.copy(rot);
}

export function updateCOMMarker(
  marker: THREE.Mesh,
  comPos: [number, number, number] | undefined,
  visible: boolean
): void {
  marker.visible = visible && !!comPos;
  if (comPos) {
    marker.position.set(comPos[0], comPos[1], comPos[2]);
  }
}
