import * as THREE from 'three';
import { RobotSpec, VisualSpec } from '../model/schemas';

export function buildVisualPrimitive(
  vis: VisualSpec,
  robot: RobotSpec
): THREE.Mesh | null {
  if (vis.kind !== 'primitive' || !vis.primitive) return null;
  const prim = vis.primitive;

  let radius = prim.dimensions[0];
  let length = prim.dimensions[1];
  let localOrigin = [...vis.originM];
  let localRpy = [...vis.rpyRad];

  if (robot.kind === 'CR6') {
    const geom6 = robot.geometry as any;
    const joints = geom6.joints || [];
    const j1 = joints.find((j: any) => j.name === 'J1');
    const j2 = joints.find((j: any) => j.name === 'J2');
    const j3 = joints.find((j: any) => j.name === 'J3');
    const j4 = joints.find((j: any) => j.name === 'J4');
    const j6 = joints.find((j: any) => j.name === 'J6');

    if (vis.body === 'LINK1' && j1) {
      length = Math.abs(j1.d_m);
      localOrigin = [j1.a_m, 0, j1.d_m / 2];
      localRpy = [0, 0, 0];
    } else if (vis.body === 'BASE' && j1) {
      if (Math.abs(j1.a_m) <= 1e-6 && Math.abs(j1.d_m) <= 1e-6) return null;
      length = prim.dimensions[1];
      radius = Math.max(0.06, Math.abs(j1.a_m) + 0.06);
    } else if (vis.body === 'LINK2' && j2) {
      length = j2.a_m;
      localOrigin = [0, 0, length / 2];
    } else if (vis.body === 'LINK3' && j3) {
      length = j3.a_m;
      localOrigin = [0, 0, length / 2];
      localRpy = [0, 0, 0];
    } else if (vis.body === 'LINK4' && j4) {
      length = j4.d_m;
      localOrigin = [length / 2, 0, 0];
    } else if (vis.body === 'LINK6' && j6) {
      length = j6.d_m;
      localOrigin = [length / 2, 0, 0];
    }
  } else if (robot.kind === 'CR4') {
    const geom4 = robot.geometry as any;
    const tKey = vis.frameName || vis.body;
    if (vis.body === 'SWING' && tKey === 'SWING') {
      const height = Math.abs(prim.dimensions[1]);
      if (Math.abs(geom4.O[0]) <= 1e-6 && Math.abs(geom4.O[2]) <= 1e-6) return null;
      if (height < 0.2) {
        length = height;
        radius = Math.max(0.06, Math.abs(geom4.O[0]) + 0.06);
        localOrigin = [0, 0, height / 2];
      } else {
        length = Math.abs(geom4.O[2]);
        localOrigin = [geom4.O[0], 0, geom4.O[2] / 2];
      }
      localRpy = [0, 0, 0];
    } else if (tKey.startsWith('seg_')) {
      const parts = tKey.split('_');
      const first = parts[1];
      const second = parts[2];
      if (geom4[first] && geom4[second]) {
        length = dist(geom4[first], geom4[second]);
        localOrigin = [length / 2, 0, 0];
        localRpy = [0, Math.PI / 2.0, 0];
      }
    }
  }

  let geometry: THREE.BufferGeometry;
  if (prim.type === 'cylinder') {
    if (Math.abs(length) <= 1e-6 || radius <= 1e-6) return null;
    geometry = new THREE.CylinderGeometry(radius, radius, Math.abs(length), 16);
    // Rotate cylinder to align cylinder axis (Y in ThreeJS) with Z axis
    geometry.rotateX(Math.PI / 2);
  } else if (prim.type === 'box') {
    let dims = [...prim.dimensions];
    if (robot.kind === 'CR6') {
      if (vis.body === 'LINK2' || vis.body === 'LINK3') {
        dims = [dims[0], dims[1], Math.abs(length)];
      }
    }
    geometry = new THREE.BoxGeometry(dims[0], dims[1], dims[2]);
  } else {
    geometry = new THREE.SphereGeometry(radius, 16, 16);
  }

  const material = new THREE.MeshStandardMaterial({
    color: robot.kind === 'CR6' ? 0xef4444 : 0xf97316, // Red for CR6, orange for CR4
    roughness: 0.4,
    metalness: 0.1,
    transparent: true,
    opacity: 0.85
  });

  const mesh = new THREE.Mesh(geometry, material);
  
  // Apply local origin overrides
  mesh.position.set(localOrigin[0], localOrigin[1], localOrigin[2]);
  mesh.rotation.set(localRpy[0], localRpy[1], localRpy[2]);
  mesh.scale.set(vis.scale[0], vis.scale[1], vis.scale[2]);

  return mesh;
}

function dist(p1: [number, number, number], p2: [number, number, number]): number {
  return Math.sqrt(
    Math.pow(p1[0] - p2[0], 2) +
    Math.pow(p1[1] - p2[1], 2) +
    Math.pow(p1[2] - p2[2], 2)
  );
}
