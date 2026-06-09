import * as THREE from 'three';
import { RobotSpec } from '../model/schemas';
import { Serial6Engine } from '../math/serial6';
import { PalletizerEngine } from '../math/palletizer';
import { getTranslation } from '../math/matrix';
import { TrajectoryPoint } from '../math/trajectory';

export function buildTrajectoryLine(
  playbackPoints: TrajectoryPoint[],
  robot: RobotSpec
): THREE.Line {
  const points: THREE.Vector3[] = [];
  
  let engine6: Serial6Engine | null = null;
  let engine4: PalletizerEngine | null = null;
  if (robot.kind === 'CR6') {
    engine6 = new Serial6Engine(robot);
  } else {
    engine4 = new PalletizerEngine(robot);
  }

  playbackPoints.forEach(pt => {
    if (engine6) {
      const ptFk = engine6.forwardKinematics(pt.q);
      const t = getTranslation(ptFk.tcp_transform);
      points.push(new THREE.Vector3(t[0], t[1], t[2]));
    } else if (engine4) {
      const ptFk = engine4.forwardKinematics(pt.q);
      const t = ptFk.points['TCP'];
      points.push(new THREE.Vector3(t[0], t[1], t[2]));
    }
  });

  const geom = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineBasicMaterial({ color: 0x4f46e5 });
  return new THREE.Line(geom, mat);
}
