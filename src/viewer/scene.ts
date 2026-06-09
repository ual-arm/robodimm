import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export interface SceneContext {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  gridHelper: THREE.GridHelper;
  axesGizmo: THREE.AxesHelper;
  robotGroup: THREE.Group;
}

export function createSceneContext(canvas: HTMLCanvasElement, container: HTMLDivElement): SceneContext {
  const width = container.clientWidth;
  const height = container.clientHeight;

  // Scene
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0c10); // Isaac Sim Slate Dark background

  // Parent group rotated to align robot Z-up with ThreeJS Y-up
  const robotGroup = new THREE.Group();
  robotGroup.rotation.x = -Math.PI / 2;
  scene.add(robotGroup);

  // Camera
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
  camera.position.set(2.5, 2.0, 2.5);

  // Renderer
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;

  // Controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2; // Clip at ground level

  // Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0x4f46e5, 1.2); // Cool blue directional light
  dirLight1.position.set(5, 10, 5);
  dirLight1.castShadow = true;
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0xf59e0b, 0.5); // Warm yellow backlight
  dirLight2.position.set(-5, 2, -5);
  scene.add(dirLight2);

  // Floor Grid
  const gridHelper = new THREE.GridHelper(10, 50, 0x4f46e5, 0x1f2937);
  scene.add(gridHelper);

  // Coordinate system gizmo (X Red, Y Green, Z Blue) aligned to robot Z-up
  const axesGizmo = new THREE.AxesHelper(0.3);
  robotGroup.add(axesGizmo);

  return { scene, camera, renderer, controls, gridHelper, axesGizmo, robotGroup };
}
