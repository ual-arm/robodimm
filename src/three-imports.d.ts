declare module 'three/examples/jsm/controls/OrbitControls' {
  import { Camera } from 'three';
  export class OrbitControls {
    constructor(camera: Camera, domElement: HTMLElement);
    enableDamping: boolean;
    dampingFactor: number;
    maxPolarAngle: number;
    update(): void;
    dispose(): void;
  }
}

declare module 'three/examples/jsm/loaders/STLLoader' {
  import { BufferGeometry } from 'three';
  export class STLLoader {
    constructor();
    load(
      url: string,
      onLoad: (geometry: BufferGeometry) => void,
      onProgress?: (event: ProgressEvent) => void,
      onError?: (event: ErrorEvent) => void
    ): void;
  }
}

declare module 'three/examples/jsm/loaders/GLTFLoader' {
  import { Object3D } from 'three';
  export interface GLTF {
    animations: any[];
    scene: Object3D;
    scenes: Object3D[];
    cameras: any[];
    asset: any;
  }
  export class GLTFLoader {
    constructor();
    load(
      url: string,
      onLoad: (gltf: GLTF) => void,
      onProgress?: (event: ProgressEvent) => void,
      onError?: (event: ErrorEvent) => void
    ): void;
  }
}
