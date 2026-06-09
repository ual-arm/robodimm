import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { VisualSpec } from '../model/schemas';

const gltfLoader = new GLTFLoader();
const stlLoader = new STLLoader();

export function loadMeshVisual(
  vis: VisualSpec,
  group: THREE.Group,
  onSuccess?: () => void,
  onFailure?: (err: any) => void,
  shouldAttach?: () => boolean
): void {
  if (!vis.meshUrl) return;
  const url = vis.meshUrl;

  const disposeObject = (object: THREE.Object3D) => {
    const disposeMaterial = (material: THREE.Material) => {
      Object.values(material).forEach((value) => {
        if (value && typeof value === 'object' && 'isTexture' in value) {
          (value as THREE.Texture).dispose();
        }
      });
      material.dispose();
    };

    object.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (mesh.geometry) {
        mesh.geometry.dispose();
      }
      const material = mesh.material;
      if (Array.isArray(material)) {
        material.forEach(disposeMaterial);
      } else if (material) {
        disposeMaterial(material);
      }
    });
  };

  if (url.toLowerCase().endsWith('.stl')) {
    stlLoader.load(url, (geometry) => {
      const material = new THREE.MeshStandardMaterial({
        color: 0xcccccc,
        roughness: 0.5,
        metalness: 0.2
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(vis.originM[0], vis.originM[1], vis.originM[2]);
      mesh.rotation.set(vis.rpyRad[0], vis.rpyRad[1], vis.rpyRad[2]);
      mesh.scale.set(vis.scale[0], vis.scale[1], vis.scale[2]);
      if (shouldAttach && !shouldAttach()) {
        geometry.dispose();
        material.dispose();
        return;
      }
      group.add(mesh);
      if (onSuccess) onSuccess();
    }, undefined, (err) => {
      console.error('STL Load Error:', err);
      if (onFailure) onFailure(err);
    });
  } else {
    gltfLoader.load(url, (gltf) => {
      gltf.scene.traverse((node) => {
        if (node instanceof THREE.Mesh) {
          node.castShadow = true;
          node.receiveShadow = true;
        }
      });
      gltf.scene.position.set(vis.originM[0], vis.originM[1], vis.originM[2]);
      gltf.scene.rotation.set(vis.rpyRad[0], vis.rpyRad[1], vis.rpyRad[2]);
      gltf.scene.scale.set(vis.scale[0], vis.scale[1], vis.scale[2]);
      if (shouldAttach && !shouldAttach()) {
        disposeObject(gltf.scene);
        return;
      }
      group.add(gltf.scene);
      if (onSuccess) onSuccess();
    }, undefined, (err) => {
      console.error('GLTF Load Error:', err);
      if (onFailure) onFailure(err);
    });
  }
}

/**
 * Loads a .glb from a blob/object URL and attaches the scene to `parent`.
 * Returns a cancel function — call it to abort if the object is removed
 * before the async load completes.
 */
export function loadStationGlb(
  url: string,
  parent: THREE.Group,
  onLoaded: (root: THREE.Object3D) => void,
  onError?: (err: any) => void
): () => void {
  let cancelled = false;
  gltfLoader.load(
    url,
    (gltf) => {
      if (cancelled) {
        gltf.scene.traverse((node) => {
          const mesh = node as THREE.Mesh;
          if (mesh.geometry) mesh.geometry.dispose();
          const mat = mesh.material;
          if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
          else if (mat) (mat as THREE.Material).dispose();
        });
        return;
      }
      gltf.scene.traverse((node) => {
        if (node instanceof THREE.Mesh) {
          node.castShadow = true;
          node.receiveShadow = true;
        }
      });
      parent.add(gltf.scene);
      onLoaded(gltf.scene);
    },
    undefined,
    (err) => {
      if (!cancelled) {
        console.warn('Station GLB load error:', err);
        if (onError) onError(err);
      }
    }
  );
  return () => { cancelled = true; };
}
