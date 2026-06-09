import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { useRobodimmStore } from '../model/state';
import { Serial6Engine } from '../math/serial6';
import { PalletizerEngine } from '../math/palletizer';
import { Vector3, Matrix4, createIdentity4 } from '../math/matrix';
import { Eye, EyeOff } from 'lucide-react';
import { createSceneContext } from './scene';
import { loadMeshVisual, loadStationGlb } from './meshLoaders';
import { buildVisualPrimitive } from './robotVisuals';
import { getCadAlignedTransform, updateFrameHelper, updateCOMMarker } from './frameHelpers';
import { buildTrajectoryLine } from './trajectoryLayer';
import { Cr4Geometry } from '../model/schemas';
import { getDefaultPalletizerBodyCOM } from '../math/palletizerGeometry';

function disposeObject3D(object: THREE.Object3D) {
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
}

export const RobotViewer: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const {
    activeRobot,
    isSet,
    q,
    playbackPoints,
    playbackIndex,
    program,
    showGrid,
    showAxes,
    showCOMs,
    showTrajectory,
    showTCPFrame,
    visualMode,
    addMeshWarning,
    stationObjects
  } = useRobodimmStore();

  // References to update ThreeJS objects from React state
  const stateRef = useRef({
    activeRobot,
    isSet,
    q,
    playbackPoints,
    playbackIndex,
    program,
    showGrid,
    showAxes,
    showCOMs,
    showTrajectory,
    showTCPFrame,
    visualMode,
    addMeshWarning,
    stationObjects
  });

  useEffect(() => {
    stateRef.current = {
      activeRobot,
      isSet,
      q,
      playbackPoints,
      playbackIndex,
      program,
      showGrid,
      showAxes,
      showCOMs,
      showTrajectory,
      showTCPFrame,
      visualMode,
      addMeshWarning,
      stationObjects
    };
  }, [activeRobot, isSet, q, playbackPoints, playbackIndex, program, showGrid, showAxes, showCOMs, showTrajectory, showTCPFrame, visualMode, addMeshWarning, stationObjects]);

  useEffect(() => {
    if (!containerRef.current || !canvasRef.current) return;

    // Create standard scene, camera, lights, controls, grid and global axes aligned to robot Z-up
    const context = createSceneContext(canvasRef.current, containerRef.current);
    const { scene, camera, renderer, controls, gridHelper, axesGizmo, robotGroup } = context;

    // ── Station Objects group (world-space, static) ─────────────────────────
    const stationGroup = new THREE.Group();
    stationGroup.name = 'stationGroup';
    scene.add(stationGroup);

    // Track loaded station meshes by station object id
    const stationMeshes = new Map<string, THREE.Object3D>();
    // Track which URL is currently loaded for each id (to detect changes)
    const stationLoadedUrls = new Map<string, string>();
    // Track cancellation functions for in-flight loads
    const stationCancelFns = new Map<string, () => void>();

    // Map to hold link 3D groups, axes helpers, and COM spheres
    const linkGroups = new Map<string, THREE.Group[]>();
    const axesHelpers = new Map<string, THREE.AxesHelper>();
    const comMarkers = new Map<string, THREE.Mesh>();

    let trajectoryLine: THREE.Line | null = null;

    // Initialize or rebuild visual representations
    const rebuildRobotVisuals = () => {
      // Clear previous robot meshes
      linkGroups.forEach(arr => arr.forEach(g => {
        robotGroup.remove(g);
        disposeObject3D(g);
      }));
      linkGroups.clear();
      axesHelpers.forEach(a => {
        robotGroup.remove(a);
        disposeObject3D(a);
      });
      axesHelpers.clear();
      comMarkers.forEach(c => {
        robotGroup.remove(c);
        disposeObject3D(c);
      });
      comMarkers.clear();
      if (trajectoryLine) {
        robotGroup.remove(trajectoryLine);
        disposeObject3D(trajectoryLine);
        trajectoryLine = null;
      }

      const robot = stateRef.current.activeRobot;

      robot.visuals.forEach((vis) => {
        const group = new THREE.Group();
        group.matrixAutoUpdate = true;
        robotGroup.add(group);
        
        // Resolve coordinate frame name lookup key
        const visKey = vis.frameName || vis.body;

        if (!linkGroups.has(visKey)) {
          linkGroups.set(visKey, []);
        }
        linkGroups.get(visKey)!.push(group);

        // Visual Axes helper (create only one helper per body)
        if (!axesHelpers.has(vis.body)) {
          const axes = new THREE.AxesHelper(0.15);
          robotGroup.add(axes);
          axesHelpers.set(vis.body, axes);
        }

        // COM Sphere representation (create only one COM per body)
        if (!comMarkers.has(vis.body)) {
          const comGeo = new THREE.SphereGeometry(0.015, 8, 8);
          const comMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
          const comMesh = new THREE.Mesh(comGeo, comMat);
          robotGroup.add(comMesh);
          comMarkers.set(vis.body, comMesh);
        }

        if (!vis.visible) return;

        // Render Primitive or Load mesh file
        const mode = stateRef.current.visualMode;
        if (mode === 'primitives' || !vis.meshUrl) {
          if (vis.primitive) {
            const mesh = buildVisualPrimitive(vis, robot);
            if (mesh) {
              group.add(mesh);
            }
          }
        } else {
          loadMeshVisual(vis, group, undefined, (err) => {
            if (!group.parent) return;
            // Fallback to primitive
            if (vis.primitive) {
              const tempVis = { ...vis, kind: 'primitive' as const };
              const mesh = buildVisualPrimitive(tempVis, robot);
              if (mesh) {
                group.add(mesh);
              }
            }
            stateRef.current.addMeshWarning(vis.body, err.message || String(err));
          }, () => !!group.parent);
        }
      });

      // Create TCP axes helper
      const tcpAxes = new THREE.AxesHelper(0.25);
      robotGroup.add(tcpAxes);
      axesHelpers.set('TCP', tcpAxes);
    };

    rebuildRobotVisuals();

    // Resize Handler
    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // Animation Loop
    let animationFrameId: number;
    
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();

      const {
        activeRobot: robot,
        q: currentQ,
        playbackPoints,
        playbackIndex,
        showGrid: sGrid,
        showAxes: sAxes,
        showCOMs: sCOMs,
        showTrajectory: sTraj,
        showTCPFrame: sTCPFrame,
        visualMode
      } = stateRef.current;

      // Toggle floor grid visibility
      gridHelper.visible = sGrid;
      axesGizmo.visible = sAxes;

      // Determine joint vector
      let activeQ = [...currentQ];
      if (playbackPoints.length > 0 && playbackIndex < playbackPoints.length) {
        activeQ = playbackPoints[playbackIndex].q;
      }

      // Calculate forward kinematics transforms
      let fkTransforms: Record<string, Matrix4> = {};
      let homeTransforms: Record<string, Matrix4> = {};
      let comPositions: Record<string, Vector3> = {};

      if (robot.kind === 'CR6') {
        const eng = new Serial6Engine(robot);
        const fk = eng.forwardKinematics(activeQ);
        const home_fk = eng.forwardKinematics(Array(6).fill(0));
        
        // Transform mapping for CR6 links
        for (let i = 0; i < 6; i++) {
          const bodyName = `LINK${i + 1}`;
          
          const transform = fk.joint_body_transforms[i];
          const home_transform = home_fk.joint_body_transforms[i];
          
          // Apply CAD alignment
          fkTransforms[bodyName] = getCadAlignedTransform(transform, home_transform);

          // Compute world COM position
          const T = fkTransforms[bodyName];
          const inertial = robot.inertials[bodyName] || robot.inertials[eng.joints[i].name];
          const com = inertial?.comM ?? [0, 0, 0];
          comPositions[bodyName] = [
            T[0][0] * com[0] + T[0][1] * com[1] + T[0][2] * com[2] + T[0][3],
            T[1][0] * com[0] + T[1][1] * com[1] + T[1][2] * com[2] + T[1][3],
            T[2][0] * com[0] + T[2][1] * com[1] + T[2][2] * com[2] + T[2][3]
          ];
        }
        fkTransforms['BASE'] = createIdentity4(); // Base is fixed at origin
        fkTransforms['base_link'] = createIdentity4();
        comPositions['BASE'] = [0, 0, 0];
        fkTransforms['TCP'] = fk.tcp_transform;
      } else {
        const eng = new PalletizerEngine(robot);
        const fk = eng.forwardKinematics(activeQ);
        const home_fk = eng.forwardKinematics(Array(4).fill(0));
        
        // Transform mapping for CR4 links
        for (const [bodyName, trans] of Object.entries(fk.transforms)) {
          fkTransforms[bodyName] = trans;
 
          const inertial = robot.inertials[bodyName];
          const com = inertial?.comM ?? getDefaultPalletizerBodyCOM(robot.geometry as Cr4Geometry, bodyName);
          comPositions[bodyName] = [
            trans[0][0] * com[0] + trans[0][1] * com[1] + trans[0][2] * com[2] + trans[0][3],
            trans[1][0] * com[0] + trans[1][1] * com[1] + trans[1][2] * com[2] + trans[1][3],
            trans[2][0] * com[0] + trans[2][1] * com[1] + trans[2][2] * com[2] + trans[2][3]
          ];
        }
        fkTransforms['TCP'] = fk.transforms['TCP_frame'];

        // Map home transforms for CR4
        for (const [bodyName, trans] of Object.entries(home_fk.transforms)) {
          homeTransforms[bodyName] = trans;
        }
        homeTransforms['TCP'] = home_fk.transforms['TCP_frame'];
      }

      // Hide all groups by default, then we'll show the ones with valid transforms
      linkGroups.forEach(groups => {
        groups.forEach(g => { g.visible = false; });
      });
      axesHelpers.forEach(h => { h.visible = false; });
      comMarkers.forEach(m => { m.visible = false; });

      // Update 3D elements based on FK transforms
      const updatedAxisBodies = new Set<string>();
      robot.visuals.forEach((vis) => {
        const visKey = vis.frameName || vis.body;
        let T = fkTransforms[visKey];
        if (T) {
          if (robot.kind === 'CR4' && visualMode === 'meshes' && vis.kind === 'mesh') {
            const home_trans = homeTransforms[visKey] || createIdentity4();
            T = getCadAlignedTransform(T, home_trans);
          }
          // Convert row-major Matrix4 (TS) to column-major (ThreeJS)
          const elements = [
            T[0][0], T[1][0], T[2][0], T[3][0],
            T[0][1], T[1][1], T[2][1], T[3][1],
            T[0][2], T[1][2], T[2][2], T[3][2],
            T[0][3], T[1][3], T[2][3], T[3][3]
          ];
          
          // Decompose the matrix to update the position, rotation and scale of the group
          const matrix = new THREE.Matrix4().fromArray(elements);
          const pos = new THREE.Vector3();
          const rot = new THREE.Quaternion();
          const scl = new THREE.Vector3();
          matrix.decompose(pos, rot, scl);

          const groups = linkGroups.get(visKey);
          if (groups) {
            groups.forEach(group => {
              group.position.copy(pos);
              group.quaternion.copy(rot);
              group.scale.copy(scl);
              group.visible = true;
            });
          }

          // Update axes helper matching the body link (respect individual axesVisible settings)
          const helper = axesHelpers.get(vis.body);
          if (helper && !updatedAxisBodies.has(vis.body)) {
            updateFrameHelper(helper, pos, rot, sAxes && (vis.axesVisible !== false));
            updatedAxisBodies.add(vis.body);
          }

          // Update COM Marker matching the body link
          const comMarker = comMarkers.get(vis.body);
          const comPos = comPositions[vis.body];
          if (comMarker) {
            updateCOMMarker(comMarker, comPos, sCOMs);
          }
        }
      });

      // Update TCP Helper explicitly (if axes visible)
      const T_tcp = fkTransforms['TCP'];
      const tcpHelper = axesHelpers.get('TCP');
      if (T_tcp && tcpHelper) {
        const elements = [
          T_tcp[0][0], T_tcp[1][0], T_tcp[2][0], T_tcp[3][0],
          T_tcp[0][1], T_tcp[1][1], T_tcp[2][1], T_tcp[3][1],
          T_tcp[0][2], T_tcp[1][2], T_tcp[2][2], T_tcp[3][2],
          T_tcp[0][3], T_tcp[1][3], T_tcp[2][3], T_tcp[3][3]
        ];
        const matrix = new THREE.Matrix4().fromArray(elements);
        const pos = new THREE.Vector3();
        const rot = new THREE.Quaternion();
        const scl = new THREE.Vector3();
        matrix.decompose(pos, rot, scl);
        updateFrameHelper(tcpHelper, pos, rot, sTCPFrame);
      }

      // Render Trajectory Path
      if (sTraj && playbackPoints.length > 0) {
        if (!trajectoryLine) {
          trajectoryLine = buildTrajectoryLine(playbackPoints, robot);
          robotGroup.add(trajectoryLine);
        }
        trajectoryLine.visible = true;
      } else {
        if (trajectoryLine) {
          trajectoryLine.visible = false;
        }
      }

      // ── Station Objects reconciler ──────────────────────────────────────
      const currentStationObjects = stateRef.current.stationObjects;
      const currentIds = new Set(currentStationObjects.map((o) => o.id));

      // Remove meshes no longer in the list
      stationMeshes.forEach((mesh, id) => {
        if (!currentIds.has(id)) {
          // Cancel in-flight load if any
          stationCancelFns.get(id)?.();
          stationCancelFns.delete(id);
          stationGroup.remove(mesh);
          disposeObject3D(mesh);
          stationMeshes.delete(id);
          stationLoadedUrls.delete(id);
        }
      });

      // Add or update existing station objects
      for (const obj of currentStationObjects) {
        const loaded = stationMeshes.get(obj.id);
        const loadedUrl = stationLoadedUrls.get(obj.id);

        if (!loaded || loadedUrl !== obj.meshUrl) {
          // Cancel previous in-flight load for this id
          stationCancelFns.get(obj.id)?.();
          stationCancelFns.delete(obj.id);

          // Remove previous mesh if any
          if (loaded) {
            stationGroup.remove(loaded);
            disposeObject3D(loaded);
            stationMeshes.delete(obj.id);
          }

          if (obj.meshUrl) {
            const objId = obj.id;
            const loadGroup = new THREE.Group();
            stationGroup.add(loadGroup);
            stationMeshes.set(objId, loadGroup);
            stationLoadedUrls.set(objId, obj.meshUrl);

            const cancel = loadStationGlb(
              obj.meshUrl,
              loadGroup,
              (root) => {
                // Apply initial transform once loaded
                const curObj = stateRef.current.stationObjects.find((o) => o.id === objId);
                if (curObj) {
                  const [px, py, pz] = curObj.positionM ?? [0, 0, 0];
                  const [rx, ry, rz] = curObj.rotationRpyRad ?? [0, 0, 0];
                  const [sx, sy, sz] = curObj.scale ?? [1, 1, 1];
                  root.position.set(px, py, pz);
                  root.rotation.set(rx, ry, rz, 'XYZ');
                  root.scale.set(sx, sy, sz);
                  root.visible = curObj.visible !== false;
                  // Replace loadGroup with root directly in stationMeshes
                  stationGroup.remove(loadGroup);
                  stationGroup.add(root);
                  stationMeshes.set(objId, root);
                }
              }
            );
            stationCancelFns.set(objId, cancel);
          }
        } else {
          // Mesh already loaded — just update transform and visibility every frame
          const [px, py, pz] = obj.positionM ?? [0, 0, 0];
          const [rx, ry, rz] = obj.rotationRpyRad ?? [0, 0, 0];
          const [sx, sy, sz] = obj.scale ?? [1, 1, 1];
          loaded.position.set(px, py, pz);
          loaded.rotation.set(rx, ry, rz, 'XYZ');
          loaded.scale.set(sx, sy, sz);
          loaded.visible = obj.visible !== false;
        }
      }

      renderer.render(scene, camera);
    };

    animate();

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      controls.dispose();
      disposeObject3D(robotGroup);
      // Cleanup station objects
      stationCancelFns.forEach((cancel) => cancel());
      stationCancelFns.clear();
      stationMeshes.forEach((mesh) => {
        disposeObject3D(mesh);
      });
      stationMeshes.clear();
      scene.remove(stationGroup);
      renderer.dispose();
    };
  }, [activeRobot, visualMode]); // Re-init on robot spec or visualMode changes

  return (
    <div ref={containerRef} className="relative w-full h-full flex flex-col bg-slate-950 overflow-hidden">
      {/* 3D Canvas */}
      <canvas ref={canvasRef} className="w-full h-full block" />

      {!isSet && (
        <div className="absolute top-4 right-4 bg-amber-900/90 text-amber-100 backdrop-blur border border-amber-600 py-2 px-4 rounded-lg shadow-lg z-10 flex items-center gap-2 font-semibold text-sm">
          <span>⚠️ Robot configuration not locked. Click "Lock Robot (Set)" in the editor tab to run simulations and calculations.</span>
        </div>
      )}
    </div>
  );
};
