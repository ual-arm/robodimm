# Frontend Guide

The Robodimm frontend is a Vite-built SPA written in TypeScript and React 18.
It uses **Three.js** for the 3D viewer, **Zustand** for state management, and
**Tailwind CSS** (v3 plugin pipeline) for styling. The code lives under
`src/`; the build entry point is `src/main.tsx`.

```
src/
├── main.tsx              React 18 createRoot + StrictMode
├── App.tsx               Tab shell + DEMO/PRO engine switcher
├── api/backend.ts        PRO backend fetch wrappers
├── math/                 Pure TS FK/IK/dynamics (no React)
│   ├── serial6.ts        CR6 RNEA + DH FK
│   ├── palletizer.ts     CR4 closed-chain FK + approximate ID
│   ├── palletizerGeometry.ts
│   ├── matrix.ts         4×4 helpers, standardDH, rotz, etc.
│   ├── trajectory.ts     Quintic blend trajectory
│   ├── recording.ts      generateSimulationTorques
│   └── actuators.ts      selectActuatorsForLog (sizing)
├── model/
│   ├── state.ts          Zustand store (canonical state, ~943 lines)
│   └── schemas.ts        RobotSpec, ProgramSpec, TorqueLog, SizingMargins, ...
├── ui/                   Tabs: ActuatorsTab, EditorTab, JogTab, ProgramTab
├── viewer/
│   ├── scene.ts          Scene / camera / lights / controls factory
│   ├── meshLoaders.ts    loadMeshVisual, loadStationGlb
│   ├── frameHelpers.ts   CAD-aligned transforms, axes helper updates
│   ├── robotVisuals.ts   Primitive geometry builders
│   ├── trajectoryLayer.ts
│   └── RobotViewer.tsx   The Three.js canvas React component
└── index.css
```

---

## 1. State management — `src/model/state.ts`

All app state lives in a single Zustand store created by
`useRobodimmStore` (`src/model/state.ts:140`). It is the **single source of
truth**; UI components read it directly and never duplicate state
locally for things that are already there.

### 1.1 Core state slices

| Slice | Type | Purpose |
|---|---|---|
| `editRobot` | `RobotSpec` | The spec currently being edited. |
| `activeRobot` | `RobotSpec` | The locked spec used by the viewer & dynamics. |
| `isSet` | `boolean` | `true` once the user clicks **Lock Robot (Set)**. Gates Jog, Program, Sizing. |
| `program` | `ProgramSpec` | Targets and instructions. |
| `torqueLog` | `TorqueLog \| null` | Last simulated torque log. |
| `sizingResults` | `ActuatorSizingReport \| null` | Last sizing report. |
| `actuatorLibrary` | `ActuatorLibrary \| null` | Catalog loaded from `/actuators_library.json`. |
| `activeTab` | `'editor' \| 'jog' \| 'program' \| 'actuators'` | UI tab. |
| `activeEngine` | `'frontend' \| 'backend'` | DEMO vs PRO. |
| `hasBackend` | `boolean` | Cached result of last `pingBackendUrl` call. |
| `q` | `number[]` | Current joint vector. |
| `playbackPoints` | `TrajectoryPoint[]` | Sampled points from last recording. |
| `playbackIndex` | `number` | Playback cursor. |
| `visualMode` | `'primitives' \| 'meshes'` | Render mode. |
| `stationObjects` | `StationObject[]` | World-space static GLB objects. |
| `meshWarnings` | `string[]` | Mesh load failures (recoverable). |

### 1.2 Schemas — `src/model/schemas.ts`

The strict TypeScript types mirroring the backend Pydantic models:

- `RobotSpec` — `kind`, `name`, `geometry` (a `Cr4Geometry` or
  `Cr6Geometry`), `inertials`, `payload`, `visuals`, `station`, `limits`.
- `Cr4Geometry` — twelve `[x, 0, z]` planar hardpoints
  (`A, O, B, C, D, E, F, G, H, P, J4, EE, TCP`).
- `Cr6Geometry` — `{ joints: DHJointSpec[]; tool_transform: number[][] }`.
- `InertialSpec` — `massKg`, `comM`, `inertiaKgM2`, `frame`
  (`'cad' \| 'link' \| 'tcp'`).
- `VisualSpec` — `body`, `frameName`, `kind: 'primitive' \| 'mesh'`,
  `meshUrl`, `primitive`, `originM`, `rpyRad`, `scale`, `visible`.
- `StationObject` — `id`, `name`, `meshUrl`, `positionM`,
  `rotationRpyRad`, `scale`, `visible`.
- `JointLimit` — `name`, `lowerLimitRad`, `upperLimitRad`,
  `maxVelocityRadS`, `maxAccelerationRadS2`, `frictionCoeffNmSPerRad`.
- `ProgramTarget` / `ProgramInstruction` (`MoveJ | MoveL | Pause`).
- `SizingMargins` — `continuous`, `peak`, `speed`, `power`,
  `motorPeakFactor`, `enforcePowerLimit?`, `sizingObjective?`.
- `ActuatorSizingReport` — `schema: "robodimm.actuator_sizing_report.v1"`.

`cloneRobotSpec` (schemas.ts:268) is the canonical deep-clone helper used
by every action that mutates the spec.

### 1.3 Actions of interest

- `changeRobotKind(kind)` — swaps the entire spec, validates CR4 geometry,
  and resets `isSet` to `false` (`state.ts:178-208`).
- `updateEditSpec(updater)` — runs an arbitrary mutator on a deep clone
  and re-validates CR4 invariants.
- `updateCr4HardpointXZ(name, x, z)` and `updateCr4DesignParameter(key, v)`
  — maintain the `P = B + C - O` and `E = D + C - O` parallelogram
  closures automatically (`state.ts:239-298`).
- `setRobot()` — moves `editRobot` to `activeRobot` and sets `isSet = true`.
- `checkBackendStatus()` — pings `VITE_PRO_BACKEND_URL_PRIMARY` then the
  fallback after consecutive failures, updating `hasBackend`,
  `backendState`, `backendUrl`, `backendCapabilities`,
  `backendVersion`, `pinocchioVersion`, `licenseStatus`.
- `loadActuatorLibrary()` — fetches `/actuators_library.json`.
- `runSignalRecording()` — solves the program trajectory in the active
  engine, populates `torqueLog` and `playbackPoints`.
- `runIterativeActuatorSizing(margins, gearboxTypeFilter)` — runs the
  passive deterministic sizing in `src/math/actuators.ts`.

---

## 2. The 3D viewer — `src/viewer/`

`RobotViewer` is a `React.FC` that owns a `<canvas>` and a single Three.js
`WebGLRenderer`. The whole effect re-runs only when `activeRobot` or
`visualMode` change; per-frame state is read through a `stateRef` mirror
(`RobotViewer.tsx:62-96`) so React re-renders are not required for each
animation frame.

### 2.1 Scene factory — `src/viewer/scene.ts`

`createSceneContext` builds the full scene in one call:

- `Scene` with background `#0a0c10` ("Isaac Sim slate dark").
- A `robotGroup` rotated by $-\pi/2$ about world X to align the **robot
  Z-up** convention with **Three.js Y-up** (`scene.ts:22-25`).
- `PerspectiveCamera` at `(2.5, 2.0, 2.5)`, $45°$ FoV, near $0.01$, far
  $100$.
- `WebGLRenderer` with `antialias: true` and `setPixelRatio(min(devicePixelRatio, 2))`.
  Shadow map enabled.
- `OrbitControls` with damping ($0.05$), polar angle clipped at $\pi/2$ so
  the camera never goes under the ground.
- Two directional lights — a cool blue key from `(5, 10, 5)` and a warm
  yellow rim from `(-5, 2, -5)`.
- A `GridHelper` (10 m, 50 divisions) using `#4f46e5` (indigo) and
  `#1f2937` (slate).
- A world `AxesHelper(0.3)` attached to `robotGroup` so the gizmo
  inherits the Z-up rotation.

### 2.2 CAD-aligned frame helper

`src/viewer/frameHelpers.ts:getCadAlignedTransform(T, T_home)` returns
the rotation

$$
R' = R(T) \cdot R(T_\text{home})^\top
$$

with the live translation. This keeps user-authored CAD meshes stationary
in orientation at the home configuration and only rotating *relative* to
home during motion — the same semantics as Kineforge's
`_cad_aligned_frame` (`KINEFORGE_MIGRATION_REFERENCE.md:88-94`).

### 2.3 Mesh loading

`src/viewer/meshLoaders.ts` exposes two loaders.

#### `loadMeshVisual(vis, group, onSuccess?, onFailure?, shouldAttach?)`

For robot-link meshes. Dispatches on file extension:

- `.stl` → `STLLoader` with a default `MeshStandardMaterial`
  (color `#cccccc`, roughness $0.5$, metalness $0.2$).
- everything else → `GLTFLoader` (covers `.glb` and `.gltf`). Walks the
  loaded scene and enables `castShadow` / `receiveShadow` on every Mesh.

If `shouldAttach` returns `false` at completion (e.g. the parent group
has been unmounted), the geometry, material, and textures are disposed
before returning — preventing WebGL leaks. On failure, the caller can
fall back to the primitive representation and call
`addMeshWarning(body, err)`.

#### `loadStationGlb(url, parent, onLoaded, onError?)`

Loads a `.glb` for a **station object** (world-space environment mesh).
Returns a **cancel function** that flips an internal `cancelled` flag;
when the loader resolves after cancellation, the just-loaded scene is
fully disposed. This is the key to the station-object reconciler below.

### 2.4 The station-object reconciler

Station objects (tables, fences, fixtures) live in `stationObjects` in
the Zustand store. They are world-space and have no relation to the
kinematic tree, so they must be loaded asynchronously while the user is
free to add, remove, or change them. The reconciler
(`RobotViewer.tsx:404-478`) runs every animation frame and:

1. **Detects deletions.** If an ID is no longer in `stationObjects`:
   - Call `stationCancelFns.get(id)()` to abort any in-flight load.
   - `stationGroup.remove(mesh)` and `disposeObject3D(mesh)` to free GPU
     resources.
   - Drop the entries from `stationMeshes`, `stationLoadedUrls`, and
     `stationCancelFns`.
2. **Detects additions and URL changes.** If the ID is new or the URL
   changed:
   - Cancel any in-flight load for that ID.
   - Dispose the previous mesh if any.
   - Allocate a placeholder `THREE.Group` (`loadGroup`) and add it to
     `stationGroup` *immediately* (so the next frame has a stable
     transform target).
   - Call `loadStationGlb(url, loadGroup, onLoaded)` and stash the cancel
     function in `stationCancelFns`.
   - In `onLoaded`, set position/rotation/scale/visibility from the
     current `StationObject`, **replace** the placeholder with the actual
     GLB scene in `stationMeshes`, and add it to `stationGroup`.
3. **Updates every frame.** For objects that are already loaded, the
   transform is reapplied (`position.set`, `rotation.set(..., 'XYZ')`,
   `scale.set`, `visible = ...`) on every animation tick. The reconciler
   thus has no React lifecycle — it is a pure Three.js scene-graph
   reconciler driven by the current store snapshot.

The component cleanup function (line 486-499) cancels all in-flight
loads and disposes every remaining station mesh, the entire
`robotGroup`, the scene's `stationGroup`, the `WebGLRenderer`, and the
`OrbitControls`. This guarantees no WebGL context leaks across hot
reloads.

---

## 3. Tabs

The four right-hand tabs in `App.tsx` correspond to four React
components in `src/ui/`. The tab order is:

1. **Editor Spec** (`EditorTab.tsx`) — robot kind, geometry, inertias,
   limits, visuals. The **Lock Robot (Set)** button is here. On click,
   `setRobot()` copies `editRobot` into `activeRobot` and sets
   `isSet = true`.
2. **Jog Panel** (`JogTab.tsx`) — joint sliders and Cartesian XYZ+yaw
   jog in either World or TCP frame. Disabled until `isSet`.
3. **Program** (`ProgramTab.tsx`) — targets and instructions
   (`MoveJ`, `MoveL`, `Pause`). The **Program** button calls
   `runSignalRecording()` to populate the torque log and playback buffer.
4. **Sizing** (`ActuatorsTab.tsx`) — picks the sizing objective, runs the
   passive deterministic selection, displays the report, and exposes
   JSON / CSV export.

The **DEMO (Browser)** / **PRO (Python API)** switcher in the top-right
of `App.tsx` is gated by `hasBackend` and shows an *Offline* badge when
no PRO backend is reachable (`App.tsx:70-107`).

---

## 4. Conventions and gotchas

- **SI internally.** UI displays degrees and millimetres; conversions
  happen at the boundary. There are no inline rad↔deg or m↔mm
  conversions inside `src/math/`.
- **One q-space per call site.** The CR4 palletizer exposes several
  equivalent joint coordinates (actuated, dependent, TCP). All math
  helpers in `src/math/palletizer.ts` take and return one convention; do
  not mix.
- **`playbackIndex` vs `q`.** The viewer prefers `playbackPoints[i].q`
  when the playback buffer is non-empty (`RobotViewer.tsx:250-253`).
  Stopping playback (or never starting it) means the viewer tracks the
  `q` slider vector.
- **`visualMode` is global.** Toggling between `primitives` and `meshes`
  re-runs the entire viewer effect (`RobotViewer.tsx:501`); the previous
  meshes are disposed in `rebuildRobotVisuals`.
- **Mesh warnings are recoverable.** A failed mesh load produces a
  warning and the primitive representation is substituted; the app does
  not crash.
- **Bundle:** Tailwind CSS v3 plugin pipeline (`postcss.config.js`),
  despite `@tailwindcss/vite` being on v4 in `package.json`. Do not
  "modernize" without coordination — see `AGENTS.md` § Gotchas.
