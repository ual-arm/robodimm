import { create } from 'zustand';
import {
  RobotSpec,
  ProgramSpec,
  TorqueLog,
  ActuatorLibrary,
  ProgramInstruction,
  ProgramTarget,
  cloneRobotSpec,
  RobotActuatorSelection,
  ActuatorSizingReport,
  SizingMargins,
  Cr4Geometry,
  VisualSpec,
  SizingObjective,
  StationObject
} from './schemas';
import { irb4600Serial6Spec, Serial6Engine } from '../math/serial6';
import { irb460PalletizerSpec, PalletizerEngine } from '../math/palletizer';
import { TrajectoryPoint, buildProgramDynamicsTrajectory } from '../math/trajectory';
import { pingBackendUrl, setApiBackendUrl, calculateBackendBatchDynamics, uploadPackageToBackend } from '../api/backend';
import { generateSimulationTorques, runIterativeSizingAlgorithm } from '../math/recording';
import { selectActuatorsForLog } from '../math/actuators';
import {
  Cr4GeometryValidationIssue,
  sanitizeCr4Geometry,
  validateCr4Geometry,
  Cr4DesignParameters,
  designFromCr4Geometry,
  cr4GeometryFromDesign,
  validateCr4DesignParameters
} from '../math/palletizerGeometry';

interface RobodimmState {
  editRobot: RobotSpec;
  activeRobot: RobotSpec;
  isSet: boolean;
  program: ProgramSpec;
  torqueLog: TorqueLog | null;
  sizingResults: ActuatorSizingReport | null;
  actuatorLibrary: ActuatorLibrary | null;
  activeTab: 'editor' | 'jog' | 'program' | 'actuators';
  activeEngine: 'frontend' | 'backend';
  hasBackend: boolean;
  backendState: 'detecting' | 'connected' | 'disconnected' | 'incompatible';
  consecutiveBackendFailures: number;
  backendUrl: string;
  backendCapabilities: any | null;
  backendVersion: string | null;
  pinocchioVersion: string | null;
  licenseStatus: string | null;
  q: number[];
  playbackPoints: TrajectoryPoint[];
  playbackIndex: number;
  isRecording: boolean;
  showGrid: boolean;
  showAxes: boolean;
  showCOMs: boolean;
  showTrajectory: boolean;
  showTCPFrame: boolean;
  cr4ValidationIssues: Cr4GeometryValidationIssue[];
  cr4DesignParams: Cr4DesignParameters | null;
  cr4ConstrainParallelogram: boolean;
  visualMode: 'primitives' | 'meshes';
  meshesStale: boolean;
  meshWarnings: string[];
  primitiveVisualsBackup: VisualSpec[] | null;
  meshVisualsBackup: VisualSpec[] | null;
  objectUrls: string[];
  stationObjects: StationObject[];

  // Actions
  changeRobotKind: (kind: 'CR4' | 'CR6') => void;
  updateEditSpec: (updater: (spec: RobotSpec) => void) => void;
  updateCr4HardpointXZ: (name: keyof Cr4Geometry, x: number, z: number) => void;
  updateCr4DesignParameter: (key: keyof Cr4DesignParameters, value: number) => void;
  setCr4ConstrainParallelogram: (value: boolean) => void;
  loadRobotSpec: (spec: RobotSpec) => void;
  setRobot: () => void;
  updateQ: (q: number[]) => void;
  updateTCP: (position: [number, number, number], targetYaw?: number) => boolean;
  setTab: (tab: 'editor' | 'jog' | 'program' | 'actuators') => void;
  setEngine: (engine: 'frontend' | 'backend') => void;
  setHasBackend: (has: boolean) => void;
  setBackendUrl: (url: string) => void;
  toggleSolidVisibility: (bodyName: string) => void;
  toggleAxesVisibility: (bodyName: string) => void;
  toggleGrid: () => void;
  toggleAxes: () => void;
  toggleCOMs: () => void;
  toggleTrajectory: () => void;
  toggleTCPFrame: () => void;
  setVisualMode: (mode: 'primitives' | 'meshes') => void;
  applyMeshPreset: (presetName: string) => void;
  usePrimitiveVisuals: () => void;
  clearMeshWarnings: () => void;
  addMeshWarning: (body: string, warning: string) => void;
  revokeAllObjectUrls: () => void;

  // Program Actions
  addTarget: (target: ProgramTarget) => void;
  removeTarget: (name: string) => void;
  addInstruction: (instruction: ProgramInstruction) => void;
  removeInstruction: (index: number) => void;
  clearProgram: () => void;
  loadProgram: (program: ProgramSpec) => void;

  // Sizing & Dynamics
  runSignalRecording: () => Promise<void>;
  runIterativeActuatorSizing: (
    margins: {
      continuous: number;
      peak: number;
      speed: number;
      power?: number;
      motorPeakFactor?: number;
      enforcePowerLimit?: boolean;
      sizingObjective?: SizingObjective;
    },
    gearboxTypeFilter?: 'harmonic' | 'cycloidal' | 'any'
  ) => Promise<void>;
  loadActuatorLibrary: () => Promise<void>;
  checkBackendStatus: () => Promise<void>;
  uploadPackage: (files: File[]) => Promise<boolean>;

  // Station Object Actions
  addStationObject: (obj: StationObject) => void;
  updateStationObject: (id: string, patch: Partial<StationObject>) => void;
  removeStationObject: (id: string) => void;
  toggleStationObjectVisible: (id: string) => void;
}

const defaultProgram = (): ProgramSpec => ({
  schema: 'robodimm.program.v1',
  name: 'default_program',
  targets: [],
  instructions: []
});

export const useRobodimmStore = create<RobodimmState>((set, get) => ({
  editRobot: irb4600Serial6Spec(),
  activeRobot: irb4600Serial6Spec(),
  isSet: false,
  program: defaultProgram(),
  torqueLog: null,
  sizingResults: null,
  actuatorLibrary: null,
  activeTab: 'editor',
  activeEngine: 'frontend',
  hasBackend: false,
  backendState: 'detecting',
  consecutiveBackendFailures: 0,
  backendUrl: import.meta.env.VITE_PRO_BACKEND_URL_PRIMARY || 'http://127.0.0.1:8001',
  backendCapabilities: null,
  backendVersion: null,
  pinocchioVersion: null,
  licenseStatus: null,
  q: Array(6).fill(0),
  playbackPoints: [],
  playbackIndex: 0,
  isRecording: false,
  showGrid: true,
  showAxes: true,
  showCOMs: true,
  showTrajectory: true,
  showTCPFrame: true,
  cr4ValidationIssues: [],
  cr4DesignParams: null,
  cr4ConstrainParallelogram: true,
  visualMode: 'primitives',
  meshesStale: false,
  meshWarnings: [],
  primitiveVisualsBackup: null,
  meshVisualsBackup: null,
  objectUrls: [],
  stationObjects: [],

  changeRobotKind: (kind) => {
    get().revokeAllObjectUrls();
    const defaultSpec = kind === 'CR6' ? irb4600Serial6Spec() : irb460PalletizerSpec();
    const isCr4 = kind === 'CR4';
    let params = isCr4 ? designFromCr4Geometry(defaultSpec.geometry as Cr4Geometry) : null;
    let issues: Cr4GeometryValidationIssue[] = [];
    if (isCr4 && params) {
      if (get().cr4ConstrainParallelogram) {
        params.L_FG = params.L_CH;
        params.L_HG = Math.hypot(params.F_offset_x, params.F_offset_z);
      }
      const valParams = validateCr4DesignParameters(params);
      const valGeom = validateCr4Geometry(defaultSpec.geometry as Cr4Geometry);
      issues = [...valParams.issues, ...valGeom.issues];
      const hasErrors = issues.some(issue => issue.severity === 'error');
      if (!hasErrors) {
        defaultSpec.geometry = cr4GeometryFromDesign(params);
      }
    }
    set({
      editRobot: defaultSpec,
      isSet: false,
      cr4DesignParams: params,
      cr4ValidationIssues: issues,
      visualMode: 'primitives',
      meshesStale: false,
      meshWarnings: [],
      primitiveVisualsBackup: null,
      meshVisualsBackup: null
    });
  },

  updateEditSpec: (updater) => {
    const updated = cloneRobotSpec(get().editRobot);
    updater(updated);
    
    let issues = get().cr4ValidationIssues;
    let params = get().cr4DesignParams;
    if (updated.kind === 'CR4') {
      params = designFromCr4Geometry(updated.geometry as Cr4Geometry);
      if (get().cr4ConstrainParallelogram) {
        params.L_FG = params.L_CH;
        params.L_HG = Math.hypot(params.F_offset_x, params.F_offset_z);
      }
      const valParams = validateCr4DesignParameters(params);
      const valGeom = validateCr4Geometry(updated.geometry as Cr4Geometry);
      issues = [...valParams.issues, ...valGeom.issues];
      const hasErrors = issues.some(issue => issue.severity === 'error');
      if (!hasErrors) {
        updated.geometry = cr4GeometryFromDesign(params);
      }
    }
    
    set({
      editRobot: updated,
      cr4DesignParams: params,
      cr4ValidationIssues: issues,
      meshesStale: get().visualMode === 'meshes' ? true : get().meshesStale
    });
  },

  updateCr4HardpointXZ: (name, x, z) => {
    const edit = cloneRobotSpec(get().editRobot);
    if (edit.kind !== 'CR4') return;
    
    const geom = edit.geometry as Cr4Geometry;
    if (!geom[name]) return;
    
    geom[name][0] = x;
    geom[name][2] = z;
    
    geom.P[0] = geom.B[0] + geom.C[0] - geom.O[0];
    geom.P[1] = 0.0;
    geom.P[2] = geom.B[2] + geom.C[2] - geom.O[2];
    
    geom.E[0] = geom.D[0] + geom.C[0] - geom.O[0];
    geom.E[1] = 0.0;
    geom.E[2] = geom.D[2] + geom.C[2] - geom.O[2];
    
    let designParams = designFromCr4Geometry(geom);
    if (get().cr4ConstrainParallelogram) {
      designParams.L_FG = designParams.L_CH;
      designParams.L_HG = Math.hypot(designParams.F_offset_x, designParams.F_offset_z);
      edit.geometry = cr4GeometryFromDesign(designParams);
    }

    const validation = validateCr4Geometry(edit.geometry as Cr4Geometry);
    
    set({
      editRobot: edit,
      cr4DesignParams: designParams,
      cr4ValidationIssues: validation.issues,
      isSet: false,
      meshesStale: get().visualMode === 'meshes' ? true : get().meshesStale
    });
  },

  updateCr4DesignParameter: (key, value) => {
    const edit = cloneRobotSpec(get().editRobot);
    if (edit.kind !== 'CR4') return;

    const currentParams = get().cr4DesignParams;
    if (!currentParams) return;

    const updatedParams = {
      ...currentParams,
      [key]: value
    };

    if (get().cr4ConstrainParallelogram) {
      updatedParams.L_FG = updatedParams.L_CH;
      updatedParams.L_HG = Math.hypot(updatedParams.F_offset_x, updatedParams.F_offset_z);
    }

    const derivedGeom = cr4GeometryFromDesign(updatedParams);
    edit.geometry = derivedGeom;

    const valParams = validateCr4DesignParameters(updatedParams);
    const valGeom = validateCr4Geometry(derivedGeom);
    const combinedIssues = [...valParams.issues, ...valGeom.issues];

    set({
      editRobot: edit,
      cr4DesignParams: updatedParams,
      cr4ValidationIssues: combinedIssues,
      isSet: false,
      meshesStale: get().visualMode === 'meshes' ? true : get().meshesStale
    });
  },

  setCr4ConstrainParallelogram: (value) => {
    set({ cr4ConstrainParallelogram: value });
    const edit = cloneRobotSpec(get().editRobot);
    if (edit.kind !== 'CR4') return;

    const currentParams = get().cr4DesignParams;
    if (!currentParams) return;

    const updatedParams = { ...currentParams };
    if (value) {
      updatedParams.L_FG = updatedParams.L_CH;
      updatedParams.L_HG = Math.hypot(updatedParams.F_offset_x, updatedParams.F_offset_z);
    }

    const derivedGeom = cr4GeometryFromDesign(updatedParams);
    edit.geometry = derivedGeom;

    const valParams = validateCr4DesignParameters(updatedParams);
    const valGeom = validateCr4Geometry(derivedGeom);
    const combinedIssues = [...valParams.issues, ...valGeom.issues];

    set({
      editRobot: edit,
      cr4DesignParams: updatedParams,
      cr4ValidationIssues: combinedIssues,
      isSet: false,
      meshesStale: get().visualMode === 'meshes' ? true : get().meshesStale
    });
  },

  loadRobotSpec: (spec) => {
    get().revokeAllObjectUrls();
    const edit = cloneRobotSpec(spec);
    let issues: Cr4GeometryValidationIssue[] = [];
    let params: Cr4DesignParameters | null = null;
    if (edit.kind === 'CR4') {
      params = designFromCr4Geometry(edit.geometry as Cr4Geometry);
      if (get().cr4ConstrainParallelogram) {
        params.L_FG = params.L_CH;
        params.L_HG = Math.hypot(params.F_offset_x, params.F_offset_z);
      }
      const valParams = validateCr4DesignParameters(params);
      const valGeom = validateCr4Geometry(edit.geometry as Cr4Geometry);
      issues = [...valParams.issues, ...valGeom.issues];
      const hasErrors = issues.some(issue => issue.severity === 'error');
      if (!hasErrors) {
        edit.geometry = cr4GeometryFromDesign(params);
      }
    }
    const hasMeshes = edit.visuals.some(vis => vis.kind === 'mesh');
    const meshVisualsBackup = hasMeshes ? JSON.parse(JSON.stringify(edit.visuals)) : null;
    const urlsToTrack: string[] = [];
    edit.visuals.forEach(vis => {
      if (vis.meshUrl && vis.meshUrl.startsWith('blob:')) {
        urlsToTrack.push(vis.meshUrl);
      }
    });
    set({
      editRobot: edit,
      isSet: false,
      cr4DesignParams: params,
      cr4ValidationIssues: issues,
      visualMode: hasMeshes ? 'meshes' : 'primitives',
      meshesStale: false,
      meshWarnings: [],
      primitiveVisualsBackup: null,
      meshVisualsBackup,
      objectUrls: urlsToTrack
    });
  },

  setRobot: () => {
    const edit = cloneRobotSpec(get().editRobot);
    if (edit.kind === 'CR4') {
      const params = get().cr4DesignParams;
      if (params) {
        const valParams = validateCr4DesignParameters(params);
        if (!valParams.ok) {
          set({ cr4ValidationIssues: valParams.issues });
          return;
        }
        edit.geometry = cr4GeometryFromDesign(params);
      }
      const validation = validateCr4Geometry(edit.geometry as Cr4Geometry);
      set({ cr4ValidationIssues: validation.issues });
      if (!validation.ok) {
        return;
      }
    }
    const initialQ = edit.kind === 'CR6' ? Array(6).fill(0) : Array(4).fill(0);
    set({
      editRobot: edit,
      activeRobot: cloneRobotSpec(edit),
      isSet: true,
      q: initialQ,
      torqueLog: null,
      sizingResults: null,
      playbackPoints: [],
      playbackIndex: 0
    });
  },

  updateQ: (newQ) => {
    const active = get().activeRobot;
    let clampedQ = [...newQ];
    if (active.kind === 'CR6') {
      const eng = new Serial6Engine(active);
      clampedQ = eng.clampConfiguration(newQ);
    } else {
      const eng = new PalletizerEngine(active);
      clampedQ = eng.clampConfiguration(newQ);
    }
    set({ q: clampedQ });
  },

  updateTCP: (position, targetYaw) => {
    const active = get().activeRobot;
    const currentQ = get().q;
    if (active.kind === 'CR6') {
      const eng = new Serial6Engine(active);
      const currentFK = eng.forwardKinematics(currentQ);
      const T = [...currentFK.tcp_transform.map(r => [...r])];
      T[0][3] = position[0];
      T[1][3] = position[1];
      T[2][3] = position[2];
      const ik = eng.solveSphericalWristIK(T, currentQ);
      if (ik.success) {
        set({ q: ik.q });
        return true;
      }
    } else {
      const eng = new PalletizerEngine(active);
      const ik = eng.solveIK(position, currentQ, targetYaw);
      if (ik.success) {
        set({ q: ik.q });
        return true;
      }
    }
    return false;
  },

  setTab: (tab) => set({ activeTab: tab }),
  setEngine: (engine) => set({ activeEngine: engine }),
  setHasBackend: (has) => set({ hasBackend: has }),
  setBackendUrl: (url) => {
    set({ backendUrl: url });
    setApiBackendUrl(url);
  },

  toggleGrid: () => set({ showGrid: !get().showGrid }),
  toggleAxes: () => set({ showAxes: !get().showAxes }),
  toggleCOMs: () => set({ showCOMs: !get().showCOMs }),
  toggleTrajectory: () => set({ showTrajectory: !get().showTrajectory }),
  toggleTCPFrame: () => set({ showTCPFrame: !get().showTCPFrame }),

  checkBackendStatus: async () => {
    if (import.meta.env.VITE_ENABLE_PRO_BACKEND === 'false') {
      set({ hasBackend: false, backendState: 'disconnected' });
      return;
    }
    const wasDisconnected = get().backendState === 'disconnected';
    if (get().backendState === 'disconnected' || get().backendState === 'detecting') {
      set({ backendState: 'detecting' });
    }
    const currentUrl = get().backendUrl;
    let res = await pingBackendUrl(currentUrl, 800);
    
    if (!res.connected) {
      const primary = import.meta.env.VITE_PRO_BACKEND_URL_PRIMARY || 'http://127.0.0.1:8001';
      const fallback = import.meta.env.VITE_PRO_BACKEND_URL_FALLBACK || 'http://localhost:8001';
      const fallbackUrl = currentUrl.includes(primary) ? fallback : primary;
      res = await pingBackendUrl(fallbackUrl, 800);
    }
    
    if (res.connected) {
      set({
        backendState: res.incompatible ? 'incompatible' : 'connected',
        backendUrl: res.url,
        backendVersion: res.version,
        pinocchioVersion: res.pinocchioVersion,
        licenseStatus: res.licenseStatus,
        backendCapabilities: res.capabilities,
        hasBackend: true,
        consecutiveBackendFailures: 0
      });
      setApiBackendUrl(res.url);
      if (wasDisconnected && !res.incompatible) {
        set({ activeEngine: 'backend' });
      }
    } else {
      const nextFailures = get().consecutiveBackendFailures + 1;
      set({ consecutiveBackendFailures: nextFailures });
      
      const isRecording = get().isRecording;
      if (nextFailures >= 3 && !isRecording) {
        set({
          backendState: 'disconnected',
          backendVersion: null,
          pinocchioVersion: null,
          licenseStatus: null,
          backendCapabilities: null,
          hasBackend: false
        });
        if (get().activeEngine === 'backend') {
          set({ activeEngine: 'frontend' });
        }
      }
    }
  },

  toggleSolidVisibility: (bodyName) => {
    const editRobot = cloneRobotSpec(get().editRobot);
    const activeRobot = cloneRobotSpec(get().activeRobot);

    const editVisible = editRobot.visuals.find(v => v.body === bodyName)?.visible;
    editRobot.visuals.forEach((vis) => {
      if (vis.body === bodyName) {
        vis.visible = !(editVisible ?? true);
      }
    });

    const activeVisible = activeRobot.visuals.find(v => v.body === bodyName)?.visible;
    activeRobot.visuals.forEach((vis) => {
      if (vis.body === bodyName) {
        vis.visible = !(activeVisible ?? true);
      }
    });

    set({ editRobot, activeRobot });
  },

  toggleAxesVisibility: (bodyName) => {
    const editRobot = cloneRobotSpec(get().editRobot);
    const activeRobot = cloneRobotSpec(get().activeRobot);

    const editAxesVisible = editRobot.visuals.find(v => v.body === bodyName)?.axesVisible !== false;
    editRobot.visuals.forEach((vis) => {
      if (vis.body === bodyName) {
        vis.axesVisible = !editAxesVisible;
      }
    });

    const activeAxesVisible = activeRobot.visuals.find(v => v.body === bodyName)?.axesVisible !== false;
    activeRobot.visuals.forEach((vis) => {
      if (vis.body === bodyName) {
        vis.axesVisible = !activeAxesVisible;
      }
    });

    set({ editRobot, activeRobot });
  },

  addTarget: (target) => {
    const updatedTargets = [...get().program.targets];
    const idx = updatedTargets.findIndex(t => t.name === target.name);
    if (idx !== -1) {
      updatedTargets[idx] = target;
    } else {
      updatedTargets.push(target);
    }
    set({
      program: {
        ...get().program,
        targets: updatedTargets
      }
    });
  },

  removeTarget: (name) => {
    set({
      program: {
        ...get().program,
        targets: get().program.targets.filter(t => t.name !== name),
        instructions: get().program.instructions.filter(i => i.type === 'Pause' || i.target_name !== name)
      }
    });
  },

  addInstruction: (instruction) => {
    set({
      program: {
        ...get().program,
        instructions: [...get().program.instructions, instruction]
      }
    });
  },

  removeInstruction: (index) => {
    set({
      program: {
        ...get().program,
        instructions: get().program.instructions.filter((_, idx) => idx !== index)
      }
    });
  },

  clearProgram: () => set({ program: defaultProgram(), cr4ValidationIssues: [] }),
  loadProgram: (program) => set({ program, cr4ValidationIssues: [] }),

  runSignalRecording: async () => {
    const active = get().activeRobot;
    const program = get().program;
    if (program.instructions.length === 0) return;

    set({ isRecording: true });

    try {
      if (get().activeEngine === 'backend') {
        const startQ = active.kind === 'CR6' ? Array(6).fill(0) : Array(4).fill(0);
        const trajectoryPoints = buildProgramDynamicsTrajectory(startQ, program, active, 0.005);
        const samples = trajectoryPoints.map(pt => ({
          time_s: pt.time_s,
          q: pt.q,
          qd: pt.qd,
          qdd: pt.qdd
        }));

        const data = await calculateBackendBatchDynamics(active, samples);
        if (data) {
          // Normalize: backend samples use `velocity`/`acceleration`, but TorqueSample
          // requires `joint_velocity`/`joint_acceleration` for downstream chart/CSV rendering.
          const normalizedSamples = data.samples.map((s: any) => ({
            ...s,
            joint_velocity: s.velocity,
            joint_acceleration: s.acceleration
          }));
          const trajectory = normalizedSamples.map((s: any) => ({
            time_s: s.time_s,
            q: s.q,
            qd: s.velocity,
            qdd: s.acceleration,
            instruction_index: -1
          }));
          set({
            torqueLog: { ...data, samples: normalizedSamples },
            playbackPoints: trajectory,
            playbackIndex: 0,
            isRecording: false
          });
          return;
        } else {
          set({ isRecording: false });
          alert("PRO Dynamics calculation failed. Verify backend server connection.");
          return;
        }
      }

      const { torqueLog, playbackPoints } = generateSimulationTorques(active, program);
      set({
        torqueLog,
        playbackPoints,
        playbackIndex: 0,
        isRecording: false
      });
    } catch (err) {
      console.error('runSignalRecording error:', err);
      set({ isRecording: false });
      alert(`Recording failed unexpectedly: ${err instanceof Error ? err.message : String(err)}`);
    }
  },


  runIterativeActuatorSizing: async (margins, gearboxTypeFilter?: 'harmonic' | 'cycloidal' | 'any') => {
    const active = get().activeRobot;
    const lib = get().actuatorLibrary;
    const torqueLog = get().torqueLog;
    if (!lib || !torqueLog) return;

    const fullMargins: SizingMargins = {
      continuous: margins.continuous,
      peak: margins.peak,
      speed: margins.speed,
      power: margins.power ?? 1.1,
      motorPeakFactor: margins.motorPeakFactor ?? 5.0,
      enforcePowerLimit: margins.enforcePowerLimit ?? false,
      sizingObjective: margins.sizingObjective
    };

    const report = selectActuatorsForLog(
      torqueLog,
      lib,
      fullMargins,
      active.kind,
      active.name,
      { gearboxType: gearboxTypeFilter }
    );

    set({
      sizingResults: report
    });
  },

  loadActuatorLibrary: async () => {
    try {
      const resp = await fetch('/actuators_library.json');
      if (resp.ok) {
        const lib = await resp.json();
        set({ actuatorLibrary: lib });
      }
    } catch (err) {
      console.error('Failed to load actuators library', err);
    }
  },

  setVisualMode: (mode) => {
    if (mode === 'primitives') {
      get().usePrimitiveVisuals();
    } else {
      const hasMeshes = get().editRobot.visuals.some(vis => vis.meshUrl);
      if (hasMeshes) {
        set({ visualMode: 'meshes' });
      } else if (get().meshVisualsBackup) {
        const edit = cloneRobotSpec(get().editRobot);
        const active = cloneRobotSpec(get().activeRobot);
        const restoreMeshes = () => JSON.parse(JSON.stringify(get().meshVisualsBackup));
        edit.visuals = restoreMeshes();
        if (get().isSet) {
          active.visuals = restoreMeshes();
        }
        set({
          editRobot: edit,
          activeRobot: active,
          visualMode: 'meshes',
          meshWarnings: [],
          meshesStale: false
        });
      } else if (get().editRobot.kind === 'CR6') {
        get().applyMeshPreset('irb4600_20kg_250');
      } else if (get().editRobot.kind === 'CR4') {
        get().applyMeshPreset('irb460_palletizer');
      } else {
        set({ visualMode: 'meshes' });
      }
    }
  },

  applyMeshPreset: (presetName) => {
    get().revokeAllObjectUrls();
    if (presetName === 'irb4600_20kg_250') {
      const edit = cloneRobotSpec(get().editRobot);
      const active = cloneRobotSpec(get().activeRobot);
      
      let backup = get().primitiveVisualsBackup;
      if (!backup) {
        backup = JSON.parse(JSON.stringify(edit.visuals));
      }

      const applyMeshes = (visuals: any[]) => {
        return visuals.map(vis => {
          const url = IRB4600_MESH_URLS[vis.body];
          if (url) {
            return {
              ...vis,
              kind: 'mesh' as const,
              meshUrl: url,
              originM: [0, 0, 0] as [number, number, number],
              rpyRad: [0, 0, 0] as [number, number, number],
              scale: [1, 1, 1] as [number, number, number]
            };
          }
          return vis;
        });
      };

      edit.visuals = applyMeshes(edit.visuals);
      if (get().isSet) {
        active.visuals = applyMeshes(active.visuals);
      }

      set({
        editRobot: edit,
        activeRobot: active,
        visualMode: 'meshes',
        primitiveVisualsBackup: backup,
        meshVisualsBackup: JSON.parse(JSON.stringify(edit.visuals)),
        meshWarnings: [],
        meshesStale: false
      });
    } else if (presetName === 'irb460_palletizer') {
      const edit = cloneRobotSpec(get().editRobot);
      const active = cloneRobotSpec(get().activeRobot);

      let backup = get().primitiveVisualsBackup;
      if (!backup) {
        backup = JSON.parse(JSON.stringify(edit.visuals));
      }

      const applyIrb460Meshes = () => {
        const bodies = [
          'FOOT', 'SWING', 'LOWER_ARM', 'P_ARM', 'P_LINK',
          'UPPER_ARM', 'LOWER_LINK', 'LINK_PLATE', 'UPPER_LINK',
          'TILT', 'DISK'
        ];
        return bodies.map(body => ({
          body,
          frameName: body,
          kind: 'mesh' as const,
          meshUrl: IRB460_MESH_URLS[body],
          originM: [0, 0, 0] as [number, number, number],
          rpyRad: [0, 0, 0] as [number, number, number],
          scale: [1, 1, 1] as [number, number, number],
          visible: true
        }));
      };

      edit.visuals = applyIrb460Meshes();
      if (get().isSet) {
        active.visuals = applyIrb460Meshes();
      }

      set({
        editRobot: edit,
        activeRobot: active,
        visualMode: 'meshes',
        primitiveVisualsBackup: backup,
        meshVisualsBackup: JSON.parse(JSON.stringify(edit.visuals)),
        meshWarnings: [],
        meshesStale: false
      });
    }
  },

  usePrimitiveVisuals: () => {
    const edit = cloneRobotSpec(get().editRobot);
    const active = cloneRobotSpec(get().activeRobot);
    const backup = get().primitiveVisualsBackup;

    const restorePrimitives = (currentVisuals: any[]) => {
      if (backup) {
        return JSON.parse(JSON.stringify(backup));
      }
      return currentVisuals.map(vis => ({
        ...vis,
        kind: 'primitive' as const,
        meshUrl: undefined
      }));
    };

    edit.visuals = restorePrimitives(edit.visuals);
    if (get().isSet) {
      active.visuals = restorePrimitives(active.visuals);
    }

    set({
      editRobot: edit,
      activeRobot: active,
      visualMode: 'primitives',
      meshesStale: false,
      meshWarnings: []
    });
  },

  clearMeshWarnings: () => set({ meshWarnings: [] }),

  addMeshWarning: (body, msg) => {
    const current = get().meshWarnings;
    const warnStr = `${body}: ${msg}`;
    if (!current.includes(warnStr)) {
      set({ meshWarnings: [...current, warnStr] });
    }
  },

  uploadPackage: async (files: File[]) => {
    const data = await uploadPackageToBackend(files);
    if (data) {
      get().loadRobotSpec(data);
      set({ visualMode: 'meshes' });
      get().setRobot();
      return true;
    }
    return false;
  },

  revokeAllObjectUrls: () => {
    const urls = get().objectUrls;
    for (const url of urls) {
      try {
        URL.revokeObjectURL(url);
      } catch (err) {
        console.error('Failed to revoke Object URL:', url, err);
      }
    }
    set({ objectUrls: [] });
  },

  // ── Station Objects ──────────────────────────────────────────────────────
  addStationObject: (obj) => {
    set((state) => ({ stationObjects: [...state.stationObjects, obj] }));
  },

  updateStationObject: (id, patch) => {
    set((state) => ({
      stationObjects: state.stationObjects.map((o) =>
        o.id === id ? { ...o, ...patch } : o
      )
    }));
  },

  removeStationObject: (id) => {
    set((state) => ({
      stationObjects: state.stationObjects.filter((o) => o.id !== id)
    }));
  },

  toggleStationObjectVisible: (id) => {
    set((state) => ({
      stationObjects: state.stationObjects.map((o) =>
        o.id === id ? { ...o, visible: !o.visible } : o
      )
    }));
  }
}));

const IRB4600_MESH_URLS: Record<string, string> = {
  BASE: '/meshes/irb4600/IRB4600_20kg-250_BASE.glb',
  LINK1: '/meshes/irb4600/IRB4600_20kg-250_LINK1.glb',
  LINK2: '/meshes/irb4600/IRB4600_20kg-250_LINK2.glb',
  LINK3: '/meshes/irb4600/IRB4600_20kg-250_LINK3.glb',
  LINK4: '/meshes/irb4600/IRB4600_20kg-250_LINK4.glb',
  LINK5: '/meshes/irb4600/IRB4600_20kg-250_LINK5.glb',
  LINK6: '/meshes/irb4600/IRB4600_20kg-250_LINK6.glb',
};

const IRB460_MESH_URLS: Record<string, string> = {
  FOOT: '/meshes/irb460/IRB460_FOOT.glb',
  SWING: '/meshes/irb460/IRB460_SWING.glb',
  LOWER_ARM: '/meshes/irb460/IRB460_LOWER_ARM.glb',
  P_ARM: '/meshes/irb460/IRB460_P_ARM.glb',
  P_LINK: '/meshes/irb460/IRB460_P_LINK.glb',
  UPPER_ARM: '/meshes/irb460/IRB460_UPPER_ARM.glb',
  LOWER_LINK: '/meshes/irb460/IRB460_LOWER_LINK.glb',
  LINK_PLATE: '/meshes/irb460/IRB460_LINK_PLATE.glb',
  UPPER_LINK: '/meshes/irb460/IRB460_UPPER_LINK.glb',
  TILT: '/meshes/irb460/IRB460_TILT.glb',
  DISK: '/meshes/irb460/IRB460_DISK.glb',
};
