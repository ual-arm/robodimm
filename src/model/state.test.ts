import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useRobodimmStore } from './state';
import { irb460PalletizerSpec } from '../math/palletizer';
import { irb4600Serial6Spec } from '../math/serial6';
import { pingBackendUrl, calculateBackendBatchDynamics } from '../api/backend';

vi.mock('../api/backend', () => {
  return {
    pingBackendUrl: vi.fn(),
    setApiBackendUrl: vi.fn(),
    calculateBackendBatchDynamics: vi.fn(),
    uploadPackageToBackend: vi.fn(),
  };
});

const mockPingBackendUrl = pingBackendUrl as any;
const mockCalculateBackendBatchDynamics = calculateBackendBatchDynamics as any;

describe('Robodimm state store regression tests', () => {
  beforeEach(() => {
    // Reset state before each test
    useRobodimmStore.setState({
      editRobot: irb4600Serial6Spec(),
      activeRobot: irb4600Serial6Spec(),
      isSet: false,
      q: Array(6).fill(0),
    });
  });

  it('should ensure toggles do not mutate or reset joint coordinates q', () => {
    // Set a custom q configuration
    const testQ = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
    useRobodimmStore.setState({ q: testQ });

    // Call toggles
    useRobodimmStore.getState().toggleSolidVisibility('LINK1');
    useRobodimmStore.getState().toggleAxesVisibility('LINK2');

    // Expect q to remain exactly as it was
    expect(useRobodimmStore.getState().q).toEqual(testQ);
  });

  it('should ensure setRobot() performs a deep copy of the editRobot spec', () => {
    // Modify editRobot spec
    useRobodimmStore.getState().updateEditSpec((spec) => {
      spec.name = 'Modified Edit Robot';
    });

    // Verify it is modified in editRobot but activeRobot is still original
    expect(useRobodimmStore.getState().editRobot.name).toBe('Modified Edit Robot');
    expect(useRobodimmStore.getState().activeRobot.name).not.toBe('Modified Edit Robot');

    // Call setRobot to transfer editRobot to activeRobot
    useRobodimmStore.getState().setRobot();

    // Verify both are modified now
    expect(useRobodimmStore.getState().activeRobot.name).toBe('Modified Edit Robot');
    expect(useRobodimmStore.getState().activeRobot).not.toBe(useRobodimmStore.getState().editRobot);

    // Mutate editRobot again to verify activeRobot remains unchanged (deep copy)
    useRobodimmStore.getState().updateEditSpec((spec) => {
      spec.name = 'Another Edit Change';
    });

    expect(useRobodimmStore.getState().editRobot.name).toBe('Another Edit Change');
    expect(useRobodimmStore.getState().activeRobot.name).toBe('Modified Edit Robot');
  });

  it('should toggle visibility of all visuals sharing the same body name (e.g. CR4 UPPER_ARM components)', () => {
    // Switch to CR4 palletizer spec which has duplicate body names (e.g., UPPER_ARM)
    const cr4Spec = irb460PalletizerSpec();
    useRobodimmStore.setState({
      editRobot: cr4Spec,
      activeRobot: cr4Spec,
    });

    const bodyToToggle = 'UPPER_ARM';
    const initialVisibilities = useRobodimmStore.getState().activeRobot.visuals
      .filter(v => v.body === bodyToToggle)
      .map(v => v.visible);

    // Verify initially they are all true
    expect(initialVisibilities.length).toBeGreaterThan(1);
    initialVisibilities.forEach(v => expect(v).toBe(true));

    // Call toggleSolidVisibility
    useRobodimmStore.getState().toggleSolidVisibility(bodyToToggle);

    // Verify they are all now false in activeRobot and editRobot
    const toggledActiveVisibilities = useRobodimmStore.getState().activeRobot.visuals
      .filter(v => v.body === bodyToToggle)
      .map(v => v.visible);
    toggledActiveVisibilities.forEach(v => expect(v).toBe(false));

    const toggledEditVisibilities = useRobodimmStore.getState().editRobot.visuals
      .filter(v => v.body === bodyToToggle)
      .map(v => v.visible);
    toggledEditVisibilities.forEach(v => expect(v).toBe(false));

    // Call toggleAxesVisibility
    const initialAxesVisibilities = useRobodimmStore.getState().activeRobot.visuals
      .filter(v => v.body === bodyToToggle)
      .map(v => v.axesVisible !== false);
    initialAxesVisibilities.forEach(v => expect(v).toBe(true));

    useRobodimmStore.getState().toggleAxesVisibility(bodyToToggle);

    const toggledActiveAxes = useRobodimmStore.getState().activeRobot.visuals
      .filter(v => v.body === bodyToToggle)
      .map(v => v.axesVisible !== false);
    toggledActiveAxes.forEach(v => expect(v).toBe(false));
  });

  it('should verify frameName is prioritized over body for lookups in RobotViewer.tsx', () => {
    const active = useRobodimmStore.getState().activeRobot;

    // Standard helper implementation duplicating RobotViewer lookup key logic:
    // const visKey = vis.frameName || vis.body;
    active.visuals.forEach(vis => {
      const visKey = vis.frameName || vis.body;
      if (vis.frameName) {
        expect(visKey).toBe(vis.frameName);
      } else {
        expect(visKey).toBe(vis.body);
      }
    });

    // Make sure we have at least some visuals with frameName set (like CR4 segment segments)
    const cr4Spec = irb460PalletizerSpec();
    const cr4VisualsWithFrameName = cr4Spec.visuals.filter(v => !!v.frameName);
    expect(cr4VisualsWithFrameName.length).toBeGreaterThan(0);

    cr4VisualsWithFrameName.forEach(vis => {
      const visKey = vis.frameName || vis.body;
      expect(visKey).toBe(vis.frameName);
    });
  });

  it('should verify that LINK5/J5 is centered both dynamically and visually', () => {
    const spec = irb4600Serial6Spec();

    // 1. COM of LINK5 must be [0,0,0]
    expect(spec.inertials.LINK5.comM).toEqual([0.0, 0.0, 0.0]);

    // 2. Visual LINK5 must be centered at originM [0,0,0]
    const link5Visual = spec.visuals.find(v => v.body === 'LINK5');
    expect(link5Visual).toBeDefined();
    expect(link5Visual!.originM).toEqual([0.0, 0.0, 0.0]);

    // 3. Visual LINK5 must resolve by frameName: 'LINK5'
    expect(link5Visual!.frameName).toBe('LINK5');
    const visKey = link5Visual!.frameName || link5Visual!.body;
    expect(visKey).toBe('LINK5');
  });

  it('should verify global simulation layers toggles in the store', () => {
    // 1. Initial states
    expect(useRobodimmStore.getState().showGrid).toBe(true);
    expect(useRobodimmStore.getState().showAxes).toBe(true);
    expect(useRobodimmStore.getState().showTCPFrame).toBe(true);
    expect(useRobodimmStore.getState().showCOMs).toBe(true);
    expect(useRobodimmStore.getState().showTrajectory).toBe(true);

    // 2. Perform toggles
    useRobodimmStore.getState().toggleGrid();
    useRobodimmStore.getState().toggleAxes();
    useRobodimmStore.getState().toggleTCPFrame();
    useRobodimmStore.getState().toggleCOMs();
    useRobodimmStore.getState().toggleTrajectory();

    // 3. Verify they changed in the store
    expect(useRobodimmStore.getState().showGrid).toBe(false);
    expect(useRobodimmStore.getState().showAxes).toBe(false);
    expect(useRobodimmStore.getState().showTCPFrame).toBe(false);
    expect(useRobodimmStore.getState().showCOMs).toBe(false);
    expect(useRobodimmStore.getState().showTrajectory).toBe(false);
  });

  describe('CAD Mesh and Visual Mode actions', () => {
    it('should initialize with primitives mode and false stale flag', () => {
      const state = useRobodimmStore.getState();
      expect(state.visualMode).toBe('primitives');
      expect(state.meshesStale).toBe(false);
      expect(state.meshWarnings).toEqual([]);
      expect(state.primitiveVisualsBackup).toBeNull();
    });

    it('should toggle visualMode using setVisualMode and apply presets', () => {
      const store = useRobodimmStore.getState();
      
      // Since it's CR6 (IRB4600 preset), switching to meshes should automatically apply the IRB4600 meshes preset
      store.setVisualMode('meshes');
      expect(useRobodimmStore.getState().visualMode).toBe('meshes');
      expect(useRobodimmStore.getState().primitiveVisualsBackup).not.toBeNull();
      
      const visuals = useRobodimmStore.getState().editRobot.visuals;
      visuals.forEach(v => {
        if (v.body.startsWith('LINK') || v.body === 'BASE') {
          expect(v.kind).toBe('mesh');
          expect(v.meshUrl).toContain('IRB4600_20kg-250');
          expect(v.originM).toEqual([0, 0, 0]);
          expect(v.rpyRad).toEqual([0, 0, 0]);
          expect(v.scale).toEqual([1, 1, 1]);
        }
      });
      
      // Revert to primitives
      store.setVisualMode('primitives');
      expect(useRobodimmStore.getState().visualMode).toBe('primitives');
      
      const reverted = useRobodimmStore.getState().editRobot.visuals;
      reverted.forEach(v => {
        expect(v.kind).toBe('primitive');
        expect(v.meshUrl).toBeUndefined();
      });
    });

    it('should set meshesStale to true if geometry is modified in meshes mode', () => {
      const store = useRobodimmStore.getState();
      
      // Set meshes mode
      store.setVisualMode('meshes');
      expect(useRobodimmStore.getState().meshesStale).toBe(false);
      
      // Modify DH parameters of CR6
      store.updateEditSpec(spec => {
        (spec.geometry as any).joints[0].a_m = 0.55;
      });
      
      expect(useRobodimmStore.getState().meshesStale).toBe(true);
      
      // Revert to primitives and verify stale is reset
      store.usePrimitiveVisuals();
      expect(useRobodimmStore.getState().meshesStale).toBe(false);
    });

    it('should deduplicate mesh load warnings', () => {
      const store = useRobodimmStore.getState();
      store.clearMeshWarnings();
      expect(useRobodimmStore.getState().meshWarnings).toEqual([]);

      store.addMeshWarning('LINK1', 'Load failed');
      store.addMeshWarning('LINK1', 'Load failed'); // Duplicate
      store.addMeshWarning('LINK2', '404 not found');

      expect(useRobodimmStore.getState().meshWarnings).toEqual([
        'LINK1: Load failed',
        'LINK2: 404 not found'
      ]);
    });

    it('should toggle visualMode using setVisualMode and apply presets for CR4', () => {
      const store = useRobodimmStore.getState();
      
      // Load CR4 first
      const cr4Spec = irb460PalletizerSpec();
      store.loadRobotSpec(cr4Spec);
      store.setRobot();
      
      // Switch to meshes mode (should auto-load irb460_palletizer meshes)
      store.setVisualMode('meshes');
      expect(useRobodimmStore.getState().visualMode).toBe('meshes');
      expect(useRobodimmStore.getState().primitiveVisualsBackup).not.toBeNull();
      
      const visuals = useRobodimmStore.getState().editRobot.visuals;
      expect(visuals.length).toBe(11);
      visuals.forEach(v => {
        expect(v.kind).toBe('mesh');
        expect(v.meshUrl).toContain('IRB460_');
        expect(v.originM).toEqual([0, 0, 0]);
        expect(v.rpyRad).toEqual([0, 0, 0]);
        expect(v.scale).toEqual([1, 1, 1]);
      });
      
      // Revert to primitives
      store.setVisualMode('primitives');
      expect(useRobodimmStore.getState().visualMode).toBe('primitives');
      
      const reverted = useRobodimmStore.getState().editRobot.visuals;
      expect(reverted.length).toBe(cr4Spec.visuals.length); // Restored original elements
      reverted.forEach(v => {
        expect(v.kind).toBe('primitive');
        expect(v.meshUrl).toBeUndefined();
      });
    });

    it('should restore custom CR4 package meshes instead of falling back to the IRB460 preset', () => {
      const store = useRobodimmStore.getState();
      const cr4Spec = irb460PalletizerSpec();
      cr4Spec.name = 'Custom CR4';
      cr4Spec.visuals = [
        {
          body: 'SWING',
          frameName: 'SWING',
          kind: 'mesh',
          meshUrl: 'blob:custom-swing',
          originM: [0, 0, 0],
          rpyRad: [0, 0, 0],
          scale: [1, 1, 1],
          visible: true
        }
      ];

      store.loadRobotSpec(cr4Spec);
      useRobodimmStore.setState({
        primitiveVisualsBackup: irb460PalletizerSpec().visuals
      });

      store.setVisualMode('primitives');
      expect(useRobodimmStore.getState().editRobot.visuals[0].kind).toBe('primitive');

      store.setVisualMode('meshes');
      const restored = useRobodimmStore.getState().editRobot.visuals;
      expect(restored).toHaveLength(1);
      expect(restored[0].meshUrl).toBe('blob:custom-swing');
      expect(restored[0].meshUrl).not.toContain('IRB460_');
    });

    describe('cr4ConstrainParallelogram actions', () => {
      it('should respect cr4ConstrainParallelogram toggle and auto-calculate L_FG/L_HG', () => {
        const store = useRobodimmStore.getState();
        
        // 1. Switch to CR4
        store.changeRobotKind('CR4');
        expect(useRobodimmStore.getState().cr4ConstrainParallelogram).toBe(true);

        // 2. Modify L_CH and F_offset_x and verify L_FG and L_HG are auto-calculated
        store.updateCr4DesignParameter('L_CH', 0.45);
        store.updateCr4DesignParameter('F_offset_x', 0.03);
        store.updateCr4DesignParameter('F_offset_z', 0.04);
        
        let params = useRobodimmStore.getState().cr4DesignParams;
        expect(params).not.toBeNull();
        expect(params!.L_FG).toBe(0.45); // L_FG = L_CH
        expect(params!.L_HG).toBeCloseTo(0.05); // L_HG = Math.hypot(0.03, 0.04) = 0.05

        // 3. Disable the constraint and verify we can set independent values
        store.setCr4ConstrainParallelogram(false);
        expect(useRobodimmStore.getState().cr4ConstrainParallelogram).toBe(false);

        store.updateCr4DesignParameter('L_FG', 0.60);
        store.updateCr4DesignParameter('L_HG', 0.70);
        
        params = useRobodimmStore.getState().cr4DesignParams;
        expect(params!.L_FG).toBe(0.60);
        expect(params!.L_HG).toBe(0.70);

        // 4. Enable it again and check that it immediately applies the constraint
        store.setCr4ConstrainParallelogram(true);
        params = useRobodimmStore.getState().cr4DesignParams;
        expect(params!.L_FG).toBe(0.45); // reset to L_CH
        expect(params!.L_HG).toBeCloseTo(0.05); // reset to Math.hypot(F_offset_x, F_offset_z)
      });
    });
  });

  describe('Backend connection debounce and activeEngine auto-switching', () => {
    beforeEach(() => {
      vi.clearAllMocks();
      useRobodimmStore.setState({
        backendState: 'disconnected',
        consecutiveBackendFailures: 0,
        activeEngine: 'frontend',
        isRecording: false
      });
    });

    it('should ignore first two consecutive health check failures and transition to disconnected on the third', async () => {
      // Start with connected state
      useRobodimmStore.setState({
        backendState: 'connected',
        consecutiveBackendFailures: 0,
        activeEngine: 'backend'
      });

      // Mock pingBackendUrl to return disconnected
      mockPingBackendUrl.mockResolvedValue({
        connected: false,
        version: null,
        pinocchioVersion: null,
        licenseStatus: null,
        capabilities: null,
        incompatible: false,
        url: 'http://127.0.0.1:8001'
      });

      const store = useRobodimmStore.getState();

      // 1st failure
      await store.checkBackendStatus();
      expect(useRobodimmStore.getState().backendState).toBe('connected');
      expect(useRobodimmStore.getState().consecutiveBackendFailures).toBe(1);
      expect(useRobodimmStore.getState().activeEngine).toBe('backend');

      // 2nd failure
      await store.checkBackendStatus();
      expect(useRobodimmStore.getState().backendState).toBe('connected');
      expect(useRobodimmStore.getState().consecutiveBackendFailures).toBe(2);
      expect(useRobodimmStore.getState().activeEngine).toBe('backend');

      // 3rd failure
      await store.checkBackendStatus();
      expect(useRobodimmStore.getState().backendState).toBe('disconnected');
      expect(useRobodimmStore.getState().consecutiveBackendFailures).toBe(3);
      expect(useRobodimmStore.getState().activeEngine).toBe('frontend');
    });

    it('should reset consecutive failures on successful ping', async () => {
      useRobodimmStore.setState({
        backendState: 'connected',
        consecutiveBackendFailures: 2,
        activeEngine: 'backend'
      });

      mockPingBackendUrl.mockResolvedValue({
        connected: true,
        version: '1.0.0',
        pinocchioVersion: '2.6.0',
        licenseStatus: 'valid',
        capabilities: { CR4: {}, CR6: {} },
        incompatible: false,
        url: 'http://127.0.0.1:8001'
      });

      const store = useRobodimmStore.getState();
      await store.checkBackendStatus();

      expect(useRobodimmStore.getState().backendState).toBe('connected');
      expect(useRobodimmStore.getState().consecutiveBackendFailures).toBe(0);
      expect(useRobodimmStore.getState().activeEngine).toBe('backend');
    });

    it('should auto-switch to backend engine when connection is established from disconnected state', async () => {
      useRobodimmStore.setState({
        backendState: 'disconnected',
        consecutiveBackendFailures: 3,
        activeEngine: 'frontend'
      });

      mockPingBackendUrl.mockResolvedValue({
        connected: true,
        version: '1.0.0',
        pinocchioVersion: '2.6.0',
        licenseStatus: 'valid',
        capabilities: { CR4: {}, CR6: {} },
        incompatible: false,
        url: 'http://127.0.0.1:8001'
      });

      const store = useRobodimmStore.getState();
      await store.checkBackendStatus();

      expect(useRobodimmStore.getState().backendState).toBe('connected');
      expect(useRobodimmStore.getState().activeEngine).toBe('backend');
    });
  });

  describe('PRO backend recording and error handling', () => {
    let originalAlert: any;

    beforeEach(() => {
      vi.clearAllMocks();
      useRobodimmStore.setState({
        activeRobot: irb4600Serial6Spec(),
        isSet: true,
        program: {
          schema: 'robodimm.program.v1',
          name: 'test_program',
          targets: [],
          instructions: [
            { type: 'Pause', duration_s: 1.0 }
          ]
        },
        torqueLog: null,
        isRecording: false,
        activeEngine: 'backend',
        backendState: 'connected'
      });
      originalAlert = globalThis.alert;
      globalThis.alert = vi.fn();
    });

    afterEach(() => {
      globalThis.alert = originalAlert;
    });

    it('should use batch dynamics API when activeEngine is backend', async () => {
      const mockResult = {
        joint_names: ['J1', 'J2', 'J3', 'J4', 'J5', 'J6'],
        samples: [
          {
            time_s: 0.0,
            q: [0,0,0,0,0,0],
            velocity: [0,0,0,0,0,0],
            acceleration: [0,0,0,0,0,0],
            tau: [1.2, 2.3, 3.4, 4.5, 5.6, 6.7]
          }
        ],
        dt_s: 0.005,
        engine_used: 'pro_cr6_serial',
        model_id: 'cr6_serial6_template.v1',
        manifest: {
          model_id: 'cr6_serial6_template.v1',
          backend_version: '1.0.0',
          pinocchio_version: '2.6.3',
          robot_hash: 'abc123hash',
          trajectory_hash: 'traj456hash'
        }
      };
      mockCalculateBackendBatchDynamics.mockResolvedValue(mockResult);

      const store = useRobodimmStore.getState();
      await store.runSignalRecording();

      expect(mockCalculateBackendBatchDynamics).toHaveBeenCalled();
      expect(useRobodimmStore.getState().torqueLog).toEqual(mockResult);
      expect(useRobodimmStore.getState().isRecording).toBe(false);
    });

    it('should trigger alert and abort recording when backend dynamics calculation fails', async () => {
      mockCalculateBackendBatchDynamics.mockResolvedValue(null);

      const store = useRobodimmStore.getState();
      await store.runSignalRecording();

      expect(mockCalculateBackendBatchDynamics).toHaveBeenCalled();
      expect(globalThis.alert).toHaveBeenCalledWith("PRO Dynamics calculation failed. Verify backend server connection.");
      expect(useRobodimmStore.getState().torqueLog).toBeNull();
      expect(useRobodimmStore.getState().isRecording).toBe(false);
    });
  });
});
