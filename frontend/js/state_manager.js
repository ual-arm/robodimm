/**
 * state_manager.js - Unified State Management for DEMO and PRO modes
 * ===================================================================
 * 
 * Provides a factory function that returns a state manager appropriate
 * for the current mode (demo or pro). Both managers expose the same
 * interface, allowing the UI to work without mode-specific conditionals.
 * 
 * Usage:
 *   import { createStateManager } from './js/state_manager.js';
 *   const state = createStateManager('demo'); // or 'pro'
 *   const targets = await state.getTargets();
 */

import { computeFK } from './kinematics_lite.js';
import { getBenchmarkData } from './demo_robot_data.js';

const STORAGE_KEYS = {
  targets: 'robodimm_targets',
  program: 'robodimm_program',
  config: 'robodimm_config',
  currentQ: 'robodimm_currentQ'
};

/**
 * Compute position and rotation from joint angles for benchmark targets.
 * Uses kinematics_lite FK to fill in missing position/rotation data.
 */
function enrichBenchmarkTargets(targets, robotType) {
  console.log('[DEMO] Enriching targets for robot:', robotType, 'targets:', targets.length);
  
  return targets.map(target => {
    // If target already has position and rotation, return as-is
    if (target.position && target.rotation) {
      return target;
    }
    
    // Compute FK to get position and rotation
    try {
      console.log('[DEMO] Computing FK for target:', target.name, 'q:', target.q);
      const result = computeFK(robotType, target.q);
      
      if (!result || !result.transform) {
        throw new Error('FK returned invalid result');
      }
      
      const T = result.transform;
      
      // Extract position from 4x4 transform (last column, first 3 rows)
      const position = [T[0][3], T[1][3], T[2][3]];
      
      // Extract rotation matrix (3x3 top-left)
      const rotation = [
        T[0][0], T[0][1], T[0][2],
        T[1][0], T[1][1], T[1][2],
        T[2][0], T[2][1], T[2][2]
      ];
      
      console.log('[DEMO] FK computed for', target.name, 'pos:', position);
      
      return {
        ...target,
        position,
        rotation
      };
    } catch (e) {
      console.error('[DEMO] Failed to compute FK for target', target.name, e);
      // Return target with default position/rotation to avoid crashes
      return {
        ...target,
        position: target.position || [0, 0, 0],
        rotation: target.rotation || [1, 0, 0, 0, 1, 0, 0, 0, 1]
      };
    }
  });
}

/**
 * Create a state manager for DEMO mode.
 * Uses localStorage for persistence and local FK for kinematics.
 */
function createDemoManager() {
  // Check for benchmark mode URL parameter
  const urlParams = new URLSearchParams(window.location.search);
  const benchmarkMode = urlParams.get('benchmark') === 'true';
  const keepSaved = urlParams.get('keep_saved') === 'true';

  // Keep DEMO defaults aligned with PRO startup unless explicitly overridden.
  if (!keepSaved) {
    localStorage.removeItem(STORAGE_KEYS.targets);
    localStorage.removeItem(STORAGE_KEYS.program);
    localStorage.removeItem(STORAGE_KEYS.config);
    localStorage.removeItem(STORAGE_KEYS.currentQ);
  }

  // Check if we have saved data (after optional cleanup)
  const savedTargets = localStorage.getItem(STORAGE_KEYS.targets);
  const savedProgram = localStorage.getItem(STORAGE_KEYS.program);
  
  // Load config to determine robot type
  const config = JSON.parse(localStorage.getItem(STORAGE_KEYS.config) || '{"robot_type":"CR4","scale":1.0}');
  
  // If no saved data OR benchmark mode requested, use benchmark program
  let defaultTargets = '[]';
  let defaultProgram = '[]';
  let defaultQ = '[0,0,0,0]';
  
  if (!savedTargets && !savedProgram || benchmarkMode) {
    if (benchmarkMode) {
      console.log('[DEMO] Benchmark mode requested - clearing saved data and loading benchmark program');
      localStorage.removeItem(STORAGE_KEYS.targets);
      localStorage.removeItem(STORAGE_KEYS.program);
    } else {
      console.log('[DEMO] No saved data - loading benchmark program for PRO/DEMO comparison');
    }
    const benchmark = getBenchmarkData(config.robot_type);
    // Enrich benchmark targets with position/rotation via FK
    const enrichedTargets = enrichBenchmarkTargets(benchmark.targets, config.robot_type);
    defaultTargets = JSON.stringify(enrichedTargets);
    defaultProgram = JSON.stringify(benchmark.program);
    if (benchmark.targets && benchmark.targets.length > 0 && benchmark.targets[0].q) {
      defaultQ = JSON.stringify(benchmark.targets[0].q);
    }
  }
  
  // Load state from localStorage or use defaults
  let loadedTargets = JSON.parse(localStorage.getItem(STORAGE_KEYS.targets) || defaultTargets);
  
  // Check if loaded targets need enrichment (missing position/rotation)
  if (loadedTargets.length > 0 && (!loadedTargets[0].position || !loadedTargets[0].rotation)) {
    console.log('[DEMO] Enriching loaded targets with FK data');
    loadedTargets = enrichBenchmarkTargets(loadedTargets, config.robot_type);
  }
  
  // Final safety check: if still no valid targets, load benchmark
  if (!loadedTargets || loadedTargets.length === 0 || !loadedTargets[0].position) {
    console.log('[DEMO] No valid targets, forcing benchmark load');
    const benchmark = getBenchmarkData(config.robot_type);
    loadedTargets = enrichBenchmarkTargets(benchmark.targets, config.robot_type);
    defaultProgram = JSON.stringify(benchmark.program);
  }
  
  const state = {
    q: JSON.parse(localStorage.getItem(STORAGE_KEYS.currentQ) || defaultQ),
    targets: loadedTargets,
    program: JSON.parse(localStorage.getItem(STORAGE_KEYS.program) || defaultProgram),
    config: config,
  };

  function persist(key, value) {
    localStorage.setItem(STORAGE_KEYS[key], JSON.stringify(value));
  }

  return {
    mode: 'demo',

    // =================================================================
    // Joint State
    // =================================================================
    
    getQ: () => [...state.q],
    
    setQ: (q) => {
      state.q = [...q];
      persist('currentQ', state.q);
      return { ok: true, q: state.q };
    },

    jogJoint: (index, delta) => {
      const newQ = [...state.q];
      if (index >= 0 && index < newQ.length) {
        newQ[index] += delta;
      }
      state.q = newQ;
      persist('currentQ', state.q);
      
      // Compute FK for EE position
      const fk = computeFK(state.config.robot_type, state.q, state.config.scale);
      return { 
        ok: true, 
        q: state.q,
        ee_pos: fk ? [fk.ee.x, fk.ee.y, fk.ee.z] : null
      };
    },

    // =================================================================
    // Targets
    // =================================================================
    
    getTargets: () => [...state.targets],
    
    saveTarget: (name) => {
      // Compute current pose via FK
      const fk = computeFK(state.config.robot_type, state.q, state.config.scale);
      
      const target = {
        name: name,
        q: [...state.q],
        position: fk ? [fk.ee.x, fk.ee.y, fk.ee.z] : [0, 0, 0],
        rotation: fk ? flattenRotation(fk.transform) : [1,0,0,0,1,0,0,0,1]
      };
      
      // Replace if exists
      const idx = state.targets.findIndex(t => t.name === name);
      if (idx >= 0) {
        state.targets[idx] = target;
      } else {
        state.targets.push(target);
      }
      persist('targets', state.targets);
      return { ok: true, target };
    },
    
    deleteTarget: (name) => {
      state.targets = state.targets.filter(t => t.name !== name);
      // Also remove from program
      state.program = state.program.filter(p => p.target_name !== name);
      persist('targets', state.targets);
      persist('program', state.program);
      return { ok: true };
    },

    // =================================================================
    // Program
    // =================================================================
    
    getProgram: () => [...state.program],
    
    addInstruction: (instr) => {
      state.program.push(instr);
      persist('program', state.program);
      return { ok: true, instruction: instr, index: state.program.length - 1 };
    },
    
    deleteInstruction: (index) => {
      if (index >= 0 && index < state.program.length) {
        state.program.splice(index, 1);
        persist('program', state.program);
        return { ok: true };
      }
      return { ok: false, error: 'Index out of range' };
    },
    
    clearProgram: () => {
      state.program = [];
      persist('program', state.program);
      return { ok: true };
    },

    // =================================================================
    // Configuration
    // =================================================================
    
    getConfig: () => ({ ...state.config }),
    
    setConfig: (cfg) => {
      Object.assign(state.config, cfg);
      persist('config', state.config);
      return { ok: true, config: state.config };
    },

    // =================================================================
    // Import/Export
    // =================================================================
    
    exportProgram: () => {
      return JSON.stringify({
        robot_type: state.config.robot_type,
        scale: state.config.scale,
        targets: state.targets,
        program: state.program,
        exported_at: new Date().toISOString()
      }, null, 2);
    },
    
    importProgram: (json) => {
      try {
        const data = JSON.parse(json);
        if (data.targets) {
          state.targets = data.targets;
          persist('targets', state.targets);
        }
        if (data.program) {
          state.program = data.program;
          persist('program', state.program);
        }
        if (data.robot_type) {
          state.config.robot_type = data.robot_type;
        }
        if (data.scale) {
          state.config.scale = data.scale;
        }
        persist('config', state.config);
        return { ok: true };
      } catch (e) {
        return { ok: false, error: e.message };
      }
    },

    // =================================================================
    // Cartesian jog / Orientation jog (not available in DEMO)
    // =================================================================

    jogCartesian: (delta, frame) => {
      return { ok: false, error: 'Cartesian jog requires PRO mode (IK solver)' };
    },

    jogOrientation: (delta, frame) => {
      return { ok: false, error: 'Orientation jog requires PRO mode (IK solver)' };
    },

    // =================================================================
    // Execution / Trajectory (handled by interpolation.js in DEMO)
    // =================================================================

    executeProgram: async (speedFactor = 1.0) => {
      return { ok: false, error: 'Use interpolation.js executeProgram() in DEMO mode' };
    },

    getTrajectoryData: () => {
      return { ok: false, error: 'Dynamics analysis requires PRO mode (Pinocchio)' };
    },
  };
}

/**
 * Create a state manager for PRO mode.
 * Delegates all operations to the backend via fetch API.
 */
function createProManager(apiBase, getHeaders) {
  const baseUrl = apiBase || window.location.origin;
  
  async function fetchJson(endpoint, options = {}) {
    const url = baseUrl + endpoint;
    const res = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...getHeaders(),
        ...options.headers
      }
    });
    return res.json();
  }

  return {
    mode: 'pro',

    // =================================================================
    // Joint State
    // =================================================================
    
    getQ: async () => {
      const data = await fetchJson('/robot_info');
      return data.q || [];
    },
    
    setQ: async (q) => {
      return fetchJson('/set_joint', {
        method: 'POST',
        body: JSON.stringify({ q })
      });
    },

    jogJoint: async (index, delta) => {
      return fetchJson('/jog_joint', {
        method: 'POST',
        body: JSON.stringify({ index, delta })
      });
    },

    jogCartesian: async (delta, frame) => {
      return fetchJson('/jog_cartesian', {
        method: 'POST',
        body: JSON.stringify({ delta, frame })
      });
    },

    jogOrientation: async (delta, frame) => {
      return fetchJson('/jog_orientation', {
        method: 'POST',
        body: JSON.stringify({ delta, frame })
      });
    },

    // =================================================================
    // Targets
    // =================================================================
    
    getTargets: async () => {
      const data = await fetchJson('/targets');
      return data.targets || [];
    },
    
    saveTarget: async (name) => {
      return fetchJson('/save_target', {
        method: 'POST',
        body: JSON.stringify({ name })
      });
    },
    
    deleteTarget: async (name) => {
      return fetchJson('/delete_target', {
        method: 'POST',
        body: JSON.stringify({ name })
      });
    },

    // =================================================================
    // Program
    // =================================================================
    
    getProgram: async () => {
      const data = await fetchJson('/program');
      return data.program || [];
    },
    
    addInstruction: async (instr) => {
      return fetchJson('/add_instruction', {
        method: 'POST',
        body: JSON.stringify(instr)
      });
    },
    
    deleteInstruction: async (index) => {
      return fetchJson('/delete_instruction', {
        method: 'POST',
        body: JSON.stringify({ index })
      });
    },
    
    clearProgram: async () => {
      return fetchJson('/clear_program', { method: 'POST' });
    },

    // =================================================================
    // Configuration
    // =================================================================
    
    getConfig: async () => {
      const data = await fetchJson('/robot_config');
      return data;
    },
    
    setConfig: async (cfg) => {
      return fetchJson('/robot_config', {
        method: 'POST',
        body: JSON.stringify(cfg)
      });
    },

    // =================================================================
    // Import/Export
    // =================================================================
    
    exportProgram: async () => {
      const targets = await fetchJson('/targets');
      const program = await fetchJson('/program');
      return JSON.stringify({
        ...targets,
        ...program,
        exported_at: new Date().toISOString()
      }, null, 2);
    },
    
    importProgram: async (json) => {
      // Parse and re-create targets + program via individual API calls
      const data = JSON.parse(json);
      // Note: Backend /save_target saves the current robot position under the given name.
      // Full target import with stored joint values requires a batch import endpoint
      // that doesn't exist yet. For now, we save the instructions only.
      if (data.program) {
        await fetchJson('/clear_program', { method: 'POST' });
        for (const instr of data.program) {
          await fetchJson('/add_instruction', {
            method: 'POST',
            body: JSON.stringify(instr)
          });
        }
      }
      return { ok: true, warning: 'Target positions not restored (requires batch import endpoint)' };
    },

    // =================================================================
    // Execution
    // =================================================================
    
    executeProgram: async (speedFactor = 1.0) => {
      return fetchJson('/execute_program', {
        method: 'POST',
        body: JSON.stringify({ speed_factor: speedFactor })
      });
    },

    getTrajectoryData: async () => {
      return fetchJson('/trajectory_data');
    }
  };
}

/**
 * Factory function to create the appropriate state manager.
 * 
 * @param {string} mode - 'demo' or 'pro'
 * @param {Object} options - Configuration options (for PRO mode)
 * @returns {Object} State manager with unified interface
 */
export function createStateManager(mode, options = {}) {
  if (mode === 'demo') {
    return createDemoManager();
  }
  return createProManager(options.apiBase, options.getHeaders || (() => ({})));
}

// Helper function to flatten rotation matrix from 4x4 to 9 elements
function flattenRotation(transform) {
  // transform is 4x4 array, we want the 3x3 rotation part
  return [
    transform[0][0], transform[0][1], transform[0][2],
    transform[1][0], transform[1][1], transform[1][2],
    transform[2][0], transform[2][1], transform[2][2]
  ];
}

export default { createStateManager };
