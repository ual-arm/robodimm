/**
 * app.js - Robot Commander Frontend
 * ==================================
 * 
 * Three.js-based 3D visualization and control interface for the robot.
 * 
 * Supports two modes:
 * - DEMO (mode=demo): No backend, Math.js kinematics, limited features
 * - PRO (default): Full backend with Pinocchio/Pink, all features
 * 
 * Sections:
 * 1. IMPORTS & INITIALIZATION - ES modules, API base URL, mode detection
 * 2. DOM ELEMENTS - References to HTML controls
 * ...
 */

// =============================================================================
// SECTION 1: IMPORTS & INITIALIZATION
// =============================================================================

import * as THREE from 'three';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.149.0/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.149.0/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'https://cdn.jsdelivr.net/npm/three@0.149.0/examples/jsm/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'https://cdn.jsdelivr.net/npm/three@0.149.0/examples/jsm/libs/meshopt_decoder.module.js';
import { getDemoRobotData, getRobotList } from './js/demo_robot_data.js';
import { computeFK, getRobotDH, getRobotInfo } from './js/kinematics_lite.js';
import { createStateManager } from './js/state_manager.js';
import { executeProgram, calculateMoveJDuration, generateProgramTrajectory } from './js/interpolation.js';
import { computeInverseDynamics, getDynamicsInfo } from './js/dynamics_lite.js';
import { analyzeTrajectoryRequirementsLite, selectActuatorsLite, validateSelectionLite } from './js/actuators_lite.js';

console.log('[APP] Initialization started');

// =============================================================================
// MODE DETECTION - DEMO vs PRO
// =============================================================================

const urlParams = new URLSearchParams(window.location.search);
const APP_MODE = urlParams.get('mode') || 'pro';

const FEATURES = {
  // Now available in both modes
  TARGETS: true,                      // Both: save/load targets
  PROGRAM_EDITOR: true,               // Both: edit programs
  MOVEJ: true,                        // Both: joint motion
  
  // PRO only features
  LINEAR_JOG: APP_MODE !== 'demo',    // PRO: cartesian jog needs IK
  MOVEL: APP_MODE !== 'demo',         // PRO: linear motion needs IK
  MOVEC: APP_MODE !== 'demo',         // PRO: circular motion
  DYNAMICS: APP_MODE !== 'demo',      // PRO: dynamics analysis
  ACTUATORS: true,                    // Both: actuator sizing (PRO backend / DEMO lite)
  BACKEND: APP_MODE !== 'demo',       // PRO: has backend
  LOGIN: APP_MODE !== 'demo',         // PRO: requires login
  
  // Configuration
  CONFIG_ROBOT_TYPE: true,            // Both: CR4/CR6 selection
  CONFIG_SCALE: true,                 // Both: robot scaling
  CONFIG_PAYLOAD: APP_MODE !== 'demo' // PRO: payload affects dynamics
};

console.log('[APP] Mode:', APP_MODE);
console.log('[APP] Features:', FEATURES);

// API base URL - auto-detect environment
const apiBase = window.location.origin;
console.log('[APP] apiBase:', apiBase);
console.log('[APP] Running in', window.location.pathname.includes('/robot') ? 'server mode' : 'local mode');

// Global robot data (used by DEMO mode)
let robotData = null;
let robotInfo = null;  // Cached robot info from kinematics_lite
let debugSpheres = [];
let simplifiedLinks = {};  // Stores simplified Three.js cylinder shapes for DEMO mode links
let eeAxes = null;

// Store last dynamics data for CSV export (DEMO mode)
let lastDemoDynamicsData = null;

// Global function reference to update robot visualization
window.updateRobotViz = null;

// Global function reference to update EE (TCP) visualization
window.updateEE = null;

// Create state manager (unified interface for DEMO and PRO)
const stateManager = createStateManager(APP_MODE, {
  apiBase: apiBase,
  getHeaders: () => ({
    'Content-Type': 'application/json',
    'X-Session-ID': sessionId
  })
});

// Control object for program execution (allows stopping)
const executionControl = { cancelled: false, speedFactor: 1.0 };

// Hide login overlay in DEMO mode immediately
if (!FEATURES.LOGIN) {
  const loginOverlay = document.getElementById('login-overlay');
  if (loginOverlay) loginOverlay.style.display = 'none';
}

// Session Management
// Check if we have a stored session
let sessionId = localStorage.getItem('robodimm_session_id');

function getHeaders() {
  return {
    'Content-Type': 'application/json',
    'X-Session-ID': sessionId
  };
}



// =============================================================================
// SECTION 2: DOM ELEMENTS
// =============================================================================

// JOG controls
const jointInput = document.getElementById('joint_input');
const setJointBtn = document.getElementById('set_joint');
const targetInput = document.getElementById('target_input');
const moveLinearBtn = document.getElementById('move_linear');
const qDisplay = document.getElementById('q_display');
const jogOriDeltaInput = document.getElementById('jog_ori_delta');

// Programming DOM
const targetNameInput = document.getElementById('target_name');
const saveTargetBtn = document.getElementById('save_target');
const targetsTree = document.getElementById('targets-tree');
const programTree = document.getElementById('program-tree');
const instrTypeSelect = document.getElementById('instr_type');
const instrSpeedInput = document.getElementById('instr_speed');
const instrZoneInput = document.getElementById('instr_zone');
const instrPauseTimeInput = document.getElementById('instr_pause_time');
const instrViaTargetSelect = document.getElementById('instr_via_target');
const addPauseBtn = document.getElementById('add_pause_btn');
const speedRow = document.getElementById('speed_row');
const zoneRow = document.getElementById('zone_row');
const pauseRow = document.getElementById('pause_row');
const viaRow = document.getElementById('via_row');
const playProgramBtn = document.getElementById('play_program');
const clearProgramBtn = document.getElementById('clear_program');
const speedFactorInput = document.getElementById('speed_factor');
const plotDynamicsBtn = document.getElementById('plot_dynamics');
const exportCsvBtn = document.getElementById('export_csv');
const plotVariableSelect = document.getElementById('plot_variable');

// Save/Load program DOM
const programNameInput = document.getElementById('program_name');
const programDescInput = document.getElementById('program_desc');
const saveProgramBtn = document.getElementById('save_program_btn');
const refreshProgramsBtn = document.getElementById('refresh_programs_btn');
const savedProgramsList = document.getElementById('saved_programs_list');

// Edit modal DOM
const editModal = document.getElementById('edit-modal');
const modalOverlay = document.getElementById('modal-overlay');
const editInstrType = document.getElementById('edit_instr_type');
const editInstrTarget = document.getElementById('edit_instr_target');
const editInstrSpeed = document.getElementById('edit_instr_speed');
const editInstrZone = document.getElementById('edit_instr_zone');
const editInstrPauseTime = document.getElementById('edit_instr_pause_time');
const editTargetRow = document.getElementById('edit_target_row');
const editSpeedRow = document.getElementById('edit_speed_row');
const editZoneRow = document.getElementById('edit_zone_row');
const editPauseRow = document.getElementById('edit_pause_row');
const cancelEditBtn = document.getElementById('cancel_edit');
const saveEditBtn = document.getElementById('save_edit');


// =============================================================================
// SECTION 3: TAB SYSTEM
// =============================================================================

// Tab switching logic
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  });
});

// Fullscreen toggle for dynamics
const toggleFullscreenBtn = document.getElementById('toggle_fullscreen');
const controlPanel = document.getElementById('control-panel');
let isFullscreen = false;

toggleFullscreenBtn.addEventListener('click', () => {
  isFullscreen = !isFullscreen;
  if (isFullscreen) {
    controlPanel.classList.add('fullscreen-dynamics');
    toggleFullscreenBtn.textContent = '⛶ Exit Fullscreen';
    // Resize chart
    if (dynamicsChart) {
      setTimeout(() => dynamicsChart.resize(), 100);
    }
  } else {
    controlPanel.classList.remove('fullscreen-dynamics');
    toggleFullscreenBtn.textContent = '⛶ Fullscreen';
    if (dynamicsChart) {
      setTimeout(() => dynamicsChart.resize(), 100);
    }
  }
});


// =============================================================================
// SECTION 4: INSTRUCTION TYPE HANDLING
// =============================================================================

// Show/hide relevant fields based on instruction type (MoveJ, MoveL, MoveC, Pause)
function updateInstrFields() {
  const type = instrTypeSelect.value;
  if (type === 'Pause') {
    speedRow.style.display = 'none';
    zoneRow.style.display = 'none';
    pauseRow.style.display = 'flex';
    viaRow.style.display = 'none';
    addPauseBtn.style.display = 'block';
  } else if (type === 'MoveC') {
    speedRow.style.display = 'flex';
    zoneRow.style.display = 'flex';
    pauseRow.style.display = 'none';
    viaRow.style.display = 'flex';
    addPauseBtn.style.display = 'none';
  } else {
    speedRow.style.display = 'flex';
    zoneRow.style.display = 'flex';
    pauseRow.style.display = 'none';
    viaRow.style.display = 'none';
    addPauseBtn.style.display = 'none';
  }
}

instrTypeSelect.addEventListener('change', updateInstrFields);
updateInstrFields();

// Add Pause button handler
addPauseBtn.addEventListener('click', async () => {
  const pauseTime = parseFloat(instrPauseTimeInput.value) || 1;
  const result = await stateManager.addInstruction({
    type: 'Pause',
    pause_time: pauseTime
  });
  if (result.ok) {
    await loadProgram();
  }
});

// Edit modal type change handler
function updateEditInstrFields() {
  const type = editInstrType.value;
  if (type === 'Pause') {
    editTargetRow.style.display = 'none';
    editSpeedRow.style.display = 'none';
    editZoneRow.style.display = 'none';
    editPauseRow.style.display = 'flex';
  } else {
    editTargetRow.style.display = 'flex';
    editSpeedRow.style.display = 'flex';
    editZoneRow.style.display = 'flex';
    editPauseRow.style.display = 'none';
  }
}

editInstrType.addEventListener('change', updateEditInstrFields);


// =============================================================================
// SECTION 5: FRAME REFERENCE TOGGLE
// =============================================================================

// Toggle between BASE and TOOL reference frames for jogging
let currentFrame = 'base';
document.querySelectorAll('.frame-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.frame-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFrame = btn.dataset.frame;
  });
});

function currentJogFrame() {
  return currentFrame;
}


// =============================================================================
// SECTION 6: UTILITY FUNCTIONS
// =============================================================================

// Chart.js instance for dynamics plots
let dynamicsChart = null;

// Local state tracking
let localTargets = [];
let localProgram = [];
let isAnimating = false;
let currentQ = [0, 0, 0, 0];  // Current joint values for slider sync (4 user joints for CR4)
let editingInstructionIndex = -1;
// let sliderInteractionActive = false;  // Disabled - not the right solution

/**
 * Get float value from input element with fallback.
 */
function getFloat(id, fallback) {
  const v = document.getElementById(id)?.value;
  return parseFloat(v) || fallback;
}

/**
 * Sync slider UI with current joint values.
 * Works with any number of joints based on current robot.
 */
function updateSlidersFromQ(q) {
  currentQ = q.slice();
  const nq = q.length;
  for (let i = 0; i < nq; i++) {
    const slider = document.getElementById(`slider_j${i}`);
    const valDisplay = document.getElementById(`slider_j${i}_val`);
    if (slider && valDisplay && q[i] !== undefined) {
      slider.value = q[i];
      valDisplay.textContent = (q[i] * 180 / Math.PI).toFixed(1) + '°';
    }
  }
}

/**
 * Convert 3x3 rotation matrix (flat array) to ZYX Euler angles in degrees.
 * Used for displaying target orientations in a human-readable format.
 */
function rotMatToEulerZYX(rot) {
  // rot is a flat array [r00, r01, r02, r10, r11, r12, r20, r21, r22]
  const r00 = rot[0], r01 = rot[1], r02 = rot[2];
  const r10 = rot[3], r11 = rot[4], r12 = rot[5];
  const r20 = rot[6], r21 = rot[7], r22 = rot[8];

  // ZYX Euler angles
  const sy = Math.sqrt(r00 * r00 + r10 * r10);
  const singular = sy < 1e-6;

  let rx, ry, rz;
  if (!singular) {
    rx = Math.atan2(r21, r22);
    ry = Math.atan2(-r20, sy);
    rz = Math.atan2(r10, r00);
  } else {
    rx = Math.atan2(-r12, r11);
    ry = Math.atan2(-r20, sy);
    rz = 0;
  }

  // Convert to degrees
  return [
    (rz * 180 / Math.PI).toFixed(1),
    (ry * 180 / Math.PI).toFixed(1),
    (rx * 180 / Math.PI).toFixed(1)
  ];
}


// =============================================================================
// SECTION 7: API CALLS (JOG Endpoints)
// =============================================================================

// Set Joint handler - direct joint input
setJointBtn.onclick = async () => {
  const q = jointInput.value.split(',').map(s => parseFloat(s.trim()));
  await setJointDirect(q);
};

// Move Linear handler (PRO only - requires IK)
moveLinearBtn.onclick = async () => {
  if (!FEATURES.LINEAR_JOG) return;
  const t = targetInput.value.split(',').map(s => parseFloat(s.trim()));
  const res = await fetch(apiBase + '/move_linear', {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ target: t })
  });
  const r = await res.json();
  if (r.q) {
    updateRobotViz(r.q);
    updateSlidersFromQ(r.q);
  }
};

// Joint jog
async function jogJoint(index, delta) {
  const r = await stateManager.jogJoint(index, delta);
  if (r.q) {
    currentQ = r.q;
    updateSlidersFromQ(r.q);
    updateRobotViz(r.q);
  }
}

// Set joint directly (for sliders)
async function setJointDirect(q) {
  const qNumeric = q.map(v => parseFloat(v) || 0);
  
  // Update state manager
  const result = await stateManager.setQ(qNumeric);
  
  // Use server's q if provided (confirmation), otherwise use sent values
  if (result.ok && result.q) {
    currentQ = result.q;
  } else {
    currentQ = qNumeric.slice();
  }
  
  // Update visualization
  if (window.updateRobotViz) {
    window.updateRobotViz(currentQ);
  }
  
  // Update TCP visualization if EE pose provided (PRO mode)
  if (result.ee_pos && window.updateEE) {
    window.updateEE(result.ee_pos, result.ee_rot);
  }
  
  return result;
}

// Cartesian jog
async function jogCartesian(dx, dy, dz) {
  if (!FEATURES.LINEAR_JOG) return;
  const r = await stateManager.jogCartesian([dx, dy, dz], currentJogFrame());
  if (r.q) {
    updateRobotViz(r.q);
    updateSlidersFromQ(r.q);
  }
  if (r.ee_pos && window.updateEE) window.updateEE(r.ee_pos, r.ee_rot);
}

// Orientation jog
async function jogOrientation(delta) {
  if (!FEATURES.LINEAR_JOG) return;
  const r = await stateManager.jogOrientation(delta, currentJogFrame());
  if (r.q) {
    updateRobotViz(r.q);
    updateSlidersFromQ(r.q);
  }
  if (r.ee_pos && window.updateEE) window.updateEE(r.ee_pos, r.ee_rot);
}


// =============================================================================
// SECTION 8: HOLD-TO-JOG FUNCTIONALITY
// =============================================================================

/**
 * Setup continuous action while button is held down.
 * Triggers immediate action on press, then repeats every 100ms.
 */
function setupHoldButton(btn, action) {
  let intervalId = null;
  let isHolding = false;

  const startAction = (e) => {
    e.preventDefault();
    if (isHolding) return;
    isHolding = true;
    action(); // Immediate first action
    intervalId = setInterval(action, 100); // Repeat every 100ms
  };

  const stopAction = () => {
    isHolding = false;
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  };

  btn.addEventListener('mousedown', startAction);
  btn.addEventListener('mouseup', stopAction);
  btn.addEventListener('mouseleave', stopAction);
  btn.addEventListener('touchstart', startAction);
  btn.addEventListener('touchend', stopAction);
}

// Wire up joint jog buttons with hold functionality
for (let i = 0; i < 6; i++) {
  const plus = document.getElementById(`jog_j${i}_plus`);
  const minus = document.getElementById(`jog_j${i}_minus`);
  if (plus) setupHoldButton(plus, () => jogJoint(i, Math.abs(getFloat('jog_joint_delta', 0.05))));
  if (minus) setupHoldButton(minus, () => jogJoint(i, -Math.abs(getFloat('jog_joint_delta', 0.05))));
}

// Wire up cartesian jog buttons with hold functionality
setupHoldButton(document.getElementById('jog_x_plus'), () => jogCartesian(getFloat('jog_cart_delta', 0.01), 0, 0));
setupHoldButton(document.getElementById('jog_x_minus'), () => jogCartesian(-getFloat('jog_cart_delta', 0.01), 0, 0));
setupHoldButton(document.getElementById('jog_y_plus'), () => jogCartesian(0, getFloat('jog_cart_delta', 0.01), 0));
setupHoldButton(document.getElementById('jog_y_minus'), () => jogCartesian(0, -getFloat('jog_cart_delta', 0.01), 0));
setupHoldButton(document.getElementById('jog_z_plus'), () => jogCartesian(0, 0, getFloat('jog_cart_delta', 0.01)));
setupHoldButton(document.getElementById('jog_z_minus'), () => jogCartesian(0, 0, -getFloat('jog_cart_delta', 0.01)));

// Wire up orientation jog buttons with hold functionality
const oriButtons = [
  ['jog_r_minus', 0, -1], ['jog_r_plus', 0, 1],
  ['jog_p_minus', 1, -1], ['jog_p_plus', 1, 1],
  ['jog_yaw_minus', 2, -1], ['jog_yaw_plus', 2, 1]
];

oriButtons.forEach(([id, axis, sign]) => {
  const btn = document.getElementById(id);
  if (btn) {
    setupHoldButton(btn, () => {
      const delta = [0, 0, 0];
      delta[axis] = sign * getFloat('jog_ori_delta', 0.05);
      jogOrientation(delta);
    });
  }
});


// =============================================================================
// SECTION 9: JOINT SLIDERS (Real-time control)
// =============================================================================

/**
 * Get the number of user-controlled joints for current robot
 */
function getUserJointCount() {
  // DEMO mode: use robotInfo from kinematics_lite
  if (robotInfo) return robotInfo.nq;
  // PRO mode: use robotData from backend
  if (robotData && robotData.nq) return robotData.nq;
  return 5;
}

/**
 * Check if robot has dependent joints (like CR4's J4_aux)
 */
function hasDependentJoints() {
  // DEMO mode: use robotInfo from kinematics_lite
  if (robotInfo) return robotInfo.hasDependentJoints || false;
  // PRO mode: infer from robot type
  if (robotData && robotData.robot_type) {
    return robotData.robot_type === 'CR4';  // CR4 has dependent J4_aux
  }
  return false;
}

// Unified function to hide/show sliders and jog buttons based on DOF (nq)
function updateJointSlidersVisibility(nq) {
  // Hide J5, J6 controls if robot has fewer joints
  for (let i = 0; i < 6; i++) {
    const container = document.getElementById(`slider_j${i}`)?.parentElement;
    const btnMinus = document.getElementById(`jog_j${i}_minus`);
    const btnPlus = document.getElementById(`jog_j${i}_plus`);
    
    const visible = (i < nq);
    if (container) container.style.display = visible ? 'flex' : 'none';
    if (btnMinus) btnMinus.style.display = visible ? 'inline-block' : 'none';
    if (btnPlus) btnPlus.style.display = visible ? 'inline-block' : 'none';
  }
}

// Setup slider event listeners with proper closure capture
// Number of sliders matches user joint count (nq_user from robot model)
function setupSliders() {
  const nq = getUserJointCount();
  
  // Ensure currentQ has the correct size for this robot
  if (currentQ.length !== nq) {
    console.log(`[APP] Resizing currentQ from ${currentQ.length} to ${nq} joints`);
    currentQ = new Array(nq).fill(0).map((_, i) => currentQ[i] || 0);
  }
  
  for (let idx = 0; idx < 6; idx++) {
    const slider = document.getElementById(`slider_j${idx}`);
    const valDisplay = document.getElementById(`slider_j${idx}_val`);

    if (slider) {
      // Clean up existing listeners by cloning (simple way to reset)
      const newSlider = slider.cloneNode(true);
      slider.parentNode.replaceChild(newSlider, slider);
      
      // Update valDisplay reference after cloning (parent might have changed)
      const newValDisplay = document.getElementById(`slider_j${idx}_val`);
      
      if (idx < nq) {
        // Both modes: immediate update with backend (fluid motion)
        newSlider.addEventListener('input', async () => {
          if (isAnimating) return;
          
          const value = parseFloat(newSlider.value);
          if (newValDisplay) newValDisplay.textContent = (value * 180 / Math.PI).toFixed(1) + '°';
          
          // Ensure currentQ has correct size before copying
          if (currentQ.length !== nq) {
            currentQ = new Array(nq).fill(0).map((_, i) => currentQ[i] || 0);
          }
          
          const newQ = [...currentQ];
          newQ[idx] = value;
          
          // Update robot immediately (both modes)
          await setJointDirect(newQ);
        });
      }
    }
  }
}


// =============================================================================
// SECTION 10: THREE.JS SCENE SETUP
// =============================================================================

console.log('[APP] Creating Three.js scene...');
const container = document.getElementById('scene');

if (!container) {
  console.error('[APP] ERROR: scene container not found!');
} else {
  // Login UI Elements (Inside scope to see loadRobotStructure)
  const loginOverlay = document.getElementById('login-overlay');
  const loginUser = document.getElementById('login_user');
  const loginPass = document.getElementById('login_pass');
  const loginBtn = document.getElementById('login_btn');
  const loginError = document.getElementById('login_error');

  async function performLogin() {
    const user = loginUser.value;
    const pass = loginPass.value;
    loginError.textContent = 'Logging in...';
    loginBtn.disabled = true;

    try {
      const res = await fetch(apiBase + '/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user, password: pass })
      });

      if (res.ok) {
        const data = await res.json();
        sessionId = data.session_id;
        localStorage.setItem('robodimm_session_id', sessionId);
        console.log('[AUTH] Login success, session:', sessionId);

        // Hide login, init app
        loginOverlay.classList.remove('active');
        initApp();

      } else {
        loginError.textContent = 'Invalid credentials';
        loginBtn.disabled = false;
      }
    } catch (e) {
      console.error('Login error', e);
      loginError.textContent = 'Connection error';
      loginBtn.disabled = false;
    }
  }

  loginBtn.addEventListener('click', performLogin);

  loginPass.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performLogin();
  });

  async function initApp() {
    if (!sessionId) {
      if (!FEATURES.LOGIN) {
        loadRobotStructure();
        loadStationGeometries();
        loadTargets();
        loadProgram();
        loadRobotConfig();
        return;
      }
      loginOverlay.classList.add('active');
      return;
    }

    try {
      if (!FEATURES.BACKEND) {
        loadRobotStructure();
        loadStationGeometries();
        loadTargets();
        loadProgram();
        loadRobotConfig();
        return;
      }
      
      const config = await stateManager.getConfig();
      if (config.status === 401 || (config.detail && config.detail.includes('Not authenticated'))) {
        localStorage.removeItem('robodimm_session_id');
        sessionId = null;
        loginOverlay.classList.add('active');
      } else {
        // App can now see these functions inside the closure
        loadRobotStructure();
        loadStationGeometries();
        loadTargets();
        loadProgram();
        loadRobotConfig();
        setupWebSocket();
      }
    } catch (e) {
      console.error('[AUTH] Validation check failed', e);
    }
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  // Get actual container size
  const rect = container.getBoundingClientRect();
  const width = rect.width || 800;
  const height = rect.height || 600;

  const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 20);
  camera.position.set(1.2, 1.2, 1.2);
  camera.up.set(0, 0, 1); // Z up to match Pinocchio
  camera.lookAt(0, 0, 0.4);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  // Better color appearance (especially for imported GLTF materials)
  // Keep compatibility across Three.js versions.
  if ('outputColorSpace' in renderer) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  } else {
    renderer.outputEncoding = THREE.sRGBEncoding;
  }
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.90;
  container.appendChild(renderer.domElement);

  // Handle window resize
  window.addEventListener('resize', () => {
    const rect = container.getBoundingClientRect();
    camera.aspect = rect.width / rect.height;
    camera.updateProjectionMatrix();
    renderer.setSize(rect.width, rect.height);
  });

  // OrbitControls (ES module)
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = true;
  controls.minDistance = 0.5;
  controls.maxDistance = 10;
  controls.maxPolarAngle = Math.PI;
  if (controls.target) controls.target.set(0, 0, 0.4);
  controls.update();

  // Lights
  const ambient = new THREE.AmbientLight(0xffffff, 0.50);
  scene.add(ambient);

  const dir = new THREE.DirectionalLight(0xffffff, 0.75);
  dir.position.set(2, 2, 3);
  dir.castShadow = true;
  scene.add(dir);

  const dir2 = new THREE.DirectionalLight(0xffffff, 0.3);
  dir2.position.set(-2, -2, 1);
  scene.add(dir2);

  // Grid (XY plane with Z up)
  const grid = new THREE.GridHelper(2, 20, 0x444444, 0x333333);
  grid.rotation.x = -Math.PI / 2;
  scene.add(grid);

  // Origin axes
  const axesHelper = new THREE.AxesHelper(0.4);
  scene.add(axesHelper);

  // Robot visualization group
  const robotGroup = new THREE.Group();
  scene.add(robotGroup);

  // Debug: spheres for DH reference frames (DEMO mode visualization aid)
  function createDebugSpheres() {
    debugSpheres.forEach(s => scene.remove(s));
    debugSpheres.length = 0;
    
    if (!robotData || !robotData.dh) return;
    
    const colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff];
    // Create one sphere per DH joint with uniform size
    const sphereRadius = 0.03;  // Uniform radius for all debug spheres
    for (let i = 0; i < robotData.dh.length; i++) {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(sphereRadius, 32, 24),
        new THREE.MeshStandardMaterial({ 
          color: colors[i % colors.length], 
          emissive: colors[i % colors.length], 
          emissiveIntensity: 0.8,
          transparent: true,
          opacity: 0.9
        })
      );
      scene.add(sphere);
      debugSpheres.push(sphere);
    }
    console.log('[APP] Created', debugSpheres.length, 'debug spheres for', robotData.dh.length, 'joints');
  }

  // Create simplified link shapes using Three.js primitives (connecting DH frames)
  // Works for any robot type (CR4, CR6, etc.) - creates cylinders between DH frames
  function createSimplifiedLinks() {
    // Remove existing simplified links
    Object.values(simplifiedLinks).forEach(link => scene.remove(link));
    simplifiedLinks = {};
    
    if (!robotData || !robotData.dh) return;
    
    const scale = robotData.scale || 1.0;
    const colors = [0xff8800, 0x00ff88, 0x8800ff, 0xff0088, 0xff00ff, 0x0088ff];
    
    // Create one cylinder for each joint (connecting to next joint)
    // The actual length is computed dynamically in updateRobotVizDemo based on FK frames
    for (let i = 0; i < robotData.dh.length; i++) {
      const geometry = new THREE.CylinderGeometry(0.0125 * scale, 0.0125 * scale, 1.0, 16);
      const material = new THREE.MeshStandardMaterial({ 
        color: colors[i % colors.length],
        metalness: 0.3,
        roughness: 0.7,
        transparent: true,
        opacity: 0.8
      });
      
      const mesh = new THREE.Mesh(geometry, material);
      mesh.userData.linkIndex = i;
      
      scene.add(mesh);
      simplifiedLinks[`link${i+1}`] = mesh;
    }
    
    console.log('[APP] Created', Object.keys(simplifiedLinks).length, 'simplified cylinder links');
  }

  // TCP axes helper only (no red sphere)
  eeAxes = new THREE.AxesHelper(0.1);
  scene.add(eeAxes);
  eeAxes.position.set(0.25, 0, 0.8);

  // Update EE visualization and overlay
  function updateEE(pos, rot) {
    eeAxes.position.set(pos[0], pos[1], pos[2]);

    // Update overlay
    document.getElementById('ee_x').textContent = pos[0].toFixed(3);
    document.getElementById('ee_y').textContent = pos[1].toFixed(3);
    document.getElementById('ee_z').textContent = pos[2].toFixed(3);

    if (rot && rot.length === 9) {
      const mat3 = new THREE.Matrix3();
      mat3.set(rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8]);
      const quat = new THREE.Quaternion();
      quat.setFromRotationMatrix(new THREE.Matrix4().setFromMatrix3(mat3));
      eeAxes.quaternion.copy(quat);
    }
  }
  
  // Expose globally
  window.updateEE = updateEE;


  // ===========================================================================
  // SECTION 11: GLTF LOADING (Custom meshes with caching)
  // ===========================================================================

  // Robot structure data from backend (now global)
  let linkMeshes = {};

  // GLTF loader and cache for efficient mesh reuse
  const gltfLoader = new GLTFLoader();
  const dracoLoader = new DRACOLoader();
  dracoLoader.setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.149.0/examples/js/libs/draco/');
  gltfLoader.setDRACOLoader(dracoLoader);
  gltfLoader.setMeshoptDecoder(MeshoptDecoder);
  const gltfCache = new Map();

  /**
   * Load GLTF scene with caching and material tweaking.
   * Returns a deep clone so each instance can be independently transformed.
   */
  async function loadGltfScene(url) {
    if (!url) return null;
    // =========================================================================
    // [!!! OPTIMIZATION DEBT !!!] 
    // DEBUG ONLY: Cache-buster timestamp added to force reload of GLTF meshes.
    // Once the robot model designs are FINAL, remove the `?t=${timestamp}` and 
    // this variable to allow efficient browser caching (ETag / 304).
    // =========================================================================
    const timestamp = new Date().getTime();
    const absUrl = (url.startsWith('http') ? url : (apiBase + (url.startsWith('/') ? '' : '/') + url)) + `?t=${timestamp}`;

    function deepCloneForInstance(src) {
      const cloned = src.clone(true);
      cloned.traverse(obj => {
        if (obj.isMesh) {
          if (obj.geometry) obj.geometry = obj.geometry.clone();
          if (obj.material) {
            if (Array.isArray(obj.material)) {
              obj.material = obj.material.map(m => (m ? m.clone() : m));
            } else {
              obj.material = obj.material.clone();
            }
          }
        }
      });
      return cloned;
    }

    if (gltfCache.has(absUrl)) return deepCloneForInstance(gltfCache.get(absUrl));

    const gltf = await new Promise((resolve, reject) => {
      gltfLoader.load(absUrl, resolve, undefined, reject);
    });
    const sceneObj = gltf.scene || gltf.scenes?.[0];
    if (!sceneObj) return null;

    gltfCache.set(absUrl, sceneObj);
    const instance = deepCloneForInstance(sceneObj);

    // Tweak materials for better visibility under our simple lighting
    instance.traverse(obj => {
      if (!obj.isMesh) return;
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      const tweaked = mats.map(m => {
        if (!m) return m;
        const mm = m;
        // CAD exports can look darker than expected; lift moderately (not too bright).
        if (mm.color) mm.color.multiplyScalar(1.15);
        if (typeof mm.metalness === 'number') mm.metalness = Math.min(mm.metalness, 0.15);
        if (typeof mm.roughness === 'number') mm.roughness = Math.min(mm.roughness, 0.65);
        mm.needsUpdate = true;
        return mm;
      });
      obj.material = (tweaked.length === 1 ? tweaked[0] : tweaked);
    });

    return instance;
  }


  // ===========================================================================
  // SECTION 12: ROBOT VISUALIZATION
  // ===========================================================================

  /**
   * Fetch robot structure from backend and build 3D mesh.
   */
  async function loadRobotStructure() {
    try {
      console.log('[APP] Loading robot structure...');
      
      // Use demo robot data if backend not available
      if (!FEATURES.BACKEND) {
        console.log('[APP] DEMO mode: using local robot data');
        
        // Get config from state manager (includes user-set scale)
        const demoConfig = stateManager.getConfig();
        const robotType = demoConfig.robot_type || 'CR4';
        const scale = demoConfig.scale || 1.0;
        
        console.log(`[APP] DEMO mode init: robotType=${robotType}, config=`, demoConfig);
        
        robotData = getDemoRobotData(robotType);
        robotInfo = getRobotInfo(robotType, scale);
        
        console.log(`[APP] DEMO mode: robotData.nq_user=${robotData.nq_user}, robotInfo.nq=${robotInfo.nq}`);
        
        // Override scale with user config
        robotData.scale = scale;
        
        // DEMO mode: create simplified cylinder links (no GLTF meshes)
        createDebugSpheres();
        createSimplifiedLinks();
        
        // Initialize currentQ with correct size for user joints
        const nq = robotInfo.nq;
        currentQ = robotData.q_home ? robotData.q_home.slice(0, nq) : new Array(nq).fill(0);
        
        console.log(`[APP] DEMO mode: initializing with nq=${nq}, currentQ=[${currentQ.join(', ')}]`);
        
        // Setup UI for this robot
        setupSliders();
        updateJointSlidersVisibility(nq);
        
        updateRobotViz(currentQ);
        updateSlidersFromQ(currentQ);
        return;
      }
      
      // Clear GLTF cache to ensure we get fresh models if they've changed on disk
      gltfCache.clear();
      
      // PRO mode: fetch full robot info (includes geometries for mesh building)
      const robotInfoData = await fetch(apiBase + '/robot_info', {
        headers: getHeaders()
      }).then(r => r.json());
      
      if (!robotInfoData) {
        console.error('[APP] Failed to fetch robot info');
        return;
      }
      
      robotData = robotInfoData;
      console.log('[APP] Robot data loaded:', robotData);
      buildRobotMesh();
      
      // Setup UI for this robot (PRO mode)
      const nq = getUserJointCount();
      setupSliders();
      updateJointSlidersVisibility(nq);
      
      // Initialize robot at home position (backend may not send q_home, use zeros)
      const qHome = robotData.q_home || new Array(nq).fill(0);
      currentQ = qHome.slice();
      updateRobotViz(qHome);
      updateSlidersFromQ(qHome);
    } catch (e) {
      console.error('[APP] Failed to load robot info:', e);
    }
  }

  /**
   * Construct Three.js meshes for all robot geometries.
   * Handles both primitive shapes (cylinder, box, sphere) and custom GLTF meshes.
   */
  function buildRobotMesh() {
    robotGroup.traverse(obj => {
      if (obj !== robotGroup && obj.geometry) obj.geometry.dispose();
    });
    robotGroup.clear();
    linkMeshes = {};

    if (!robotData) return;

    const geoms = robotData.geometries;
    const robotScale = robotData.scale || 1.0;  // Get scale factor from backend
    console.log('[APP] Building mesh for', geoms.length, 'geometries, scale:', robotScale);

    // Wrist joint IDs (typically 4, 5, 6 for j_w1, j_w2, j_w3)
    const wristJointIds = new Set();
    robotData.joints.forEach(j => {
      if (j.name && (j.name.includes('w1') || j.name.includes('w2') || j.name.includes('w3'))) {
        wristJointIds.add(j.id);
      }
    });

    for (const geom of geoms) {
      const linkId = geom.link_id;
      const pos = new THREE.Vector3(...geom.position);
      const r = geom.rotation;
      const mat3 = new THREE.Matrix3();
      mat3.set(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
      let quat = new THREE.Quaternion();
      quat.setFromRotationMatrix(new THREE.Matrix4().setFromMatrix3(mat3));

      let mesh = null;
      const shape = geom.shape;

      // If a mesh is specified, use it for visualization
      if (geom.mesh_url) {
        const meshGroup = new THREE.Group();
        mesh = meshGroup;

        // Placeholder so link never "disappears" while GLTF loads/fails.
        if (shape && shape.type === 'box') {
          const [sx, sy, sz] = shape.halfSide;
          const geo = new THREE.BoxGeometry(sx * 2, sy * 2, sz * 2);
          const mat = new THREE.MeshStandardMaterial({ color: 0x0066cc, metalness: 0.2, roughness: 0.6 });
          const placeholder = new THREE.Mesh(geo, mat);
          placeholder.castShadow = true;
          placeholder.receiveShadow = true;
          meshGroup.add(placeholder);
        }

        // Load async; keep placeholder group so placements still update.
        // Capture robotScale for use in async closure
        const capturedScale = robotScale;
        (async () => {
          try {
            const gltfScene = await loadGltfScene(geom.mesh_url);
            if (!gltfScene) return;
            // Apply robot scale factor to GLTF
            gltfScene.scale.set(capturedScale, capturedScale, capturedScale);
            // Normalize: ensure meshes cast/receive shadows
            gltfScene.traverse(obj => {
              if (obj.isMesh) {
                obj.castShadow = true;
                obj.receiveShadow = true;
              }
            });
            meshGroup.clear();
            meshGroup.add(gltfScene);
            console.log('[APP] GLTF loaded:', geom.mesh_url, 'scale:', capturedScale);
          } catch (e) {
            console.error('[APP] Failed to load GLTF', geom.mesh_url, e);
          }
        })();
      }

      // Check if this is a wrist geometry
      const isWrist = wristJointIds.has(linkId);

      // Use color from backend if available, otherwise default colors
      let matColor = isWrist ? 0xff6600 : 0x00aa00;
      if (geom.color && geom.color.length >= 3) {
        const r = Math.floor(geom.color[0] * 255);
        const g = Math.floor(geom.color[1] * 255);
        const b = Math.floor(geom.color[2] * 255);
        matColor = (r << 16) | (g << 8) | b;
      }

      if (!mesh && shape.type === 'cylinder') {
        const geo = new THREE.CylinderGeometry(shape.radius, shape.radius, shape.height, 16);
        geo.rotateX(Math.PI / 2);

        const mat = new THREE.MeshStandardMaterial({ color: matColor, metalness: 0.4, roughness: 0.3 });
        mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      } else if (!mesh && shape.type === 'sphere') {
        const geo = new THREE.SphereGeometry(shape.radius, 16, 12);
        const mat = new THREE.MeshStandardMaterial({ color: matColor, metalness: 0.5, roughness: 0.3 });
        mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      } else if (!mesh && shape.type === 'box') {
        const [sx, sy, sz] = shape.halfSide;
        const geo = new THREE.BoxGeometry(sx * 2, sy * 2, sz * 2);
        const mat = new THREE.MeshStandardMaterial({ color: matColor, metalness: 0.3, roughness: 0.4 });
        mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      }

      if (mesh) {
        mesh.position.copy(pos);
        mesh.quaternion.copy(quat);
        mesh.userData.linkId = linkId;
        mesh.userData.geomId = geom.id;
        robotGroup.add(mesh);
        if (!linkMeshes[linkId]) linkMeshes[linkId] = [];
        linkMeshes[linkId].push({ mesh, localPos: pos.clone(), localQuat: quat.clone() });
      }
    }

    console.log('[APP] Robot mesh built with', Object.keys(linkMeshes).length, 'links');
  }

  /**
   * Update robot visualization by fetching current world placements from backend.
   */
  /**
   * Update robot visualization.
   * In PRO mode: fetch placements from backend.
   * In DEMO mode: compute using local FK with DH parameters.
   */
  function updateRobotViz(q) {
    if (!FEATURES.BACKEND) {
      updateRobotVizDemo(q);
      return;
    }
    
    (async () => {
      try {
        const res = await fetch(apiBase + '/robot_placements', { headers: getHeaders() });
        if (!res.ok) return;
        const data = await res.json();
        const placements = data.placements || [];

        for (const p of placements) {
          const geomId = p.id;
          const pos = p.position;
          const r = p.rotation;

          const mat3 = new THREE.Matrix3();
          mat3.set(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
          let quat = new THREE.Quaternion();
          quat.setFromRotationMatrix(new THREE.Matrix4().setFromMatrix3(mat3));

          robotGroup.traverse(obj => {
            if (obj.userData && obj.userData.geomId === geomId) {
              obj.position.set(pos[0], pos[1], pos[2]);
              obj.quaternion.copy(quat);
            }
          });
        }
      } catch (e) {
        console.error('[APP] updateRobotViz error', e);
      }
    })();
  }
  
  // Expose globally for setJointDirect
  window.updateRobotViz = updateRobotViz;

  /**
   * Update robot visualization using local FK (DEMO mode).
   * Uses generic FK from kinematics_lite.js which handles all robot types.
   */
  function updateRobotVizDemo(q) {
    if (!robotData || !robotInfo) return;
    
    try {
      const scale = robotData.scale || 1.0;
      const qNumeric = q.map(v => parseFloat(v) || 0);
      
      // Use generic FK - handles offsets, relative joints, dependent joints internally
      const fk = computeFK(robotInfo.type, qNumeric, scale);
      if (!fk) return;
      
      const frames = fk.frames;
      
      // DEMO mode: update debug spheres at each DH frame
      for (let i = 0; i < debugSpheres.length; i++) {
        if (debugSpheres[i] && frames[i]) {
          const tf = frames[i];
          debugSpheres[i].position.set(tf.x, tf.y, tf.z);
        }
      }
      
      // Update simplified links (connecting DH joint frames)
      for (const [name, linkMesh] of Object.entries(simplifiedLinks)) {
        const i = linkMesh.userData.linkIndex;
        if (frames[i] && frames[i + 1]) {
          const start = new THREE.Vector3(frames[i].x, frames[i].y, frames[i].z);
          const end = new THREE.Vector3(frames[i + 1].x, frames[i + 1].y, frames[i + 1].z);
          
          const direction = new THREE.Vector3().subVectors(end, start);
          const length = direction.length();
          
          if (length < 0.001) {
            linkMesh.visible = false;
            continue;
          }
          linkMesh.visible = true;
          
          // Position at midpoint
          const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
          linkMesh.position.copy(midpoint);
          
          // Scale Y to match length (cylinder is height 1.0 along Y)
          linkMesh.scale.y = length;
          
          // Align cylinder (Y axis) with direction vector
          const yAxis = new THREE.Vector3(0, 1, 0);
          const orientation = new THREE.Quaternion().setFromUnitVectors(yAxis, direction.clone().normalize());
          linkMesh.quaternion.copy(orientation);
        }
      }
      
      // Update EE position and orientation
      updateEEDemo([fk.ee.x, fk.ee.y, fk.ee.z], fk.ee.matrix);
      
    } catch (e) {
      console.error('[APP] updateRobotVizDemo error:', e);
    }
  }

  function updateEEDemo(pos, rot) {
    eeAxes.position.set(pos[0], pos[1], pos[2]);
    document.getElementById('ee_x').textContent = pos[0].toFixed(3);
    document.getElementById('ee_y').textContent = pos[1].toFixed(3);
    document.getElementById('ee_z').textContent = pos[2].toFixed(3);
    
    if (rot) {
      const m = new THREE.Matrix4();
      
      // Handle 4x4 matrix (array of arrays) from computeFK
      if (rot.length === 4 && Array.isArray(rot[0])) {
        m.set(
          rot[0][0], rot[0][1], rot[0][2], rot[0][3],
          rot[1][0], rot[1][1], rot[1][2], rot[1][3],
          rot[2][0], rot[2][1], rot[2][2], rot[2][3],
          rot[3][0], rot[3][1], rot[3][2], rot[3][3]
        );
      } 
      // Handle flat array (9 for rotation or 16 for transform)
      else if (rot.length === 9) {
        const m3 = new THREE.Matrix3().fromArray(rot);
        m.setFromMatrix3(m3);
      } else if (rot.length === 16) {
        m.fromArray(rot);
      }
      
      const quat = new THREE.Quaternion();
      quat.setFromRotationMatrix(m);
      eeAxes.quaternion.copy(quat);
    }
  }

  // Load robot on startup
  // Moved to initApp()
  // loadRobotStructure();


  // ===========================================================================
  // SECTION 13: WEBSOCKET (Real-time state streaming)
  // ===========================================================================

  // const ws = new WebSocket('ws://127.0.0.1:8000/ws/state?session_id=' + sessionId); // Moved to setupWebSocket
  let ws = null;
  const wsStatusText = document.getElementById('ws_status_text');
  const wsStatusDot = document.getElementById('ws_status_dot');

  function setupWebSocket() {
    if (ws) ws.close();
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/state?session_id=${sessionId}`);

    ws.onopen = () => {
      console.log('[APP] ws open');
      wsStatusText.textContent = 'Connected';
      wsStatusDot.classList.add('connected');
    };

    ws.onmessage = (ev) => {
      // Skip WebSocket updates during animation to prevent picotazos
      if (isAnimating) return;

      try {
        const msg = JSON.parse(ev.data);
        if (msg.q && Array.isArray(msg.q)) {
          // Skip all-zero messages silently (server sends these when idle)
          const allZeros = msg.q.every(v => Math.abs(v) < 0.001);
          if (allZeros) return;
          
          // Use J1-J6 naming (1-indexed)
          const qLabels = msg.q.map((v, i) => `J${i + 1}: ${v.toFixed(3)}`).join('\n');
          qDisplay.textContent = qLabels;
          updateRobotViz(msg.q);
          updateSlidersFromQ(msg.q);
        }
        if (msg.ee_pos) {
          if (window.updateEE) window.updateEE(msg.ee_pos, msg.ee_rot);
        }
      } catch (e) {
        console.error('[APP] ws parse error:', e);
      }
    };

    ws.onclose = () => {
      console.log('[APP] ws closed');
      wsStatusText.textContent = 'Disconnected';
      wsStatusDot.classList.remove('connected');
    };
  }

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
  console.log('[APP] Animation loop started');


  // ===========================================================================
  // SECTION 14: TARGETS & PROGRAM UI
  // ===========================================================================

  // Group for visualizing saved target axes in scene
  const savedTargetsGroup = new THREE.Group();
  scene.add(savedTargetsGroup);

  function createTargetAxes(target) {
    const axes = new THREE.AxesHelper(0.05);
    const pos = target.position;
    const rot = target.rotation;

    axes.position.set(pos[0], pos[1], pos[2]);

    if (rot && rot.length === 9) {
      const mat3 = new THREE.Matrix3();
      mat3.set(rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8]);
      const quat = new THREE.Quaternion();
      quat.setFromRotationMatrix(new THREE.Matrix4().setFromMatrix3(mat3));
      axes.quaternion.copy(quat);
    }

    axes.userData.targetName = target.name;
    return axes;
  }

  function refreshTargetAxesVisuals() {
    while (savedTargetsGroup.children.length > 0) {
      savedTargetsGroup.remove(savedTargetsGroup.children[0]);
    }
    for (const t of localTargets) {
      const axes = createTargetAxes(t);
      savedTargetsGroup.add(axes);
    }
  }

  function renderTargetsTree() {
    targetsTree.innerHTML = '';
    for (const t of localTargets) {
      const div = document.createElement('div');
      div.className = 'tree-item';
      div.draggable = true;
      div.dataset.targetName = t.name;

      // Get Euler angles if rotation exists
      let eulerStr = '';
      if (t.rotation && t.rotation.length === 9) {
        const euler = rotMatToEulerZYX(t.rotation);
        eulerStr = `Rz:${euler[0]}° Ry:${euler[1]}° Rx:${euler[2]}°`;
      }

      div.innerHTML = `
        <div class="target-info">
          <span class="target-name">${t.name}</span>
          <span class="target-coords">XYZ: ${t.position[0].toFixed(3)}, ${t.position[1].toFixed(3)}, ${t.position[2].toFixed(3)}</span>
          <span class="target-coords">${eulerStr}</span>
        </div>
        <div class="target-actions">
          <span class="jump-target-btn" data-name="${t.name}" title="Jump to target">⤳</span>
          <span class="delete-target-btn" data-name="${t.name}">✕</span>
        </div>
      `;

      div.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('text/plain', t.name);
        div.classList.add('dragging');
      });
      div.addEventListener('dragend', () => {
        div.classList.remove('dragging');
      });

      // Double-click to go to target
      div.addEventListener('dblclick', async () => {
        const target = localTargets.find(x => x.name === t.name);
        if (target) {
          const r = await stateManager.setQ(target.q);
          if (r.ok) {
             updateRobotViz(target.q);
             updateSlidersFromQ(target.q);
          }
        }
      });

      targetsTree.appendChild(div);
    }

    // Wire up jump buttons
    targetsTree.querySelectorAll('.jump-target-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const name = e.target.dataset.name;
        const target = localTargets.find(x => x.name === name);
        if (target) {
          const r = await stateManager.setQ(target.q);
          if (r.ok) {
            updateRobotViz(target.q);
            updateSlidersFromQ(target.q);
          }
        }
      });
    });

    // Wire up delete buttons
    targetsTree.querySelectorAll('.delete-target-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const name = e.target.dataset.name;
        await stateManager.deleteTarget(name);
        await loadTargets();
        await loadProgram(); // Refresh program too since instructions may reference deleted target
      });
    });

    refreshTargetAxesVisuals();
  }

  function renderProgramTree() {
    programTree.innerHTML = '';
    if (localProgram.length === 0) {
      programTree.innerHTML = '<div class="drop-placeholder">Drag targets here to create instructions</div>';
      return;
    }

    localProgram.forEach((instr, idx) => {
      const div = document.createElement('div');
      div.className = 'program-item';
      div.dataset.index = idx;

      // Format display text based on instruction type
      let displayText;
      if (instr.type === 'Pause') {
        displayText = `⏸ Pause ${instr.pause_time || 1}s`;
      } else if (instr.type === 'MoveC') {
        displayText = `${instr.type} via:${instr.via_target_name || '?'} → ${instr.target_name} v=${instr.speed} z=${instr.zone}`;
      } else {
        displayText = `${instr.type} ${instr.target_name} v=${instr.speed} z=${instr.zone}`;
      }

      div.innerHTML = `
        <span class="instr-text">${displayText}</span>
        <span class="delete-btn" data-index="${idx}">✕</span>
      `;

      // Click to edit instruction
      div.querySelector('.instr-text').addEventListener('click', () => {
        openEditModal(idx);
      });

      programTree.appendChild(div);
    });

    programTree.querySelectorAll('.delete-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const idx = parseInt(e.target.dataset.index);
        await stateManager.deleteInstruction(idx);
        await loadProgram();
      });
    });

    // Also update the via target dropdown for MoveC
    updateViaTargetDropdown();
  }

  function updateViaTargetDropdown() {
    if (instrViaTargetSelect) {
      instrViaTargetSelect.innerHTML = '';
      for (const t of localTargets) {
        const opt = document.createElement('option');
        opt.value = t.name;
        opt.textContent = t.name;
        instrViaTargetSelect.appendChild(opt);
      }
    }
  }

  // Edit instruction modal
  function openEditModal(index) {
    const instr = localProgram[index];
    if (!instr) return;

    editingInstructionIndex = index;
    editInstrType.value = instr.type;
    editInstrSpeed.value = instr.speed || 100;
    editInstrZone.value = instr.zone || 50;
    editInstrPauseTime.value = instr.pause_time || 1;

    // Populate target dropdown
    editInstrTarget.innerHTML = '';
    localTargets.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t.name;
      opt.textContent = t.name;
      if (t.name === instr.target_name) opt.selected = true;
      editInstrTarget.appendChild(opt);
    });

    // Show/hide fields based on type
    updateEditInstrFields();

    editModal.classList.add('active');
    modalOverlay.classList.add('active');
  }

  function closeEditModal() {
    editModal.classList.remove('active');
    modalOverlay.classList.remove('active');
    editingInstructionIndex = -1;
  }

  cancelEditBtn.addEventListener('click', closeEditModal);
  modalOverlay.addEventListener('click', closeEditModal);

  saveEditBtn.addEventListener('click', async () => {
    if (editingInstructionIndex < 0) return;

    // Delete old instruction
    await stateManager.deleteInstruction(editingInstructionIndex);

    // Create new instruction
    let newInstr;
    if (editInstrType.value === 'Pause') {
      newInstr = {
        type: 'Pause',
        pause_time: parseFloat(editInstrPauseTime.value) || 1
      };
    } else {
      newInstr = {
        type: editInstrType.value,
        target_name: editInstrTarget.value,
        speed: parseFloat(editInstrSpeed.value) || 100,
        zone: parseFloat(editInstrZone.value) || 50
      };
    }

    // Add the new instruction
    await stateManager.addInstruction(newInstr);

    closeEditModal();
    await loadProgram();
  });

  async function loadTargets() {
    try {
      localTargets = await stateManager.getTargets();
      renderTargetsTree();
    } catch (e) {
      console.error('Failed to load targets', e);
    }
  }

  async function loadProgram() {
    try {
      localProgram = await stateManager.getProgram();
      renderProgramTree();
    } catch (e) {
      console.error('Failed to load program', e);
    }
  }

  // Save target
  saveTargetBtn.addEventListener('click', async () => {
    const name = targetNameInput.value.trim();
    if (!name) {
      alert('Please enter a target name');
      return;
    }

    const result = await stateManager.saveTarget(name);

    if (result.ok) {
      const match = name.match(/^(.*)(\d+)$/);
      if (match) {
        targetNameInput.value = match[1] + (parseInt(match[2]) + 1);
      } else {
        targetNameInput.value = name + '2';
      }
      await loadTargets();
    }
  });

  // Drop zone
  programTree.addEventListener('dragover', (e) => {
    e.preventDefault();
    programTree.classList.add('drag-over');
  });

  programTree.addEventListener('dragleave', () => {
    programTree.classList.remove('drag-over');
  });

  programTree.addEventListener('drop', async (e) => {
    e.preventDefault();
    programTree.classList.remove('drag-over');

    const targetName = e.dataTransfer.getData('text/plain');
    if (!targetName) return;

    const type = instrTypeSelect.value;

    // Don't allow dropping targets for Pause
    if (type === 'Pause') {
      alert('Use the "Add Pause" button to add Pause instructions');
      return;
    }

    const speed = parseFloat(instrSpeedInput.value) || 100;
    const zone = parseFloat(instrZoneInput.value) || 50;

    let instrData = { type, target_name: targetName, speed, zone };

    // For MoveC, add via_target
    if (type === 'MoveC') {
      const viaTarget = instrViaTargetSelect?.value;
      if (!viaTarget) {
        alert('Select a via point for MoveC');
        return;
      }
      instrData.via_target_name = viaTarget;
    }

    const result = await stateManager.addInstruction(instrData);
    if (result.ok) {
      await loadProgram();
    }
  });

  // Clear program
  clearProgramBtn.addEventListener('click', async () => {
    await stateManager.clearProgram();
    await loadProgram();
  });

  // ==========================================================================
  // Save/Load Program Functionality
  // ==========================================================================

  async function loadSavedProgramsList() {
    try {
      const res = await fetch(apiBase + '/list_programs', { headers: getHeaders() });
      const data = await res.json();

      if (!data.ok || !data.programs || data.programs.length === 0) {
        savedProgramsList.innerHTML = '<div style="color:#888;">No saved programs</div>';
        return;
      }

      let html = '';
      for (const prog of data.programs) {
        html += `
          <div style="display:flex; align-items:center; padding:4px 6px; background:rgba(100,150,200,0.15); margin-bottom:3px; border-radius:3px;">
            <div style="flex:1; cursor:pointer;" onclick="window.loadSavedProgram('${prog.filename}')">
              <div style="color:#8cf; font-weight:bold;">${prog.filename}</div>
              <div style="font-size:9px; color:#888;">
                ${prog.robot_type} | ${prog.num_targets} targets, ${prog.num_instructions} instr
                ${prog.description ? ' | ' + prog.description : ''}
              </div>
            </div>
            <button onclick="window.deleteSavedProgram('${prog.filename}')" 
                    style="background:#a33; color:#fff; border:none; padding:2px 6px; border-radius:3px; cursor:pointer; font-size:10px;">
              ✕
            </button>
          </div>
        `;
      }
      savedProgramsList.innerHTML = html;
    } catch (e) {
      console.error('Error loading programs list:', e);
      savedProgramsList.innerHTML = '<div style="color:#f88;">Error loading list</div>';
    }
  }

  window.loadSavedProgram = async function (filename) {
    try {
      const res = await fetch(apiBase + '/load_program', {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({ filename })
      });
      const data = await res.json();

      if (!data.ok) {
        alert('Error loading program: ' + data.error);
        return;
      }

      // Refresh UI
      await loadTargets();
      await loadProgram();

      // Update config fields if robot changed
      if (robotTypeSelect) robotTypeSelect.value = data.robot_type || '6dof';
      if (robotScaleInput) robotScaleInput.value = data.scale || 1;
      if (robotPayloadInput) robotPayloadInput.value = data.payload_kg || 0;
      populatePayloadInputs(data);
      updateRobotConfigInfo(data);

      alert(`Loaded program: ${filename}\n${data.num_targets} targets, ${data.num_instructions} instructions`);
    } catch (e) {
      console.error('Error loading program:', e);
      alert('Error loading program');
    }
  };

  window.deleteSavedProgram = async function (filename) {
    if (!confirm(`Delete program "${filename}"?`)) return;

    try {
      const res = await fetch(apiBase + `/delete_program/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
        headers: getHeaders()
      });
      const data = await res.json();

      if (data.ok) {
        await loadSavedProgramsList();
      } else {
        alert('Error deleting: ' + data.error);
      }
    } catch (e) {
      console.error('Error deleting program:', e);
    }
  };

  if (saveProgramBtn) {
    saveProgramBtn.addEventListener('click', async () => {
      const name = programNameInput?.value?.trim();
      if (!name) {
        alert('Enter a program name');
        return;
      }

      try {
        const res = await fetch(apiBase + '/save_program', {
          method: 'POST',
          headers: getHeaders(),
          body: JSON.stringify({
            filename: name,
            description: programDescInput?.value?.trim() || ''
          })
        });
        const data = await res.json();

        if (data.ok) {
          alert(`Program saved: ${data.filename}`);
          await loadSavedProgramsList();
        } else {
          alert('Error saving: ' + data.error);
        }
      } catch (e) {
        console.error('Error saving program:', e);
        alert('Error saving program');
      }
    });
  }

  if (refreshProgramsBtn) {
    refreshProgramsBtn.addEventListener('click', loadSavedProgramsList);
  }
  // ==========================================================================

  // Play program with smooth animation
  playProgramBtn.addEventListener('click', async () => {
    if (isAnimating) return;

    const speedFactor = parseFloat(speedFactorInput.value) || 1;
    executionControl.speedFactor = speedFactor;
    executionControl.cancelled = false;

    playProgramBtn.disabled = true;
    playProgramBtn.textContent = '⏳ Running...';
    isAnimating = true;

    try {
      if (APP_MODE === 'demo') {
        // DEMO mode: use local interpolation
        await executeProgram(
          localProgram,
          localTargets,
          currentQ,
          (q) => {
            // onFrame: update visualization
            currentQ = q;
            updateRobotViz(q);
            updateSlidersFromQ(q);
          },
          (index, instr) => {
            // onInstruction: highlight current instruction
            highlightInstruction(index);
          },
          executionControl
        );
        finishAnimation();
      } else {
        // PRO mode: use backend
        const res = await fetch(apiBase + '/execute_program', {
          method: 'POST',
          headers: getHeaders(),
          body: JSON.stringify({ speed_factor: speedFactor })
        });
        const result = await res.json();
        console.log('Execute program', result);

        if (result.ok && result.trajectory && result.trajectory.length > 0) {
          const trajectory = result.trajectory;
          const baseDt = result.dt * 1000;

          let frameIndex = 0;
          let lastTime = performance.now();
          const frameDt = Math.max(16, baseDt / speedFactor);

          function animateFrame(currentTime) {
            if (frameIndex >= trajectory.length) {
              finishAnimation();
              return;
            }

            const elapsed = currentTime - lastTime;
            if (elapsed >= frameDt) {
              lastTime = currentTime;
              updateRobotVizDirect(trajectory[frameIndex]);
              frameIndex++;
            }

            requestAnimationFrame(animateFrame);
          }

          requestAnimationFrame(animateFrame);
        } else if (!result.ok) {
          alert('Execution failed: ' + (result.error || 'Unknown error'));
          finishAnimation();
        } else {
          finishAnimation();
        }
      }
    } catch (e) {
      console.error('Play program error', e);
      alert('Error executing program');
      finishAnimation();
    }
  });

  async function finishAnimation() {
    isAnimating = false;
    playProgramBtn.disabled = false;
    playProgramBtn.textContent = '▶ PLAY';
    highlightInstruction(-1); // Clear highlight

    // Notify backend that animation is done (PRO mode only)
    if (APP_MODE !== 'demo') {
      try {
        await fetch(apiBase + '/animation_done', { method: 'POST', headers: getHeaders() });
      } catch (e) {
        console.error('Failed to notify animation done', e);
      }
    }
  }

  /**
   * Highlight the currently executing instruction in the program tree.
   * @param {number} index - instruction index, or -1 to clear
   */
  function highlightInstruction(index) {
    const items = programTree.querySelectorAll('.program-item');
    items.forEach((item, i) => {
      if (i === index) {
        item.style.background = '#f39c1233';
        item.style.borderLeftColor = '#f39c12';
      } else {
        item.style.background = '';
        item.style.borderLeftColor = '';
      }
    });
  }

  // Direct robot viz update
  async function updateRobotVizDirect(qFrame) {
    await fetch(apiBase + '/set_joint', {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ q: qFrame })
    });

    const res = await fetch(apiBase + '/robot_placements', { headers: getHeaders() });
    if (!res.ok) return;
    const data = await res.json();
    const placements = data.placements || [];

    for (const p of placements) {
      const geomId = p.id;
      const pos = p.position;
      const r = p.rotation;

      const mat3 = new THREE.Matrix3();
      mat3.set(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
      let quat = new THREE.Quaternion();
      quat.setFromRotationMatrix(new THREE.Matrix4().setFromMatrix3(mat3));

      robotGroup.traverse(obj => {
        if (obj.userData && obj.userData.geomId === geomId) {
          obj.position.set(pos[0], pos[1], pos[2]);
          obj.quaternion.copy(quat);
        }
      });
    }

    // Update TCP using the real end-effector frame (same reference as JOG/targets)
    if (data.ee_pos && data.ee_rot) {
      if (window.updateEE) window.updateEE(data.ee_pos, data.ee_rot);
    }

    // Update sliders during animation too
    updateSlidersFromQ(qFrame);
  }

  // Plot dynamics
  plotDynamicsBtn.addEventListener('click', async () => {
    try {
      let data;
      
      if (FEATURES.DYNAMICS) {
        // PRO mode: use backend data
        const result = await stateManager.getTrajectoryData();
        if (!result.ok) {
          alert(result.error || 'No trajectory data available');
          return;
        }
        data = result.data;
      } else {
        // DEMO mode: compute locally
        const program = stateManager.getProgram();
        const targets = stateManager.getTargets();
        
        if (!program || program.length === 0) {
          alert('No program to analyze. Create and execute a program first.');
          return;
        }
        
        console.log('[APP] DEMO mode: computing trajectory dynamics...');
        
        // Generate trajectory
        const traj = generateProgramTrajectory(program, targets, currentQ, 0.02);
        
        // Compute inverse dynamics for each point
        const robotType = robotInfo ? robotInfo.type : 'CR4';
        const demoConfig = stateManager.getConfig ? stateManager.getConfig() : {};
        traj.tau = [];
        for (let i = 0; i < traj.q.length; i++) {
          const result = computeInverseDynamics(robotType, traj.q[i], traj.qd[i], traj.qdd[i], demoConfig);
          traj.tau.push(result.tau);
        }
        
        data = {
          t: traj.time,
          q: traj.q,
          v: traj.qd,
          a: traj.qdd,
          tau: traj.tau
        };
        
        // Store for CSV export
        lastDemoDynamicsData = data;
        
        console.log(`[APP] DEMO mode: computed ${traj.q.length} trajectory points`);
      }
      
      const variable = plotVariableSelect.value;
      const values = data[variable];
      const times = data.t;

      if (!values || values.length === 0) {
        alert('No data to plot');
        return;
      }

      const nJoints = values[0].length;
      const colors = ['#ff6384', '#36a2eb', '#ffce56', '#4bc0c0', '#9966ff', '#ff9f40'];
      const datasets = [];

      for (let j = 0; j < nJoints; j++) {
        datasets.push({
          label: `J${j + 1}`,  // 1-indexed joint names
          data: values.map(v => v[j]),
          borderColor: colors[j % colors.length],
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          pointRadius: 0
        });
      }

      const ctx = document.getElementById('dynamics-chart').getContext('2d');

      if (dynamicsChart) {
        dynamicsChart.destroy();
      }

      const labels = {
        tau: 'Torques (N·m)',
        q: 'Positions (rad)',
        v: 'Velocities (rad/s)',
        a: 'Accelerations (rad/s²)'
      };

      dynamicsChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: times.map(t => t.toFixed(3)),
          datasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: labels[variable] || variable,
              color: '#fff'
            },
            legend: {
              position: 'bottom',
              labels: { font: { size: 10 }, color: '#aaa' }
            }
          },
          scales: {
            x: {
              title: { display: true, text: 'Time (s)', color: '#888' },
              ticks: { maxTicksLimit: 10, color: '#888' },
              grid: { color: '#333' }
            },
            y: {
              title: { display: true, text: labels[variable] || variable, color: '#888' },
              ticks: { color: '#888' },
              grid: { color: '#333' }
            }
          }
        }
      });
    } catch (e) {
      console.error('Plot dynamics error', e);
      alert('Error fetching trajectory data');
    }
  });

  // Export CSV
  exportCsvBtn.addEventListener('click', () => {
    if (!FEATURES.BACKEND) {
      // DEMO mode: generate CSV from stored data
      if (!lastDemoDynamicsData) {
        alert('No dynamics data to export. Run "Plot Dynamics" first.');
        return;
      }
      
      // Generate CSV content
      const data = lastDemoDynamicsData;
      const nq = data.q[0].length;
      
      // Header
      let csv = 't';
      for (let i = 0; i < nq; i++) csv += `,q${i+1}`;
      for (let i = 0; i < nq; i++) csv += `,v${i+1}`;
      for (let i = 0; i < nq; i++) csv += `,a${i+1}`;
      for (let i = 0; i < nq; i++) csv += `,tau${i+1}`;
      csv += '\n';
      
      // Data rows
      for (let i = 0; i < data.t.length; i++) {
        let row = data.t[i].toFixed(6);
        for (let j = 0; j < nq; j++) row += `,${data.q[i][j].toFixed(6)}`;
        for (let j = 0; j < nq; j++) row += `,${data.v[i][j].toFixed(6)}`;
        for (let j = 0; j < nq; j++) row += `,${data.a[i][j].toFixed(6)}`;
        for (let j = 0; j < nq; j++) row += `,${data.tau[i][j].toFixed(6)}`;
        csv += row + '\n';
      }
      
      // Download CSV
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dynamics_demo_${robotInfo?.type || 'CR4'}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      console.log('[APP] DEMO mode: CSV exported');
      return;
    }
    window.open(apiBase + '/export_csv?session_id=' + sessionId, '_blank');
  });

  // Re-plot when variable changes
  plotVariableSelect.addEventListener('change', () => {
    plotDynamicsBtn.click();
  });


  // ===========================================================================
  // SECTION 17: CONFIG TAB (Robot Configuration & Station Geometries)
  // ===========================================================================

  // DOM elements for config tab
  const robotTypeSelect = document.getElementById('robot_type_select');
  const robotScaleInput = document.getElementById('robot_scale_input');
  const robotPayloadInput = document.getElementById('robot_payload_input');
  const robotPayloadBoxXInput = document.getElementById('robot_payload_box_x_input');
  const robotPayloadBoxYInput = document.getElementById('robot_payload_box_y_input');
  const robotPayloadBoxZInput = document.getElementById('robot_payload_box_z_input');
  const robotPayloadComXInput = document.getElementById('robot_payload_com_x_input');
  const robotPayloadComYInput = document.getElementById('robot_payload_com_y_input');
  const robotPayloadComZInput = document.getElementById('robot_payload_com_z_input');
  const robotFrictionInput = document.getElementById('robot_friction_input');
  const robotReflectedInertiaInput = document.getElementById('robot_reflected_inertia_input');
  const robotCoulombInput = document.getElementById('robot_coulomb_input');
  const applyRobotConfigBtn = document.getElementById('apply_robot_config');
  const robotConfigInfo = document.getElementById('robot_config_info');
  const stationGltfFileInput = document.getElementById('station_gltf_file');
  const importStationGeometryBtn = document.getElementById('import_station_geometry');
  const stationGeometriesList = document.getElementById('station-geometries-list');
  const stationGeomEditSection = document.getElementById('station-geom-edit-section');
  const editGeomName = document.getElementById('edit_geom_name');
  const editGeomX = document.getElementById('edit_geom_x');
  const editGeomY = document.getElementById('edit_geom_y');
  const editGeomZ = document.getElementById('edit_geom_z');
  const editGeomScale = document.getElementById('edit_geom_scale');
  const editGeomRx = document.getElementById('edit_geom_rx');
  const editGeomRy = document.getElementById('edit_geom_ry');
  const editGeomRz = document.getElementById('edit_geom_rz');
  const updateStationGeomBtn = document.getElementById('update_station_geom');
  const deleteStationGeomBtn = document.getElementById('delete_station_geom');

  let stationGeometries = [];
  let stationMeshes = {}; // id -> THREE.Group
  let selectedStationGeomId = null;

  function parseVec3Inputs(inputs) {
    const values = inputs.map(input => parseFloat(input?.value));
    return values.every(v => Number.isFinite(v)) ? values : null;
  }

  function setVec3Inputs(inputs, values) {
    const arr = Array.isArray(values) ? values : [];
    inputs.forEach((input, idx) => {
      if (!input) return;
      const value = arr[idx];
      input.value = Number.isFinite(value)
        ? Number(value).toFixed(3).replace(/\.000$/, '')
        : '';
    });
  }

  function populatePayloadInputs(config) {
    const payloadInertia = config?.payload_inertia || {};
    setVec3Inputs(
      [robotPayloadBoxXInput, robotPayloadBoxYInput, robotPayloadBoxZInput],
      payloadInertia.box_size_xyz_m
    );
    setVec3Inputs(
      [robotPayloadComXInput, robotPayloadComYInput, robotPayloadComZInput],
      payloadInertia.com_from_tcp
    );
  }

  // Group for station geometries in Three.js scene
  const stationGroup = new THREE.Group();
  scene.add(stationGroup);

  // Load robot config on init
  async function loadRobotConfig() {
    try {
      const config = await stateManager.getConfig();
      if (!config) return null;

      if (robotTypeSelect) robotTypeSelect.value = config.robot_type || 'CR4';
      if (robotScaleInput) robotScaleInput.value = config.scale || 1.0;
      if (robotPayloadInput) robotPayloadInput.value = config.payload_kg || 0.0;
      populatePayloadInputs(config);

      // Display friction coefficients if available
      if (robotFrictionInput && config.friction_coeffs) {
        robotFrictionInput.value = config.friction_coeffs.map(f => f.toFixed(2)).join(',');
      } else if (robotFrictionInput) {
        robotFrictionInput.value = '';
      }

      if (robotReflectedInertiaInput && config.reflected_inertia) {
        robotReflectedInertiaInput.value = config.reflected_inertia.map(v => Number(v).toFixed(4)).join(',');
      } else if (robotReflectedInertiaInput) {
        robotReflectedInertiaInput.value = '';
      }

      if (robotCoulombInput && config.coulomb_friction) {
        robotCoulombInput.value = config.coulomb_friction.map(v => Number(v).toFixed(3)).join(',');
      } else if (robotCoulombInput) {
        robotCoulombInput.value = '';
      }

      // Update info display
      updateRobotConfigInfo(config);

      // Update joint sliders visibility based on robot type
      const nq_user = config.robot_type === 'CR6' ? 6 : 4;
      updateJointSlidersVisibility(nq_user);

      return config;
    } catch (e) {
      console.error('Failed to load robot config', e);
      return null;
    }
  }

  function updateRobotConfigInfo(config) {
    if (!robotConfigInfo) return;
    const details = [];
    let info = `DOF: ${config.nq || '?'}\n`;
    if (config.payload_kg > 0) details.push(`Payload: ${config.payload_kg} kg`);
    if (config.payload_inertia?.box_size_xyz_m) {
      const dims = config.payload_inertia.box_size_xyz_m
        .map(v => Number(v).toFixed(3))
        .join(', ');
      details.push(`Box: [${dims}] m`);
    }
    if (config.payload_inertia?.com_from_tcp) {
      const com = config.payload_inertia.com_from_tcp
        .map(v => Number(v).toFixed(3))
        .join(', ');
      details.push(`COM: [${com}] m`);
    }
    if (config.friction_coeffs) details.push('Friction: active');
    if (config.reflected_inertia) details.push('Iref: active');
    if (config.coulomb_friction) details.push('Coulomb: active');
    if (details.length > 0) info += details.join(' | ');
    if (config.ee_always_down) info += '\n⬇️ Tool always pointing down';
    robotConfigInfo.textContent = info;
  }

  function parseNumericList(inputEl) {
    const s = inputEl?.value?.trim();
    if (!s) return null;
    const nums = s.split(',').map(v => parseFloat(v.trim())).filter(v => !Number.isNaN(v));
    return nums.length > 0 ? nums : null;
  }

  // updateJointSlidersVisibility was moved up
  
  // Apply robot configuration
  if (applyRobotConfigBtn) {
    applyRobotConfigBtn.addEventListener('click', async () => {
      const robotType = robotTypeSelect?.value || 'CR4';
      const scale = parseFloat(robotScaleInput?.value) || 1.0;
      const payload = parseFloat(robotPayloadInput?.value) || 0.0;
      const payloadBox = parseVec3Inputs([
        robotPayloadBoxXInput,
        robotPayloadBoxYInput,
        robotPayloadBoxZInput
      ]);
      const payloadCom = parseVec3Inputs([
        robotPayloadComXInput,
        robotPayloadComYInput,
        robotPayloadComZInput
      ]);

      // Parse friction coefficients
      const frictionCoeffs = parseNumericList(robotFrictionInput);
      const reflectedInertia = parseNumericList(robotReflectedInertiaInput);
      const coulombFriction = parseNumericList(robotCoulombInput);
      const payloadInertia = payload > 0 && (payloadBox || payloadCom)
        ? {
            ...(payloadBox ? { box_size_xyz_m: payloadBox } : {}),
            ...(payloadCom ? { com_from_tcp: payloadCom } : {})
          }
        : null;

      applyRobotConfigBtn.disabled = true;
      applyRobotConfigBtn.textContent = 'Applying...';

      try {
        const result = await stateManager.setConfig({
          robot_type: robotType,
          scale,
          payload_kg: payload,
          payload_inertia: payloadInertia,
          friction_coeffs: frictionCoeffs,
          reflected_inertia: reflectedInertia,
          coulomb_friction: coulombFriction
        });

        if (result.ok) {
          // Reload robot structure
          await loadRobotStructure();
          invalidateActuatorCaches();
          // Update config info
          const fullConfig = await stateManager.getConfig();
          populatePayloadInputs(fullConfig);
          updateRobotConfigInfo(fullConfig);
          // Update slider visibility
          const nq_user = fullConfig.nq || (fullConfig.robot_type === 'CR6' ? 6 : 4);
          updateJointSlidersVisibility(nq_user);
          // Refresh UI components
          await loadTargets();
          await loadProgram();
        } else {
          alert('Failed to apply config: ' + (result.error || 'Unknown error'));
        }
      } catch (e) {
        console.error('Apply robot config error', e);
        alert('Error applying robot configuration');
      }

      applyRobotConfigBtn.disabled = false;
      applyRobotConfigBtn.textContent = 'Apply Configuration';
    });
  }


  // ===========================================================================
  // SECTION 17B: STATION GEOMETRIES (External 3D Models)
  // ===========================================================================

  // Import button triggers file picker
  if (importStationGeometryBtn && stationGltfFileInput) {
    importStationGeometryBtn.addEventListener('click', () => {
      stationGltfFileInput.click();
    });

    stationGltfFileInput.addEventListener('change', async (e) => {
      const files = e.target.files;
      if (!files || files.length === 0) return;

      for (const file of files) {
        await uploadStationFile(file);
      }

      // Clear input so same file can be re-selected
      stationGltfFileInput.value = '';

      // Reload station geometries
      await loadStationGeometries();
    });
  }

  async function uploadStationFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Determine endpoint based on file type
    const filename = file.name.toLowerCase();
    let endpoint = '/upload_station_geometry';
    if (filename.endsWith('.bin')) {
      endpoint = '/upload_station_bin';
    }

    try {
      const res = await fetch(apiBase + endpoint, {
        method: 'POST',
        headers: { 'X-Session-ID': sessionId },
        body: formData
      });
      const result = await res.json();

      if (result.ok) {
        console.log('[APP] Station file uploaded:', result);
      } else {
        console.error('Upload failed:', result.error);
        alert('Upload failed: ' + (result.error || 'Unknown error'));
      }
    } catch (e) {
      console.error('Upload error', e);
      alert('Error uploading file');
    }
  }

  async function loadStationGeometries() {
    try {
      if (!FEATURES.BACKEND) {
        stationGeometries = [];
        renderStationGeometriesList();
        return;
      }
      const res = await fetch(apiBase + '/station_geometries', { headers: getHeaders() });
      const data = await res.json();
      stationGeometries = data.geometries || [];

      // Render list
      renderStationGeometriesList();

      // Update 3D scene
      updateStationMeshes();
    } catch (e) {
      console.error('Failed to load station geometries', e);
    }
  }

  function renderStationGeometriesList() {
    if (!stationGeometriesList) return;

    if (stationGeometries.length === 0) {
      stationGeometriesList.innerHTML = '<div class="drop-placeholder" style="padding:10px;">No geometries imported</div>';
      return;
    }

    stationGeometriesList.innerHTML = '';

    for (const geom of stationGeometries) {
      const div = document.createElement('div');
      div.className = 'tree-item';
      if (geom.id === selectedStationGeomId) div.style.borderLeft = '3px solid #00d4ff';

      div.innerHTML = `
        <div class="target-info">
          <span class="target-name">${geom.name}</span>
          <span class="target-coords">Pos: ${geom.position[0].toFixed(2)}, ${geom.position[1].toFixed(2)}, ${geom.position[2].toFixed(2)}</span>
        </div>
      `;

      div.addEventListener('click', () => selectStationGeometry(geom.id));

      stationGeometriesList.appendChild(div);
    }
  }

  function selectStationGeometry(id) {
    selectedStationGeomId = id;
    const geom = stationGeometries.find(g => g.id === id);

    if (!geom) {
      stationGeomEditSection.style.display = 'none';
      return;
    }

    // Populate edit form
    editGeomName.textContent = geom.name;
    editGeomX.value = geom.position[0];
    editGeomY.value = geom.position[1];
    editGeomZ.value = geom.position[2];
    editGeomScale.value = geom.scale || 1;
    editGeomRx.value = (geom.rotation[0] * 180 / Math.PI).toFixed(1);
    editGeomRy.value = (geom.rotation[1] * 180 / Math.PI).toFixed(1);
    editGeomRz.value = (geom.rotation[2] * 180 / Math.PI).toFixed(1);

    stationGeomEditSection.style.display = 'block';

    // Highlight in list
    renderStationGeometriesList();
  }

  // Update station geometry position/rotation
  if (updateStationGeomBtn) {
    updateStationGeomBtn.addEventListener('click', async () => {
      if (!selectedStationGeomId) return;

      const update = {
        id: selectedStationGeomId,
        position: [
          parseFloat(editGeomX.value) || 0,
          parseFloat(editGeomY.value) || 0,
          parseFloat(editGeomZ.value) || 0
        ],
        rotation: [
          (parseFloat(editGeomRx.value) || 0) * Math.PI / 180,
          (parseFloat(editGeomRy.value) || 0) * Math.PI / 180,
          (parseFloat(editGeomRz.value) || 0) * Math.PI / 180
        ],
        scale: parseFloat(editGeomScale.value) || 1
      };

      try {
        const res = await fetch(apiBase + '/update_station_geometry', {
          method: 'POST',
          headers: getHeaders(),
          body: JSON.stringify(update)
        });
        const result = await res.json();

        if (result.ok) {
          await loadStationGeometries();
        } else {
          alert('Update failed: ' + (result.error || 'Unknown error'));
        }
      } catch (e) {
        console.error('Update station geometry error', e);
      }
    });
  }

  // Delete station geometry
  if (deleteStationGeomBtn) {
    deleteStationGeomBtn.addEventListener('click', async () => {
      if (!selectedStationGeomId) return;

      if (!confirm('Delete this geometry?')) return;

      try {
        const res = await fetch(apiBase + '/delete_station_geometry', {
          method: 'POST',
          headers: getHeaders(),
          body: JSON.stringify({ id: selectedStationGeomId })
        });
        const result = await res.json();

        if (result.ok) {
          selectedStationGeomId = null;
          stationGeomEditSection.style.display = 'none';
          await loadStationGeometries();
        } else {
          alert('Delete failed: ' + (result.error || 'Unknown error'));
        }
      } catch (e) {
        console.error('Delete station geometry error', e);
      }
    });
  }

  // Update 3D meshes for station geometries
  async function updateStationMeshes() {
    // Remove meshes that no longer exist
    const currentIds = new Set(stationGeometries.map(g => g.id));
    for (const id of Object.keys(stationMeshes)) {
      if (!currentIds.has(id)) {
        stationGroup.remove(stationMeshes[id]);
        delete stationMeshes[id];
      }
    }

    // Add/update meshes
    for (const geom of stationGeometries) {
      let meshGroup = stationMeshes[geom.id];

      if (!meshGroup) {
        // Create new mesh group
        meshGroup = new THREE.Group();
        stationMeshes[geom.id] = meshGroup;
        stationGroup.add(meshGroup);

        // Load GLTF
        try {
          const gltfScene = await loadGltfScene(geom.url);
          if (gltfScene) {
            gltfScene.traverse(obj => {
              if (obj.isMesh) {
                obj.castShadow = true;
                obj.receiveShadow = true;
              }
            });
            meshGroup.add(gltfScene);
            console.log('[APP] Station GLTF loaded:', geom.url);
          }
        } catch (e) {
          console.error('Failed to load station GLTF/GLB', geom.url, e);
          console.error('Hint: if .gltf uses external .bin/textures, upload those files too.');
          // Add placeholder box
          const placeholder = new THREE.Mesh(
            new THREE.BoxGeometry(0.1, 0.1, 0.1),
            new THREE.MeshStandardMaterial({ color: 0x888888, wireframe: true })
          );
          meshGroup.add(placeholder);
        }
      }

      // Update transform
      meshGroup.position.set(geom.position[0], geom.position[1], geom.position[2]);
      meshGroup.rotation.set(geom.rotation[0], geom.rotation[1], geom.rotation[2]);
      const s = geom.scale || 1;
      meshGroup.scale.set(s, s, s);
    }
  }

  // Load config and station on init
  // Moved to initApp() for authentication
  // loadRobotConfig();
  // loadStationGeometries();

  // ===========================================================================
  // SECTION 18: ACTUATOR SELECTION SYSTEM
  // ===========================================================================

  // DOM Elements for Actuators tab
  const sfTorqueInput = document.getElementById('safety_factor_torque');
  const sfSpeedInput = document.getElementById('safety_factor_speed');
  const btnAnalyzeReq = document.getElementById('btn_analyze_requirements');
  const btnSelectActuators = document.getElementById('btn_select_actuators');
  const btnValidateSelection = document.getElementById('btn_validate_selection');
  const validationStatus = document.getElementById('validation-status');
  const btnShowMotors = document.getElementById('btn_show_motors');
  const btnShowGearboxes = document.getElementById('btn_show_gearboxes');
  const btnAddComponent = document.getElementById('btn_add_component');
  const requirementsDisplay = document.getElementById('requirements-display');
  const recommendationsDisplay = document.getElementById('actuator-recommendations');
  const libraryDisplay = document.getElementById('library-display');

  // Component modal elements
  const componentModal = document.getElementById('component-modal');
  const componentModalTitle = document.getElementById('component-modal-title');
  const motorFields = document.getElementById('motor-fields');
  const gearboxFields = document.getElementById('gearbox-fields');
  const cancelComponentBtn = document.getElementById('cancel_component');
  const saveComponentBtn = document.getElementById('save_component');

  // Current view state
  let currentLibraryView = 'motors'; // 'motors' or 'gearboxes'
  let actuatorLibrary = { motors: [], gearboxes: [], compatibility_matrix: {} };
  let lastRequirements = null;
  let lastSelection = null;

  function invalidateActuatorCaches() {
    lastDemoDynamicsData = null;
    lastRequirements = null;
    lastSelection = null;
  }

  // Load actuator library from backend
  async function loadActuatorLibrary() {
    if (!FEATURES.BACKEND) {
      const isValidLibrary = (lib) => {
        const motors = lib?.motors || [];
        const gearboxes = lib?.gearboxes || [];
        const compat = lib?.compatibility_matrix || {};
        return motors.length > 0 && gearboxes.length > 0 && Object.keys(compat).length > 0;
      };

      try {
        const cached = localStorage.getItem('robodimm_actuator_library');
        if (cached) {
          const parsed = JSON.parse(cached);
          if (isValidLibrary(parsed)) {
            actuatorLibrary = parsed;
            return;
          }
          console.warn('Cached actuator library is incomplete, reloading bundled file');
        }
      } catch (e) {
        console.warn('Invalid local actuator library cache:', e);
      }
      try {
        const res = await fetch('./actuators_library.json');
        const data = await res.json();
        actuatorLibrary = {
          motors: data.motors || [],
          gearboxes: data.gearboxes || [],
          compatibility_matrix: data.compatibility_matrix || {}
        };
        if (!isValidLibrary(actuatorLibrary)) {
          console.error('Bundled actuator library is incomplete for lite mode', actuatorLibrary);
        }
        localStorage.setItem('robodimm_actuator_library', JSON.stringify(actuatorLibrary));
      } catch (e) {
        console.error('Failed to load local actuator library:', e);
      }
      return;
    }
    try {
      const res = await fetch(apiBase + '/actuators_library', { headers: getHeaders() });
      const data = await res.json();
      if (data.ok) {
        actuatorLibrary = {
          motors: data.motors || [],
          gearboxes: data.gearboxes || [],
          compatibility_matrix: data.compatibility_matrix || {}
        };
      }
    } catch (e) {
      console.error('Failed to load actuator library:', e);
    }
  }

  // Analyze trajectory requirements
  async function analyzeRequirements() {
    if (!FEATURES.BACKEND) {
      try {
        const program = stateManager.getProgram();
        const targets = stateManager.getTargets();
        if (!program || program.length === 0) {
          requirementsDisplay.innerHTML = '<p style="color:#f88;">No program to analyze. Create and execute a program first.</p>';
          return null;
        }

        const traj = generateProgramTrajectory(program, targets, currentQ, 0.02);
        const robotType = robotInfo ? robotInfo.type : 'CR4';
        const demoConfig = stateManager.getConfig ? stateManager.getConfig() : {};
        traj.tau = [];
        for (let i = 0; i < traj.q.length; i++) {
          const result = computeInverseDynamics(robotType, traj.q[i], traj.qd[i], traj.qdd[i], demoConfig);
          traj.tau.push(result.tau);
        }
        const data = { t: traj.time, q: traj.q, v: traj.qd, a: traj.qdd, tau: traj.tau };
        lastDemoDynamicsData = data;

        const requirements = analyzeTrajectoryRequirementsLite(data, robotType);
        lastRequirements = requirements;

        let html = '';
        for (const [joint, req] of Object.entries(requirements)) {
          html += `
            <div style="margin-bottom:8px; padding:6px; background:rgba(0,100,200,0.1); border-radius:4px;">
              <div style="color:#00d4ff; font-weight:bold;">${joint.replace('_', ' ').toUpperCase()}</div>
              <div>Peak τ: <span style="color:#f80;">${req.peak_torque_Nm.toFixed(2)}</span> Nm</div>
              <div>RMS τ: <span style="color:#fa0;">${req.rms_torque_Nm.toFixed(2)}</span> Nm</div>
              <div>Peak ω: <span style="color:#0f8;">${req.peak_velocity_rpm.toFixed(1)}</span> rpm</div>
            </div>
          `;
        }
        requirementsDisplay.innerHTML = html || '<p style="color:#888;">No data</p>';
        return requirements;
      } catch (e) {
        console.error('Analyze requirements error (lite):', e);
        requirementsDisplay.innerHTML = '<p style="color:#f88;">Error analyzing requirements</p>';
        return null;
      }
    }

    try {
      const res = await fetch(apiBase + '/trajectory_requirements', { headers: getHeaders() });
      const data = await res.json();

      if (!data.ok) {
        requirementsDisplay.innerHTML = `<p style="color:#f88;">${data.error}</p>`;
        return null;
      }

      let html = '';
      for (const [joint, req] of Object.entries(data.requirements)) {
        html += `
          <div style="margin-bottom:8px; padding:6px; background:rgba(0,100,200,0.1); border-radius:4px;">
            <div style="color:#00d4ff; font-weight:bold;">${joint.replace('_', ' ').toUpperCase()}</div>
            <div>Peak τ: <span style="color:#f80;">${req.peak_torque_Nm.toFixed(2)}</span> Nm</div>
            <div>RMS τ: <span style="color:#fa0;">${req.rms_torque_Nm.toFixed(2)}</span> Nm</div>
            <div>Peak ω: <span style="color:#0f8;">${req.peak_velocity_rpm.toFixed(1)}</span> rpm</div>
          </div>
        `;
      }
      requirementsDisplay.innerHTML = html || '<p style="color:#888;">No data</p>';
      lastRequirements = data.requirements;
      return data.requirements;
    } catch (e) {
      console.error('Analyze requirements error:', e);
      requirementsDisplay.innerHTML = '<p style="color:#f88;">Error fetching requirements</p>';
      return null;
    }
  }

  // Select actuators based on requirements
  async function selectActuators() {
    const sfTorque = parseFloat(sfTorqueInput?.value) || 1.5;
    const sfSpeed = parseFloat(sfSpeedInput?.value) || 1.2;

    if (!FEATURES.BACKEND) {
      try {
        if (!actuatorLibrary?.motors?.length || !actuatorLibrary?.gearboxes?.length || !Object.keys(actuatorLibrary?.compatibility_matrix || {}).length) {
          await loadActuatorLibrary();
        }
        const requirements = lastRequirements || await analyzeRequirements();
        if (!requirements) {
          recommendationsDisplay.innerHTML = '<p style="color:#f88;">No requirements available</p>';
          return;
        }

        const selection = selectActuatorsLite(
          requirements,
          actuatorLibrary.motors || [],
          actuatorLibrary.gearboxes || [],
          actuatorLibrary.compatibility_matrix || {},
          sfTorque,
          sfSpeed
        );
        lastSelection = selection;
        displaySelectionResults(selection);
        return;
      } catch (e) {
        console.error('Select actuators error (lite):', e);
        recommendationsDisplay.innerHTML = '<p style="color:#f88;">Error selecting actuators</p>';
        return;
      }
    }

    try {
      const res = await fetch(apiBase + '/select_actuators', {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({
          safety_factor_torque: sfTorque,
          safety_factor_speed: sfSpeed,
          apply_to_robot_config: true
        })
      });
      const data = await res.json();

      if (!data.ok) {
        recommendationsDisplay.innerHTML = `<p style="color:#f88;">${data.error}</p>`;
        return;
      }

      let html = '';
      for (const [joint, sel] of Object.entries(data.selection)) {
        const rec = sel.recommended;
        const required = sel.required;

        html += `
          <div style="margin-bottom:12px; padding:8px; background:rgba(0,200,100,0.1); border-radius:4px; border-left:3px solid ${rec ? '#0f8' : '#f88'};">
            <div style="color:#00d4ff; font-weight:bold; margin-bottom:4px;">${joint.replace('_', ' ').toUpperCase()}</div>
            <div style="font-size:10px; color:#888;">
              Original: ${required.original_peak_torque_Nm?.toFixed(2) || '?'} Nm @ ${required.original_peak_speed_rpm?.toFixed(1) || '?'} rpm
              (avg: ${required.mean_velocity_rpm?.toFixed(1) || '?'} rpm)
            </div>
            <div style="font-size:10px; color:#aa0;">
              With SF: ${required.torque_Nm.toFixed(2)} Nm @ ${required.speed_rpm.toFixed(1)} rpm
            </div>
        `;

        if (rec) {
          html += `
            <div style="margin-top:6px; padding:6px; background:rgba(0,255,100,0.1); border-radius:3px;">
              <div style="color:#0f8;">✓ RECOMMENDED</div>
              <div><strong>${rec.motor_desc}</strong></div>
              <div>+ <strong>${rec.gearbox_desc}</strong> @ ${rec.ratio}:1</div>
              <div style="font-size:10px; margin-top:4px;">
                Output: ${rec.output_torque_Nm} Nm @ ${rec.max_output_speed_rpm} rpm
              </div>
              <div style="font-size:10px; color:#0a0;">
                Margin: τ +${rec.margin_torque_pct}% | ω +${rec.margin_speed_pct}%
              </div>
              ${rec.natural_match ? '<div style="font-size:9px; color:#0f8;">★ Natural match</div>' : ''}
            </div>
          `;

          // Show alternatives if any
          if (sel.candidates.length > 1) {
            html += `<details style="margin-top:4px;"><summary style="cursor:pointer; font-size:10px; color:#888;">Other options (${sel.candidates.length - 1})</summary>`;
            for (let i = 1; i < sel.candidates.length; i++) {
              const alt = sel.candidates[i];
              html += `
                <div style="font-size:10px; padding:4px; background:rgba(100,100,100,0.2); margin-top:2px; border-radius:2px;">
                  ${alt.motor_id} + ${alt.gearbox_id} @ ${alt.ratio}:1
                  (${alt.output_torque_Nm} Nm, +${alt.margin_torque_pct}%)
                </div>
              `;
            }
            html += '</details>';
          }
        } else {
          html += `
            <div style="margin-top:6px; padding:6px; background:rgba(255,100,100,0.2); border-radius:3px;">
              <div style="color:#f88;">✗ NO SUITABLE ACTUATOR FOUND</div>
              <div style="font-size:10px;">Requirements exceed available motor+gearbox combinations</div>
          `;
          // Show rejected samples for debugging
          if (sel.rejected_samples && sel.rejected_samples.length > 0) {
            html += `<details style="margin-top:4px;"><summary style="cursor:pointer; font-size:10px; color:#f88;">Why rejected? (samples)</summary>`;
            for (const rej of sel.rejected_samples) {
              html += `
                <div style="font-size:9px; padding:3px; background:rgba(255,50,50,0.1); margin-top:2px; border-radius:2px;">
                  ${rej.motor_id} + ${rej.gearbox_id} @ ${rej.ratio}:1 → 
                  ${rej.output_torque_Nm} Nm, ${rej.output_speed_rpm} rpm<br/>
                  <span style="color:#f66;">❌ ${rej.fail_reason}</span>
                </div>
              `;
            }
            html += '</details>';
          }
          html += '</div>';
        }

        html += '</div>';
      }

      recommendationsDisplay.innerHTML = html || '<p style="color:#888;">No recommendations</p>';
      lastSelection = data.selection;

    } catch (e) {
      console.error('Select actuators error:', e);
      recommendationsDisplay.innerHTML = '<p style="color:#f88;">Error selecting actuators</p>';
    }
  }

  // Display library (motors or gearboxes)
  function displayLibrary(type) {
    currentLibraryView = type;
    const items = type === 'motors' ? actuatorLibrary.motors : actuatorLibrary.gearboxes;

    if (!items || items.length === 0) {
      libraryDisplay.innerHTML = `<p style="color:#888;">No ${type} in library</p>`;
      return;
    }

    let html = `<div style="font-weight:bold; color:#00d4ff; margin-bottom:6px;">${type.toUpperCase()} (${items.length})</div>`;

    for (const item of items) {
      if (type === 'motors') {
        html += `
          <div style="padding:6px; background:rgba(100,100,200,0.1); margin-bottom:4px; border-radius:3px; cursor:pointer;" 
               onclick="window.editMotor && window.editMotor('${item.id}')">
            <div style="color:#8cf;"><strong>${item.id}</strong></div>
            <div style="font-size:9px; color:#aaa;">${item.description || ''}</div>
            <div style="font-size:9px;">
              ${item.nominal_torque_Nm} Nm @ ${item.nominal_speed_rpm} rpm | Flange: ${item.flange_mm || '?'}mm
            </div>
            <div style="font-size:9px; color:#666;">
              Compatible: ${(item.compatible_gearboxes || []).join(', ') || 'none'}
            </div>
          </div>
        `;
      } else {
        html += `
          <div style="padding:6px; background:rgba(200,100,100,0.1); margin-bottom:4px; border-radius:3px; cursor:pointer;"
               onclick="window.editGearbox && window.editGearbox('${item.id}')">
            <div style="color:#fc8;"><strong>${item.id}</strong></div>
            <div style="font-size:9px; color:#aaa;">${item.description || ''}</div>
            <div style="font-size:9px;">
              Ratios: ${(item.ratios || []).join(', ')} | Servo: ${item.for_servo_mm || '?'}mm
            </div>
            ${item.efficiency ? `<div style="font-size:9px;">Efficiency: ${(item.efficiency * 100).toFixed(0)}%</div>` : ''}
          </div>
        `;
      }
    }

    libraryDisplay.innerHTML = html;
  }

  // Show component modal for adding new motor/gearbox
  function showAddComponentModal(type) {
    componentModalTitle.textContent = type === 'motor' ? 'Add Motor' : 'Add Gearbox';

    // Show appropriate fields
    motorFields.style.display = type === 'motor' ? 'block' : 'none';
    gearboxFields.style.display = type === 'gearbox' ? 'block' : 'none';

    // Clear fields
    if (type === 'motor') {
      document.getElementById('motor_id').value = '';
      document.getElementById('motor_desc').value = '';
      document.getElementById('motor_flange').value = '';
      document.getElementById('motor_nom_torque').value = '';
      document.getElementById('motor_peak_torque').value = '';
      document.getElementById('motor_nom_speed').value = '';
      document.getElementById('motor_max_speed').value = '';
      document.getElementById('motor_mass').value = '';
      document.getElementById('motor_length').value = '';
      document.getElementById('motor_compatible').value = '';
    } else {
      document.getElementById('gearbox_id').value = '';
      document.getElementById('gearbox_desc').value = '';
      document.getElementById('gearbox_servo').value = '';
      document.getElementById('gearbox_ratios').value = '';
      document.getElementById('gearbox_rated_torque').value = '';
      document.getElementById('gearbox_peak_torque').value = '';
      document.getElementById('gearbox_efficiency').value = '0.9';
      document.getElementById('gearbox_mass').value = '';
      document.getElementById('gearbox_backlash').value = '';
    }

    componentModal.style.display = 'block';
    componentModal.dataset.type = type;
    componentModal.dataset.editing = '';
    modalOverlay.style.display = 'block';
  }

  // Edit existing motor
  window.editMotor = function (id) {
    const motor = actuatorLibrary.motors.find(m => m.id === id);
    if (!motor) return;

    componentModalTitle.textContent = 'Edit Motor';
    motorFields.style.display = 'block';
    gearboxFields.style.display = 'none';

    document.getElementById('motor_id').value = motor.id;
    document.getElementById('motor_desc').value = motor.description || '';
    document.getElementById('motor_flange').value = motor.flange_mm || '';
    document.getElementById('motor_nom_torque').value = motor.nominal_torque_Nm || '';
    document.getElementById('motor_peak_torque').value = motor.peak_torque_Nm || '';
    document.getElementById('motor_nom_speed').value = motor.nominal_speed_rpm || '';
    document.getElementById('motor_max_speed').value = motor.max_speed_rpm || '';
    document.getElementById('motor_mass').value = motor.mass_kg || '';
    document.getElementById('motor_length').value = motor.length_mm || '';
    document.getElementById('motor_compatible').value = (motor.compatible_gearboxes || []).join(',');

    componentModal.style.display = 'block';
    componentModal.dataset.type = 'motor';
    componentModal.dataset.editing = id;
    modalOverlay.style.display = 'block';
  };

  // Edit existing gearbox
  window.editGearbox = function (id) {
    const gb = actuatorLibrary.gearboxes.find(g => g.id === id);
    if (!gb) return;

    componentModalTitle.textContent = 'Edit Gearbox';
    motorFields.style.display = 'none';
    gearboxFields.style.display = 'block';

    document.getElementById('gearbox_id').value = gb.id;
    document.getElementById('gearbox_desc').value = gb.description || '';
    document.getElementById('gearbox_servo').value = gb.for_servo_mm || '';
    document.getElementById('gearbox_ratios').value = (gb.ratios || []).join(',');
    document.getElementById('gearbox_rated_torque').value = gb.rated_torque_Nm || '';
    document.getElementById('gearbox_peak_torque').value = gb.peak_torque_Nm || '';
    document.getElementById('gearbox_efficiency').value = gb.efficiency || 0.9;
    document.getElementById('gearbox_mass').value = gb.mass_kg || '';
    document.getElementById('gearbox_backlash').value = gb.backlash_arcmin || '';

    componentModal.style.display = 'block';
    componentModal.dataset.type = 'gearbox';
    componentModal.dataset.editing = id;
    modalOverlay.style.display = 'block';
  };

  // Save component (motor or gearbox)
  async function saveComponent() {
    const type = componentModal.dataset.type;

    if (type === 'motor') {
      const id = document.getElementById('motor_id').value.trim();
      const nomTorque = parseFloat(document.getElementById('motor_nom_torque').value);
      const nomSpeed = parseFloat(document.getElementById('motor_nom_speed').value);

      if (!id || isNaN(nomTorque) || isNaN(nomSpeed)) {
        alert('ID, Nominal Torque and Nominal Speed are required');
        return;
      }

      const compatibleStr = document.getElementById('motor_compatible').value;
      const compatible = compatibleStr ? compatibleStr.split(',').map(s => s.trim()).filter(s => s) : [];

      const motorData = {
        id: id,
        description: document.getElementById('motor_desc').value || '',
        flange_mm: parseFloat(document.getElementById('motor_flange').value) || null,
        nominal_torque_Nm: nomTorque,
        peak_torque_Nm: parseFloat(document.getElementById('motor_peak_torque').value) || null,
        nominal_speed_rpm: nomSpeed,
        max_speed_rpm: parseFloat(document.getElementById('motor_max_speed').value) || null,
        mass_kg: parseFloat(document.getElementById('motor_mass').value) || null,
        length_mm: parseFloat(document.getElementById('motor_length').value) || null,
        compatible_gearboxes: compatible
      };

      try {
        if (!FEATURES.BACKEND) {
          const motors = actuatorLibrary.motors || [];
          const idx = motors.findIndex(m => m.id === motorData.id);
          if (idx >= 0) motors[idx] = motorData;
          else motors.push(motorData);
          actuatorLibrary.motors = motors;
          localStorage.setItem('robodimm_actuator_library', JSON.stringify(actuatorLibrary));
          await loadActuatorLibrary();
          displayLibrary('motors');
          closeComponentModal();
          return;
        }
        const res = await fetch(apiBase + '/add_motor', {
          method: 'POST',
          headers: getHeaders(),
          body: JSON.stringify(motorData)
        });
        const result = await res.json();
        if (result.ok) {
          await loadActuatorLibrary();
          displayLibrary('motors');
          closeComponentModal();
        } else {
          alert('Failed to save motor: ' + (result.error || 'Unknown error'));
        }
      } catch (e) {
        console.error('Save motor error:', e);
        alert('Error saving motor');
      }

    } else if (type === 'gearbox') {
      const id = document.getElementById('gearbox_id').value.trim();
      const ratiosStr = document.getElementById('gearbox_ratios').value;

      if (!id || !ratiosStr) {
        alert('ID and Ratios are required');
        return;
      }

      const ratios = ratiosStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

      const gearboxData = {
        id: id,
        description: document.getElementById('gearbox_desc').value || '',
        for_servo_mm: parseFloat(document.getElementById('gearbox_servo').value) || null,
        ratios: ratios,
        rated_torque_Nm: parseFloat(document.getElementById('gearbox_rated_torque').value) || null,
        peak_torque_Nm: parseFloat(document.getElementById('gearbox_peak_torque').value) || null,
        efficiency: parseFloat(document.getElementById('gearbox_efficiency').value) || null,
        mass_kg: parseFloat(document.getElementById('gearbox_mass').value) || null,
        backlash_arcmin: parseFloat(document.getElementById('gearbox_backlash').value) || null
      };

      try {
        if (!FEATURES.BACKEND) {
          const gearboxes = actuatorLibrary.gearboxes || [];
          const idx = gearboxes.findIndex(g => g.id === gearboxData.id);
          if (idx >= 0) gearboxes[idx] = gearboxData;
          else gearboxes.push(gearboxData);
          actuatorLibrary.gearboxes = gearboxes;
          localStorage.setItem('robodimm_actuator_library', JSON.stringify(actuatorLibrary));
          await loadActuatorLibrary();
          displayLibrary('gearboxes');
          closeComponentModal();
          return;
        }
        const res = await fetch(apiBase + '/add_gearbox', {
          method: 'POST',
          headers: getHeaders(),
          body: JSON.stringify(gearboxData)
        });
        const result = await res.json();
        if (result.ok) {
          await loadActuatorLibrary();
          displayLibrary('gearboxes');
          closeComponentModal();
        } else {
          alert('Failed to save gearbox: ' + (result.error || 'Unknown error'));
        }
      } catch (e) {
        console.error('Save gearbox error:', e);
        alert('Error saving gearbox');
      }
    }
  }

  function closeComponentModal() {
    componentModal.style.display = 'none';
    modalOverlay.style.display = 'none';
  }

  // Event listeners for actuator tab
  if (btnAnalyzeReq) {
    btnAnalyzeReq.addEventListener('click', analyzeRequirements);
  }

  if (btnSelectActuators) {
    btnSelectActuators.addEventListener('click', async () => {
      // First analyze, then select
      await analyzeRequirements();
      await selectActuators();
      // Hide validation status until user clicks validate
      if (validationStatus) validationStatus.style.display = 'none';
    });
  }

  if (btnValidateSelection) {
    btnValidateSelection.addEventListener('click', async () => {
      await validateSelection();
    });
  }

  // Validate selection (2nd round with actuator masses)
  async function validateSelection() {
    const sfTorque = parseFloat(sfTorqueInput?.value) || 1.5;
    const sfSpeed = parseFloat(sfSpeedInput?.value) || 1.2;

    if (!FEATURES.BACKEND) {
      try {
        btnValidateSelection.disabled = true;
        btnValidateSelection.textContent = '⏳ Validating...';

        const req = lastRequirements || await analyzeRequirements();
        if (!req) {
          validationStatus.innerHTML = '<span style="color:#f88;">❌ No requirements to validate</span>';
          validationStatus.style.display = 'block';
          return;
        }
        const sel = lastSelection || selectActuatorsLite(
          req,
          actuatorLibrary.motors || [],
          actuatorLibrary.gearboxes || [],
          actuatorLibrary.compatibility_matrix || {},
          sfTorque,
          sfSpeed
        );

        const cfg = stateManager.getConfig ? stateManager.getConfig() : {};
        const data = validateSelectionLite(
          req,
          sel,
          actuatorLibrary.motors || [],
          actuatorLibrary.gearboxes || [],
          actuatorLibrary.compatibility_matrix || {},
          sfTorque,
          sfSpeed,
          (robotInfo && robotInfo.type) || 'CR4',
          Number(cfg.payload_kg || 0)
        );

        if (data.validated) {
          validationStatus.innerHTML = `
            <span style="color:#0f8;">✅ VALIDATED: Selection passes 2nd round</span><br/>
            <span style="color:#888; font-size:9px;">
              Actuator masses add ${data.total_actuator_contribution_kg} kg to J1 load.
              No changes needed.
            </span>
          `;
          validationStatus.style.background = 'rgba(0,255,100,0.1)';
        } else {
          let changesHtml = data.changes_needed.map(c =>
            `<div style="color:#fa0;">${c.joint}: ${c.round1} → ${c.round2}</div>`
          ).join('');
          validationStatus.innerHTML = `
            <span style="color:#fa0;">⚠️ CHANGES NEEDED after adding actuator masses</span><br/>
            <span style="color:#888; font-size:9px;">
              Actuator masses add ${data.total_actuator_contribution_kg} kg to J1 load.
            </span>
            <div style="margin-top:4px;">${changesHtml}</div>
          `;
          validationStatus.style.background = 'rgba(255,150,0,0.15)';
          displaySelectionResults(data.round2_selection);
        }
        validationStatus.style.display = 'block';
      } catch (e) {
        console.error('Validate selection error (lite):', e);
        validationStatus.innerHTML = '<span style="color:#f88;">Error validating</span>';
        validationStatus.style.display = 'block';
      } finally {
        btnValidateSelection.disabled = false;
        btnValidateSelection.textContent = '🔄 Validate (2nd round with masses)';
      }
      return;
    }

    try {
      btnValidateSelection.disabled = true;
      btnValidateSelection.textContent = '⏳ Validating...';

      const res = await fetch(apiBase + '/validate_selection', {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({
          safety_factor_torque: sfTorque,
          safety_factor_speed: sfSpeed
        })
      });
      const data = await res.json();

      if (!data.ok) {
        validationStatus.innerHTML = `<span style="color:#f88;">❌ ${data.error}</span>`;
        validationStatus.style.display = 'block';
        return;
      }

      // Show validation result
      if (data.validated) {
        validationStatus.innerHTML = `
          <span style="color:#0f8;">✅ VALIDATED: Selection passes 2nd round</span><br/>
          <span style="color:#888; font-size:9px;">
            Actuator masses add ${data.total_actuator_contribution_kg} kg to J1 load.
            No changes needed.
          </span>
        `;
        validationStatus.style.background = 'rgba(0,255,100,0.1)';
      } else {
        let changesHtml = data.changes_needed.map(c =>
          `<div style="color:#fa0;">${c.joint}: ${c.round1} → ${c.round2}</div>`
        ).join('');
        validationStatus.innerHTML = `
          <span style="color:#fa0;">⚠️ CHANGES NEEDED after adding actuator masses</span><br/>
          <span style="color:#888; font-size:9px;">
            Actuator masses add ${data.total_actuator_contribution_kg} kg to J1 load.
          </span>
          <div style="margin-top:4px;">${changesHtml}</div>
        `;
        validationStatus.style.background = 'rgba(255,150,0,0.15)';

        // Update recommendations with round2 selection
        displaySelectionResults(data.round2_selection);
      }
      validationStatus.style.display = 'block';

    } catch (e) {
      console.error('Validate selection error:', e);
      validationStatus.innerHTML = '<span style="color:#f88;">Error validating</span>';
      validationStatus.style.display = 'block';
    } finally {
      btnValidateSelection.disabled = false;
      btnValidateSelection.textContent = '🔄 Validate (2nd round with masses)';
    }
  }

  // Helper to display selection results (used by both select and validate)
  function displaySelectionResults(selection) {
    let html = '';
    for (const [joint, sel] of Object.entries(selection)) {
      const rec = sel.recommended;
      const required = sel.required;

      html += `
        <div style="margin-bottom:12px; padding:8px; background:rgba(0,200,100,0.1); border-radius:4px; border-left:3px solid ${rec ? '#0f8' : '#f88'};">
          <div style="color:#00d4ff; font-weight:bold; margin-bottom:4px;">${joint.replace('_', ' ').toUpperCase()}</div>
          <div style="font-size:10px; color:#888;">
            Original: ${required.original_peak_torque_Nm?.toFixed(2) || '?'} Nm @ ${required.original_peak_speed_rpm?.toFixed(1) || '?'} rpm
            (avg: ${required.mean_velocity_rpm?.toFixed(1) || '?'} rpm)
          </div>
          <div style="font-size:10px; color:#aa0;">
            With SF: ${required.torque_Nm.toFixed(2)} Nm @ ${required.speed_rpm.toFixed(1)} rpm
          </div>
      `;

      if (rec) {
        html += `
          <div style="margin-top:6px; padding:6px; background:rgba(0,255,100,0.1); border-radius:3px;">
            <div style="color:#0f8;">✓ RECOMMENDED</div>
            <div><strong>${rec.motor_desc}</strong></div>
            <div>+ <strong>${rec.gearbox_desc}</strong> @ ${rec.ratio}:1</div>
            <div style="font-size:10px; margin-top:4px;">
              Output: ${rec.output_torque_Nm} Nm @ ${rec.max_output_speed_rpm} rpm
            </div>
            <div style="font-size:10px; color:#0a0;">
              Margin: τ +${rec.margin_torque_pct}% | ω +${rec.margin_speed_pct}%
            </div>
            ${rec.natural_match ? '<div style="font-size:9px; color:#0f8;">★ Natural match</div>' : ''}
          </div>
        `;
      } else {
        html += `
          <div style="margin-top:6px; padding:6px; background:rgba(255,100,100,0.2); border-radius:3px;">
            <div style="color:#f88;">✗ NO SUITABLE ACTUATOR FOUND</div>
            <div style="font-size:10px;">Requirements exceed available motor+gearbox combinations</div>
          </div>
        `;
      }

      html += '</div>';
    }

    recommendationsDisplay.innerHTML = html || '<p style="color:#888;">No recommendations</p>';
  }

  if (btnShowMotors) {
    btnShowMotors.addEventListener('click', async () => {
      await loadActuatorLibrary();
      displayLibrary('motors');
    });
  }

  if (btnShowGearboxes) {
    btnShowGearboxes.addEventListener('click', async () => {
      await loadActuatorLibrary();
      displayLibrary('gearboxes');
    });
  }

  if (btnAddComponent) {
    btnAddComponent.addEventListener('click', () => {
      // Show choice dialog
      const type = prompt('Add component type:\n1 = Motor\n2 = Gearbox', '1');
      if (type === '1') {
        showAddComponentModal('motor');
      } else if (type === '2') {
        showAddComponentModal('gearbox');
      }
    });
  }

  if (cancelComponentBtn) {
    cancelComponentBtn.addEventListener('click', closeComponentModal);
  }

  if (saveComponentBtn) {
    saveComponentBtn.addEventListener('click', saveComponent);
  }

  // Initial load of actuator library
  // Moved to initApp -> loadRobotConfig() if we decide to load it there, or keep it lazy
  // But let's verify if initApp handles it.
  // Actually, loadActuatorLibrary is lazy load on tab click.

  // Run initial check
  initApp();
}
