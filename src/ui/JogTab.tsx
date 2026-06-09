import React, { useState, useEffect, useRef } from 'react';
import { useRobodimmStore } from '../model/state';
import { Serial6Engine } from '../math/serial6';
import { PalletizerEngine } from '../math/palletizer';
import { getTranslation, getMatrix3, multiply3x3AndVector, Vector3, multiply3x3, createIdentity4 } from '../math/matrix';
import { Save, RefreshCw, AlertTriangle } from 'lucide-react';

interface JogButtonProps {
  onClickAction: () => void;
  className?: string;
  children: React.ReactNode;
}

const JogButton: React.FC<JogButtonProps> = ({ onClickAction, className, children }) => {
  const callbackRef = useRef(onClickAction);
  useEffect(() => {
    callbackRef.current = onClickAction;
  }, [onClickAction]);

  const timeoutRef = useRef<any>(null);
  const intervalRef = useRef<any>(null);

  const startJogging = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    callbackRef.current();

    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    if (intervalRef.current) clearInterval(intervalRef.current);

    timeoutRef.current = setTimeout(() => {
      intervalRef.current = setInterval(() => {
        callbackRef.current();
      }, 70);
    }, 300);
  };

  const stopJogging = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <button
      onMouseDown={startJogging}
      onMouseUp={stopJogging}
      onMouseLeave={stopJogging}
      onTouchStart={startJogging}
      onTouchEnd={stopJogging}
      className={className}
    >
      {children}
    </button>
  );
};

export const JogTab: React.FC = () => {
  const {
    activeRobot,
    isSet,
    q,
    updateQ,
    updateTCP,
    program,
    addTarget
  } = useRobodimmStore();

  const [refFrame, setRefFrame] = useState<'world' | 'tcp'>('world');
  const [jogStep, setJogStep] = useState<number>(0.05); // meters
  const [yawStep, setYawStep] = useState<number>(5); // degrees
  const [targetName, setTargetName] = useState<string>('');

  if (!isSet) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center bg-slate-900/40 backdrop-blur">
        <AlertTriangle size={48} className="text-amber-500 mb-4 animate-pulse" />
        <h3 className="text-lg font-bold text-slate-200">Robot Not Locked</h3>
        <p className="text-sm text-slate-400 max-w-xs mt-2">
          Please, go to the <strong>Editor</strong> tab and click the <strong>Lock Robot (Set)</strong> button to initialize interactive jogging.
        </p>
      </div>
    );
  }

  // Compute live TCP coordinates and orientations
  let tcpPosition: Vector3 = [0, 0, 0];
  let currentYaw = 0; // for CR4

  let toolTransform: number[][] | null = null;
  if (activeRobot.kind === 'CR6') {
    const eng = new Serial6Engine(activeRobot);
    const fk = eng.forwardKinematics(q);
    toolTransform = fk.tcp_transform;
    tcpPosition = getTranslation(toolTransform);
  } else {
    const eng = new PalletizerEngine(activeRobot);
    const fk = eng.forwardKinematics(q);
    toolTransform = fk.transforms['TCP_frame'];
    tcpPosition = fk.points['TCP'];
    currentYaw = q[0] + q[3] + Math.PI; // CR4 tool yaw
  }

  // Set default target name if empty
  const defaultTargetName = `P${program.targets.length}`;
  const activeTargetName = targetName.trim() || defaultTargetName;

  // Joint sliders handlers
  const handleJointChange = (idx: number, degVal: number) => {
    const newQ = [...q];
    newQ[idx] = (degVal * Math.PI) / 180;
    updateQ(newQ);
  };

  // Cartesian Jog Handler
  const handleCartesianJog = (axis: 'X' | 'Y' | 'Z', direction: 1 | -1) => {
    const step = jogStep * direction;
    let dp: Vector3 = [0, 0, 0];
    if (axis === 'X') dp = [step, 0, 0];
    if (axis === 'Y') dp = [0, step, 0];
    if (axis === 'Z') dp = [0, 0, step];

    let dpWorld: Vector3 = [...dp];

    if (refFrame === 'tcp' && toolTransform) {
      const R = getMatrix3(toolTransform);
      dpWorld = multiply3x3AndVector(R, dp);
    }

    const newPos: Vector3 = [
      tcpPosition[0] + dpWorld[0],
      tcpPosition[1] + dpWorld[1],
      tcpPosition[2] + dpWorld[2]
    ];

    if (activeRobot.kind === 'CR6') {
      updateTCP(newPos);
    } else {
      updateTCP(newPos, currentYaw);
    }
  };

  // Yaw Jog (CR4 only)
  const handleYawJog = (direction: 1 | -1) => {
    const dyaw = (yawStep * direction * Math.PI) / 180;
    const newYaw = currentYaw + dyaw;
    updateTCP(tcpPosition, newYaw);
  };

  // Reorientation Jog Handler (CR6 only)
  const handleReorientJog = (axis: 'Roll' | 'Pitch' | 'Yaw', direction: 1 | -1) => {
    if (activeRobot.kind !== 'CR6' || !toolTransform) return;

    const dtheta = (yawStep * direction * Math.PI) / 180;
    let T_step: number[][];
    const c = Math.cos(dtheta);
    const s = Math.sin(dtheta);

    if (axis === 'Roll') {
      T_step = [
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
      ];
    } else if (axis === 'Pitch') {
      T_step = [
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
      ];
    } else {
      T_step = [
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ];
    }

    const R_current = getMatrix3(toolTransform);
    const R_step = getMatrix3(T_step);
    let R_new: number[][];

    if (refFrame === 'world') {
      // Rotation around world axes: R_new = R_step @ R_current
      R_new = multiply3x3(R_step, R_current);
    } else {
      // Rotation around TCP axes: R_new = R_current @ R_step
      R_new = multiply3x3(R_current, R_step);
    }

    // Construct full T target matrix
    const T_new = createIdentity4();
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        T_new[r][c] = R_new[r][c];
      }
      T_new[r][3] = tcpPosition[r];
    }

    const eng = new Serial6Engine(activeRobot);
    const ik = eng.solveSphericalWristIK(T_new, q);
    if (ik.success) {
      updateQ(ik.q);
    } else {
      console.warn("Reorient IK failed");
    }
  };

  // Zero joints
  const handleResetJoints = () => {
    const zeroQ = Array(q.length).fill(0);
    updateQ(zeroQ);
  };

  // Save current pose as target
  const handleSaveTarget = () => {
    addTarget({
      name: activeTargetName,
      q: [...q],
      tcpPose: toolTransform ? toolTransform.map(row => [...row]) : undefined
    });
    setTargetName(''); // clear input for next target
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/40 backdrop-blur select-none">
      {/* Header */}
      <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/20">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200">Jogging Panel</h2>
        <button
          onClick={handleResetJoints}
          className="flex items-center gap-1 text-[10px] uppercase font-bold text-slate-400 hover:text-white px-2 py-1 bg-slate-800 rounded border border-slate-700 transition"
        >
          <RefreshCw size={10} />
          <span>Home (0s)</span>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
        {/* Joint Space Control Panel */}
        <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-400">Joint Control</h3>
          
          <div className="flex flex-col gap-3.5">
            {activeRobot.limits.map((limit, idx) => {
              const qValRad = q[idx] ?? 0;
              const qValDeg = (qValRad * 180) / Math.PI;
              
              const lowerLimDeg = isFinite(limit.lowerLimitRad) ? (limit.lowerLimitRad * 180) / Math.PI : -180;
              const upperLimDeg = isFinite(limit.upperLimitRad) ? (limit.upperLimitRad * 180) / Math.PI : 180;

              return (
                <div key={limit.name} className="flex flex-col gap-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-mono font-bold text-slate-300">{limit.name}</span>
                    <span className="font-mono font-bold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                      {qValDeg.toFixed(1)}°
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] font-mono text-slate-500 w-10 text-right">
                      {lowerLimDeg.toFixed(0)}°
                    </span>
                    <input
                      type="range"
                      min={lowerLimDeg}
                      max={upperLimDeg}
                      step={0.1}
                      value={qValDeg}
                      onChange={(e) => handleJointChange(idx, parseFloat(e.target.value))}
                      className="flex-1 accent-indigo-500 h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                    />
                    <span className="text-[10px] font-mono text-slate-500 w-10 text-left">
                      {upperLimDeg.toFixed(0)}°
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Cartesian Control Panel */}
        <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-4">
          <div className="flex justify-between items-center">
            <h3 className="text-xs font-bold uppercase tracking-wider text-amber-500">Cartesian Control (TCP)</h3>
            
            {/* Frame Selector */}
            <div className="flex p-0.5 bg-slate-900 border border-slate-800 rounded-lg">
              <button
                onClick={() => setRefFrame('world')}
                className={`text-[10px] font-bold uppercase px-2.5 py-1 rounded-md transition ${
                  refFrame === 'world' ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-white'
                }`}
              >
                World
              </button>
              <button
                onClick={() => setRefFrame('tcp')}
                className={`text-[10px] font-bold uppercase px-2.5 py-1 rounded-md transition ${
                  refFrame === 'tcp' ? 'bg-amber-600 text-white shadow' : 'text-slate-400 hover:text-white'
                }`}
              >
                TCP
              </button>
            </div>
          </div>

          {/* Jog Step configuration */}
          <div className="grid grid-cols-2 gap-3 text-xs bg-slate-950/30 border border-slate-900 p-3 rounded-lg">
            <div className="flex flex-col gap-1">
              <span className="text-slate-400">Linear Step (m)</span>
              <select
                value={jogStep}
                onChange={(e) => setJogStep(parseFloat(e.target.value))}
                className="bg-slate-900 border border-slate-800 rounded p-1 text-slate-200 text-xs font-semibold focus:outline-none focus:border-amber-500"
              >
                <option value={0.001}>1 mm</option>
                <option value={0.01}>10 mm</option>
                <option value={0.05}>50 mm</option>
                <option value={0.1}>100 mm</option>
                <option value={0.25}>250 mm</option>
              </select>
            </div>

            {activeRobot.kind === 'CR4' && (
              <div className="flex flex-col gap-1">
                <span className="text-slate-400">Angular Step (Yaw)</span>
                <select
                  value={yawStep}
                  onChange={(e) => setYawStep(parseFloat(e.target.value))}
                  className="bg-slate-900 border border-slate-800 rounded p-1 text-slate-200 text-xs font-semibold focus:outline-none focus:border-amber-500"
                >
                  <option value={1}>1°</option>
                  <option value={5}>5°</option>
                  <option value={15}>15°</option>
                  <option value={45}>45°</option>
                </select>
              </div>
            )}
          </div>

          {/* Jog Buttons Grid */}
          <div className="flex flex-col gap-2">
            <div className="grid grid-cols-3 gap-2">
              {/* Row X */}
              <div className="flex items-center text-xs font-mono font-bold text-slate-400">X Axis</div>
              <JogButton
                onClickAction={() => handleCartesianJog('X', -1)}
                className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
              >
                -X
              </JogButton>
              <JogButton
                onClickAction={() => handleCartesianJog('X', 1)}
                className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
              >
                +X
              </JogButton>

              {/* Row Y */}
              <div className="flex items-center text-xs font-mono font-bold text-slate-400">Y Axis</div>
              <JogButton
                onClickAction={() => handleCartesianJog('Y', -1)}
                className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
              >
                -Y
              </JogButton>
              <JogButton
                onClickAction={() => handleCartesianJog('Y', 1)}
                className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
              >
                +Y
              </JogButton>

              {/* Row Z */}
              <div className="flex items-center text-xs font-mono font-bold text-slate-400">Z Axis</div>
              <JogButton
                onClickAction={() => handleCartesianJog('Z', -1)}
                className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
              >
                -Z
              </JogButton>
              <JogButton
                onClickAction={() => handleCartesianJog('Z', 1)}
                className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
              >
                +Z
              </JogButton>

              {/* Row Yaw (CR4 only) */}
              {activeRobot.kind === 'CR4' && (
                <>
                  <div className="flex items-center text-xs font-mono font-bold text-slate-400">Yaw Rotation</div>
                  <JogButton
                    onClickAction={() => handleYawJog(-1)}
                    className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
                  >
                    -Yaw
                  </JogButton>
                  <JogButton
                    onClickAction={() => handleYawJog(1)}
                    className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
                  >
                    +Yaw
                  </JogButton>
                </>
              )}

              {/* Roll, Pitch, Yaw (CR6 only) */}
              {activeRobot.kind === 'CR6' && (
                <>
                  <div className="flex items-center text-xs font-mono font-bold text-slate-400">Roll (Rx)</div>
                  <JogButton
                    onClickAction={() => handleReorientJog('Roll', -1)}
                    className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
                  >
                    -Roll
                  </JogButton>
                  <JogButton
                    onClickAction={() => handleReorientJog('Roll', 1)}
                    className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
                  >
                    +Roll
                  </JogButton>

                  <div className="flex items-center text-xs font-mono font-bold text-slate-400">Pitch (Ry)</div>
                  <JogButton
                    onClickAction={() => handleReorientJog('Pitch', -1)}
                    className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
                  >
                    -Pitch
                  </JogButton>
                  <JogButton
                    onClickAction={() => handleReorientJog('Pitch', 1)}
                    className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
                  >
                    +Pitch
                  </JogButton>

                  <div className="flex items-center text-xs font-mono font-bold text-slate-400">Yaw (Rz)</div>
                  <JogButton
                    onClickAction={() => handleReorientJog('Yaw', -1)}
                    className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
                  >
                    -Yaw
                  </JogButton>
                  <JogButton
                    onClickAction={() => handleReorientJog('Yaw', 1)}
                    className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-600 text-slate-200 font-bold py-2 rounded-lg text-sm transition"
                  >
                    +Yaw
                  </JogButton>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Target Saving Footer */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/60 flex items-center justify-between gap-3">
        <input
          type="text"
          placeholder={defaultTargetName}
          className="flex-1 text-xs py-2 px-3 bg-slate-900 border border-slate-800 rounded-lg text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500 font-semibold"
          value={targetName}
          onChange={(e) => setTargetName(e.target.value)}
        />
        <button
          onClick={handleSaveTarget}
          className="flex items-center gap-1.5 text-xs py-2 px-4 rounded-lg bg-indigo-600 text-white font-semibold hover:bg-indigo-700 transition shadow-lg shadow-indigo-600/20"
        >
          <Save size={14} />
          <span>Save Pose</span>
        </button>
      </div>
    </div>
  );
};
