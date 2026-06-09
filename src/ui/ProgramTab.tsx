import React, { useState, useEffect } from 'react';
import { useRobodimmStore } from '../model/state';
import { ProgramInstruction, ProgramTarget } from '../model/schemas';
import {
  Play,
  Pause,
  Plus,
  Trash2,
  Download,
  Activity,
  Award,
  Database,
  Grid,
  FileSpreadsheet,
  FileCode,
  Sparkles,
  ArrowRight,
  Clock
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';

export const ProgramTab: React.FC = () => {
  const {
    activeRobot,
    isSet,
    program,
    torqueLog,
    playbackPoints,
    playbackIndex,
    isRecording,
    addTarget,
    removeTarget,
    addInstruction,
    removeInstruction,
    clearProgram,
    loadProgram,
    runSignalRecording
  } = useRobodimmStore();

  const [instructionType, setInstructionType] = useState<'MoveJ' | 'MoveL' | 'Pause'>('MoveJ');
  const [selectedTarget, setSelectedTarget] = useState<string>('');
  const [speed, setSpeed] = useState<number>(1.0);
  const [zone, setZone] = useState<number>(0.01);
  const [pauseDuration, setPauseDuration] = useState<number>(0.5);

  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [selectedJointIdx, setSelectedJointIdx] = useState<number>(0);
  const [chartMetric, setChartMetric] = useState<'position' | 'velocity' | 'acceleration' | 'torque' | 'power'>('torque');

  // Handle visualizer play loop aligned to real-time wall clock
  useEffect(() => {
    if (!isPlaying || playbackPoints.length === 0) return;

    let animFrameId: number;
    const realStart = performance.now();
    let simStart = useRobodimmStore.getState().playbackIndex;
    
    // If play is clicked at the very end, reset to start automatically
    if (simStart >= playbackPoints.length - 1) {
      simStart = 0;
      useRobodimmStore.setState({ playbackIndex: 0 });
    }

    let lastIndex = simStart;

    const tick = () => {
      const elapsedReal = (performance.now() - realStart) / 1000;
      const targetSimTime = simStart + elapsedReal;

      const points = useRobodimmStore.getState().playbackPoints;
      let nextIndex = lastIndex;
      
      // Fast forward to match the target simulation time
      while (
        nextIndex < points.length - 1 &&
        points[nextIndex].time_s < targetSimTime
      ) {
        nextIndex++;
      }

      // If the index changed, update the store
      if (nextIndex !== lastIndex) {
        lastIndex = nextIndex;
        useRobodimmStore.setState({ playbackIndex: nextIndex });
      }

      if (nextIndex >= points.length - 1) {
        setIsPlaying(false);
      } else {
        animFrameId = requestAnimationFrame(tick);
      }
    };

    animFrameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animFrameId);
  }, [isPlaying, playbackPoints]);

  if (!isSet) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center bg-slate-900/40 backdrop-blur">
        <Activity size={48} className="text-indigo-500 mb-4 animate-pulse" />
        <h3 className="text-lg font-bold text-slate-200">Robot Not Locked</h3>
        <p className="text-sm text-slate-400 max-w-xs mt-2">
          Please, go to the <strong>Editor</strong> tab and click the <strong>Lock Robot (Set)</strong> button to program trajectories.
        </p>
      </div>
    );
  }

  // Load preset by robot kind/model
  const loadPresetForKind = (kind: 'CR4' | 'CR6') => {
    clearProgram();
    
    let targets: ProgramTarget[] = [];
    let instructions: ProgramInstruction[] = [];

    if (kind === 'CR6') {
      // IRB4600 RobotStudio Like Program
      targets = [
        { name: 'Target_10', q: [0.0, 1.0722089882619912e-11, -2.361844053666573e-11, 0.0, 0.5235987758267564, 0.0] },
        { name: 'Target_20', q: [1.5804304748785105, 2.2565949109321082e-11, -5.1034731995969196e-11, 1.235403335186902e-09, 0.5235987761026415, -8.561014119834454e-10] },
        { name: 'Target_30', q: [1.580430507648714, 0.2434994616489643, 0.400597179480914, 8.767514447782787e-08, 0.9219047962443057, 1.5039532286209578e-07] },
        { name: 'Target_40', q: [1.5804305076115015, 0.6464364528495863, 0.47370406988485847, 8.944880791617038e-08, 0.4458608921622291, 1.4801686987198082e-07] },
        { name: 'Target_50', q: [1.5778309106627488, 1.0035079717482627, -0.21871052679419023, 1.7731087740280316e-05, 0.7812039849641046, -0.0026131561892150934] },
        { name: 'Target_60', q: [0.0035983491777540344, 0.6464364528323592, 0.473704099720357, 1.5287200305635906e-07, 0.4458608917669524, 3.871281202272314e-07] },
        { name: 'Target_70', q: [0.005533549476578692, 0.38237583638703576, 0.9934422969529288, -4.878883833292491e-05, 0.1901833120887435, 0.0019835562404031393] },
        { name: 'Target_80', q: [0.005533549988742337, -0.1371909231392512, 0.9559214115523806, -1.1834440861946405e-05, 0.7472707624169974, 0.0019449315377673138] }
      ];
      instructions = [
        { type: 'MoveJ', target_name: 'Target_10', speed_rad_s: 1.305, zone_m: 0.1 },
        { type: 'MoveJ', target_name: 'Target_20', speed_rad_s: 1.305, zone_m: 0.1 },
        { type: 'MoveJ', target_name: 'Target_30', speed_rad_s: 1.305, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_40', tcp_speed_m_s: 0.5, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_50', tcp_speed_m_s: 0.5, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_40', tcp_speed_m_s: 0.5, zone_m: 0.1 },
        { type: 'MoveJ', target_name: 'Target_60', speed_rad_s: 1.305, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_70', tcp_speed_m_s: 0.5, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_80', tcp_speed_m_s: 0.5, zone_m: 0.1 },
        { type: 'MoveJ', target_name: 'Target_10', speed_rad_s: 1.305, zone_m: 0.1 }
      ];
    } else {
      // IRB460 Palletizing RAPID Emulation Program
      targets = [
        { name: 'Target_10', q: [0.0, 0.0, 0.0, 0.0] },
        { name: 'Target_20', q: [1.5415536976287865, 0.898877593289879, 0.9990302960023246, -9.042258053426622e-10] },
        { name: 'Target_30', q: [1.541553735733068, 0.9609910004498775, 1.039232676381714, -1.1129717281121999e-07] },
        { name: 'Target_40', q: [1.5474723577497647, 1.2991925307033179, 0.6203535464305907, 0.005918496578899646] },
        { name: 'Target_50', q: [1.5474723577500065, 0.8652531502215532, -0.0014909738061408373, 0.005918469691279293] },
        { name: 'Target_60', q: [1.52930450439473, -0.22091347495599514, 0.3049250883785243, -0.012249351119885343] },
        { name: 'Target_70', q: [0.0, -0.22091347495599514, 0.3049250883785243, -0.012249350584953987] },
        { name: 'Target_80', q: [0.0, -0.22091347495599514, 0.3049250883785243, -0.012249350584953987] },
        { name: 'Target_90', q: [0.0, 0.8505420354364523, 1.5119773393180762, -0.012249350584953987] },
        { name: 'Target_100', q: [0.0, 1.161062789016279, 0.8593033159371732, -0.012249350584953987] }
      ];
      instructions = [
        { type: 'MoveJ', target_name: 'Target_10', speed_rad_s: 1.0, zone_m: 0.1 },
        { type: 'MoveJ', target_name: 'Target_20', speed_rad_s: 1.0, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_30', tcp_speed_m_s: 1.0, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_40', tcp_speed_m_s: 1.0, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_50', tcp_speed_m_s: 1.0, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_60', tcp_speed_m_s: 1.0, zone_m: 0.1 },
        { type: 'MoveJ', target_name: 'Target_70', speed_rad_s: 1.0, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_90', tcp_speed_m_s: 1.0, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_100', tcp_speed_m_s: 1.0, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_90', tcp_speed_m_s: 0.5, zone_m: 0.1 },
        { type: 'MoveL', target_name: 'Target_70', tcp_speed_m_s: 0.5, zone_m: 0.1 },
        { type: 'MoveJ', target_name: 'Target_10', speed_rad_s: 1.0, zone_m: 0.1 }
      ];
    }

    targets.forEach(t => addTarget(t));
    instructions.forEach(i => addInstruction(i));
  };

  const loadExamplePickAndPlace = () => {
    loadPresetForKind(activeRobot.kind);
  };

  // Import Program file (.json or .yaml)
  const handleImportProgramFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        let prog: any = null;
        if (file.name.endsWith('.json')) {
          prog = JSON.parse(text);
          if (!prog.targets || !prog.instructions) {
            if (prog.program && prog.program.targets && prog.program.instructions) {
              prog = prog.program;
            } else {
              throw new Error("Missing targets or instructions key");
            }
          }
        } else {
          prog = parseSimpleYaml(text);
        }

        if (prog && Array.isArray(prog.targets) && Array.isArray(prog.instructions)) {
          prog.targets = prog.targets.map((t: any) => ({
            name: String(t.name).replace(/['"]/g, ''),
            q: Array.isArray(t.q) ? t.q : []
          }));
          loadProgram({
            schema: 'robodimm.program.v1',
            name: prog.name || 'imported_program',
            targets: prog.targets,
            instructions: prog.instructions
          });
        } else {
          alert('Could not find a valid program structure in the file.');
        }
      } catch (err) {
        alert('Error parsing program file: ' + (err as Error).message);
      }
    };
    reader.readAsText(file);
  };

  // Add a new instruction
  const handleAddInstruction = () => {
    if (instructionType === 'Pause') {
      addInstruction({
        type: 'Pause',
        duration_s: pauseDuration
      });
    } else {
      if (!selectedTarget) return;
      if (instructionType === 'MoveJ') {
        addInstruction({
          type: 'MoveJ',
          target_name: selectedTarget,
          speed_rad_s: speed,
          zone_m: zone
        });
      } else {
        addInstruction({
          type: 'MoveL',
          target_name: selectedTarget,
          tcp_speed_m_s: speed,
          zone_m: zone
        });
      }
    }
  };

  // Export CSV Data
  const handleExportCSV = () => {
    if (!torqueLog) return;
    const jointNames = torqueLog.joint_names;
    
    // Header
    let csvContent = 'time_s';
    jointNames.forEach(name => {
      csvContent += `,${name}_pos_rad,${name}_vel_rads,${name}_acc_rads2,${name}_torque_Nm,${name}_power_W`;
    });
    csvContent += '\n';

    // Samples
    torqueLog.samples.forEach(sample => {
      let row = `${sample.time_s.toFixed(4)}`;
      for (let j = 0; j < jointNames.length; j++) {
        const p = sample.q[j];
        const v = sample.joint_velocity[j];
        const a = sample.joint_acceleration[j];
        const t = sample.tau[j];
        const pow = Math.abs(t * v);
        row += `,${p.toFixed(5)},${v.toFixed(5)},${a.toFixed(5)},${t.toFixed(4)},${pow.toFixed(4)}`;
      }
      csvContent += row + '\n';
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `${activeRobot.name.toLowerCase().replace(/\s+/g, '_')}_torque_log.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Export JSON Manifest (robot + program specs + backend details)
  const handleExportManifest = () => {
    const manifest = {
      schema: 'robodimm.manifest.v1',
      timestamp: new Date().toISOString(),
      robot: activeRobot,
      program: program,
      engine_used: torqueLog?.engine_used || 'demo_frontend',
      model_id: torqueLog?.model_id || (activeRobot.kind === 'CR6' ? 'cr6_demo_frontend' : 'cr4_demo_frontend'),
      backend_version: torqueLog?.manifest?.backend_version || null,
      pinocchio_version: torqueLog?.manifest?.pinocchio_version || null,
      robot_hash: torqueLog?.manifest?.robot_hash || null,
      trajectory_hash: torqueLog?.manifest?.trajectory_hash || null,
      backend_manifest: torqueLog?.manifest || null,
      summary: torqueLog ? {
        duration_s: torqueLog.samples[torqueLog.samples.length - 1]?.time_s || 0,
        num_samples: torqueLog.samples.length,
        dt_s: torqueLog.dt_s
      } : null
    };

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(manifest, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `${activeRobot.name.toLowerCase().replace(/\s+/g, '_')}_reproducibility_manifest.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Prepare chart data for Recharts
  const chartData = torqueLog ? torqueLog.samples.map(sample => {
    const val = sample.q[selectedJointIdx] ?? 0;
    const vel = sample.joint_velocity[selectedJointIdx] ?? 0;
    const acc = sample.joint_acceleration[selectedJointIdx] ?? 0;
    const t = sample.tau[selectedJointIdx] ?? 0;
    const pow = Math.abs(t * vel);

    let displayVal = 0;
    if (chartMetric === 'position') displayVal = val;
    else if (chartMetric === 'velocity') displayVal = vel;
    else if (chartMetric === 'acceleration') displayVal = acc;
    else if (chartMetric === 'torque') displayVal = t;
    else displayVal = pow;

    return {
      time: sample.time_s,
      valor: parseFloat(displayVal.toFixed(4))
    };
  }) : [];

  const getMetricLabel = () => {
    if (chartMetric === 'position') return 'Position (rad)';
    if (chartMetric === 'velocity') return 'Velocity (rad/s)';
    if (chartMetric === 'acceleration') return 'Acceleration (rad/s²)';
    if (chartMetric === 'torque') return 'Joint Torque (Nm)';
    return 'Mechanical Power (W)';
  };

  const getMetricColor = () => {
    if (chartMetric === 'position') return '#6366f1'; // indigo
    if (chartMetric === 'velocity') return '#06b6d4'; // cyan
    if (chartMetric === 'acceleration') return '#ec4899'; // pink
    if (chartMetric === 'torque') return '#f59e0b'; // amber
    return '#10b981'; // emerald
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/40 backdrop-blur">
      
      {/* Program Header */}
      <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/20 gap-2">
        <div className="flex items-center gap-2">
          <Clock size={16} className="text-indigo-400" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200">Duty Cycle Editor</h2>
        </div>
        <div className="flex gap-1.5 items-center">
          <label className="flex items-center gap-1.5 text-[10px] uppercase font-bold text-slate-400 hover:text-white px-2.5 py-1.5 bg-slate-800 rounded border border-slate-700 hover:bg-slate-750 transition cursor-pointer">
            <span>Import</span>
            <input
              type="file"
              accept=".json,.yaml,.yml"
              className="hidden"
              onChange={handleImportProgramFile}
            />
          </label>
          <div className="relative">
            <select
              onChange={(e) => {
                const val = e.target.value;
                if (val === 'cr4') {
                  loadPresetForKind('CR4');
                } else if (val === 'cr6') {
                  loadPresetForKind('CR6');
                }
                e.target.value = ''; // reset selection
              }}
              defaultValue=""
              className="appearance-none pr-8 flex items-center gap-1.5 text-[10px] uppercase font-bold text-indigo-400 hover:text-white px-2.5 py-1.5 bg-indigo-500/10 hover:bg-indigo-600/20 rounded border border-indigo-500/20 transition cursor-pointer focus:outline-none"
            >
              <option value="" disabled className="bg-slate-900 text-slate-400">Load Preset...</option>
              <option value="cr4" className="bg-slate-900 text-slate-200">
                {activeRobot.kind === 'CR4' ? '★ IRB460 Palletizing (CR4)' : 'IRB460 Palletizing (CR4)'}
              </option>
              <option value="cr6" className="bg-slate-900 text-slate-200">
                {activeRobot.kind === 'CR6' ? '★ IRB4600 RobotStudio (CR6)' : 'IRB4600 RobotStudio (CR6)'}
              </option>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-indigo-400">
              <Sparkles size={11} />
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
        {/* Step 1: Program Targets Panel */}
        <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">1. Registered Poses (Targets)</h3>
          {program.targets.length === 0 ? (
            <div className="text-[11px] text-slate-500 italic p-2 border border-dashed border-slate-800 rounded text-center">
              No registered poses. Move using the Jog Panel and click "Save Pose".
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {program.targets.map(target => (
                <div
                  key={target.name}
                  className="flex items-center gap-1.5 text-[10px] font-mono bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-300"
                >
                  <span className="font-bold text-indigo-400">{target.name}</span>
                  <button
                    onClick={() => removeTarget(target.name)}
                    className="text-slate-500 hover:text-red-400 transition"
                  >
                    <Trash2 size={10} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Step 2: Form to Add Instruction */}
        <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">2. New Trajectory Instruction</h3>
          
          <div className="grid grid-cols-3 gap-3">
            {/* Instruction Type Selector */}
            <div className="flex flex-col gap-1.5">
              <span className="text-[10px] text-slate-500 font-bold uppercase">Movement Type</span>
              <select
                value={instructionType}
                onChange={(e) => setInstructionType(e.target.value as any)}
                className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200 text-xs focus:outline-none"
              >
                <option value="MoveJ">MoveJ (Joint)</option>
                <option value="MoveL">MoveL (Linear)</option>
                <option value="Pause">Pause (Wait)</option>
              </select>
            </div>

            {/* Target Selector or Duration */}
            {instructionType === 'Pause' ? (
              <div className="flex flex-col gap-1.5">
                <span className="text-[10px] text-slate-500 font-bold uppercase">Duration (s)</span>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  value={pauseDuration}
                  onChange={(e) => setPauseDuration(parseFloat(e.target.value) || 0.1)}
                  className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200 text-xs"
                />
              </div>
            ) : (
              <>
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] text-slate-500 font-bold uppercase">Target Pose</span>
                  <select
                    value={selectedTarget}
                    onChange={(e) => setSelectedTarget(e.target.value)}
                    className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200 text-xs focus:outline-none"
                  >
                    <option value="">Select...</option>
                    {program.targets.map(t => (
                      <option key={t.name} value={t.name}>{t.name}</option>
                    ))}
                  </select>
                </div>
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] text-slate-500 font-bold uppercase">
                    {instructionType === 'MoveJ' ? 'Joint Speed (rad/s)' : 'TCP Speed (m/s)'}
                  </span>
                  <input
                    type="number"
                    step="0.1"
                    min="0.01"
                    value={speed}
                    onChange={(e) => setSpeed(parseFloat(e.target.value) || 0.1)}
                    className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-200 text-xs"
                  />
                </div>
              </>
            )}
          </div>

          <button
            onClick={handleAddInstruction}
            className="flex items-center justify-center gap-1.5 text-xs py-1.5 mt-1 rounded bg-indigo-600 hover:bg-indigo-700 text-white font-semibold transition"
          >
            <Plus size={14} />
            <span>Add to Program</span>
          </button>
        </div>

        {/* Step 3: Instructions List Table */}
        <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
          <div className="flex justify-between items-center">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">3. Program Sequence</h3>
            {program.instructions.length > 0 && (
              <button
                onClick={clearProgram}
                className="text-[10px] font-bold text-red-400 hover:underline"
              >
                Clear all
              </button>
            )}
          </div>

          {program.instructions.length === 0 ? (
            <div className="text-[11px] text-slate-500 italic p-3 border border-dashed border-slate-800 rounded text-center">
              The program is empty. Add some instructions or load the example Pick & Place.
            </div>
          ) : (
            <div className="max-h-[160px] overflow-y-auto border border-slate-800 rounded bg-slate-950/20 pr-1">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-slate-850 text-slate-500 font-bold text-[10px] uppercase">
                    <th className="p-2">#</th>
                    <th className="p-2">Command</th>
                    <th className="p-2">Details</th>
                    <th className="p-2 text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {program.instructions.map((inst, idx) => (
                    <tr key={idx} className="border-b border-slate-900/50 hover:bg-slate-900/20">
                      <td className="p-2 font-mono text-slate-500">{idx + 1}</td>
                      <td className="p-2 font-semibold text-slate-200">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                          inst.type === 'MoveJ' ? 'bg-indigo-500/20 text-indigo-400' :
                          inst.type === 'MoveL' ? 'bg-cyan-500/20 text-cyan-400' :
                          'bg-slate-500/20 text-slate-400'
                        }`}>
                          {inst.type}
                        </span>
                      </td>
                      <td className="p-2 text-slate-400 font-mono text-[11px]">
                        {inst.type === 'Pause' ? (
                          <span>Wait: {inst.duration_s}s</span>
                        ) : (
                          <span>
                            Go to <strong className="text-slate-200 font-semibold">{inst.target_name}</strong>
                            {inst.type === 'MoveJ' ? ` @ ${inst.speed_rad_s} rad/s` : ` @ ${inst.tcp_speed_m_s} m/s`}
                          </span>
                        )}
                      </td>
                      <td className="p-2 text-right">
                        <button
                          onClick={() => removeInstruction(idx)}
                          className="text-slate-500 hover:text-red-400 p-1 transition"
                        >
                          <Trash2 size={12} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Step 4: Dynamics & Recording Controls */}
        {program.instructions.length > 0 && (
          <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-4">
            <div className="flex gap-3">
              <button
                onClick={runSignalRecording}
                disabled={isRecording}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-xs font-bold uppercase tracking-wider bg-emerald-600 text-white hover:bg-emerald-700 transition disabled:opacity-50"
              >
                {isRecording ? 'Recording...' : 'Record Dynamics (200Hz)'}
              </button>

              {playbackPoints.length > 0 && (
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className={`px-4 flex items-center justify-center gap-1.5 rounded-lg border text-xs font-bold uppercase transition ${
                    isPlaying
                      ? 'bg-amber-500/20 text-amber-400 border-amber-500/35 hover:bg-amber-500/30'
                      : 'bg-indigo-500/20 text-indigo-400 border-indigo-500/35 hover:bg-indigo-500/30'
                  }`}
                >
                  {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                  <span>{isPlaying ? 'Pause' : 'Play'}</span>
                </button>
              )}
            </div>

            {/* Playback Seek Bar */}
            {playbackPoints.length > 0 && (
              <div className="flex flex-col gap-1 text-[10px] bg-slate-950/30 p-2.5 rounded-lg border border-slate-900">
                <div className="flex justify-between items-center text-slate-400 font-semibold mb-1">
                  <span>3D Visual Simulation</span>
                  <span>
                    {(playbackPoints[playbackIndex]?.time_s ?? 0).toFixed(2)}s / {(playbackPoints[playbackPoints.length - 1]?.time_s ?? 0).toFixed(2)}s
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={playbackPoints.length - 1}
                  value={playbackIndex}
                  onChange={(e) => useRobodimmStore.setState({ playbackIndex: parseInt(e.target.value) || 0 })}
                  className="w-full accent-indigo-500 h-1 bg-slate-800 rounded appearance-none cursor-pointer"
                />
              </div>
            )}
          </div>
        )}

        {/* Step 5: Charts Visualizer */}
        {torqueLog && (
          <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Dynamic Analyzer</h3>
                <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                  torqueLog.engine_used === 'pro_cr4_kkt' || torqueLog.engine_used === 'pro_cr6_serial'
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : 'bg-indigo-500/20 text-indigo-400'
                }`}>
                  {(torqueLog.engine_used === 'pro_cr4_kkt' || torqueLog.engine_used === 'pro_cr6_serial')
                    ? 'PRO backend'
                    : 'Demo approx'}
                </span>
              </div>
              
              {/* Dropdowns to toggle graph */}
              <div className="flex gap-2">
                <select
                  value={selectedJointIdx}
                  onChange={(e) => setSelectedJointIdx(parseInt(e.target.value))}
                  className="bg-slate-900 border border-slate-800 rounded px-1.5 py-0.5 text-slate-300 text-[10px] focus:outline-none"
                >
                  {activeRobot.limits.map((lim, idx) => (
                    <option key={lim.name} value={idx}>{lim.name}</option>
                  ))}
                </select>

                <select
                  value={chartMetric}
                  onChange={(e) => setChartMetric(e.target.value as any)}
                  className="bg-slate-900 border border-slate-800 rounded px-1.5 py-0.5 text-slate-300 text-[10px] focus:outline-none"
                >
                  <option value="position">Position</option>
                  <option value="velocity">Velocity</option>
                  <option value="acceleration">Acceleration</option>
                  <option value="torque">Torque</option>
                  <option value="power">Power</option>
                </select>
              </div>
            </div>

            {/* Recharts Area Plot */}
            <div className="w-full h-48 mt-1">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={chartData}
                  margin={{ top: 5, right: 5, left: -25, bottom: 5 }}
                >
                  <defs>
                    <linearGradient id="colorMetric" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={getMetricColor()} stopOpacity={0.4}/>
                      <stop offset="95%" stopColor={getMetricColor()} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.3} />
                  <XAxis
                    dataKey="time"
                    stroke="#475569"
                    fontSize={9}
                    tickFormatter={(v) => `${v.toFixed(1)}s`}
                  />
                  <YAxis stroke="#475569" fontSize={9} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#94a3b8', fontSize: '10px', fontWeight: 'bold' }}
                    itemStyle={{ color: getMetricColor(), fontSize: '11px', fontWeight: 'bold' }}
                    formatter={(value: any) => [value, getMetricLabel()]}
                    labelFormatter={(label) => `Time: ${parseFloat(label).toFixed(3)}s`}
                  />
                  <Area
                    type="monotone"
                    dataKey="valor"
                    stroke={getMetricColor()}
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorMetric)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* Persistence / Export Footer */}
      {torqueLog && (
        <div className="p-4 border-t border-slate-800 bg-slate-950/60 flex items-center justify-between gap-3">
          <button
            onClick={handleExportCSV}
            className="flex-1 flex items-center justify-center gap-1.5 text-xs py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-350 font-bold border border-slate-700 hover:border-slate-650 rounded-lg transition"
          >
            <FileSpreadsheet size={13} />
            <span>Export CSV</span>
          </button>
          
          <button
            onClick={handleExportManifest}
            className="flex-1 flex items-center justify-center gap-1.5 text-xs py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-350 font-bold border border-slate-700 hover:border-slate-650 rounded-lg transition"
          >
            <FileCode size={13} />
            <span>JSON Manifest</span>
          </button>
        </div>
      )}
    </div>
  );
};

// Simple, lightweight YAML parser to support loading Kineforge .yaml program files
function parseSimpleYaml(yamlText: string): any {
  const lines = yamlText.split('\n');
  const result: any = { name: 'imported_yaml_program', targets: [], instructions: [] };
  let currentTarget: any = null;
  let currentInstruction: any = null;
  let mode: 'targets' | 'instructions' | null = null;
  let readingQ = false;

  for (let line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    // Determine current level of indentation
    const indent = line.length - line.trimStart().length;

    if (indent === 0) {
      if (trimmed.startsWith('name:')) {
        result.name = trimmed.split('name:')[1].trim().replace(/['"]/g, '');
      } else if (trimmed.startsWith('targets:')) {
        mode = 'targets';
        readingQ = false;
      } else if (trimmed.startsWith('instructions:')) {
        mode = 'instructions';
        readingQ = false;
      }
      continue;
    }

    if (mode === 'targets') {
      if (trimmed.startsWith('- name:')) {
        readingQ = false;
        currentTarget = { name: trimmed.split('name:')[1].trim().replace(/['"]/g, ''), q: [] };
        result.targets.push(currentTarget);
      } else if (trimmed.startsWith('q:')) {
        const afterColon = trimmed.split('q:')[1].trim();
        if (afterColon.startsWith('[') && afterColon.endsWith(']')) {
          const content = afterColon.slice(1, -1);
          const values = content.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
          if (currentTarget) {
            currentTarget.q = values;
          }
          readingQ = false;
        } else {
          readingQ = true;
        }
      } else if (readingQ && trimmed.startsWith('-')) {
        const val = parseFloat(trimmed.replace('-', '').trim());
        if (currentTarget && !isNaN(val)) {
          currentTarget.q.push(val);
        }
      } else if (!trimmed.startsWith('-') && readingQ) {
        readingQ = false;
      }
    } else if (mode === 'instructions') {
      if (trimmed.startsWith('- type:')) {
        currentInstruction = { type: trimmed.split('type:')[1].trim().replace(/['"]/g, '') };
        result.instructions.push(currentInstruction);
      } else if (currentInstruction) {
        const parts = trimmed.split(':');
        if (parts.length >= 2) {
          const key = parts[0].trim();
          const val = parts.slice(1).join(':').trim();
          if (key === 'target_name') {
            currentInstruction.target_name = val.replace(/['"]/g, ''); // remove any quotes
          } else if (key === 'speed_rad_s' || key === 'tcp_speed_m_s' || key === 'zone_m' || key === 'duration_s') {
            currentInstruction[key] = parseFloat(val);
          }
        }
      }
    }
  }
  return result;
}
