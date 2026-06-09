import React, { useState, useEffect } from 'react';
import { useRobodimmStore } from '../model/state';
import {
  Zap,
  Sliders,
  Shield,
  Layers,
  Database,
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  Info,
  ChevronDown,
  ChevronUp,
  Download,
  Check,
  X
} from 'lucide-react';
import { TorqueLog, SizingObjective } from '../model/schemas';

interface CyclePowerDiagnostics {
  cycleTimeS: number;
  jointStats: {
    joint_name: string;
    tau_rms: number;
    tau_peak: number;
    speed_peak: number;
    accel_peak: number;
    positive_power_mean: number;
    positive_power_peak: number;
  }[];
  positivePowerMeanTotal: number;
  positivePowerPeakTotal: number;
  energyPerCycleJ: number;
  energyPerCycleWh: number;
}

function computeCyclePowerDiagnostics(torqueLog: TorqueLog): CyclePowerDiagnostics {
  const numSamples = torqueLog.samples.length;
  const numJoints = torqueLog.joint_names.length;
  if (numSamples === 0) {
    return {
      cycleTimeS: 0,
      jointStats: [],
      positivePowerMeanTotal: 0,
      positivePowerPeakTotal: 0,
      energyPerCycleJ: 0,
      energyPerCycleWh: 0
    };
  }

  const cycleTimeS = torqueLog.samples[numSamples - 1].time_s;
  const jointStats = torqueLog.joint_names.map((name, j) => {
    let sumTauSq = 0;
    let sumPosPower = 0;
    let peakTau = 0;
    let peakSpeed = 0;
    let peakAccel = 0;
    let peakPosPower = 0;
    let totalTime = 0;

    for (let s = 0; s < numSamples; s++) {
      const sample = torqueLog.samples[s];
      const prevTime = s > 0 ? torqueLog.samples[s - 1].time_s : sample.time_s - torqueLog.dt_s;
      const dt = sample.time_s - prevTime;
      if (dt <= 0) continue;

      const tau = sample.tau[j];
      const speed = sample.joint_velocity ? sample.joint_velocity[j] : (sample.velocity ? sample.velocity[j] : 0);
      const accel = sample.joint_acceleration ? sample.joint_acceleration[j] : (sample.acceleration ? sample.acceleration[j] : 0);
      const power = tau * speed;
      const posPower = Math.max(power, 0);

      sumTauSq += tau * tau * dt;
      sumPosPower += posPower * dt;

      peakTau = Math.max(peakTau, Math.abs(tau));
      peakSpeed = Math.max(peakSpeed, Math.abs(speed));
      peakAccel = Math.max(peakAccel, Math.abs(accel));
      peakPosPower = Math.max(peakPosPower, posPower);

      totalTime += dt;
    }

    const tau_rms = totalTime > 0 ? Math.sqrt(sumTauSq / totalTime) : 0;
    const positive_power_mean = totalTime > 0 ? sumPosPower / totalTime : 0;

    return {
      joint_name: name,
      tau_rms,
      tau_peak: peakTau,
      speed_peak: peakSpeed,
      accel_peak: peakAccel,
      positive_power_mean,
      positive_power_peak: peakPosPower
    };
  });

  const positivePowerTotalTime = Array(numSamples).fill(0);
  for (let s = 0; s < numSamples; s++) {
    const sample = torqueLog.samples[s];
    for (let j = 0; j < numJoints; j++) {
      const tau = sample.tau[j];
      const speed = sample.joint_velocity ? sample.joint_velocity[j] : (sample.velocity ? sample.velocity[j] : 0);
      const power = tau * speed;
      positivePowerTotalTime[s] += Math.max(power, 0);
    }
  }

  let positivePowerPeakTotal = 0;
  let sumPositivePowerTotalTime = 0;
  let totalTimeIntegral = 0;

  for (let s = 0; s < numSamples; s++) {
    const sample = torqueLog.samples[s];
    const prevTime = s > 0 ? torqueLog.samples[s - 1].time_s : sample.time_s - torqueLog.dt_s;
    const dt = sample.time_s - prevTime;
    if (dt <= 0) continue;

    const posPowerT = positivePowerTotalTime[s];
    positivePowerPeakTotal = Math.max(positivePowerPeakTotal, posPowerT);
    sumPositivePowerTotalTime += posPowerT * dt;
    totalTimeIntegral += dt;
  }

  const positivePowerMeanTotal = totalTimeIntegral > 0 ? sumPositivePowerTotalTime / totalTimeIntegral : 0;
  const energyPerCycleJ = sumPositivePowerTotalTime;
  const energyPerCycleWh = energyPerCycleJ / 3600.0;

  return {
    cycleTimeS,
    jointStats,
    positivePowerMeanTotal,
    positivePowerPeakTotal,
    energyPerCycleJ,
    energyPerCycleWh
  };
}

export const ActuatorsTab: React.FC = () => {
  const {
    activeRobot,
    isSet,
    torqueLog,
    sizingResults,
    actuatorLibrary,
    runIterativeActuatorSizing,
    loadActuatorLibrary
  } = useRobodimmStore();

  const diagnostics = torqueLog ? computeCyclePowerDiagnostics(torqueLog) : null;


  // Safety margins state (with default values matching spec)
  const [marginCont, setMarginCont] = useState<number>(1.3);
  const [marginPeak, setMarginPeak] = useState<number>(1.2);
  const [marginSpeed, setMarginSpeed] = useState<number>(1.15);
  const [marginPower, setMarginPower] = useState<number>(1.1);
  const [sizingObjective, setSizingObjective] = useState<SizingObjective>('min_mass');
  
  // Advanced parameters
  const [motorPeakFactor, setMotorPeakFactor] = useState<number>(5.0);
  const [enforcePowerLimit, setEnforcePowerLimit] = useState<boolean>(false);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  // Reducer Type Filter
  const [gearboxType, setGearboxType] = useState<'any' | 'harmonic' | 'cycloidal'>('any');
  
  // UI Tabs
  const [subTab, setSubTab] = useState<'sizing' | 'motors' | 'gearboxes'>('sizing');
  const [expandedJointIdx, setExpandedJointIdx] = useState<number | null>(null);

  // Load library on mount if missing
  useEffect(() => {
    if (!actuatorLibrary) {
      loadActuatorLibrary();
    }
  }, [actuatorLibrary, loadActuatorLibrary]);

  // Run sizing in real-time as sliders or filters change
  useEffect(() => {
    if (torqueLog && actuatorLibrary) {
      runIterativeActuatorSizing(
        {
          continuous: marginCont,
          peak: marginPeak,
          speed: marginSpeed,
          power: marginPower,
          motorPeakFactor,
          enforcePowerLimit,
          sizingObjective
        } as any,
        gearboxType
      );
    }
  }, [
    torqueLog,
    actuatorLibrary,
    marginCont,
    marginPeak,
    marginSpeed,
    marginPower,
    motorPeakFactor,
    enforcePowerLimit,
    sizingObjective,
    gearboxType,
    runIterativeActuatorSizing
  ]);

  if (!isSet) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center bg-slate-900/40 backdrop-blur">
        <Zap size={48} className="text-amber-500 mb-4 animate-pulse" />
        <h3 className="text-lg font-bold text-slate-200">Robot Not Locked</h3>
        <p className="text-sm text-slate-400 max-w-xs mt-2">
          Please go to the <strong>Editor Spec</strong> tab and click the <strong>Lock Robot (Set)</strong> button to start sizing.
        </p>
      </div>
    );
  }

  const toggleExpandJoint = (idx: number) => {
    setExpandedJointIdx(expandedJointIdx === idx ? null : idx);
  };

  // Export Sizing Report
  const handleExportReport = () => {
    if (!sizingResults) return;
    
    const reportWithNotes = {
      ...sizingResults,
      notes: [
        `Motor peak torque is estimated as ${sizingResults.margins.motorPeakFactor.toFixed(1)} x rated torque for short robotic transients.`,
        "Gearbox peak is limited by max_intermittent_torque_Nm.",
        "Robot inertial parameters are not modified by selected actuators.",
        "Motor rated power is a sizing class, not expected average cycle consumption.",
        "Cycle energy and mean positive mechanical power estimate the motion energy demand."
      ]
    };

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(reportWithNotes, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `${activeRobot.name.toLowerCase().replace(/\s+/g, '_')}_actuator_sizing_report.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/40 backdrop-blur">
      
      {/* Sub Tabs Navigation */}
      <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/20">
        <div className="flex gap-2">
          <button
            onClick={() => setSubTab('sizing')}
            className={`text-[10px] py-1.5 px-3 rounded transition font-bold uppercase tracking-wider ${
              subTab === 'sizing' ? 'bg-indigo-650 text-white' : 'bg-slate-800 text-slate-400 hover:text-slate-300'
            }`}
          >
            Actuator Sizing
          </button>
          <button
            onClick={() => setSubTab('motors')}
            className={`text-[10px] py-1.5 px-3 rounded transition font-bold uppercase tracking-wider ${
              subTab === 'motors' ? 'bg-indigo-650 text-white' : 'bg-slate-800 text-slate-400 hover:text-slate-300'
            }`}
          >
            Motors Catalog
          </button>
          <button
            onClick={() => setSubTab('gearboxes')}
            className={`text-[10px] py-1.5 px-3 rounded transition font-bold uppercase tracking-wider ${
              subTab === 'gearboxes' ? 'bg-indigo-650 text-white' : 'bg-slate-800 text-slate-400 hover:text-slate-300'
            }`}
          >
            Reducers Catalog
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
        
        {/* VIEW 1: SIZING PROCESS */}
        {subTab === 'sizing' && (
          <>
            {/* Control Panel */}
            <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-4">
              <div className="flex justify-between items-center">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Sliders size={14} className="text-indigo-400" />
                  <span>Sizing Parameters & Safety Factors</span>
                </h3>
              </div>

              {/* Sliders Grid */}
              <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                {/* Continuous Margin */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-[10px] font-bold">
                    <span className="text-slate-400">Continuous Torque</span>
                    <span className="text-indigo-400">{marginCont.toFixed(2)}x</span>
                  </div>
                  <input
                    type="range"
                    min={1.0}
                    max={2.5}
                    step={0.05}
                    value={marginCont}
                    onChange={(e) => setMarginCont(parseFloat(e.target.value))}
                    className="accent-indigo-500 h-1 bg-slate-800 rounded appearance-none cursor-pointer"
                  />
                </div>

                {/* Peak Margin */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-[10px] font-bold">
                    <span className="text-slate-400">Peak Torque</span>
                    <span className="text-indigo-400">{marginPeak.toFixed(2)}x</span>
                  </div>
                  <input
                    type="range"
                    min={1.0}
                    max={2.5}
                    step={0.05}
                    value={marginPeak}
                    onChange={(e) => setMarginPeak(parseFloat(e.target.value))}
                    className="accent-indigo-500 h-1 bg-slate-800 rounded appearance-none cursor-pointer"
                  />
                </div>

                {/* Speed Margin */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-[10px] font-bold">
                    <span className="text-slate-400">Speed limit</span>
                    <span className="text-indigo-400">{marginSpeed.toFixed(2)}x</span>
                  </div>
                  <input
                    type="range"
                    min={1.0}
                    max={2.0}
                    step={0.05}
                    value={marginSpeed}
                    onChange={(e) => setMarginSpeed(parseFloat(e.target.value))}
                    className="accent-indigo-500 h-1 bg-slate-800 rounded appearance-none cursor-pointer"
                  />
                </div>

                {/* Power Margin */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-[10px] font-bold">
                    <span className="text-slate-400">Power rating</span>
                    <span className="text-indigo-400">{marginPower.toFixed(2)}x</span>
                  </div>
                  <input
                    type="range"
                    min={1.0}
                    max={2.0}
                    step={0.05}
                    value={marginPower}
                    onChange={(e) => setMarginPower(parseFloat(e.target.value))}
                    className="accent-indigo-500 h-1 bg-slate-800 rounded appearance-none cursor-pointer"
                  />
                </div>
              </div>

              {/* Sizing Objective */}
              <div className="flex flex-col gap-1.5 border-t border-slate-900 pt-3">
                <span className="text-[10px] font-bold text-slate-400">Sizing Objective</span>
                <select
                  value={sizingObjective}
                  onChange={(e) => setSizingObjective(e.target.value as SizingObjective)}
                  className="w-full bg-slate-900 border border-slate-800 text-slate-200 rounded-lg p-2 text-xs font-mono focus:ring-0 focus:outline-none focus:border-indigo-500 transition"
                >
                  <option value="min_mass">Minimize Total Mass</option>
                  <option value="min_power">Minimize Motor Rated Power</option>
                  <option value="min_gearbox">Minimize Gearbox Size</option>
                  <option value="max_margin">Maximize Safety Margin</option>
                </select>
              </div>

              {/* Reducer Type Filter */}
              <div className="flex flex-col gap-1.5 border-t border-slate-900 pt-3">
                <span className="text-[10px] font-bold text-slate-400">Reducer Type Filter</span>
                <div className="grid grid-cols-3 gap-2 bg-slate-900 border border-slate-850 p-0.5 rounded-lg">
                  <button
                    onClick={() => setGearboxType('any')}
                    className={`text-[9px] uppercase font-bold py-1.5 rounded transition ${
                      gearboxType === 'any' ? 'bg-slate-800 text-slate-200' : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    Any
                  </button>
                  <button
                    onClick={() => setGearboxType('harmonic')}
                    className={`text-[9px] uppercase font-bold py-1.5 rounded transition ${
                      gearboxType === 'harmonic' ? 'bg-slate-800 text-slate-200' : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    Strain wave
                  </button>
                  <button
                    onClick={() => setGearboxType('cycloidal')}
                    className={`text-[9px] uppercase font-bold py-1.5 rounded transition ${
                      gearboxType === 'cycloidal' ? 'bg-slate-800 text-slate-200' : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    Cycloidal
                  </button>
                </div>
              </div>

              {/* Advanced Parameters Toggler */}
              <div className="border-t border-slate-900 pt-2 flex flex-col gap-2.5">
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="text-[10px] font-bold text-indigo-400 hover:text-indigo-300 transition flex items-center justify-between"
                >
                  <span>{showAdvanced ? 'Hide' : 'Show'} Advanced Settings</span>
                  <span className="font-mono text-[9px] px-1 bg-slate-900 border border-slate-800 rounded">
                    Peak & Power Policies
                  </span>
                </button>

                {showAdvanced && (
                  <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-900 flex flex-col gap-3">
                    <div className="flex flex-col gap-1">
                      <div className="flex justify-between text-[10px] font-bold">
                        <span className="text-slate-400">Motor Peak Factor</span>
                        <span className="text-indigo-400">{motorPeakFactor.toFixed(1)}x</span>
                      </div>
                      <input
                        type="range"
                        min={2.0}
                        max={10.0}
                        step={0.5}
                        value={motorPeakFactor}
                        onChange={(e) => setMotorPeakFactor(parseFloat(e.target.value))}
                        className="accent-indigo-500 h-1 bg-slate-800 rounded appearance-none cursor-pointer"
                      />
                      <span className="text-[9px] text-slate-500 leading-normal mt-0.5">
                        Establishes short peak transient torque limit: T_peak = factor * T_rated.
                      </span>
                    </div>

                    <label className="flex items-start gap-2 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={enforcePowerLimit}
                        onChange={(e) => setEnforcePowerLimit(e.target.checked)}
                        className="mt-0.5 accent-indigo-500 rounded border-slate-800"
                      />
                      <div className="flex flex-col">
                        <span className="text-[10px] font-bold text-slate-350">Enforce power limits strictly</span>
                        <span className="text-[9px] text-slate-500 leading-normal mt-0.5">
                          When checked, candidates failing the power limit margin will be discarded instead of generating warnings.
                        </span>
                      </div>
                    </label>
                  </div>
                )}
              </div>

              {!torqueLog && (
                <div className="flex items-start gap-2 bg-amber-500/10 border border-amber-500/20 text-amber-400 p-2.5 rounded text-[10px] leading-relaxed">
                  <Info size={14} className="shrink-0 mt-0.5" />
                  <span>
                    Sizing requires a calculated torque log. First go to the <strong>Program</strong> tab and click <strong>Run Simulation</strong>.
                  </span>
                </div>
              )}
            </div>

            {/* Sizing Results Report */}
            {torqueLog && sizingResults && (
              <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
                <div className="flex justify-between items-center border-b border-slate-850 pb-2">
                  <div className="flex flex-col">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">Optimal selections</h3>
                    <span className="text-[9px] text-slate-500">
                      Based on {torqueLog.samples.length} trajectory samples ({sizingResults.dynamics_source})
                    </span>
                  </div>
                  <div className={`flex items-center gap-1 text-[10px] font-bold uppercase px-2 py-0.5 rounded border ${
                    sizingResults.complete
                      ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
                      : 'text-red-400 bg-red-500/10 border-red-500/20'
                  }`}>
                    {sizingResults.complete ? (
                      <>
                        <CheckCircle size={10} />
                        <span>Sized successfully</span>
                      </>
                    ) : (
                      <>
                        <AlertTriangle size={10} />
                        <span>Incomplete</span>
                      </>
                    )}
                  </div>
                </div>

                <div className="flex flex-col gap-2.5">
                  {sizingResults.joints.map((joint, idx) => {
                    const isExpanded = expandedJointIdx === idx;
                    const best = joint.best;
                    const demand = joint.demand;

                    return (
                      <div
                        key={joint.demand.joint_name}
                        className={`border rounded-lg overflow-hidden bg-slate-950/20 transition ${
                          isExpanded ? 'border-indigo-500/40' : 'border-slate-850'
                        }`}
                      >
                        {/* Summary Header */}
                        <div
                          onClick={() => toggleExpandJoint(idx)}
                          className="p-3 flex justify-between items-center hover:bg-slate-900/30 cursor-pointer transition select-none"
                        >
                          <div className="flex flex-col gap-0.5">
                            <span className="font-mono text-xs font-bold text-indigo-400">{demand.joint_name}</span>
                            <span className="text-[10px] text-slate-500 font-mono">
                              RMS: {demand.tau_rms_Nm.toFixed(1)} Nm | Peak: {demand.tau_peak_Nm.toFixed(1)} Nm
                            </span>
                          </div>

                          <div className="flex items-center gap-3">
                            {best ? (
                              <div className="flex flex-col items-end text-right">
                                <span className="text-[11px] font-semibold text-slate-250">
                                  {best.motor.name}
                                </span>
                                <span className="text-[9px] text-slate-500 font-mono">
                                  {best.gearbox_type === 'harmonic' ? 'Strain wave' : 'Cycloidal'} ({best.ratio}:1) | {best.total_mass_kg.toFixed(2)} kg
                                </span>
                              </div>
                            ) : (
                              <span className="text-[10px] font-bold text-red-400 bg-red-500/10 px-2 py-0.5 rounded border border-red-500/20 flex items-center gap-1">
                                <AlertTriangle size={10} />
                                <span>No Candidates</span>
                              </span>
                            )}
                            {isExpanded ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                          </div>
                        </div>

                        {/* Detailed Dropdown Options */}
                        {isExpanded && (
                          <div className="p-3 bg-slate-950/50 border-t border-slate-850 flex flex-col gap-3">
                            
                            {/* Demand specs details */}
                            <div className="grid grid-cols-2 gap-2 text-[10px] bg-slate-900/60 p-2.5 rounded-lg border border-slate-850 leading-relaxed text-slate-400">
                              <div className="flex flex-col gap-0.5">
                                <span>Cycle Time: <strong className="text-slate-200 font-mono">{demand.cycle_time_s.toFixed(2)} s</strong></span>
                                <span>RMS Torque: <strong className="text-slate-200 font-mono">{demand.tau_rms_Nm.toFixed(2)} Nm</strong></span>
                                <span>Peak Torque: <strong className="text-slate-200 font-mono">{demand.tau_peak_Nm.toFixed(2)} Nm</strong></span>
                              </div>
                              <div className="flex flex-col gap-0.5">
                                <span>Max Speed: <strong className="text-slate-200 font-mono">{(demand.speed_peak_rad_s * 60 / (2*Math.PI)).toFixed(0)} RPM</strong></span>
                                <span>RMS Power: <strong className="text-slate-200 font-mono">{demand.power_rms_W.toFixed(1)} W</strong></span>
                                <span>Regen Peak: <strong className="text-slate-200 font-mono">{demand.regen_peak_W.toFixed(1)} W</strong></span>
                              </div>
                            </div>

                            {/* Chosen Candidate Details */}
                            {best ? (
                              <div className="flex flex-col gap-3">
                                <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Optimal Selection Margins:</div>
                                <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[10px] text-slate-400">
                                  <div className="flex justify-between border-b border-slate-900 pb-0.5">
                                    <span>Continuous Output Torque:</span>
                                    <span className="font-bold text-emerald-400">{(best.motor.rated_torque_Nm * best.gearbox.ratio * best.gearbox.efficiency).toFixed(1)} Nm ({best.continuous_margin.toFixed(2)}x)</span>
                                  </div>
                                  <div className="flex justify-between border-b border-slate-900 pb-0.5">
                                    <span>Peak Output Torque (5x):</span>
                                    <span className="font-bold text-emerald-400">{(best.motor.rated_torque_Nm * motorPeakFactor * best.gearbox.ratio * best.gearbox.efficiency).toFixed(1)} Nm ({best.peak_margin.toFixed(2)}x)</span>
                                  </div>
                                  <div className="flex justify-between border-b border-slate-900 pb-0.5">
                                    <span>Output Speed limit:</span>
                                    <span className="font-bold text-emerald-400">{((best.motor.no_load_speed_rpm / best.gearbox.ratio)).toFixed(0)} RPM ({best.speed_margin.toFixed(2)}x)</span>
                                  </div>
                                  <div className="flex justify-between border-b border-slate-900 pb-0.5">
                                    <span>Gearbox Max Continuous:</span>
                                    <span className="font-bold text-emerald-400">{best.gearbox.max_continuous_torque_Nm} Nm ({best.gearbox_continuous_margin.toFixed(2)}x)</span>
                                  </div>
                                  <div className="flex justify-between border-b border-slate-900 pb-0.5">
                                    <span>Gearbox Max Intermittent:</span>
                                    <span className="font-bold text-emerald-400">{best.gearbox.max_intermittent_torque_Nm} Nm ({best.gearbox_peak_margin.toFixed(2)}x)</span>
                                  </div>
                                  <div className="flex justify-between border-b border-slate-900 pb-0.5">
                                    <span>Rated Motor Power:</span>
                                    <span className={`font-bold ${best.motor.rated_power_W < (demand.power_rms_W / best.gearbox.efficiency) * marginPower ? 'text-amber-400' : 'text-slate-300'}`}>
                                      {best.motor.rated_power_W} W ({best.power_margin === Infinity ? 'n/a' : `${best.power_margin.toFixed(2)}x`})
                                    </span>
                                  </div>
                                </div>

                                {/* Power check warning */}
                                {best.motor.rated_power_W < (demand.power_rms_W / best.gearbox.efficiency) * marginPower && (
                                  <div className="flex items-start gap-1 text-[9px] text-amber-400 bg-amber-500/5 border border-amber-500/10 p-2 rounded">
                                    <AlertTriangle size={12} className="shrink-0 mt-0.5" />
                                    <span>
                                      Warning: Motor rated power ({best.motor.rated_power_W}W) is below the RMS demand limit ({(demand.power_rms_W / best.gearbox.efficiency).toFixed(0)}W including efficiency and margins). Candidate passed since power limit enforcement is disabled.
                                    </span>
                                  </div>
                                )}

                                {/* Table of Alternative Candidates */}
                                {joint.candidates.length > 1 && (
                                  <div className="mt-1 flex flex-col gap-1.5">
                                    <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Alternative Passing Options:</div>
                                    <div className="overflow-x-auto border border-slate-900 rounded-md bg-slate-950/20 max-h-[120px] overflow-y-auto">
                                      <table className="w-full text-[10px] text-left text-slate-400">
                                        <thead>
                                          <tr className="bg-slate-900 text-slate-500 font-bold uppercase border-b border-slate-850">
                                            <th className="p-1.5">Motor</th>
                                            <th className="p-1.5">Gearbox</th>
                                            <th className="p-1.5">Type</th>
                                            <th className="p-1.5">Ratio</th>
                                            <th className="p-1.5">Mass</th>
                                            <th className="p-1.5">Min Margin</th>
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {joint.candidates.filter(c => c.passes && c.motor_id !== best.motor_id).slice(0, 3).map((cand, cIdx) => (
                                            <tr key={cIdx} className="border-b border-slate-900/50 hover:bg-slate-900/10">
                                              <td className="p-1.5 font-mono text-slate-300">{cand.motor.name}</td>
                                              <td className="p-1.5 font-mono text-slate-400">{cand.gearbox.name}</td>
                                              <td className="p-1.5">{cand.gearbox_type === 'harmonic' ? 'Strain wave' : 'Cycloidal'}</td>
                                              <td className="p-1.5 font-mono">{cand.gearbox.ratio}:1</td>
                                              <td className="p-1.5 font-mono">{cand.total_mass_kg.toFixed(2)} kg</td>
                                              <td className="p-1.5 font-mono text-emerald-400">{cand.min_margin === Infinity ? 'n/a' : `${cand.min_margin.toFixed(2)}x`}</td>
                                            </tr>
                                          ))}
                                          {joint.candidates.filter(c => c.passes && c.motor_id !== best.motor_id).length === 0 && (
                                            <tr>
                                              <td colSpan={6} className="p-2 text-center text-slate-500">No other passing options found.</td>
                                            </tr>
                                          )}
                                        </tbody>
                                      </table>
                                    </div>
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div className="flex flex-col gap-2">
                                <div className="text-[10px] text-red-400 bg-red-500/5 border border-red-500/10 p-2 rounded">
                                  No compatible motor/gearbox combinations found in the library that satisfy the selected safety margins.
                                </div>
                                <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Failed Candidates Diagnostics:</div>
                                <div className="overflow-x-auto border border-slate-900 rounded-md bg-slate-950/20 max-h-[140px] overflow-y-auto">
                                  <table className="w-full text-[10px] text-left text-slate-400">
                                    <thead>
                                      <tr className="bg-slate-900 text-slate-500 font-bold uppercase border-b border-slate-850">
                                        <th className="p-1.5">Motor</th>
                                        <th className="p-1.5">Gearbox</th>
                                        <th className="p-1.5">Critical Limiting Criterias</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {joint.candidates.slice(0, 3).map((cand, cIdx) => (
                                        <tr key={cIdx} className="border-b border-slate-900/50 hover:bg-slate-900/10">
                                          <td className="p-1.5 font-mono text-slate-300 shrink-0 max-w-[80px] truncate">{cand.motor.name}</td>
                                          <td className="p-1.5 font-mono text-slate-400 shrink-0 max-w-[80px] truncate">{cand.gearbox.name}</td>
                                          <td className="p-1.5 text-red-400/90 leading-tight">
                                            {cand.failure_reasons.map((r, rIdx) => (
                                              <div key={rIdx} className="flex items-start gap-0.5 my-0.5">
                                                <X size={8} className="shrink-0 mt-1" />
                                                <span>{r}</span>
                                              </div>
                                            ))}
                                          </td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                <div className="flex flex-col gap-2 mt-3 pt-3 border-t border-slate-850">
                  {/* Export Button */}
                  <button
                    onClick={handleExportReport}
                    className="w-full flex items-center justify-center gap-2 py-2 rounded-lg text-[10px] font-bold uppercase tracking-wider bg-slate-800 text-slate-200 hover:bg-slate-700 transition border border-slate-700"
                  >
                    <Download size={12} />
                    <span>Export Sizing Report (JSON)</span>
                  </button>

                  <p className="text-[9px] text-slate-500 leading-normal mt-2">
                    <strong>Sizing Policy:</strong> Motor peak torque is estimated as {motorPeakFactor.toFixed(1)}x rated torque for short robotic transients. Gearbox peak is limited by max_intermittent_torque_Nm. Robot inertial parameters are not modified by selected actuators.
                  </p>
                </div>
              </div>
            )}

            {/* Cycle Power & Energy Diagnostics */}
            {torqueLog && diagnostics && (
              <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-350 flex items-center gap-1.5">
                  <Zap size={14} className="text-amber-400 animate-pulse" />
                  <span>Cycle Power & Energy Diagnostics</span>
                </h3>
                <p className="text-[10px] text-slate-400 leading-relaxed">
                  Summary of positive mechanical power and cycle energy drawn by all joints combined:
                </p>

                <div className="grid grid-cols-3 gap-2 mt-1">
                  <div className="bg-slate-900/50 border border-slate-850 p-2.5 rounded-lg flex flex-col justify-between">
                    <span className="text-[9px] font-bold text-slate-500 uppercase">Cycle Duration</span>
                    <span className="text-xs font-bold text-slate-200 mt-1 font-mono">{diagnostics.cycleTimeS.toFixed(3)} s</span>
                  </div>
                  <div className="bg-slate-900/50 border border-slate-850 p-2.5 rounded-lg flex flex-col justify-between">
                    <span className="text-[9px] font-bold text-slate-500 uppercase">Mean Pos. Power</span>
                    <span className="text-xs font-bold text-amber-400 mt-1 font-mono">
                      {diagnostics.positivePowerMeanTotal.toFixed(1)} W <span className="text-[9px] text-slate-500">({(diagnostics.positivePowerMeanTotal / 1000).toFixed(3)} kW)</span>
                    </span>
                  </div>
                  <div className="bg-slate-900/50 border border-slate-850 p-2.5 rounded-lg flex flex-col justify-between">
                    <span className="text-[9px] font-bold text-slate-500 uppercase">Peak Pos. Power</span>
                    <span className="text-xs font-bold text-amber-400 mt-1 font-mono">
                      {diagnostics.positivePowerPeakTotal.toFixed(1)} W <span className="text-[9px] text-slate-500">({(diagnostics.positivePowerPeakTotal / 1000).toFixed(3)} kW)</span>
                    </span>
                  </div>
                </div>

                <div className="bg-slate-900/30 border border-slate-850 p-3 rounded-lg flex justify-between items-center text-[10px]">
                  <span className="font-bold text-slate-400 uppercase tracking-wide">Energy Consumed per Cycle:</span>
                  <span className="font-mono font-bold text-indigo-400 text-xs">
                    {diagnostics.energyPerCycleJ.toFixed(0)} J ({diagnostics.energyPerCycleWh.toFixed(4)} Wh)
                  </span>
                </div>

                {/* Joint-by-joint dynamics detail */}
                <div className="mt-1 flex flex-col gap-1.5">
                  <span className="text-[9px] font-bold uppercase tracking-wider text-slate-500">Joint Peak Performance & Dynamics:</span>
                  <div className="overflow-x-auto border border-slate-850 rounded-lg bg-slate-950/20">
                    <table className="w-full text-[9px] text-left text-slate-400">
                      <thead>
                        <tr className="bg-slate-900/80 text-slate-500 font-bold uppercase border-b border-slate-850">
                          <th className="p-1.5">Joint</th>
                          <th className="p-1.5">Peak Torque</th>
                          <th className="p-1.5">Peak Speed</th>
                          <th className="p-1.5">Peak Accel</th>
                          <th className="p-1.5">Mean Power</th>
                          <th className="p-1.5">Peak Power</th>
                        </tr>
                      </thead>
                      <tbody>
                        {diagnostics.jointStats.map((jStat) => (
                          <tr key={jStat.joint_name} className="border-b border-slate-900/50 hover:bg-slate-900/10">
                            <td className="p-1.5 font-bold font-mono text-indigo-400">{jStat.joint_name}</td>
                            <td className="p-1.5 font-mono text-slate-300">{jStat.tau_peak.toFixed(1)} Nm</td>
                            <td className="p-1.5 font-mono text-slate-300">{(jStat.speed_peak * 180 / Math.PI).toFixed(0)}°/s</td>
                            <td className="p-1.5 font-mono text-slate-300">{(jStat.accel_peak * 180 / Math.PI).toFixed(0)}°/s²</td>
                            <td className="p-1.5 font-mono text-amber-500/90">{jStat.positive_power_mean.toFixed(1)} W</td>
                            <td className="p-1.5 font-mono text-amber-500">{jStat.positive_power_peak.toFixed(1)} W</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="text-[9px] text-slate-500 leading-normal border-t border-slate-900 pt-2 flex flex-col gap-0.5 mt-1">
                  <span>* <strong>Motor rated power</strong> is a sizing class, not expected average cycle consumption.</span>
                  <span>* <strong>Cycle energy</strong> and <strong>mean positive mechanical power</strong> estimate the motion energy demand.</span>
                </div>
              </div>
            )}


            {/* Reducer Technology Sizing Guide */}
            <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-350 flex items-center gap-1.5">
                <Info size={14} className="text-indigo-400" />
                <span>Strain Wave vs Cycloidal Reducers</span>
              </h3>
              <p className="text-[10px] text-slate-400 leading-relaxed">
                Choosing the correct technology depends on the joint placement and torque requirements:
              </p>
              <div className="grid grid-cols-2 gap-3 mt-1">
                <div className="bg-slate-900/40 border border-slate-850 p-2.5 rounded-lg flex flex-col gap-1">
                  <span className="text-[10px] font-bold text-slate-200">Strain Wave (Harmonic)</span>
                  <ul className="list-disc list-inside text-[9px] text-slate-500 leading-normal flex flex-col gap-0.5">
                    <li>Single-stage strain wave, zero backlash.</li>
                    <li>Compact size, extremely light weight.</li>
                    <li>High precision, ideal for wrist/distal joints.</li>
                  </ul>
                </div>
                <div className="bg-slate-900/40 border border-slate-850 p-2.5 rounded-lg flex flex-col gap-1">
                  <span className="text-[10px] font-bold text-slate-200">Cycloidal Reducers</span>
                  <ul className="list-disc list-inside text-[9px] text-slate-500 leading-normal flex flex-col gap-0.5">
                    <li>Two-stage cycloidal + planetary, ultra-low backlash.</li>
                    <li>Highly robust, handles massive shock overloads.</li>
                    <li>Heavier, ideal for base/proximal axis.</li>
                  </ul>
                </div>
              </div>
            </div>
          </>
        )}

        {/* VIEW 2: MOTORS DATABASE */}
        {subTab === 'motors' && (
          <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Database size={14} className="text-indigo-400" />
              <span>Motors Available in Library</span>
            </h3>

            <div className="overflow-x-auto border border-slate-800 rounded bg-slate-950/20">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-slate-850 text-slate-500 font-bold text-[10px] uppercase bg-slate-950/30">
                    <th className="p-2">Model</th>
                    <th className="p-2">Type</th>
                    <th className="p-2 text-right">Power (W)</th>
                    <th className="p-2 text-right">Rated Torque (Nm)</th>
                    <th className="p-2 text-right">Rated Speed (RPM)</th>
                    <th className="p-2 text-right">Mass (kg)</th>
                  </tr>
                </thead>
                <tbody>
                  {actuatorLibrary?.motors.map(motor => (
                    <tr key={motor.id} className="border-b border-slate-900/50 hover:bg-slate-900/15">
                      <td className="p-2 font-mono font-bold text-indigo-400">{motor.name}</td>
                      <td className="p-2 text-slate-400">{motor.series}</td>
                      <td className="p-2 text-right font-mono text-slate-350">{motor.rated_power_W} W</td>
                      <td className="p-2 text-right font-mono text-slate-300">{motor.rated_torque_Nm.toFixed(2)} Nm</td>
                      <td className="p-2 text-right font-mono text-slate-400">{motor.rated_speed_rpm}</td>
                      <td className="p-2 text-right font-mono text-slate-300">{motor.mass_kg.toFixed(2)} kg</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* VIEW 3: GEARBOXES DATABASE */}
        {subTab === 'gearboxes' && (
          <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Database size={14} className="text-indigo-400" />
              <span>Gearboxes Available in Library</span>
            </h3>

            <div className="overflow-x-auto border border-slate-800 rounded bg-slate-950/20">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-slate-850 text-slate-500 font-bold text-[10px] uppercase bg-slate-950/30">
                    <th className="p-2">Model</th>
                    <th className="p-2">Type</th>
                    <th className="p-2 text-right">Ratio (i)</th>
                    <th className="p-2 text-right">Max Torque (Nm)</th>
                    <th className="p-2 text-right">Efficiency</th>
                    <th className="p-2 text-right">Mass (kg)</th>
                  </tr>
                </thead>
                <tbody>
                  {actuatorLibrary?.gearboxes.map(g => (
                    <tr key={g.id} className="border-b border-slate-900/50 hover:bg-slate-900/15">
                      <td className="p-2 font-mono font-bold text-amber-500">{g.name}</td>
                      <td className="p-2 text-slate-400">{g.type === 'harmonic' ? 'Strain wave' : 'Cycloidal'}</td>
                      <td className="p-2 text-right font-mono text-slate-350">{g.ratio}:1</td>
                      <td className="p-2 text-right font-mono text-slate-300">{g.max_continuous_torque_Nm} Nm</td>
                      <td className="p-2 text-right font-mono text-emerald-400">{(g.efficiency * 100).toFixed(0)}%</td>
                      <td className="p-2 text-right font-mono text-slate-300">{g.mass_kg.toFixed(2)} kg</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};
