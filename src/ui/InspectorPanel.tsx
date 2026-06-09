import React, { useRef, useCallback } from 'react';
import { useRobodimmStore } from '../model/state';
import { Eye, EyeOff, Info, Layers, Crosshair, Box, ChevronDown, ChevronRight, Download, Server, AlertTriangle, X, CheckCircle, MapPin, Trash2, Upload, RotateCcw } from 'lucide-react';
import { Serial6Engine } from '../math/serial6';
import { PalletizerEngine } from '../math/palletizer';
import { getTranslation } from '../math/matrix';
import { getDefaultPalletizerBodyCOM } from '../math/palletizerGeometry';
import { Cr4Geometry } from '../model/schemas';

function formatInertiaValue(val: number): string {
  if (val === 0) return '0';
  const abs = Math.abs(val);
  if (abs < 1e-4 || abs >= 1e4) {
    return val.toExponential(3);
  }
  return val.toFixed(5);
}

export const InspectorPanel: React.FC = () => {
  const {
    activeRobot,
    isSet,
    q,
    activeEngine,
    backendState,
    backendVersion,
    backendUrl,
    pinocchioVersion,
    licenseStatus,
    setEngine,
    toggleSolidVisibility,
    toggleAxesVisibility,
    showGrid,
    showAxes,
    showCOMs,
    showTrajectory,
    showTCPFrame,
    toggleGrid,
    toggleAxes,
    toggleCOMs,
    toggleTrajectory,
    toggleTCPFrame,
    stationObjects,
    addStationObject,
    updateStationObject,
    removeStationObject,
    toggleStationObjectVisible
  } = useRobodimmStore();

  const [expandedBody, setExpandedBody] = React.useState<string | null>(null);
  const [expandedStation, setExpandedStation] = React.useState<string | null>(null);
  const [showWalkthrough, setShowWalkthrough] = React.useState(false);
  const glbInputRef = useRef<HTMLInputElement>(null);

  // Object URL registry for station GLBs (keyed by id)
  const stationUrlsRef = useRef<Map<string, string>>(new Map());

  const handleGlbUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Validate extension
    if (!file.name.toLowerCase().endsWith('.glb')) {
      alert('Please select a .glb file.');
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    const id = `station_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    stationUrlsRef.current.set(id, objectUrl);
    addStationObject({
      id,
      name: file.name.replace(/\.glb$/i, ''),
      meshUrl: objectUrl,
      positionM: [0, 0, 0],
      rotationRpyRad: [0, 0, 0],
      scale: [1, 1, 1],
      visible: true
    });
    // Reset so same file can be re-uploaded
    e.target.value = '';
  }, [addStationObject]);

  const getInertialData = (bodyName: string) => {
    let data = {
      massKg: 0,
      comM: [0, 0, 0] as [number, number, number],
      inertiaKgM2: [[0, 0, 0], [0, 0, 0], [0, 0, 0]] as [number, number, number][],
      frame: 'cad' as 'cad' | 'link' | 'tcp'
    };

    if (bodyName === 'PAYLOAD') {
      const p = activeRobot.payload;
      data = {
        massKg: p?.massKg ?? 0,
        comM: p?.comM ?? [0, 0, 0],
        inertiaKgM2: p?.inertiaKgM2 ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        frame: (p?.frame ?? 'link') as 'cad' | 'link' | 'tcp'
      };
    } else {
      const inertial = activeRobot.inertials?.[bodyName];
      if (inertial) {
        data = {
          massKg: inertial.massKg,
          comM: (inertial.comM ?? (activeRobot.kind === 'CR4' ? getDefaultPalletizerBodyCOM(activeRobot.geometry as Cr4Geometry, bodyName) : [0, 0, 0])) as [number, number, number],
          inertiaKgM2: inertial.inertiaKgM2 ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
          frame: (inertial.frame ?? 'cad') as 'cad' | 'link' | 'tcp'
        };
      } else if (activeRobot.kind === 'CR4') {
        let massKg = 0;
        switch (bodyName) {
          case 'SWING': massKg = 90.0; break;
          case 'P_ARM': massKg = 35.0; break;
          case 'LOWER_ARM': massKg = 75.0; break;
          case 'P_LINK': massKg = 25.0; break;
          case 'UPPER_ARM': massKg = 40.0; break;
          case 'LOWER_LINK': massKg = 20.0; break;
          case 'LINK_PLATE': massKg = 15.0; break;
          case 'UPPER_LINK': massKg = 15.0; break;
          case 'TILT': massKg = 15.0; break;
          case 'DISK': massKg = 10.0; break;
          default: massKg = 0.0;
        }
        data = {
          massKg,
          comM: getDefaultPalletizerBodyCOM(activeRobot.geometry as Cr4Geometry, bodyName),
          inertiaKgM2: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
          frame: 'cad'
        };
      }
    }

    // Convert link frame to cad frame for CR6 if needed
    if (data.frame === 'link' && activeRobot.kind === 'CR6') {
      let linkIdx = -1;
      if (bodyName.startsWith('LINK')) {
        linkIdx = parseInt(bodyName.replace('LINK', '')) - 1;
      } else if (bodyName === 'PAYLOAD') {
        linkIdx = 5; // payload is attached to LINK6
      }

      if (linkIdx >= 0 && linkIdx < 6) {
        const eng = new Serial6Engine(activeRobot);
        const cadSpec = eng._linkInertialToCadInertial(linkIdx, {
          body: bodyName,
          massKg: data.massKg,
          comM: data.comM,
          inertiaKgM2: data.inertiaKgM2 as any,
          frame: 'link'
        });
        data = {
          massKg: cadSpec.massKg,
          comM: cadSpec.comM ?? [0, 0, 0],
          inertiaKgM2: cadSpec.inertiaKgM2 ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
          frame: 'cad'
        };
      }
    }

    return data;
  };

  const renderInertialSubDrawer = (bodyName: string) => {
    const data = getInertialData(bodyName);
    return (
      <div className="px-3 pb-3 pt-1.5 bg-slate-950/40 border-t border-slate-900 flex flex-col gap-2.5">
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div>
            <span className="text-[9px] text-slate-500 uppercase font-semibold tracking-wider block">Mass</span>
            <span className="font-mono text-slate-300 font-semibold">{data.massKg.toFixed(3)} kg</span>
          </div>
          <div>
            <span className="text-[9px] text-slate-500 uppercase font-semibold tracking-wider block">Frame</span>
            <span className="font-mono text-slate-400 uppercase font-semibold text-[10px]">{data.frame}</span>
          </div>
        </div>

        <div>
          <span className="text-[9px] text-slate-500 uppercase font-semibold tracking-wider block mb-0.5">Center of Mass</span>
          <div className="grid grid-cols-3 gap-1 font-mono text-[10px]">
            <div className="bg-slate-900/60 px-1.5 py-0.5 rounded flex justify-between">
              <span className="text-slate-500 font-bold">X</span>
              <span className="text-slate-300">{data.comM[0].toFixed(4)}</span>
            </div>
            <div className="bg-slate-900/60 px-1.5 py-0.5 rounded flex justify-between">
              <span className="text-slate-500 font-bold">Y</span>
              <span className="text-slate-300">{data.comM[1].toFixed(4)}</span>
            </div>
            <div className="bg-slate-900/60 px-1.5 py-0.5 rounded flex justify-between">
              <span className="text-slate-500 font-bold">Z</span>
              <span className="text-slate-300">{data.comM[2].toFixed(4)}</span>
            </div>
          </div>
        </div>

        <div>
          <span className="text-[9px] text-slate-500 uppercase font-semibold tracking-wider block mb-1">Inertia Tensor (kg·m²)</span>
          <div className="grid grid-cols-3 gap-1 bg-slate-950/80 p-1.5 rounded border border-slate-800/60 font-mono text-[9px]">
            {data.inertiaKgM2.map((row, rIdx) => 
              row.map((val, cIdx) => {
                const axesNames = ['x', 'y', 'z'];
                const label = `I${axesNames[rIdx]}${axesNames[cIdx]}`;
                return (
                  <div key={`${rIdx}-${cIdx}`} className="bg-slate-900/40 p-1 rounded flex flex-col items-center">
                    <span className="text-[7px] text-slate-500 font-bold uppercase">{label}</span>
                    <span className="text-slate-300 font-medium">{formatInertiaValue(val)}</span>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    );
  };

  // Compute TCP position live for inspector display
  let tcpPosition = [0, 0, 0];
  if (isSet) {
    if (activeRobot.kind === 'CR6') {
      const eng = new Serial6Engine(activeRobot);
      const fk = eng.forwardKinematics(q);
      tcpPosition = getTranslation(fk.tcp_transform);
    } else {
      const eng = new PalletizerEngine(activeRobot);
      const fk = eng.forwardKinematics(q);
      tcpPosition = fk.points['TCP'];
    }
  }

  const solidRows = Array.from(
    activeRobot.visuals.reduce((groups, vis) => {
      const existing = groups.get(vis.body);
      if (existing) {
        existing.visible = existing.visible || vis.visible;
        existing.axesVisible = existing.axesVisible || vis.axesVisible !== false;
        existing.count += 1;
      } else {
        groups.set(vis.body, {
          body: vis.body,
          visible: vis.visible,
          axesVisible: vis.axesVisible !== false,
          count: 1
        });
      }
      return groups;
    }, new Map<string, { body: string; visible: boolean; axesVisible: boolean; count: number }>()).values()
  );

  return (
    <div className="w-80 h-full flex flex-col border-r border-slate-800 bg-slate-900/60 backdrop-blur overflow-y-auto relative">
      {/* Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Layers size={18} className="text-indigo-400" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200">CAD Inspector</h2>
        </div>
      </div>

      {/* Info Card */}
      <div className="p-4">
        <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-3 flex flex-col gap-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-400">
            <Info size={14} className="text-indigo-400" />
            <span>System Summary</span>
          </div>
          <div className="flex justify-between text-xs border-b border-slate-900 pb-1.5 mt-1">
            <span className="text-slate-400">Robot Family</span>
            <span className="font-semibold text-slate-200">{activeRobot.kind}</span>
          </div>
          <div className="flex justify-between text-xs border-b border-slate-900 pb-1.5">
            <span className="text-slate-400">Robot Name</span>
            <span className="font-semibold text-slate-200 max-w-[150px] truncate">{activeRobot.name}</span>
          </div>
          <div className="flex justify-between text-xs border-b border-slate-900 pb-1.5">
            <span className="text-slate-400">Degrees of Freedom</span>
            <span className="font-semibold text-slate-200">{activeRobot.limits.length} DoF</span>
          </div>
          
          {/* Active Engine Configuration */}
          <div className="flex flex-col gap-2 mt-2 pt-2 border-t border-slate-900">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400 font-semibold">Active Engine</span>
              {backendState === 'connected' ? (
                <select
                  value={activeEngine}
                  onChange={(e) => setEngine(e.target.value as 'frontend' | 'backend')}
                  className="bg-slate-900 border border-slate-800 text-slate-200 text-xs rounded px-2 py-1 focus:outline-none focus:border-indigo-500 font-semibold"
                >
                  <option value="frontend">DEMO (Frontend)</option>
                  <option value="backend">PRO (Local Python)</option>
                </select>
              ) : (
                <span className="text-[11px] font-semibold text-slate-500 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                  DEMO (Frontend)
                </span>
              )}
            </div>

            {/* Backend Status Alerts */}
            {backendState === 'connected' ? (
              <div className="bg-emerald-500/5 border border-emerald-500/15 rounded p-2 text-[10px] text-emerald-400 flex flex-col gap-1 mt-1">
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-1 font-semibold">
                    <CheckCircle size={10} />
                    <span>PRO Connected</span>
                  </span>
                  <span className="font-mono text-slate-500">v{backendVersion}</span>
                </div>
                {pinocchioVersion && (
                  <div className="flex justify-between border-t border-emerald-500/10 pt-1 mt-1 font-mono text-[9px] text-slate-500">
                    <span>Pinocchio:</span>
                    <span className="text-slate-400">{pinocchioVersion}</span>
                  </div>
                )}
                {licenseStatus && (
                  <div className="flex justify-between font-mono text-[9px] text-slate-500">
                    <span>License Status:</span>
                    <span className="text-slate-400">{licenseStatus}</span>
                  </div>
                )}
              </div>
            ) : backendState === 'incompatible' ? (
              <div className="bg-amber-950/20 border border-amber-900/40 rounded p-2 text-[10px] text-amber-400 flex flex-col gap-1.5 mt-1">
                <div className="flex items-center gap-1.5 font-semibold">
                  <AlertTriangle size={12} className="text-amber-500 animate-pulse" />
                  <span>PRO Backend Incompatible</span>
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Version mismatch or missing dynamic solver capabilities. Please download the latest package.
                </p>
                <button
                  onClick={() => setShowWalkthrough(true)}
                  className="bg-amber-600 hover:bg-amber-500 text-white font-bold py-1 px-2 rounded text-[10px] transition self-start"
                >
                  View Update Guide
                </button>
              </div>
            ) : (
              <div className="bg-slate-950/40 border border-slate-900 rounded p-2 text-[10px] text-slate-400 flex flex-col gap-1.5 mt-1">
                <div className="flex items-center gap-1.5 font-semibold text-slate-400">
                  <Server size={12} className="text-slate-500" />
                  <span>PRO Backend Offline</span>
                </div>
                <p className="text-slate-500 leading-relaxed">
                  Calculate full KKT dynamics locally without uploading geometries or trajectories.
                </p>
                <button
                  onClick={() => setShowWalkthrough(true)}
                  className="bg-indigo-600/90 hover:bg-indigo-600 text-white font-semibold py-1 px-2.5 rounded text-[10px] transition self-start shadow-md flex items-center gap-1"
                >
                  <Download size={10} />
                  <span>Install local backend</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* TCP Live Position */}
      {isSet && (
        <div className="px-4 pb-4">
          <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-3 flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-400">
              <Crosshair size={14} className="text-amber-500" />
              <span>TCP Coordinates (m)</span>
            </div>
            <div className="grid grid-cols-3 gap-2 mt-1">
              <div className="flex flex-col bg-slate-900 p-1.5 rounded text-center">
                <span className="text-[10px] text-slate-500 font-semibold">X</span>
                <span className="text-xs font-mono font-bold text-slate-200">{tcpPosition[0].toFixed(4)}</span>
              </div>
              <div className="flex flex-col bg-slate-900 p-1.5 rounded text-center">
                <span className="text-[10px] text-slate-500 font-semibold">Y</span>
                <span className="text-xs font-mono font-bold text-slate-200">{tcpPosition[1].toFixed(4)}</span>
              </div>
              <div className="flex flex-col bg-slate-900 p-1.5 rounded text-center">
                <span className="text-[10px] text-slate-500 font-semibold">Z</span>
                <span className="text-xs font-mono font-bold text-slate-200">{tcpPosition[2].toFixed(4)}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Joints Monitor */}
      {isSet && (
        <div className="px-4 pb-4">
          <div className="bg-slate-950/50 border border-slate-800 rounded-lg p-3 flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-400">
              <Crosshair size={14} className="text-indigo-400" />
              <span>Joints Monitor</span>
            </div>
            <div className="flex flex-col gap-1.5 mt-1">
              {activeRobot.limits.map((limit, idx) => {
                const valRad = q[idx] ?? 0;
                const valDeg = (valRad * 180) / Math.PI;
                const friction = limit.frictionCoeffNmSPerRad ?? 0.5;
                return (
                  <div
                    key={limit.name}
                    className="flex justify-between items-center text-[11px] border-b border-slate-900/40 pb-1.5 last:border-b-0 last:pb-0"
                  >
                    <div className="flex items-center gap-1.5 font-mono">
                      <span className="font-bold text-indigo-400">{limit.name}</span>
                      <span className="text-slate-300">{(valRad).toFixed(4)} rad</span>
                      <span className="text-slate-500">({(valDeg).toFixed(1)}°)</span>
                    </div>
                    <div className="text-right">
                      <span className="text-slate-500 font-mono text-[9px] mr-1">B:</span>
                      <span className="font-mono font-semibold text-slate-350">
                        {friction.toFixed(2)} N·m·s/rad
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Solids Visibility Tree */}
      <div className="flex-1 px-4 pb-4 flex flex-col gap-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 mb-1">
          <Box size={14} className="text-slate-400" />
          <span>CAD Solids</span>
        </div>
        <div className="flex flex-col border border-slate-800 rounded-lg overflow-hidden bg-slate-950/30">
          <div className="max-h-[350px] overflow-y-auto">
            {solidRows.map((solid) => {
              const isExpanded = expandedBody === solid.body;
              return (
                <div key={solid.body} className="border-b border-slate-900/50 last:border-b-0">
                  <div
                    onClick={() => setExpandedBody(isExpanded ? null : solid.body)}
                    className="flex items-center justify-between p-2.5 hover:bg-slate-900/20 text-xs cursor-pointer select-none"
                  >
                    <div className="flex items-center gap-1.5">
                      {isExpanded ? (
                        <ChevronDown size={14} className="text-slate-400" />
                      ) : (
                        <ChevronRight size={14} className="text-slate-400" />
                      )}
                      <div className={`w-2.5 h-2.5 rounded-full ${
                        activeRobot.kind === 'CR6' ? 'bg-red-500' : 'bg-orange-500'
                      }`} />
                      <span className="font-mono text-slate-300 font-medium">{solid.body}</span>
                      {solid.count > 1 && (
                        <span className="text-[9px] text-slate-500">{solid.count} parts</span>
                      )}
                    </div>
                    <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                      <button
                        onClick={() => toggleAxesVisibility(solid.body)}
                        className="p-1"
                        title="Toggle CAD Frame (Axes)"
                      >
                        {solid.axesVisible ? (
                          <Crosshair size={13} className="text-indigo-400 hover:text-white" />
                        ) : (
                          <Crosshair size={13} className="text-slate-600 hover:text-slate-400" />
                        )}
                      </button>
                      <button
                        onClick={() => toggleSolidVisibility(solid.body)}
                        className="text-slate-400 hover:text-white p-1"
                        title="Toggle Solid Mesh"
                      >
                        {solid.visible ? <Eye size={13} /> : <EyeOff size={13} className="text-slate-600" />}
                      </button>
                    </div>
                  </div>
                  {isExpanded && renderInertialSubDrawer(solid.body)}
                </div>
              );
            })}
            
            {activeRobot.payload && activeRobot.payload.massKg > 0 && (() => {
              const isExpanded = expandedBody === 'PAYLOAD';
              return (
                <div className="border-t border-slate-900/50">
                  <div
                    onClick={() => setExpandedBody(isExpanded ? null : 'PAYLOAD')}
                    className="flex items-center justify-between p-2.5 hover:bg-slate-900/20 text-xs cursor-pointer select-none"
                  >
                    <div className="flex items-center gap-1.5">
                      {isExpanded ? (
                        <ChevronDown size={14} className="text-slate-400" />
                      ) : (
                        <ChevronRight size={14} className="text-slate-400" />
                      )}
                      <div className="w-2.5 h-2.5 rounded-full bg-slate-400" />
                      <span className="font-mono text-slate-300 font-medium">PAYLOAD</span>
                    </div>
                    <span className="text-[10px] text-slate-500 italic pr-1 select-none">Fixed</span>
                  </div>
                  {isExpanded && renderInertialSubDrawer('PAYLOAD')}
                </div>
              );
            })()}
          </div>
        </div>
      </div>

      {/* Station Objects */}
      <div className="px-4 pb-4 border-t border-slate-800/80 pt-4 flex flex-col gap-2">
        {/* Hidden file input */}
        <input
          ref={glbInputRef}
          type="file"
          accept=".glb"
          className="hidden"
          onChange={handleGlbUpload}
        />

        {/* Section Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400">
            <MapPin size={14} className="text-emerald-400" />
            <span>Station Objects</span>
          </div>
          <button
            onClick={() => glbInputRef.current?.click()}
            className="flex items-center gap-1 bg-emerald-600/80 hover:bg-emerald-500 text-white text-[10px] font-bold px-2 py-1 rounded transition shadow-sm"
            title="Import a .glb CAD file as a static station element"
          >
            <Upload size={10} />
            <span>Add GLB</span>
          </button>
        </div>

        {stationObjects.length === 0 ? (
          <div className="text-[11px] text-slate-600 italic text-center py-3 bg-slate-950/20 border border-dashed border-slate-800 rounded-lg">
            No station elements — click <span className="font-semibold text-slate-500">Add GLB</span> to import environment objects
          </div>
        ) : (
          <div className="flex flex-col border border-slate-800 rounded-lg overflow-hidden bg-slate-950/30">
            {stationObjects.map((obj) => {
              const isExp = expandedStation === obj.id;
              const posM = obj.positionM ?? [0, 0, 0];
              const rotRad = obj.rotationRpyRad ?? [0, 0, 0];
              const rzDeg = (rotRad[2] * 180 / Math.PI);
              return (
                <div key={obj.id} className="border-b border-slate-900/60 last:border-b-0">
                  {/* Row header */}
                  <div
                    onClick={() => setExpandedStation(isExp ? null : obj.id)}
                    className="flex items-center justify-between p-2.5 hover:bg-slate-900/20 text-xs cursor-pointer select-none"
                  >
                    <div className="flex items-center gap-1.5 min-w-0">
                      {isExp ? <ChevronDown size={13} className="text-slate-400 flex-shrink-0" /> : <ChevronRight size={13} className="text-slate-400 flex-shrink-0" />}
                      <div className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0" />
                      <span className="font-mono text-slate-300 font-medium truncate max-w-[100px]" title={obj.name}>{obj.name}</span>
                    </div>
                    <div className="flex items-center gap-1 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
                      {/* Position summary */}
                      <span className="text-[9px] text-slate-600 font-mono hidden">
                        ({posM[0].toFixed(2)},{posM[1].toFixed(2)},{posM[2].toFixed(2)})
                      </span>
                      <button
                        onClick={() => toggleStationObjectVisible(obj.id)}
                        className="p-1 text-slate-400 hover:text-white"
                        title="Toggle visibility"
                      >
                        {obj.visible ? <Eye size={12} /> : <EyeOff size={12} className="text-slate-600" />}
                      </button>
                      <button
                        onClick={() => removeStationObject(obj.id)}
                        className="p-1 text-slate-600 hover:text-red-400 transition"
                        title="Remove station object"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>

                  {/* Expanded transform controls */}
                  {isExp && (
                    <div className="px-3 pb-3 pt-1 bg-slate-950/50 border-t border-slate-900 flex flex-col gap-2.5">
                      {/* Position */}
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase font-semibold tracking-wider block mb-1">Position (m)</span>
                        <div className="grid grid-cols-3 gap-1">
                          {(['X', 'Y', 'Z'] as const).map((axis, axIdx) => (
                            <div key={axis} className="flex flex-col">
                              <label className="text-[8px] text-slate-500 font-bold uppercase mb-0.5 text-center">{axis}</label>
                              <input
                                type="number"
                                step="0.01"
                                value={posM[axIdx].toFixed(3)}
                                onChange={(e) => {
                                  const newPos: [number, number, number] = [...posM] as [number, number, number];
                                  newPos[axIdx] = parseFloat(e.target.value) || 0;
                                  updateStationObject(obj.id, { positionM: newPos });
                                }}
                                className="bg-slate-900 border border-slate-700 text-slate-200 text-[10px] font-mono rounded px-1.5 py-1 text-center focus:outline-none focus:border-emerald-500 w-full"
                              />
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Rotation */}
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase font-semibold tracking-wider block mb-1">Rotation</span>
                        <div className="grid grid-cols-3 gap-1">
                          {/* Roll (Rx) */}
                          <div className="flex flex-col">
                            <label className="text-[8px] text-slate-500 font-bold uppercase mb-0.5 text-center">Rx (°)</label>
                            <input
                              type="number"
                              step="1"
                              value={(rotRad[0] * 180 / Math.PI).toFixed(1)}
                              onChange={(e) => {
                                const newRot: [number, number, number] = [...rotRad] as [number, number, number];
                                newRot[0] = (parseFloat(e.target.value) || 0) * Math.PI / 180;
                                updateStationObject(obj.id, { rotationRpyRad: newRot });
                              }}
                              className="bg-slate-900 border border-slate-700 text-slate-200 text-[10px] font-mono rounded px-1.5 py-1 text-center focus:outline-none focus:border-emerald-500 w-full"
                            />
                          </div>
                          {/* Pitch (Ry) */}
                          <div className="flex flex-col">
                            <label className="text-[8px] text-slate-500 font-bold uppercase mb-0.5 text-center">Ry (°)</label>
                            <input
                              type="number"
                              step="1"
                              value={(rotRad[1] * 180 / Math.PI).toFixed(1)}
                              onChange={(e) => {
                                const newRot: [number, number, number] = [...rotRad] as [number, number, number];
                                newRot[1] = (parseFloat(e.target.value) || 0) * Math.PI / 180;
                                updateStationObject(obj.id, { rotationRpyRad: newRot });
                              }}
                              className="bg-slate-900 border border-slate-700 text-slate-200 text-[10px] font-mono rounded px-1.5 py-1 text-center focus:outline-none focus:border-emerald-500 w-full"
                            />
                          </div>
                          {/* Yaw (Rz) */}
                          <div className="flex flex-col">
                            <label className="text-[8px] text-slate-500 font-bold uppercase mb-0.5 text-center">Rz (°)</label>
                            <input
                              type="number"
                              step="1"
                              value={rzDeg.toFixed(1)}
                              onChange={(e) => {
                                const newRot: [number, number, number] = [...rotRad] as [number, number, number];
                                newRot[2] = (parseFloat(e.target.value) || 0) * Math.PI / 180;
                                updateStationObject(obj.id, { rotationRpyRad: newRot });
                              }}
                              className="bg-slate-900 border border-slate-700 text-slate-200 text-[10px] font-mono rounded px-1.5 py-1 text-center focus:outline-none focus:border-emerald-500 w-full"
                            />
                          </div>
                        </div>
                      </div>

                      {/* Reset */}
                      <button
                        onClick={() => updateStationObject(obj.id, {
                          positionM: [0, 0, 0],
                          rotationRpyRad: [0, 0, 0]
                        })}
                        className="flex items-center gap-1 self-end text-[10px] text-slate-500 hover:text-slate-300 transition py-0.5"
                      >
                        <RotateCcw size={10} />
                        <span>Reset transform</span>
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Simulation Layers Embedded Panel */}
      <div className="px-4 pb-4 border-t border-slate-800/80 pt-4 flex flex-col gap-2 bg-slate-950/20">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400">
          <Layers size={14} className="text-slate-400" />
          <span>Simulation Layers</span>
        </div>
        <div className="flex flex-col gap-1.5">
          <button
            onClick={toggleGrid}
            className={`flex items-center justify-between w-full text-xs py-1.5 px-2.5 rounded border transition ${
              showGrid 
                ? 'bg-indigo-600/10 text-indigo-300 border-indigo-500/30' 
                : 'bg-slate-900/40 text-slate-500 border-transparent hover:text-slate-400'
            }`}
          >
            <span className="font-semibold">Base Floor Grid</span>
            {showGrid ? <Eye size={13} /> : <EyeOff size={13} />}
          </button>

          <button
            onClick={toggleAxes}
            className={`flex items-center justify-between w-full text-xs py-1.5 px-2.5 rounded border transition ${
              showAxes 
                ? 'bg-indigo-600/10 text-indigo-300 border-indigo-500/30' 
                : 'bg-slate-900/40 text-slate-500 border-transparent hover:text-slate-400'
            }`}
          >
            <span className="font-semibold">CAD Link Frames (Axes)</span>
            {showAxes ? <Eye size={13} /> : <EyeOff size={13} />}
          </button>

          <button
            onClick={toggleTCPFrame}
            className={`flex items-center justify-between w-full text-xs py-1.5 px-2.5 rounded border transition ${
              showTCPFrame 
                ? 'bg-amber-600/10 text-amber-300 border-amber-500/30' 
                : 'bg-slate-900/40 text-slate-500 border-transparent hover:text-slate-400'
            }`}
            title="Toggle tool center point (TCP) coordinate frame RGB helper"
          >
            <span className="font-semibold">TCP Frame (Axes)</span>
            {showTCPFrame ? <Eye size={13} /> : <EyeOff size={13} />}
          </button>

          <button
            onClick={toggleCOMs}
            className={`flex items-center justify-between w-full text-xs py-1.5 px-2.5 rounded border transition ${
              showCOMs 
                ? 'bg-indigo-600/10 text-indigo-300 border-indigo-500/30' 
                : 'bg-slate-900/40 text-slate-500 border-transparent hover:text-slate-400'
            }`}
          >
            <span className="font-semibold">Center of Mass (COM)</span>
            {showCOMs ? <Eye size={13} /> : <EyeOff size={13} />}
          </button>

          <button
            onClick={toggleTrajectory}
            className={`flex items-center justify-between w-full text-xs py-1.5 px-2.5 rounded border transition ${
              showTrajectory 
                ? 'bg-indigo-600/10 text-indigo-300 border-indigo-500/30' 
                : 'bg-slate-900/40 text-slate-500 border-transparent hover:text-slate-400'
            }`}
          >
            <span className="font-semibold">Sizing Trajectory Path</span>
            {showTrajectory ? <Eye size={13} /> : <EyeOff size={13} />}
          </button>
        </div>
      </div>

      {/* Beautiful Modal Walkthrough for Backend Installer */}
      {showWalkthrough && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
          <div className="bg-slate-900 border border-slate-800 rounded-xl w-full max-w-lg overflow-hidden shadow-2xl flex flex-col max-h-[85vh]">
            {/* Modal Header */}
            <div className="px-5 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-950/30">
              <div className="flex items-center gap-2">
                <Server className="text-indigo-400" size={18} />
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">Robodimm PRO Setup</h3>
              </div>
              <button 
                onClick={() => setShowWalkthrough(false)}
                className="text-slate-500 hover:text-slate-300 transition"
              >
                <X size={18} />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto flex flex-col gap-4 text-xs text-slate-300 leading-relaxed">
              <p>
                To calculate physical closed-chain loop constraints (CR4) and serial multi-link dynamics (CR6) with highest precision, 
                you can couple a local, secure python calculator to this web app.
              </p>

              <div className="flex flex-col gap-3.5 mt-2">
                {/* Step 1 */}
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-indigo-500/20 text-indigo-400 font-bold flex items-center justify-center border border-indigo-500/30 font-mono">
                    1
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-200 mb-0.5">Download Local Backend</h4>
                    <p className="text-slate-400">
                      Download the latest release package from GitHub at <a href="https://github.com/customrobotics/robodimm-pro-backend" target="_blank" rel="noopener noreferrer" className="text-indigo-400 hover:underline">robodimm-pro-backend</a>.
                    </p>
                  </div>
                </div>

                {/* Step 2 */}
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-indigo-500/20 text-indigo-400 font-bold flex items-center justify-center border border-indigo-500/30 font-mono">
                    2
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-200 mb-0.5">Launch the Application</h4>
                    <p className="text-slate-400"> Extract the ZIP and execute the launcher:</p>
                    <ul className="list-disc pl-4 text-slate-500 mt-1 flex flex-col gap-0.5">
                      <li><strong className="text-slate-400">Windows:</strong> Double-click <code className="bg-slate-950 px-1 py-0.5 rounded font-mono text-[10px]">robodimm-backend.exe</code></li>
                      <li><strong className="text-slate-400">Linux:</strong> Run AppImage or execute <code className="bg-slate-950 px-1 py-0.5 rounded font-mono text-[10px]">./setup_backend.sh</code></li>
                      <li><strong className="text-slate-400">Docker:</strong> <code className="bg-slate-950 px-1 py-0.5 rounded font-mono text-[10px]">docker run -p 127.0.0.1:8001:8001 robodimm/backend-pro</code></li>
                    </ul>
                  </div>
                </div>

                {/* Step 3 */}
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-indigo-500/20 text-indigo-400 font-bold flex items-center justify-center border border-indigo-500/30 font-mono">
                    3
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-200 mb-0.5">Automatic Recognition</h4>
                    <p className="text-slate-400">
                      Once running on <code className="bg-slate-950 px-1 py-0.5 rounded font-mono text-[10px]">127.0.0.1:8001</code>, this tab will automatically update status to <span className="text-emerald-400 font-semibold">PRO Connected</span>.
                    </p>
                  </div>
                </div>

                {/* Step 4 */}
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-indigo-500/20 text-indigo-400 font-bold flex items-center justify-center border border-indigo-500/30 font-mono">
                    4
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-200 mb-0.5">Enable and Calculate</h4>
                    <p className="text-slate-400">
                      Select <strong>PRO (Local Python)</strong> in the Active Engine selector above to process robot physics on your own CPU.
                    </p>
                  </div>
                </div>
              </div>

              {/* Privacy Notice */}
              <div className="mt-4 bg-slate-950/40 p-3 rounded-lg border border-slate-800 text-[11px] text-slate-400 leading-normal">
                <strong className="text-slate-300 block mb-0.5">🔒 Privacy Notice</strong>
                Trajectory dynamics and torque computations are calculated strictly locally on your loopback device (127.0.0.1). Your CAD geometries and motion paths never leave your computer.
              </div>
            </div>

            {/* Modal Footer */}
            <div className="px-5 py-4 border-t border-slate-800 bg-slate-950/20 flex justify-end">
              <button
                onClick={() => setShowWalkthrough(false)}
                className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-1.5 px-4 rounded text-xs transition"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
