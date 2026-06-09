import React, { useState, useEffect } from 'react';
import { useRobodimmStore } from '../model/state';
import { Save, FolderOpen, Play, Settings, ShieldAlert, Check, ChevronDown, ChevronRight } from 'lucide-react';
import { irb4600Serial6Spec } from '../math/serial6';
import { irb460PalletizerSpec, PalletizerEngine } from '../math/palletizer';
import { getDefaultPalletizerBodyCOM } from '../math/palletizerGeometry';
import { RobotSpec, RobotPackageSpec, Cr4Geometry } from '../model/schemas';

export const EditorTab: React.FC = () => {
  const {
    editRobot,
    isSet,
    changeRobotKind,
    updateEditSpec,
    updateCr4HardpointXZ,
    updateCr4DesignParameter,
    setCr4ConstrainParallelogram,
    cr4ConstrainParallelogram,
    loadRobotSpec,
    setRobot,
    cr4ValidationIssues,
    cr4DesignParams,
    visualMode,
    meshesStale,
    meshWarnings,
    setVisualMode,
    applyMeshPreset,
    usePrimitiveVisuals,
    hasBackend,
    uploadPackage
  } = useRobodimmStore();

  const [activeSection, setActiveSection] = useState<'geometry' | 'inertials' | 'station'>('geometry');
  const [cr4EditorMode, setCr4EditorMode] = useState<'basic' | 'linkage'>('basic');
  const hasErrors = editRobot.kind === 'CR4' && cr4ValidationIssues.some(issue => issue.severity === 'error');

  const [expandedInertialBody, setExpandedInertialBody] = useState<string | null>(null);
  const [localInertialData, setLocalInertialData] = useState<{
    massKg: string;
    comM: [string, string, string];
    inertiaKgM2: string[][];
  } | null>(null);

  useEffect(() => {
    if (expandedInertialBody) {
      let massKg = 0;
      let comM: [number, number, number] = [0, 0, 0];
      let inertiaKgM2: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      
      if (expandedInertialBody === 'PAYLOAD') {
        const p = editRobot.payload;
        massKg = p.massKg;
        comM = p.comM || [0, 0, 0];
        inertiaKgM2 = p.inertiaKgM2 || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      } else {
        const inertial = editRobot.inertials[expandedInertialBody];
        if (inertial) {
          massKg = inertial.massKg;
          comM = inertial.comM || (editRobot.kind === 'CR4' ? getDefaultPalletizerBodyCOM(editRobot.geometry as Cr4Geometry, expandedInertialBody) : [0, 0, 0]);
          inertiaKgM2 = inertial.inertiaKgM2 || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        } else if (editRobot.kind === 'CR6') {
          const preset = irb4600Serial6Spec();
          const d = preset.inertials[expandedInertialBody];
          if (d) {
            massKg = d.massKg;
            comM = d.comM ?? [0, 0, 0];
            inertiaKgM2 = d.inertiaKgM2 ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
          }
        } else if (editRobot.kind === 'CR4') {
          const preset = irb460PalletizerSpec();
          const d = preset.inertials[expandedInertialBody];
          if (d) {
            massKg = d.massKg;
            comM = d.comM ?? getDefaultPalletizerBodyCOM(editRobot.geometry as Cr4Geometry, expandedInertialBody);
            inertiaKgM2 = d.inertiaKgM2 ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
          } else {
            massKg = new PalletizerEngine(editRobot)._getDefaultBodyMass(expandedInertialBody);
            comM = getDefaultPalletizerBodyCOM(editRobot.geometry as Cr4Geometry, expandedInertialBody);
          }
        }
      }
      
      setLocalInertialData({
        massKg: massKg.toString(),
        comM: [comM[0].toString(), comM[1].toString(), comM[2].toString()],
        inertiaKgM2: inertiaKgM2.map(row => row.map(v => v.toString()))
      });
    } else {
      setLocalInertialData(null);
    }
  }, [expandedInertialBody, editRobot.kind]);

  const handleLocalChange = (
    field: 'massKg' | 'com' | 'inertia',
    strVal: string,
    extra?: { index: number; col?: number }
  ) => {
    if (!localInertialData || !expandedInertialBody) return;
    
    const next = { ...localInertialData };
    if (field === 'massKg') {
      next.massKg = strVal;
    } else if (field === 'com' && extra) {
      const nextCom = [...next.comM] as [string, string, string];
      nextCom[extra.index] = strVal;
      next.comM = nextCom;
    } else if (field === 'inertia' && extra && extra.col !== undefined) {
      const nextInertia = next.inertiaKgM2.map(row => [...row]);
      nextInertia[extra.index][extra.col] = strVal;
      if (extra.index !== extra.col) {
        nextInertia[extra.col][extra.index] = strVal;
      }
      next.inertiaKgM2 = nextInertia;
    }
    setLocalInertialData(next);

    const val = parseFloat(strVal);
    if (isNaN(val) || !isFinite(val)) return;
    
    // Validations: mass >= 0, diagonals >= 0
    if (field === 'massKg' && val < 0) return;
    if (field === 'inertia' && extra && extra.index === extra.col && val < 0) return;

    updateEditSpec((spec) => {
      const body = expandedInertialBody;
      if (body === 'PAYLOAD') {
        if (!spec.payload) {
          spec.payload = {
            body: 'PAYLOAD',
            massKg: 0,
            comM: [0, 0, 0],
            inertiaKgM2: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            frame: 'link'
          };
        }
        const p = spec.payload;
        if (!p.comM) p.comM = [0, 0, 0];
        if (!p.inertiaKgM2) p.inertiaKgM2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        if (field === 'massKg') {
          p.massKg = val;
        } else if (field === 'com' && extra) {
          p.comM[extra.index] = val;
        } else if (field === 'inertia' && extra && extra.col !== undefined) {
          p.inertiaKgM2[extra.index][extra.col] = val;
          if (extra.index !== extra.col) {
            p.inertiaKgM2[extra.col][extra.index] = val;
          }
        }
      } else {
        if (!spec.inertials[body]) {
          if (spec.kind === 'CR6') {
            const preset = irb4600Serial6Spec();
            const d = preset.inertials[body];
            spec.inertials[body] = {
              body,
              massKg: d ? d.massKg : 0,
              comM: d ? d.comM : [0, 0, 0],
              inertiaKgM2: d ? d.inertiaKgM2 as any : [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              frame: 'cad'
            };
          } else {
            const preset = irb460PalletizerSpec();
            const d = preset.inertials[body];
            const defaultMass = new PalletizerEngine(spec)._getDefaultBodyMass(body);
            const defaultCom = getDefaultPalletizerBodyCOM(spec.geometry as Cr4Geometry, body);
            spec.inertials[body] = {
              body,
              massKg: d ? d.massKg : defaultMass,
              comM: d && d.comM ? d.comM : defaultCom,
              inertiaKgM2: d && d.inertiaKgM2 ? d.inertiaKgM2 as any : [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              frame: 'cad'
            };
          }
        }
        const inertial = spec.inertials[body];
        if (field === 'massKg') {
          inertial.massKg = val;
        } else {
          if (!inertial.comM) {
            inertial.comM = spec.kind === 'CR4'
              ? getDefaultPalletizerBodyCOM(spec.geometry as Cr4Geometry, body)
              : [0, 0, 0];
          }
          if (!inertial.inertiaKgM2) {
            inertial.inertiaKgM2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
          }
          if (field === 'com' && extra) {
            inertial.comM[extra.index] = val;
          } else if (field === 'inertia' && extra && extra.col !== undefined) {
            inertial.inertiaKgM2[extra.index][extra.col] = val;
            if (extra.index !== extra.col) {
              inertial.inertiaKgM2[extra.col][extra.index] = val;
            }
          }
        }
      }
    });
  };

  const handleBlur = () => {
    if (!expandedInertialBody) return;
    
    let massKg = 0;
    let comM: [number, number, number] = [0, 0, 0];
    let inertiaKgM2: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    
    if (expandedInertialBody === 'PAYLOAD') {
      const p = editRobot.payload;
      massKg = p.massKg;
      comM = p.comM || [0, 0, 0];
      inertiaKgM2 = p.inertiaKgM2 || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    } else {
      const inertial = editRobot.inertials[expandedInertialBody];
      if (inertial) {
        massKg = inertial.massKg;
        comM = inertial.comM || (editRobot.kind === 'CR4' ? getDefaultPalletizerBodyCOM(editRobot.geometry as Cr4Geometry, expandedInertialBody) : [0, 0, 0]);
        inertiaKgM2 = inertial.inertiaKgM2 || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      } else if (editRobot.kind === 'CR6') {
        const preset = irb4600Serial6Spec();
        const d = preset.inertials[expandedInertialBody];
        if (d) {
          massKg = d.massKg;
          comM = d.comM ?? [0, 0, 0];
          inertiaKgM2 = d.inertiaKgM2 ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        }
      } else if (editRobot.kind === 'CR4') {
        const preset = irb460PalletizerSpec();
        const d = preset.inertials[expandedInertialBody];
        if (d) {
          massKg = d.massKg;
          comM = d.comM ?? getDefaultPalletizerBodyCOM(editRobot.geometry as Cr4Geometry, expandedInertialBody);
          inertiaKgM2 = d.inertiaKgM2 ?? [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        } else {
          massKg = new PalletizerEngine(editRobot)._getDefaultBodyMass(expandedInertialBody);
          comM = getDefaultPalletizerBodyCOM(editRobot.geometry as Cr4Geometry, expandedInertialBody);
        }
      }
    }
    
    setLocalInertialData({
      massKg: massKg.toString(),
      comM: [comM[0].toString(), comM[1].toString(), comM[2].toString()],
      inertiaKgM2: inertiaKgM2.map(row => row.map(v => v.toString()))
    });
  };

  const renderInertialEditorSubDrawer = (bodyName: string) => {
    if (!localInertialData) return null;
    const frame = bodyName === 'PAYLOAD' ? 'Payload/TCP Frame' : 'CAD';

    return (
      <div className="px-3 pb-3 pt-2 bg-slate-950/40 border-t border-slate-900 flex flex-col gap-3 text-xs">
        <div className="grid grid-cols-2 gap-2">
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-slate-500 uppercase font-semibold">Mass (kg)</span>
            <input
              type="text"
              className="w-full bg-slate-900 border border-slate-800 text-slate-200 rounded p-1 text-xs text-center font-mono focus:ring-0 focus:outline-none"
              value={localInertialData.massKg}
              onChange={(e) => handleLocalChange('massKg', e.target.value)}
              onBlur={handleBlur}
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-slate-500 uppercase font-semibold">Frame</span>
            <span className="bg-slate-900/60 border border-transparent text-slate-400 rounded p-1 text-xs text-center font-mono uppercase select-none">
              {frame}
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <span className="text-[10px] text-slate-500 uppercase font-semibold">Center of Mass (m)</span>
          <div className="grid grid-cols-3 gap-1.5 font-mono">
            <div className="flex items-center bg-slate-900 border border-slate-850 rounded px-1.5">
              <span className="text-[10px] text-slate-500 font-bold mr-1">X</span>
              <input
                type="text"
                className="w-full bg-transparent border-0 text-slate-200 p-0.5 text-xs text-right font-mono focus:ring-0 focus:outline-none"
                value={localInertialData.comM[0]}
                onChange={(e) => handleLocalChange('com', e.target.value, { index: 0 })}
                onBlur={handleBlur}
              />
            </div>
            <div className="flex items-center bg-slate-900 border border-slate-850 rounded px-1.5">
              <span className="text-[10px] text-slate-500 font-bold mr-1">Y</span>
              <input
                type="text"
                className="w-full bg-transparent border-0 text-slate-200 p-0.5 text-xs text-right font-mono focus:ring-0 focus:outline-none"
                value={localInertialData.comM[1]}
                onChange={(e) => handleLocalChange('com', e.target.value, { index: 1 })}
                onBlur={handleBlur}
              />
            </div>
            <div className="flex items-center bg-slate-900 border border-slate-850 rounded px-1.5">
              <span className="text-[10px] text-slate-500 font-bold mr-1">Z</span>
              <input
                type="text"
                className="w-full bg-transparent border-0 text-slate-200 p-0.5 text-xs text-right font-mono focus:ring-0 focus:outline-none"
                value={localInertialData.comM[2]}
                onChange={(e) => handleLocalChange('com', e.target.value, { index: 2 })}
                onBlur={handleBlur}
              />
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <span className="text-[10px] text-slate-500 uppercase font-semibold">Inertia Tensor (kg·m²)</span>
          <div className="grid grid-cols-3 gap-1 bg-slate-950 p-1.5 rounded border border-slate-850 font-mono">
            {localInertialData.inertiaKgM2.map((row, rIdx) => 
              row.map((val, cIdx) => {
                const axesNames = ['x', 'y', 'z'];
                const label = `I${axesNames[rIdx]}${axesNames[cIdx]}`;
                return (
                  <div key={`${rIdx}-${cIdx}`} className="bg-slate-900/60 p-1 rounded flex flex-col items-center border border-slate-850">
                    <span className="text-[8px] text-slate-500 font-bold uppercase">{label}</span>
                    <input
                      type="text"
                      className="w-full bg-transparent border-0 text-slate-200 p-0.5 text-[10px] text-center font-mono focus:ring-0 focus:outline-none"
                      value={val}
                      onChange={(e) => handleLocalChange('inertia', e.target.value, { index: rIdx, col: cIdx })}
                      onBlur={handleBlur}
                    />
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    );
  };

  // Handle Save Spec
  const handleSaveSpec = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(editRobot, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `${editRobot.name.toLowerCase().replace(/\s+/g, '_')}_spec.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Handle Load Spec
  const handleLoadSpec = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const spec = JSON.parse(event.target?.result as string) as RobotSpec;
        if (spec.schema === 'robodimm.robot.v1' && (spec.kind === 'CR4' || spec.kind === 'CR6')) {
          loadRobotSpec(spec);
        } else {
          alert('Invalid robot specification schema');
        }
      } catch (err) {
        alert('Error reading JSON file');
      }
    };
    reader.readAsText(file);
  };

  // Handle Load Local Folder Package
  const handleLoadFolderPackage = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const fileList = Array.from(files);

    // 1. Find robot.json
    const robotJsonFile = fileList.find(f => f.name.toLowerCase() === 'robot.json');
    if (!robotJsonFile) {
      alert('Error: No "robot.json" file found in the selected folder.');
      return;
    }

    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const text = event.target?.result as string;
        const packageData = JSON.parse(text);

        if (packageData.schema === 'robodimm.package.v1') {
          const robotSpec = packageData.robot as RobotSpec;
          const meshes = packageData.assets?.meshes || [];

          let updatedVisuals = robotSpec.visuals;
          let backup = null;

          if (meshes && meshes.length > 0) {
            backup = JSON.parse(JSON.stringify(robotSpec.visuals));
            updatedVisuals = meshes.map((meshAsset: any) => {
              const cleanPath = meshAsset.path.replace(/\\/g, '/');
              const matchedFile = fileList.find(f => {
                const relPath = f.webkitRelativePath.replace(/\\/g, '/');
                return relPath.endsWith(cleanPath) || f.name.toLowerCase() === cleanPath.split('/').pop()?.toLowerCase();
              });

              const objectUrl = matchedFile ? URL.createObjectURL(matchedFile) : '';
              return {
                body: meshAsset.body,
                frameName: meshAsset.frameName || meshAsset.body,
                kind: 'mesh' as const,
                meshUrl: objectUrl,
                originM: [0, 0, 0] as [number, number, number],
                rpyRad: [0, 0, 0] as [number, number, number],
                scale: [1, 1, 1] as [number, number, number],
                visible: true
              };
            });
          }

          robotSpec.visuals = updatedVisuals;
          loadRobotSpec(robotSpec);
          if (backup) {
            useRobodimmStore.setState({
              primitiveVisualsBackup: backup,
              visualMode: 'meshes'
            });
          }
        } else if (packageData.schema === 'robodimm.robot.v1') {
          loadRobotSpec(packageData);
        } else {
          alert('Invalid package or robot specification schema');
        }
      } catch (err) {
        console.error('Error parsing robot.json:', err);
        alert('Error parsing "robot.json": ' + (err as Error).message);
      }
    };
    reader.readAsText(robotJsonFile);
  };

  // Handle Export Package
  const handleExportPackage = () => {
    const robotCopy = JSON.parse(JSON.stringify(editRobot)) as RobotSpec;

    const meshAssets = robotCopy.visuals
      .filter(vis => vis.kind === 'mesh' && vis.meshUrl)
      .map(vis => {
        let filename = `${vis.body.toLowerCase()}.glb`;
        if (vis.meshUrl && !vis.meshUrl.startsWith('blob:')) {
          const parts = vis.meshUrl.split('/');
          filename = parts[parts.length - 1];
        }
        const relativePath = `meshes/${filename}`;
        vis.meshUrl = relativePath;
        return {
          body: vis.body,
          frameName: vis.frameName || vis.body,
          path: relativePath,
          format: 'glb' as const,
          units: 'm' as const,
          frame: 'cad' as const
        };
      });

    const packageData: RobotPackageSpec = {
      schema: 'robodimm.package.v1',
      robot: robotCopy,
      assets: {
        meshes: meshAssets
      }
    };

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(packageData, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `${editRobot.name.toLowerCase().replace(/\s+/g, '_')}_package.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/40 backdrop-blur">
      {/* Tab Header Selector */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex gap-2">
          <button
            onClick={() => setActiveSection('geometry')}
            className={`text-xs py-1.5 px-3 rounded transition ${
              activeSection === 'geometry' ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400'
            }`}
          >
            DH Geometry / Hardpoints
          </button>
          <button
            onClick={() => setActiveSection('inertials')}
            className={`text-xs py-1.5 px-3 rounded transition ${
              activeSection === 'inertials' ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400'
            }`}
          >
            Inertial Parameters
          </button>
        </div>
      </div>

      {/* Main Form Fields scrollable */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
        {/* Robot Type and Presets */}
        <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Robot Family Selection</h3>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => changeRobotKind('CR6')}
              className={`p-3 rounded-lg border flex flex-col text-left transition ${
                editRobot.kind === 'CR6'
                  ? 'border-indigo-500 bg-indigo-500/10 text-white'
                  : 'border-slate-800 bg-slate-950/20 text-slate-400 hover:border-slate-700'
              }`}
            >
              <span className="font-bold text-sm">CR6</span>
              <span className="text-[10px] mt-1">6-DOF serial articulated robot (IRB 4600)</span>
            </button>
            <button
              onClick={() => changeRobotKind('CR4')}
              className={`p-3 rounded-lg border flex flex-col text-left transition ${
                editRobot.kind === 'CR4'
                  ? 'border-indigo-500 bg-indigo-500/10 text-white'
                  : 'border-slate-800 bg-slate-950/20 text-slate-400 hover:border-slate-700'
              }`}
            >
              <span className="font-bold text-sm">CR4</span>
              <span className="text-[10px] mt-1">4-DOF parallel palletizer robot (IRB 460)</span>
            </button>
          </div>
        </div>

        {/* Visual Mode Toggle */}
        <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Visual Mode</h3>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setVisualMode('primitives')}
              className={`p-2 rounded border flex flex-col items-center justify-center text-center transition ${
                visualMode === 'primitives'
                  ? 'border-indigo-500 bg-indigo-500/10 text-white'
                  : 'border-slate-800 bg-slate-950/20 text-slate-400 hover:border-slate-700'
              }`}
            >
              <span className="font-bold text-xs">Primitive Shapes</span>
              <span className="text-[9px] mt-0.5 text-slate-500">Cylinders and Boxes</span>
            </button>
            <button
              onClick={() => setVisualMode('meshes')}
              className={`p-2 rounded border flex flex-col items-center justify-center text-center transition ${
                visualMode === 'meshes'
                  ? 'border-indigo-500 bg-indigo-500/10 text-white'
                  : 'border-slate-800 bg-slate-950/20 text-slate-400 hover:border-slate-700'
              }`}
            >
              <span className="font-bold text-xs">CAD Meshes</span>
              <span className="text-[9px] mt-0.5 text-slate-500">High-fidelity 3D models</span>
            </button>
          </div>

          {/* Quick preset for IRB4600 if visualMode is primitives and it's CR6 */}
          {editRobot.kind === 'CR6' && !editRobot.visuals.some(v => v.meshUrl) && (
            <button
              onClick={() => applyMeshPreset('irb4600_20kg_250')}
              className="mt-1 text-xs py-2 px-3 bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-600/30 rounded transition text-center font-medium"
            >
              Load IRB4600 CAD Meshes
            </button>
          )}

          {/* Quick preset for IRB460 if visualMode is primitives and it's CR4 */}
          {editRobot.kind === 'CR4' && !editRobot.visuals.some(v => v.meshUrl) && (
            <button
              onClick={() => applyMeshPreset('irb460_palletizer')}
              className="mt-1 text-xs py-2 px-3 bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-600/30 rounded transition text-center font-medium"
            >
              Load IRB460 CAD Meshes
            </button>
          )}

          {/* Warning for Stale Meshes */}
          {meshesStale && (
            <div className="mt-1 p-2 bg-amber-950/15 border border-amber-900/50 text-amber-300 rounded text-xs flex flex-col gap-1.5">
              <span>⚠️ Robot geometry was modified. CAD meshes may not align perfectly with updated linkage dimensions.</span>
              <button
                onClick={usePrimitiveVisuals}
                className="py-1 px-2 bg-amber-800 hover:bg-amber-700 text-white rounded text-[10px] self-start font-medium"
              >
                Revert to Primitives
              </button>
            </div>
          )}

          {/* Mesh Warnings */}
          {meshWarnings.length > 0 && (
            <div className="mt-1 p-2 bg-red-950/15 border border-red-900/50 text-red-300 rounded text-xs flex flex-col gap-1">
              <span className="font-bold">Mesh Loading Warnings:</span>
              <div className="max-h-[100px] overflow-y-auto flex flex-col gap-0.5">
                {meshWarnings.map((warn, i) => (
                  <div key={i} className="font-mono text-[10px]">{warn}</div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Section: Geometry */}
        {activeSection === 'geometry' && (
          <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Geometric Parameters</h3>
            
            {editRobot.kind === 'CR6' ? (
              <div className="overflow-x-auto">
                <table className="custom-table">
                  <thead>
                    <tr>
                      <th>Joint</th>
                      <th>a (m)</th>
                      <th>alpha (rad)</th>
                      <th>d (m)</th>
                      <th>theta offset (rad)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(editRobot.geometry as any).joints.map((joint: any, idx: number) => (
                      <tr key={joint.name}>
                        <td className="font-mono font-semibold text-indigo-400">{joint.name}</td>
                        <td>
                          {idx < 3 ? (
                            <input
                              type="number"
                              step="0.005"
                              className="w-16 p-1 text-xs text-center bg-slate-900 border border-slate-800 rounded text-slate-200"
                              value={joint.a_m}
                              onChange={(e) => {
                                const val = parseFloat(e.target.value) || 0;
                                updateEditSpec((spec) => {
                                  (spec.geometry as any).joints[idx].a_m = val;
                                });
                              }}
                            />
                          ) : (
                            <input
                              type="number"
                              disabled
                              className="w-16 p-1 text-xs text-center bg-slate-950/40 text-slate-500 border-none cursor-not-allowed"
                              value={joint.a_m}
                            />
                          )}
                        </td>
                        <td>
                          <input
                            type="number"
                            disabled
                            className="w-16 p-1 text-xs text-center bg-slate-950/40 text-slate-500 border-none cursor-not-allowed"
                            value={joint.alpha_rad}
                          />
                        </td>
                        <td>
                          {idx === 0 || idx === 3 || idx === 5 ? (
                            <input
                              type="number"
                              step="0.005"
                              className="w-16 p-1 text-xs text-center bg-slate-900 border border-slate-800 rounded text-slate-200"
                              value={joint.d_m}
                              onChange={(e) => {
                                const val = parseFloat(e.target.value) || 0;
                                updateEditSpec((spec) => {
                                  (spec.geometry as any).joints[idx].d_m = val;
                                });
                              }}
                            />
                          ) : (
                            <input
                              type="number"
                              disabled
                              className="w-16 p-1 text-xs text-center bg-slate-950/40 text-slate-500 border-none cursor-not-allowed"
                              value={joint.d_m}
                            />
                          )}
                        </td>
                        <td>
                          <input
                            type="number"
                            disabled
                            className="w-16 p-1 text-xs text-center bg-slate-950/40 text-slate-500 border-none cursor-not-allowed"
                            value={joint.theta_offset_rad}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                {/* Editor Mode Selector */}
                <div className="flex items-center justify-between gap-4 mb-1 flex-wrap">
                  <div className="flex gap-2 bg-slate-950/40 p-1 rounded-lg border border-slate-800 self-start">
                    <button
                      type="button"
                      onClick={() => setCr4EditorMode('basic')}
                      className={`text-[9px] uppercase font-bold py-1 px-3 rounded transition ${
                        cr4EditorMode === 'basic' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Basic Geometry
                    </button>
                    <button
                      type="button"
                      onClick={() => setCr4EditorMode('linkage')}
                      className={`text-[9px] uppercase font-bold py-1 px-3 rounded transition ${
                        cr4EditorMode === 'linkage' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Linkage Details
                    </button>
                  </div>

                  <label className="flex items-center gap-2 text-xs font-semibold text-slate-350 cursor-pointer hover:text-slate-100 select-none bg-slate-950/25 py-1 px-2.5 rounded-lg border border-slate-850">
                    <input
                      type="checkbox"
                      className="rounded border-slate-850 bg-slate-900 text-indigo-600 focus:ring-indigo-500 focus:ring-offset-slate-950 h-3.5 w-3.5"
                      checked={cr4ConstrainParallelogram}
                      onChange={(e) => setCr4ConstrainParallelogram(e.target.checked)}
                    />
                    <span>Enforce perfect parallelogram</span>
                  </label>
                </div>

                {/* Parameters inputs */}
                {cr4DesignParams && (
                  <div className="grid grid-cols-2 gap-3 mb-2 text-xs bg-slate-950/20 p-3 rounded-lg border border-slate-900">
                    {cr4EditorMode === 'basic' ? (
                      <>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">Pivot X (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.Ox}
                            onChange={(e) => updateCr4DesignParameter('Ox', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">Pivot Z (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.Oz}
                            onChange={(e) => updateCr4DesignParameter('Oz', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">Crank L_OB (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.L_OB}
                            onChange={(e) => updateCr4DesignParameter('L_OB', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">Lower Arm L_OC (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.L_OC}
                            onChange={(e) => updateCr4DesignParameter('L_OC', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">Extension L_CH (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.L_CH}
                            onChange={(e) => updateCr4DesignParameter('L_CH', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">Tool Extension L_HEE (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.L_HEE}
                            onChange={(e) => updateCr4DesignParameter('L_HEE', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">Tool Offset L_TCP (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.L_TCP}
                            onChange={(e) => updateCr4DesignParameter('L_TCP', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">D Offset X (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.D_offset_x}
                            onChange={(e) => updateCr4DesignParameter('D_offset_x', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">D Offset Z (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.D_offset_z}
                            onChange={(e) => updateCr4DesignParameter('D_offset_z', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">F Offset X (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.F_offset_x}
                            onChange={(e) => updateCr4DesignParameter('F_offset_x', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">F Offset Z (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.F_offset_z}
                            onChange={(e) => updateCr4DesignParameter('F_offset_z', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold flex items-center justify-between">
                            <span>Link FG Length (m)</span>
                            {cr4ConstrainParallelogram && (
                              <span className="text-[7.5px] text-indigo-400 bg-indigo-950/40 px-1 rounded border border-indigo-900/40 uppercase tracking-wide">Auto</span>
                            )}
                          </span>
                          <input
                            type="number"
                            step="0.005"
                            className={`p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none ${
                              cr4ConstrainParallelogram ? 'opacity-55 cursor-not-allowed bg-slate-950/50 text-slate-400' : ''
                            }`}
                            value={cr4DesignParams.L_FG}
                            onChange={(e) => updateCr4DesignParameter('L_FG', parseFloat(e.target.value) || 0)}
                            disabled={cr4ConstrainParallelogram}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold flex items-center justify-between">
                            <span>Link HG Length (m)</span>
                            {cr4ConstrainParallelogram && (
                              <span className="text-[7.5px] text-indigo-400 bg-indigo-950/40 px-1 rounded border border-indigo-900/40 uppercase tracking-wide">Auto</span>
                            )}
                          </span>
                          <input
                            type="number"
                            step="0.005"
                            className={`p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none ${
                              cr4ConstrainParallelogram ? 'opacity-55 cursor-not-allowed bg-slate-950/50 text-slate-400' : ''
                            }`}
                            value={cr4DesignParams.L_HG}
                            onChange={(e) => updateCr4DesignParameter('L_HG', parseFloat(e.target.value) || 0)}
                            disabled={cr4ConstrainParallelogram}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">J4 Offset X (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.J4_offset_x}
                            onChange={(e) => updateCr4DesignParameter('J4_offset_x', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-slate-400 font-semibold">J4 Offset Z (m)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="p-1 bg-slate-900 border border-slate-800 rounded text-slate-100 font-mono text-xs focus:border-indigo-500 focus:outline-none"
                            value={cr4DesignParams.J4_offset_z}
                            onChange={(e) => updateCr4DesignParameter('J4_offset_z', parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </>
                    )}
                  </div>
                )}

                {/* Read-Only Derived Hardpoints Table */}
                <div className="flex flex-col gap-1">
                  <span className="text-[9px] uppercase font-bold text-slate-400 tracking-wider">
                    Derived Hardpoints Reference
                  </span>
                  <div className="overflow-x-auto max-h-[180px] border border-slate-800 rounded-lg">
                    <table className="min-w-full text-xs text-left text-slate-350 bg-slate-950/20">
                      <thead className="bg-slate-950 text-slate-400 font-bold uppercase tracking-wider text-[9px]">
                        <tr>
                          <th className="p-1.5 border-b border-slate-800">Point</th>
                          <th className="p-1.5 border-b border-slate-800 text-center">X (m)</th>
                          <th className="p-1.5 border-b border-slate-800 text-center">Y (m)</th>
                          <th className="p-1.5 border-b border-slate-800 text-center">Z (m)</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-850">
                        {Object.entries(editRobot.geometry as any).map(([label, coord]: [string, any]) => {
                          const isDerived = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'P', 'J4', 'EE', 'TCP'].includes(label);
                          return (
                            <tr key={label} className="hover:bg-slate-900/30">
                              <td className="p-1.5 font-mono font-bold text-slate-200 flex items-center gap-1.5">
                                <span>{label}</span>
                                {isDerived && (
                                  <span className="text-[7.5px] px-1 bg-black/40 text-slate-500 rounded uppercase font-sans tracking-wide">
                                    Derived
                                  </span>
                                )}
                              </td>
                              <td className="p-1.5 text-center font-mono text-slate-300">
                                {coord[0].toFixed(4)}
                              </td>
                              <td className="p-1.5 text-center font-mono text-slate-500">
                                0.0
                              </td>
                              <td className="p-1.5 text-center font-mono text-slate-300">
                                {coord[2].toFixed(4)}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {cr4ValidationIssues.length > 0 && (
                  <div className="mt-1 bg-slate-950/60 border border-slate-850 p-2.5 rounded-lg flex flex-col gap-1.5">
                    <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                      <ShieldAlert size={12} className="text-amber-500" />
                      <span>Design validation results</span>
                    </h4>
                    <div className="flex flex-col gap-1 max-h-[120px] overflow-y-auto pr-1">
                      {cr4ValidationIssues.map((issue) => (
                        <div
                          key={issue.id}
                          className={`text-xs p-1.5 rounded border flex items-start gap-1.5 ${
                            issue.severity === 'error'
                              ? 'bg-red-950/15 border-red-900/50 text-red-300'
                              : 'bg-amber-950/15 border-amber-900/50 text-amber-300'
                          }`}
                        >
                          <ShieldAlert size={13} className="mt-0.5 shrink-0" />
                          <div className="flex-1">
                            <span className="font-semibold uppercase tracking-wider text-[8px] mr-1 px-1 py-0.2 rounded bg-black/30">
                              {issue.severity}
                            </span>
                            {issue.message}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

          </div>
        )}

        {/* Section: Inertials */}
        {activeSection === 'inertials' && (
          <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-4">
            {(() => {
              const bodies = editRobot.kind === 'CR6'
                ? ['PAYLOAD', 'LINK1', 'LINK2', 'LINK3', 'LINK4', 'LINK5', 'LINK6']
                : ['PAYLOAD', 'SWING', 'P_ARM', 'LOWER_ARM', 'P_LINK', 'UPPER_ARM', 'LOWER_LINK', 'LINK_PLATE', 'UPPER_LINK', 'TILT', 'DISK'];
              
              const title = editRobot.kind === 'CR6' ? 'Inertial Parameters (CR6)' : 'Inertial Parameters (CR4 Palletizer)';

              return (
                <div className="flex flex-col gap-3">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">{title}</h3>
                  <div className="flex flex-col border border-slate-800 rounded-lg overflow-hidden bg-slate-950/30">
                    <div className="max-h-[400px] overflow-y-auto">
                      {bodies.map((name) => {
                        const isExpanded = expandedInertialBody === name;
                        let val = 0;
                        if (name === 'PAYLOAD') {
                          val = editRobot.payload.massKg;
                        } else {
                          val = editRobot.inertials[name]?.massKg ?? (() => {
                            if (editRobot.kind === 'CR6') {
                              const preset = irb4600Serial6Spec();
                              return preset.inertials[name]?.massKg ?? 0;
                            } else {
                              const preset = irb460PalletizerSpec();
                              return preset.inertials[name]?.massKg ?? new PalletizerEngine(editRobot)._getDefaultBodyMass(name);
                            }
                          })();
                        }

                        return (
                          <div key={name} className="border-b border-slate-900/50 last:border-b-0">
                            <div
                              onClick={() => setExpandedInertialBody(isExpanded ? null : name)}
                              className="flex items-center justify-between p-2.5 hover:bg-slate-900/20 text-xs cursor-pointer select-none"
                            >
                              <div className="flex items-center gap-1.5">
                                {isExpanded ? (
                                  <ChevronDown size={14} className="text-slate-400" />
                                ) : (
                                  <ChevronRight size={14} className="text-slate-400" />
                                )}
                                <div className={`w-2.5 h-2.5 rounded-full ${name === 'PAYLOAD' ? 'bg-amber-500' : 'bg-indigo-500'}`} />
                                <span className="font-mono text-slate-300 font-medium">{name}</span>
                              </div>
                              <span className="font-mono text-slate-400 font-semibold pr-1">
                                {val.toFixed(2)} kg
                              </span>
                            </div>
                            {isExpanded && renderInertialEditorSubDrawer(name)}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              );
            })()}

            {/* Joint Viscous Friction card */}
            <div className="mt-2 bg-slate-950/40 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 font-bold">
                Joint Damping / Viscous Friction
              </h3>
              <p className="text-[10px] text-slate-500 leading-normal">
                Set the viscous friction coefficient for each active joint in N·m·s/rad (equivalent to N·m/(rad/s)).
              </p>
              <div className="grid grid-cols-2 gap-3 mt-1">
                {editRobot.limits.map((limit, idx) => {
                  const val = limit.frictionCoeffNmSPerRad ?? 0.5;
                  return (
                    <div key={limit.name} className="flex flex-col gap-1">
                      <span className="text-[10px] text-slate-400 font-semibold font-mono">
                        {limit.name} Damping
                      </span>
                      <div className="flex items-center bg-slate-900 border border-slate-850 rounded px-2">
                        <input
                          type="number"
                          step="0.05"
                          min="0"
                          className="w-full bg-transparent border-0 text-slate-200 p-1 text-xs font-mono focus:ring-0 focus:outline-none"
                          value={val}
                          onChange={(e) => {
                            const parsed = parseFloat(e.target.value);
                            const nextVal = isNaN(parsed) ? 0.0 : parsed;
                            updateEditSpec((spec) => {
                              if (spec.limits[idx]) {
                                spec.limits[idx].frictionCoeffNmSPerRad = nextVal;
                              }
                            });
                          }}
                        />
                        <span className="text-[9px] text-slate-500 font-bold ml-1 shrink-0 font-mono">
                          N·m·s/rad
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Persistence Bar at the bottom */}
      <div className="p-4 border-t border-slate-800 flex items-center justify-between bg-slate-950/60 gap-3">
        <div className="flex gap-2 flex-wrap">
          {/* File input for Open */}
          <label className="flex items-center gap-1.5 text-xs py-2 px-3 rounded-lg bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-700 cursor-pointer">
            <FolderOpen size={14} />
            <span>Open</span>
            <input
              type="file"
              accept=".json"
              className="hidden"
              onChange={handleLoadSpec}
            />
          </label>

          {/* Folder input for Local Folder Package */}
          <label 
            className="flex items-center gap-1.5 text-xs py-2 px-3 rounded-lg bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-700 cursor-pointer"
            title="Import a folder containing robot.json and meshes (Recommended: Chromium browser)"
          >
            <FolderOpen size={14} className="text-indigo-400" />
            <span>Open Folder</span>
            <input
              type="file"
              {...{ webkitdirectory: "", directory: "" } as any}
              multiple
              className="hidden"
              onChange={handleLoadFolderPackage}
            />
          </label>

          {/* Upload Folder to Backend (Persistent) */}
          {hasBackend && (
            <label 
              className="flex items-center gap-1.5 text-xs py-2 px-3 rounded-lg bg-indigo-900/40 text-indigo-200 border border-indigo-750 hover:bg-indigo-900/60 cursor-pointer font-medium"
              title="Upload folder persistently to Python backend storage"
            >
              <FolderOpen size={14} className="text-indigo-300" />
              <span>Upload to Server</span>
              <input
                type="file"
                {...{ webkitdirectory: "", directory: "" } as any}
                multiple
                className="hidden"
                onChange={async (e) => {
                  const files = e.target.files;
                  if (files && files.length > 0) {
                    const success = await uploadPackage(Array.from(files));
                    if (success) {
                      alert('Package persistently uploaded and loaded successfully!');
                    } else {
                      alert('Failed to upload package to Python backend.');
                    }
                  }
                }}
              />
            </label>
          )}

          <button
            onClick={handleSaveSpec}
            className="flex items-center gap-1.5 text-xs py-2 px-3 rounded-lg bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-700"
          >
            <Save size={14} />
            <span>Save</span>
          </button>

          {editRobot.visuals.some(v => v.meshUrl) && (
            <button
              onClick={handleExportPackage}
              className="flex items-center gap-1.5 text-xs py-2 px-3 rounded-lg bg-indigo-950/40 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-950/60 font-medium"
            >
              <Save size={14} />
              <span>Export Package</span>
            </button>
          )}
        </div>

        <button
          onClick={setRobot}
          disabled={hasErrors}
          className={`flex items-center gap-2 py-2 px-6 rounded-lg text-sm font-semibold transition ${
            hasErrors
              ? 'bg-red-950/20 text-red-500 border border-red-500/50 cursor-not-allowed'
              : isSet
                ? 'bg-emerald-600/30 text-emerald-400 border border-emerald-500/50 hover:bg-emerald-600/40'
                : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-600/25'
          }`}
        >
          {hasErrors ? <ShieldAlert size={16} /> : isSet ? <Check size={16} /> : <Play size={16} />}
          <span>{hasErrors ? 'Lock/Set blocked' : isSet ? 'Locked (Set)' : 'Lock Robot (Set)'}</span>
        </button>
      </div>
    </div>
  );
};
