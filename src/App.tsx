import React, { useEffect } from 'react';
import { useRobodimmStore } from './model/state';
import { RobotViewer } from './viewer/RobotViewer';
import { InspectorPanel } from './ui/InspectorPanel';
import { EditorTab } from './ui/EditorTab';
import { JogTab } from './ui/JogTab';
import { ProgramTab } from './ui/ProgramTab';
import { ActuatorsTab } from './ui/ActuatorsTab';
import {
  Cpu,
  Settings,
  Activity,
  Zap,
  HardDrive,
  Monitor,
  Menu
} from 'lucide-react';

function App() {
  const {
    activeRobot,
    isSet,
    activeTab,
    setTab,
    activeEngine,
    setEngine,
    hasBackend,
    loadActuatorLibrary,
    checkBackendStatus
  } = useRobodimmStore();

  // Query Python backend on load and setup polling
  useEffect(() => {
    loadActuatorLibrary();
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 10000);
    return () => clearInterval(interval);
  }, [loadActuatorLibrary, checkBackendStatus]);

  const renderActiveTabContent = () => {
    switch (activeTab) {
      case 'editor':
        return <EditorTab />;
      case 'jog':
        return <JogTab />;
      case 'program':
        return <ProgramTab />;
      case 'actuators':
        return <ActuatorsTab />;
      default:
        return <EditorTab />;
    }
  };

  return (
    <div className="w-screen h-screen flex flex-col bg-[#090c10] text-[#e2e8f0] font-sans overflow-hidden">
      
      {/* Premium Main Header */}
      <header className="h-14 border-b border-slate-800 bg-slate-950/65 backdrop-blur flex justify-between items-center px-4 shrink-0 select-none z-25">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-1.5 rounded-lg shadow-md shadow-indigo-600/20">
            <Cpu size={18} className="text-white" />
          </div>
          <div className="flex flex-col">
            <h1 className="text-sm font-black uppercase tracking-widest text-slate-100 leading-tight">Robodimm</h1>
            <span className="text-[9px] text-slate-500 font-bold uppercase tracking-wider">Dynamic & Iterative Actuator Sizing</span>
          </div>
        </div>

        {/* Engine mode switcher */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-0.5 rounded-lg">
            <button
              onClick={() => setEngine('frontend')}
              className={`flex items-center gap-1 text-[10px] uppercase font-bold px-2.5 py-1.5 rounded-md transition ${
                activeEngine === 'frontend'
                  ? 'bg-indigo-600 text-white shadow shadow-indigo-600/20'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Monitor size={12} />
              <span>DEMO (Browser)</span>
            </button>
            <button
              onClick={() => {
                if (hasBackend) setEngine('backend');
              }}
              disabled={!hasBackend}
              className={`flex items-center gap-1 text-[10px] uppercase font-bold px-2.5 py-1.5 rounded-md transition ${
                activeEngine === 'backend'
                  ? 'bg-emerald-600 text-white shadow shadow-emerald-600/20'
                  : hasBackend
                  ? 'text-slate-400 hover:text-slate-200'
                  : 'text-slate-650 cursor-not-allowed'
              }`}
              title={!hasBackend ? "Local FastAPI backend offline. Run backend server to enable PRO mode." : ""}
            >
              <HardDrive size={12} />
              <span>PRO (Python API)</span>
              {!hasBackend && (
                <span className="text-[8px] bg-slate-950/60 text-slate-500 px-1 py-0.5 rounded ml-0.5">
                  Offline
                </span>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Left Side: Dynamic CAD Inspector Tree */}
        <InspectorPanel />

        {/* Central Area: ThreeJS Robot Visualizer */}
        <div className="flex-1 h-full relative">
          <RobotViewer />
        </div>

        {/* Right Side: Configuration & Programming Panel */}
        <div className="w-96 h-full flex flex-col border-l border-slate-800 bg-slate-950/40 backdrop-blur z-20">
          
          {/* Tab Selection Header */}
          <nav className="flex border-b border-slate-800 bg-slate-950/50 shrink-0 select-none">
            <button
              onClick={() => setTab('editor')}
              className={`flex-1 text-[10px] uppercase tracking-wider font-bold py-3.5 text-center border-b-2 transition ${
                activeTab === 'editor'
                  ? 'border-indigo-500 text-slate-100 bg-slate-900/10'
                  : 'border-transparent text-slate-500 hover:text-slate-300'
              }`}
            >
              Editor Spec
            </button>
            
            <button
              onClick={() => {
                if (isSet) setTab('jog');
              }}
              disabled={!isSet}
              className={`flex-1 text-[10px] uppercase tracking-wider font-bold py-3.5 text-center border-b-2 transition ${
                activeTab === 'jog'
                  ? 'border-indigo-500 text-slate-100 bg-slate-900/10'
                  : isSet
                  ? 'border-transparent text-slate-500 hover:text-slate-300'
                  : 'border-transparent text-slate-700 cursor-not-allowed'
              }`}
            >
              Jog Panel
            </button>

            <button
              onClick={() => {
                if (isSet) setTab('program');
              }}
              disabled={!isSet}
              className={`flex-1 text-[10px] uppercase tracking-wider font-bold py-3.5 text-center border-b-2 transition ${
                activeTab === 'program'
                  ? 'border-indigo-500 text-slate-100 bg-slate-900/10'
                  : isSet
                  ? 'border-transparent text-slate-500 hover:text-slate-300'
                  : 'border-transparent text-slate-700 cursor-not-allowed'
              }`}
            >
              Program
            </button>

            <button
              onClick={() => {
                if (isSet) setTab('actuators');
              }}
              disabled={!isSet}
              className={`flex-1 text-[10px] uppercase tracking-wider font-bold py-3.5 text-center border-b-2 transition ${
                activeTab === 'actuators'
                  ? 'border-indigo-500 text-slate-100 bg-slate-900/10'
                  : isSet
                  ? 'border-transparent text-slate-500 hover:text-slate-300'
                  : 'border-transparent text-slate-700 cursor-not-allowed'
              }`}
            >
              Sizing
            </button>
          </nav>

          {/* Active Tab Panel Content */}
          <div className="flex-1 overflow-hidden">
            {renderActiveTabContent()}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
