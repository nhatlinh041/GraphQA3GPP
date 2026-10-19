import React from 'react';
import { createRoot } from 'react-dom/client';
import { RunExplorer } from './components/RunExplorer';
import { SideNav } from './components/SideNav';
import './index.css';

// Separate Vite entry — browse one benchmark run as a run → config → question tree
// and open any single question as its own page.
function RunApp() {
  return (
    <div className="flex h-screen bg-white font-sans text-gray-900">
      {/* Same left rail as every other page — see components/SideNav. */}
      <SideNav current="run" />
      <div className="flex-1 overflow-hidden">
        <RunExplorer />
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<RunApp />);
