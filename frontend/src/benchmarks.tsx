import React from 'react';
import { createRoot } from 'react-dom/client';
import { BenchmarkViewer } from './components/BenchmarkViewer';
import { SideNav } from './components/SideNav';
import './index.css';

// Separate Vite entry — benchmark dataset/run viewer + editor, opened in a new tab.
function BenchmarksApp() {
  return (
    <div className="flex h-screen bg-white font-sans text-gray-900">
      {/* Same left rail as every other page — see components/SideNav. */}
      <SideNav current="bench" />
      <div className="flex-1 overflow-hidden">
        <BenchmarkViewer />
      </div>
    </div>
  );
}

const root = document.getElementById('root')!;
createRoot(root).render(<BenchmarksApp />);
