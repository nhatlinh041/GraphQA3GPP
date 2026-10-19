import React from 'react';
import { createRoot } from 'react-dom/client';
import { CypherTester } from './components/CypherTester';
import { ChatPopup } from './components/ChatPopup';
import { SideNav } from './components/SideNav';
import './index.css';

// Separate Vite entry — this page is opened in a new tab from the Chat header
function CypherApp() {
  return (
    <div className="flex h-screen bg-white font-sans text-gray-900">
      {/* Same left rail as every other page — see components/SideNav. */}
      <SideNav current="cypher" />
      <div className="flex-1 overflow-hidden">
        <CypherTester />
      </div>
      {/* Floating chat widget — popup test chat without leaving the page */}
      <ChatPopup />
    </div>
  );
}

const root = document.getElementById('root')!;
createRoot(root).render(<CypherApp />);
