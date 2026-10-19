import React from 'react';
import { createRoot } from 'react-dom/client';
import { ReviewApp } from './components/review/ReviewApp';
import { SideNav } from './components/SideNav';
import './index.css';

// Separate Vite entry — expert review of the 3GPP-ClauseQA sample (passkey-gated).
function App() {
  return (
    <div className="flex h-screen bg-white font-sans text-gray-900">
      <SideNav current="review" />
      <div className="flex-1 overflow-hidden">
        <ReviewApp />
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
