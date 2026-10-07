import React, { useState } from 'react';
import { ChatBox } from './ChatBox';

// Stable session ID for the popup — separate from any other ChatBox instance on the page
const POPUP_SESSION_ID = `popup-session-${Date.now()}`;

// Seed prompts shown as chips while the popup chat is empty — pick coverage across
// core 5G NFs, security, and architectural comparisons so the user can sanity-check
// the RAG pipeline quickly.
const SUGGESTED_QUESTIONS = [
  'What is AMF in 5G?',
  'Explain the role of SMF',
  'How does 5G AKA authentication work?',
  'Compare 5G NSA vs SA',
  'What is the N1 reference point?',
  'List the main 5G network functions',
];

// Floating chat widget: bubble button at bottom-right that expands into a chat panel.
// Mounted on the Cypher Tester page so users can ask the assistant without leaving the page.
export function ChatPopup() {
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* Chat panel — anchored bottom-right; mounted only when open to reset state on close */}
      {open && (
        <div
          className="fixed bottom-24 right-6 z-50 w-[400px] h-[600px]
                     max-w-[calc(100vw-2rem)] max-h-[calc(100vh-8rem)]
                     bg-white rounded-2xl shadow-2xl border border-gray-200
                     flex flex-col overflow-hidden animate-in fade-in slide-in-from-bottom-4"
          role="dialog"
          aria-label="3GPP Assistant chat"
        >
          {/* Panel header with title + close */}
          <div className="flex items-center gap-2 px-4 py-2.5 border-b border-gray-200 bg-gray-50">
            <div className="w-2 h-2 rounded-full bg-green-500" />
            <div className="flex flex-col leading-tight">
              <span className="text-sm font-semibold text-gray-800">3GPP Assistant</span>
              <span className="text-[11px] text-gray-500">Test the chat without leaving</span>
            </div>
            <div className="flex-1" />
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close chat"
              className="w-7 h-7 rounded-md flex items-center justify-center
                         text-gray-500 hover:bg-gray-200 hover:text-gray-800"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Chat body — reuse the full ChatBox so popup has identical capabilities */}
          <div className="flex-1 min-h-0 overflow-hidden">
            <ChatBox sessionId={POPUP_SESSION_ID} suggestions={SUGGESTED_QUESTIONS} />
          </div>
        </div>
      )}

      {/* Floating bubble button — toggles the panel; swaps icon between chat / close */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? 'Close chat' : 'Open chat'}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full
                   bg-gray-900 text-white shadow-lg hover:bg-gray-800
                   flex items-center justify-center transition-transform
                   hover:scale-105 active:scale-95"
      >
        {open ? (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
          </svg>
        )}
      </button>
    </>
  );
}
