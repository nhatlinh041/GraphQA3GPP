import React, { useRef } from 'react';
import { ChatBox, type ChatBoxHandle } from './components/ChatBox';
import { SideNav } from './components/SideNav';

// Stable session ID per browser session
const SESSION_ID = `session-${Date.now()}`;

// Build a JSON snapshot of the entire chat (history + settings) and trigger a
// browser download. Used by the header Export button.
function downloadChatJson(handle: ChatBoxHandle | null): void {
  if (!handle) return;
  const history = handle.getHistory();
  if (history.length === 0) {
    alert('No messages to export yet.');
    return;
  }
  const payload = {
    exportedAt: new Date().toISOString(),
    sessionId: SESSION_ID,
    settings: handle.getSettings(),
    messageCount: history.length,
    messages: history,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  // Filename like 3gpp-chat-2026-04-26T15-32-05.json — filesystem-safe.
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  a.href = url;
  a.download = `3gpp-chat-${stamp}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function App() {
  const chatRef = useRef<ChatBoxHandle>(null);

  return (
    <div className="flex h-screen bg-white font-sans text-gray-900">
      {/* Left rail replaces the per-page header links. Export lives in the rail's
          footer slot: it is an action ON this page, not a destination, so mixing
          it into the destination list was what made the old header read as a
          jumble of five unrelated controls. */}
      <SideNav current="chat">
        <button
          type="button"
          onClick={() => downloadChatJson(chatRef.current)}
          className="w-full text-xs px-2.5 py-1.5 rounded-md border border-gray-200 bg-white
                     text-gray-700 hover:bg-gray-100 hover:border-gray-300
                     inline-flex items-center justify-center gap-1.5"
          title="Download the entire chat (questions, answers, pipeline trace, sources) as JSON"
        >
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          <span className="hidden lg:inline">Export JSON</span>
        </button>
      </SideNav>
      <div className="flex-1 overflow-hidden">
        <ChatBox ref={chatRef} sessionId={SESSION_ID} />
      </div>
    </div>
  );
}
