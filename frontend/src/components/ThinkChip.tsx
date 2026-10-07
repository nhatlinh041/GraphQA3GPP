import React from 'react';

interface ThinkChipProps {
  value: boolean;
  onChange: (v: boolean) => void;
  /** Why the toggle does not apply, e.g. a model with no reasoning channel.
   *  Empty/undefined = enabled. */
  disabledReason?: string;
}

// Thinking toggle chip — single click flips between "Thinking" (on) and "No thinking" (off).
// When off, reasoning models skip the <think> phase and jump straight to the answer/Cypher.
export function ThinkChip({ value, onChange, disabledReason }: ThinkChipProps) {
  const off = !!disabledReason;
  // Active = thinking enabled (filled style); inactive = thinking disabled (muted style)
  const active = value && !off;
  return (
    <button
      type="button"
      onClick={() => !off && onChange(!value)}
      disabled={off}
      // The tooltip carries the reason: a greyed-out control with no explanation reads
      // as broken, and "this model has no reasoning channel" is not guessable.
      title={
        disabledReason ||
        (active ? 'Thinking enabled — click to disable' : 'Thinking disabled — click to enable')
      }
      className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs
                  border transition-colors
                  ${
                    off
                      ? 'bg-gray-50 border-gray-200 text-gray-400 cursor-not-allowed'
                      : active
                      ? 'bg-amber-50 border-amber-200 text-amber-700 hover:bg-amber-100'
                      : 'bg-white border-gray-200 text-gray-500 hover:border-gray-300 hover:bg-gray-50'
                  }`}
    >
      {/* Light-bulb icon — communicates "reasoning" without needing extra text */}
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7c.7.6 1 1.4 1 2.3v1h6v-1c0-.9.3-1.7 1-2.3A7 7 0 0 0 12 2z" />
      </svg>
      <span className="font-medium">{active ? 'Thinking' : 'No thinking'}</span>
    </button>
  );
}
