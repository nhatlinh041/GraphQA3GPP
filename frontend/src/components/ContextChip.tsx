import React, { useEffect, useRef, useState } from 'react';

// Ollama num_ctx override — default 16k (must match DEFAULT_NUM_CTX in
// rag-engine/llm/model_profiles.py), selectable down to 8k.
export const CONTEXT_LENGTHS = [8192, 16384, 32768] as const;
export type ContextLength = (typeof CONTEXT_LENGTHS)[number];

interface ContextChipProps {
  value: ContextLength;
  onChange: (v: ContextLength) => void;
  /** Why the control does not apply, e.g. on a hosted model. Empty/undefined = enabled. */
  disabledReason?: string;
}

function formatFull(n: number): string {
  return n.toLocaleString('en-US');
}

// Context length selector chip — pill button + popover, same pattern as ModeChip/ModelChip
export function ContextChip({ value, onChange, disabledReason }: ContextChipProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const off = !!disabledReason;

  // Close if the chip becomes disabled while its popover is open — switching to a
  // cloud model with the list showing would otherwise leave a live menu behind.
  useEffect(() => {
    if (off) setOpen(false);
  }, [off]);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => !off && setOpen((v) => !v)}
        disabled={off}
        // The tooltip carries the reason: a greyed-out control with no explanation
        // reads as broken, and the reason here is not guessable from the UI.
        title={disabledReason || 'Context length (num_ctx)'}
        className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs
                   border transition-colors ${
                     off
                       ? 'bg-gray-50 border-gray-200 text-gray-400 cursor-not-allowed'
                       : 'bg-white border-gray-200 text-gray-700 hover:border-gray-300 hover:bg-gray-50'
                   }`}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="M3 9h18" />
        </svg>
        <span className="font-medium">{formatFull(value)} context</span>
        <svg
          width="10"
          height="10"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          className={`transition-transform ${open ? 'rotate-180' : ''}`}
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {open && (
        <div
          className="absolute bottom-full mb-2 left-0 min-w-[140px] py-1
                     bg-white border border-gray-200 rounded-lg shadow-lg z-50"
        >
          {CONTEXT_LENGTHS.map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => {
                onChange(c);
                setOpen(false);
              }}
              className={`w-full text-left px-3 py-1.5 text-sm hover:bg-gray-50 flex items-center gap-2
                          ${c === value ? 'text-gray-900' : 'text-gray-700'}`}
            >
              <span className="w-3 inline-block flex-shrink-0">
                {c === value && (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <path d="M20 6L9 17l-5-5" />
                  </svg>
                )}
              </span>
              {formatFull(c)} context
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
