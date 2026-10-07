import React, { useEffect, useRef, useState } from 'react';

export type ModelOption = { id: string; where: 'local' | 'cloud' };

interface ModelChipProps {
  // Accepts bare ids too, so a caller that has no local/cloud split still works.
  models: readonly (string | ModelOption)[];
  value: string;
  onChange: (m: string) => void;
}

const asOption = (m: string | ModelOption): ModelOption =>
  typeof m === 'string' ? { id: m, where: 'local' } : m;

// Model selector chip — pill button + dropdown popover, opens upward (above the composer)
export function ModelChip({ models, value, onChange }: ModelChipProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
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
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs
                   bg-white border border-gray-200 text-gray-700 hover:border-gray-300
                   hover:bg-gray-50 transition-colors"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="3" />
          <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.93 4.93l2.12 2.12M16.95 16.95l2.12 2.12M4.93 19.07l2.12-2.12M16.95 7.05l2.12-2.12" />
        </svg>
        <span className="font-medium">{value}</span>
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
          // Capped and scrollable: the list is 18 entries once the Pro cloud catalogue
          // is in it, and the popover opens UPWARD from the composer — unbounded, the
          // top of the list runs off the top of the viewport where it cannot be reached.
          className="absolute bottom-full mb-2 left-0 min-w-[210px] py-1
                     max-h-[60vh] overflow-y-auto overscroll-contain
                     bg-white border border-gray-200 rounded-lg shadow-lg z-50"
        >
          {(['local', 'cloud'] as const).map((where) => {
            const group = models.map(asOption).filter((m) => m.where === where);
            if (group.length === 0) return null;
            return (
              <div key={where}>
                {/* Headed only when both groups exist: with one group the heading
                    labels every row and says nothing. Which machine runs the model
                    is the real difference here — it decides latency and whether the
                    request leaves this box — so it gets a heading, not a badge. */}
                {models.map(asOption).some((m) => m.where !== where) && (
                  <div className="px-3 pt-1.5 pb-0.5 text-[10px] uppercase tracking-wide text-gray-400">
                    {where === 'local' ? 'On this machine' : 'Ollama Cloud'}
                  </div>
                )}
                {group.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => {
                      onChange(m.id);
                      setOpen(false);
                    }}
                    className={`w-full text-left px-3 py-1.5 text-sm hover:bg-gray-50 flex items-center gap-2
                                ${m.id === value ? 'text-gray-900' : 'text-gray-700'}`}
                  >
                    <span className="w-3 inline-block">
                      {m.id === value && (
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                          <path d="M20 6L9 17l-5-5" />
                        </svg>
                      )}
                    </span>
                    {m.id.replace(/-cloud$|:cloud$/, '')}
                  </button>
                ))}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
