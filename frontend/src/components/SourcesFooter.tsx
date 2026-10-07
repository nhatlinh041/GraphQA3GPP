import React, { useState } from 'react';
import type { Source } from '../hooks/useSSE';

interface SourcesFooterProps {
  sources: Source[];
}

export function SourcesFooter({ sources }: SourcesFooterProps) {
  // Whole sources list is collapsible; expanded by default
  const [open, setOpen] = useState(true);
  // Per-source expansion when content is long
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});

  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-4 pt-3 border-t border-gray-100">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 text-xs uppercase tracking-wide text-gray-500
                   hover:text-gray-700 transition-colors mb-2"
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          className={`transition-transform ${open ? 'rotate-90' : ''}`}
        >
          <path d="M9 18l6-6-6-6" />
        </svg>
        <span>Sources ({sources.length})</span>
      </button>

      {open && (
        <ol className="space-y-2">
          {sources.map((s, i) => {
            const content = (s.content ?? '').trim();
            const isLong = content.length > 280;
            const isExpanded = !!expanded[i];
            const display = !content
              ? null
              : isLong && !isExpanded
              ? content.slice(0, 280).trimEnd() + '…'
              : content;

            return (
              <li
                key={i}
                className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-sm"
              >
                <div className="flex items-baseline gap-2 flex-wrap">
                  <span className="text-xs font-mono text-gray-500">[{i + 1}]</span>
                  <span className="font-medium text-gray-800">{s.spec_id}</span>
                  {s.section && <span className="text-gray-500 text-xs">§{s.section}</span>}
                  {s.chunk_id && (
                    <span
                      title="chunk_id — click to copy"
                      onClick={() => navigator.clipboard?.writeText(s.chunk_id)}
                      className="text-[11px] font-mono text-indigo-600 bg-indigo-50 border border-indigo-100
                                 rounded px-1.5 py-0.5 cursor-pointer hover:bg-indigo-100"
                    >
                      {s.chunk_id}
                    </span>
                  )}
                  {typeof s.score === 'number' && (
                    <span className="text-xs text-gray-400 ml-auto tabular-nums">
                      score {s.score.toFixed(2)}
                    </span>
                  )}
                </div>
                {display && (
                  <div className="mt-1.5 text-[13px] text-gray-700 leading-relaxed whitespace-pre-wrap">
                    {display}
                    {isLong && (
                      <button
                        type="button"
                        onClick={() =>
                          setExpanded((prev) => ({ ...prev, [i]: !prev[i] }))
                        }
                        className="ml-1 text-blue-600 hover:underline text-xs"
                      >
                        {isExpanded ? 'Show less' : 'Show more'}
                      </button>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
