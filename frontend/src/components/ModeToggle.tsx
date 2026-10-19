import React, { useEffect, useRef, useState } from 'react';

export type Mode =
  | 'fixed'
  | 'hybrid'
  | 'bm25_dense'
  | 'vector_only'
  | 'bm25'
  | 'kg_only'
  | 'llm_only'
  | 'react_agent';

interface ModeChipProps {
  value: Mode;
  onChange: (m: Mode) => void;
}

// Labels use the paper's vocabulary (Results_RAG_3GPP.xlsx) with the repo's
// internal mode name in parentheses. Note "Hybrid" is overloaded: the paper's
// "Hybrid" = BM25+Dense (sparse+dense), while the repo's `hybrid` = Dense+KG =
// the paper's "Text+KG" row — hence the explicit labels below.
const MODES: Array<{ value: Mode; label: string; desc: string }> = [
  { value: 'fixed', label: 'Text + KG + Reranker (fixed)', desc: 'Full pipeline: dense + graph, RRF fusion + cross-encoder rerank' },
  { value: 'hybrid', label: 'Text + KG (hybrid)', desc: 'Dense + graph, RRF fusion, no reranker' },
  { value: 'bm25_dense', label: 'BM25 + Dense (hybrid sparse+dense)', desc: 'Sparse + dense retrieval fused by RRF, no KG' },
  { value: 'vector_only', label: 'Dense (vector_only)', desc: 'Dense vector retrieval only, no KG' },
  { value: 'bm25', label: 'BM25 (sparse)', desc: 'Sparse lexical retrieval only, no KG' },
  { value: 'kg_only', label: 'KG (kg_only)', desc: 'Graph/Cypher retrieval only, no vector' },
  { value: 'llm_only', label: 'LLM-only', desc: 'No retrieval, parametric answer only' },
  { value: 'react_agent', label: 'ReAct Agent', desc: 'LLM-driven dynamic tool selection' },
];

// Mode selector chip — pill button + popover with descriptions
export function ModeChip({ value, onChange }: ModeChipProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [open]);

  const current = MODES.find((m) => m.value === value) ?? MODES[0];

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
          <path d="M3 6h18M3 12h18M3 18h18" />
        </svg>
        <span className="font-medium">{current.label}</span>
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
          className="absolute bottom-full mb-2 left-0 min-w-[220px] py-1
                     bg-white border border-gray-200 rounded-lg shadow-lg z-50"
        >
          {MODES.map((m) => (
            <button
              key={m.value}
              type="button"
              onClick={() => {
                onChange(m.value);
                setOpen(false);
              }}
              className={`w-full text-left px-3 py-1.5 hover:bg-gray-50 flex items-start gap-2
                          ${m.value === value ? 'text-gray-900' : 'text-gray-700'}`}
            >
              <span className="w-3 inline-block flex-shrink-0 mt-0.5">
                {m.value === value && (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <path d="M20 6L9 17l-5-5" />
                  </svg>
                )}
              </span>
              <span className="flex-1">
                <span className="block text-sm font-medium">{m.label}</span>
                <span className="block text-xs text-gray-500">{m.desc}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
