import React, { useCallback, useEffect, useState } from 'react';

/**
 * Weekly LLM request counter for this app.
 *
 * Counts every question asked from the chat page, in EVERY mode and for local models
 * as well as cloud ones — the pipeline makes the same calls either way, and a local
 * model still costs GPU time even though nothing is metered.
 *
 * It counts REQUESTS, not questions. One question is several calls: the intent
 * classifier always runs, the answer is one, and every mode that writes Cypher adds
 * another. Counting one per question would under-report by 2-3x.
 *
 * This is a LOCAL estimate, not a reading of Ollama's meter. ollama.com/api/usage
 * answers 401 — it authenticates with an ed25519 signature whose key lives inside the
 * daemon, and the daemon does not proxy that endpoint. So the real figure at
 * ollama.com/settings is always HIGHER than this: retries after a 429 are invisible
 * here (measured ~4.2 requests per question on run v52 once retries counted), and a
 * benchmark running in the background does not touch this counter at all.
 */

const KEY = 'ollama-weekly-usage';

/** Requests per question, by pipeline mode. Mirrors what the orchestrator actually
 *  calls: intent (always) + answer (always) + Cypher generation (graph modes only).
 *  A mode missing from this table falls back to 2, the floor every mode pays. */
const COST: Record<string, number> = {
  llm_only: 2,        // intent + answer
  vector_only: 2,
  bm25: 2,
  bm25_dense: 2,
  kg_only: 3,         // + Cypher
  hybrid: 3,
  fixed: 3,
  react_agent: 5,     // planner loops; a floor, not a bound
};

export const DEFAULT_LIMIT = 1000;

type Stored = { weekStart: string; used: number };

/** Monday 00:00 of the current week, as a stable key. Ollama's own week resets on a
 *  rolling 7-day clock from first use, which the browser cannot know — anchoring to
 *  Monday keeps the number reproducible instead of silently drifting. */
function weekStart(d = new Date()): string {
  const x = new Date(d);
  x.setHours(0, 0, 0, 0);
  x.setDate(x.getDate() - ((x.getDay() + 6) % 7));
  return x.toISOString().slice(0, 10);
}

function read(): Stored {
  const now = weekStart();
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const s = JSON.parse(raw) as Stored;
      // A stored week that is not this week has expired: start clean rather than
      // carrying last week's total forward.
      if (s.weekStart === now) return s;
    }
  } catch {
    /* private mode, or corrupt entry */
  }
  return { weekStart: now, used: 0 };
}

/** Add one question's cost. Exported so the sender can call it without the chip
 *  having to be mounted or lifted into shared state. */
export function recordUsage(mode: string): void {
  const s = read();
  const next = { weekStart: s.weekStart, used: s.used + (COST[mode] ?? 2) };
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* quota or private mode — the counter is a convenience, never block a query */
  }
  window.dispatchEvent(new CustomEvent('ollama-usage'));
}

export function UsageChip({ limit = DEFAULT_LIMIT }: { limit?: number }) {
  const [used, setUsed] = useState(() => read().used);

  const refresh = useCallback(() => setUsed(read().used), []);
  useEffect(() => {
    window.addEventListener('ollama-usage', refresh);
    // `storage` fires only in OTHER tabs, so a second tab chatting keeps this one
    // honest instead of showing a stale count.
    window.addEventListener('storage', refresh);
    return () => {
      window.removeEventListener('ollama-usage', refresh);
      window.removeEventListener('storage', refresh);
    };
  }, [refresh]);

  const pct = Math.min(1, used / limit);
  // Three bands rather than a gradient: the number is an estimate, and a precise-looking
  // colour ramp would suggest more accuracy than it has.
  const tone =
    pct >= 0.9
      ? 'bg-red-50 border-red-200 text-red-700'
      : pct >= 0.7
      ? 'bg-amber-50 border-amber-200 text-amber-700'
      : 'bg-white border-gray-200 text-gray-600';

  return (
    // No tooltip: the caveats (retries invisible, benchmarks not counted) live in the
    // module docstring, not on screen. The chip shows a number.
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs
                  border ${tone}`}
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           strokeWidth="2" strokeLinecap="round">
        <path d="M3 12a9 9 0 1 0 9-9" />
        <path d="M12 7v5l3 2" />
      </svg>
      <span className="tabular-nums font-medium">{used}</span>
      <span className="opacity-60">/ {limit}</span>
    </span>
  );
}
