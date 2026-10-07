import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { MessageList, type ChatMessage } from './MessageList';
import { MessageInput } from './MessageInput';
import { useSSE, type PipelineStage } from '../hooks/useSSE';
import type { Mode } from './ModeToggle';
import type { ContextLength } from './ContextChip';
import { recordUsage } from './UsageChip';

// Token-stream stage names the rag-engine SSE emits at high frequency. Persisting
// every token would blow past localStorage's 5-10 MB quota on a ReAct run (5k+ events
// per message), so each consecutive run of the same stage + iter is collapsed into one
// combined event — the trail UI still rebuilds, because buildSteps only appends tokens
// into a buffer.
const TOKEN_STREAM_STAGES = new Set([
  'thinking',
  'answer',
  'hop_thinking',
  'hop_planner_token',
  'hop_research_thinking',
  'hop_research_token',
  'graph_cypher_thinking',
  'graph_cypher_token',
]);

function compactStagesForStorage(stages: PipelineStage[]): PipelineStage[] {
  const out: PipelineStage[] = [];
  let i = 0;
  while (i < stages.length) {
    const s = stages[i];
    if (!TOKEN_STREAM_STAGES.has(s.stage)) {
      out.push(s);
      i++;
      continue;
    }
    // Collapse consecutive events with the same stage + iter into one combined event
    const stageName = s.stage;
    const iter = (s.data as { iter?: number } | undefined)?.iter;
    const startTs = s.timestamp;
    let acc = '';
    let lastAccumulated: string | undefined;
    let j = i;
    while (
      j < stages.length &&
      stages[j].stage === stageName &&
      (stages[j].data as { iter?: number } | undefined)?.iter === iter
    ) {
      const d = stages[j].data;
      if (typeof d === 'string') {
        acc += d;
      } else if (d && typeof d === 'object') {
        const tok = (d as { token?: string }).token;
        if (typeof tok === 'string') acc += tok;
        const accField = (d as { accumulated?: string }).accumulated;
        if (typeof accField === 'string') lastAccumulated = accField;
      }
      j++;
    }
    // `thinking`/`answer`: data is a raw string -> concatenate into one big string.
    // Everything else: data is an object {iter, token, accumulated?} -> keep the schema
    if (stageName === 'thinking' || stageName === 'answer') {
      out.push({ stage: stageName, data: acc, timestamp: startTs });
    } else {
      out.push({
        stage: stageName,
        data: { iter, token: acc, accumulated: lastAccumulated ?? acc },
        timestamp: startTs,
      });
    }
    i = j;
  }
  return out;
}

// Persist history into localStorage; on quota error, drop oldest assistant turn
// and retry. Worst case (a single message exceeds the quota): skip persisting, but the
// in-memory state survives so the UI does not break.
function safePersistHistory(key: string, history: ChatMessage[]): void {
  let toStore = history;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      localStorage.setItem(key, JSON.stringify(toStore));
      return;
    } catch (e) {
      if (e instanceof Error && e.name === 'QuotaExceededError' && toStore.length > 1) {
        // Drop the oldest message and retry
        toStore = toStore.slice(1);
        continue;
      }
      // Still over quota with a single message, or a different error -> log and skip
      console.warn('Failed to persist chat history:', e);
      return;
    }
  }
}

interface ChatBoxProps {
  sessionId: string;
  // Optional clickable prompts shown above the input while history is empty
  suggestions?: string[];
}

// Imperative handle the parent can use to read chat state for export, etc.
export interface ChatBoxHandle {
  getHistory: () => ChatMessage[];
  getSettings: () => { mode: Mode; model: string; think: boolean; contextLength: ContextLength };
  clearHistory: () => void;
}

// Local models run on this machine's GPU; the `-cloud` ones are served by Ollama
// Cloud through the SAME local daemon (it proxies once `ollama signin` has run), so
// no API key, no separate base URL, and the rag-engine needs no transport change —
// they are just model names.
//
// The cloud list is the Pro catalogue. Everything from deepseek-v4 down needs the
// subscription (the free tier answers HTTP 402 for those and serves only gpt-oss,
// gemma4 and nemotron-3); each one was called with think:true to confirm it really
// returns a `thinking` channel, which is what model_profiles keys the toggle on.
// Ordered by usefulness here rather than by size: deepseek-v4-flash first because it
// is the model the benchmark judge runs on, so a manual query and a scored one see
// the same generator.
//
// Pro allows 3 concurrent cloud models and bills per REQUEST, not per token — so a
// long RAG prompt costs the same as a one-liner. It does NOT host qwen3:14b: the
// catalogue is a fixed list of large models, and pushing a private model to
// ollama.com puts it in the pull registry, not on their GPUs.
const ALL_MODELS = [
  { id: 'qwen3:14b', where: 'local' },
  { id: 'deepseek-r1:14b', where: 'local' },
  { id: 'gemma4:12b', where: 'local' },
  { id: 'gemma3:12b', where: 'local' },
  { id: 'llama3:8b', where: 'local' },
  { id: 'mistral:7b', where: 'local' },
  // deepseek-v4-flash:cloud was retired on Ollama Cloud 2026-09-25 (the daemon now
  // answers "...was retired"); v4.1 is its successor and the current judge model.
  { id: 'deepseek-v4.1-flash:cloud', where: 'cloud' },
  { id: 'deepseek-v4-pro:cloud', where: 'cloud' },
  { id: 'glm-5.3:cloud', where: 'cloud' },
  { id: 'glm-5.3-flash:cloud', where: 'cloud' },
  { id: 'glm-5.2:cloud', where: 'cloud' },
  { id: 'kimi-k3:cloud', where: 'cloud' },
  { id: 'kimi-k2.6:cloud', where: 'cloud' },
  { id: 'minimax-m3:cloud', where: 'cloud' },
  { id: 'gpt-oss:120b-cloud', where: 'cloud' },
  { id: 'gpt-oss:20b-cloud', where: 'cloud' },
  { id: 'gemma4:31b-cloud', where: 'cloud' },
  { id: 'nemotron-3-super:cloud', where: 'cloud' },
] as const;

// Set false to offer only the Ollama Cloud models (local ones were hidden this way
// 2026-10-05 → 2026-10-06 at the project lead's request).
const SHOW_LOCAL_MODELS = true;
const MODELS = ALL_MODELS.filter((m) => SHOW_LOCAL_MODELS || m.where === 'cloud');

/** Hosted models ignore per-request generation options. Verified against the daemon,
 *  not assumed: a 20k-token prompt sent with num_ctx=8192 came back COMPLETE from
 *  deepseek-v4-flash:cloud (20,022 prompt tokens evaluated) while qwen3:14b truncated
 *  it to 4,098 and lost the tail. `seed` is ignored too — three calls at a fixed seed
 *  returned three different answers, where the local model repeated one verbatim.
 *  `think` is the exception and DOES work on the hosted reasoning models, so that
 *  chip stays live for them. */
const isCloud = (m: string) => m.endsWith('-cloud') || m.endsWith(':cloud');

/** Families with a reasoning channel. Mirrors ThinkingProfile/NemotronProfile in
 *  rag-engine/llm/model_profiles.py — the two MUST agree, or the UI offers a toggle
 *  the engine then declines to send (or greys out one that would have worked).
 *  Verified per model by calling the daemon with think:true and checking the reply
 *  carries a `thinking` field; gemma and the small llama/mistral builds do not. */
const THINKING_PREFIXES = [
  'deepseek-r1', 'deepseek-v3', 'deepseek-v4', 'qwen3', 'gpt-oss',
  'glm-5', 'kimi-k', 'minimax-m', 'nemotron-3',
];
const canThink = (m: string) => THINKING_PREFIXES.some((p) => m.toLowerCase().startsWith(p));

/** The toggle starts OFF for every model (2026-09-05). Not a capability claim — the
 *  chip still turns it on per query, and `canThink` still greys it out where there is
 *  no reasoning channel at all. The default flipped because the reasoning pass buys
 *  little on a retrieval-grounded answer: the evidence is already in the context, so
 *  the model mostly re-derives it while delaying the first token (measured on gpt-oss:
 *  1302-1872 generated tokens for a ~190-token answer). It was already OFF for the
 *  DeepSeek family for exactly this reason; the argument was never specific to them.
 *  Note the benchmark harness has always sent `think: false` explicitly, so the UI
 *  default now matches what every measured number was produced with. */
const defaultsToNoThink = (_m: string) => true;

export const ChatBox = forwardRef<ChatBoxHandle, ChatBoxProps>(function ChatBox(
  { sessionId, suggestions }, ref
) {
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<Mode>('fixed');
  const [model, setModel] = useState<string>(MODELS[0].id);
  // Thinking toggle — when off, reasoning models skip the <think> phase entirely.
  // Defaults OFF; see defaultsToNoThink above for why. Flip it per query with the chip.
  const [think, setThink] = useState<boolean>(!defaultsToNoThink(MODELS[0].id));
  // Ollama num_ctx override — default 16k (must match DEFAULT_NUM_CTX in
  // rag-engine/llm/model_profiles.py), selectable down to 8k
  const [contextLength, setContextLength] = useState<ContextLength>(16384);
  const startedAtRef = useRef<number>(0);

  const { answer, thinking, sources, stages, loading, error, send, reset } = useSSE('/api/query');

  // Load chat history from localStorage when sessionId changes (or on mount)
  useEffect(() => {
    const raw = localStorage.getItem(`chat-history-${sessionId}`);
    if (!raw) {
      setHistory([]);
      return;
    }
    try {
      // Compact stages on load — an earlier build may have stored thousands of token
      // events, which renders slowly and risks hitting the quota again on re-persist
      const parsed = JSON.parse(raw) as ChatMessage[];
      const slimmed = parsed.map((m) =>
        m.stages ? { ...m, stages: compactStagesForStorage(m.stages) } : m,
      );
      setHistory(slimmed);
    } catch {
      // Corrupt entry — fall back to empty history
      setHistory([]);
    }
  }, [sessionId]);

  // Persist history whenever it changes. setItem can throw QuotaExceededError once the
  // total exceeds 5-10 MB, so it must be caught or the React tree goes down.
  useEffect(() => {
    safePersistHistory(`chat-history-${sessionId}`, history);
  }, [sessionId, history]);

  // Expose imperative API so the parent (App header) can trigger export/clear
  // without lifting state up.
  useImperativeHandle(
    ref,
    () => ({
      getHistory: () => history,
      getSettings: () => ({ mode, model, think, contextLength }),
      clearHistory: () => {
        setHistory([]);
        localStorage.removeItem(`chat-history-${sessionId}`);
      },
    }),
    [history, mode, model, think, contextLength, sessionId],
  );

  // Switching model re-applies that family's thinking default, but only when the user
  // has not overridden it for the current model: a deliberate "thinking on for
  // DeepSeek" must survive, otherwise the chip appears to undo itself.
  const thinkTouched = useRef(false);
  const handleModelChange = (m: string) => {
    setModel(m);
    if (!thinkTouched.current) setThink(!defaultsToNoThink(m));
  };
  const handleThinkChange = (v: boolean) => {
    thinkTouched.current = true;
    setThink(v);
  };

  const handleSend = (message: string) => {
    setHistory((prev) => [...prev, { role: 'user', content: message }]);
    startedAtRef.current = Date.now();
    // Count before sending, not after: a query that errors out mid-stream still spent
    // the upstream calls it had already made.
    recordUsage(mode);
    send({ question: message, mode, model, think, context_length: contextLength });
  };

  // Commit the streaming assistant turn into history when it finishes
  useEffect(() => {
    if (loading) return;
    if (!answer && stages.length === 0) return;
    // Compact token streams before they enter history — avoids storing thousands of
    // per-token events in localStorage (5-10 MB quota)
    const compactStages = compactStagesForStorage(stages);
    setHistory((prev) => {
      // Avoid duplicating if effect re-runs
      const last = prev[prev.length - 1];
      if (last?.role === 'assistant' && last.content === answer) return prev;
      return [
        ...prev,
        {
          role: 'assistant',
          content: answer,
          thinking,
          stages: compactStages,
          sources,
          startedAt: startedAtRef.current,
        },
      ];
    });
    reset();
  }, [loading]); // eslint-disable-line react-hooks/exhaustive-deps

  const streaming = loading
    ? { answer, thinking, stages, sources, startedAt: startedAtRef.current }
    : null;

  return (
    <div className="flex flex-col h-full">
      {error && (
        <div className="mx-auto max-w-3xl w-full mt-2 px-4">
          <div className="px-3 py-2 bg-red-50 text-red-600 text-sm rounded-lg border border-red-200">
            {error}
          </div>
        </div>
      )}
      <MessageList messages={history} streaming={streaming} />
      {/* Suggestion chips — only while no conversation has started yet */}
      {suggestions && suggestions.length > 0 && history.length === 0 && !loading && (
        <div className="px-4 pb-1">
          <div className="max-w-3xl mx-auto flex flex-wrap gap-1.5">
            {suggestions.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => handleSend(s)}
                className="px-2.5 py-1 text-xs rounded-full border border-gray-200
                           bg-gray-50 hover:bg-gray-100 hover:border-gray-300
                           text-gray-700 text-left transition-colors"
                title={s}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}
      <MessageInput
        onSend={handleSend}
        disabled={loading}
        mode={mode}
        onModeChange={setMode}
        model={model}
        onModelChange={handleModelChange}
        models={MODELS}
        think={think}
        onThinkChange={handleThinkChange}
        thinkDisabledReason={
          canThink(model) ? undefined : `${model} has no reasoning channel`
        }
        contextLength={contextLength}
        onContextLengthChange={setContextLength}
        contextDisabledReason={
          isCloud(model)
            ? 'Context length is fixed by the hosted model — Ollama Cloud ignores num_ctx'
            : undefined
        }
      />
    </div>
  );
});
