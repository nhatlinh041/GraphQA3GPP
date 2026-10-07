import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import type { PipelineStage, Source } from '../hooks/useSSE';
import { useTypewriter } from '../hooks/useTypewriter';
import { ThinkingTrail } from './ThinkingTrail';
import { SourcesFooter } from './SourcesFooter';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  // Reasoning trace from Ollama "think" channel (deepseek-r1, qwen3, …)
  thinking?: string;
  // Pipeline stages and sources captured for this assistant message
  stages?: PipelineStage[];
  sources?: Source[];
  // When the user sent the message that produced this turn (used as elapsed-time anchor)
  startedAt?: number;
}

interface MessageListProps {
  messages: ChatMessage[];
  // The currently-streaming assistant turn (rendered after `messages`)
  streaming: {
    answer: string;
    thinking: string;
    stages: PipelineStage[];
    sources: Source[];
    startedAt: number;
  } | null;
}

// ── Clickable citations + scrollable source panel (NotebookLM-style) ───────
// Answer text carries `[spec_id §section]` citations (grounding-rules format).
// Each becomes a numbered chip; clicking opens a right-side SCROLLABLE panel that
// scrolls to + highlights the cited source chunk. Citations matching NO retrieved
// source (parametric / hallucinated) render dim & non-clickable — a faithfulness
// signal for the reader.

// Matches `[ts_23.288 §4.1]`, `[ts_23_288 §4.1]`, `[ts_29.500]` (section optional).
const CITATION_RE = /\[(ts_[0-9][\w.\-]*)\s*(?:§\s*([^\]]+?))?\]/gi;

const normSpec = (s: string) => (s || '').replace(/\./g, '_').toLowerCase();

// Build a matcher: citation (spec, section?) → { source, number } using the
// SAME numbering as SourcesFooter ([i+1], by array order). Prefer a source whose
// section also matches when several share a spec_id.
function buildCitationMatcher(sources?: Source[]) {
  const list = sources ?? [];
  return (spec: string, section?: string): { source: Source; number: number } | null => {
    const nspec = normSpec(spec);
    const matches = list
      .map((source, idx) => ({ source, number: idx + 1 }))
      .filter((e) => normSpec(e.source.spec_id) === nspec);
    if (matches.length === 0) return null;
    if (section) {
      const nsec = section.trim().toLowerCase();
      const exact = matches.find((e) => (e.source.section || '').trim().toLowerCase() === nsec);
      if (exact) return exact;
    }
    return matches[0];
  };
}

// Numbered, clickable citation chip. Click → opens the scrollable source panel
// and scrolls to / highlights the matched source (NotebookLM-style).
function CitationChip({ number, onClick }: { number: number; onClick: () => void }) {
  return (
    <button type="button" className="cite-chip" onClick={onClick}>
      {number}
    </button>
  );
}

// Walk markdown-rendered children; replace citation patterns inside string nodes
// with a CitationChip. Unmatched (parametric) citations render dim, non-clickable.
function withCitations(
  children: React.ReactNode,
  matcher: ReturnType<typeof buildCitationMatcher>,
  onCite: (number: number) => void,
): React.ReactNode {
  let key = 0;
  return React.Children.map(children, (child) => {
    if (typeof child !== 'string') return child;
    const parts: React.ReactNode[] = [];
    let last = 0;
    let m: RegExpExecArray | null;
    CITATION_RE.lastIndex = 0;
    while ((m = CITATION_RE.exec(child))) {
      if (m.index > last) parts.push(child.slice(last, m.index));
      const hit = matcher(m[1], m[2]);
      if (hit) {
        const n = hit.number;
        parts.push(<CitationChip key={`c${key++}`} number={n} onClick={() => onCite(n)} />);
      } else {
        parts.push(<span key={`u${key++}`} className="cite-unmatched">{m[0]}</span>);
      }
      last = m.index + m[0].length;
    }
    if (parts.length === 0) return child;
    if (last < child.length) parts.push(child.slice(last));
    return parts;
  });
}

// ReactMarkdown `components` overrides routing text through withCitations.
// `node` dropped so it never reaches the DOM. Rebuilt per render (cheap).
function makeCitationComponents(sources: Source[] | undefined, onCite: (number: number) => void) {
  const matcher = buildCitationMatcher(sources);
  const wrap = (Tag: keyof JSX.IntrinsicElements) =>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ({ node, children, ...props }: any) =>
      React.createElement(Tag, props, withCitations(children, matcher, onCite));
  return { p: wrap('p'), li: wrap('li'), td: wrap('td'), th: wrap('th') };
}

// Right-side SCROLLABLE source panel (NotebookLM-style). Lists every source of
// the message; when `active` changes it scrolls that card into view + highlights.
function SourcePanel({
  sources, active, onClose,
}: { sources: Source[]; active: number; onClose: () => void }) {
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);
  useEffect(() => {
    cardRefs.current[active]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [active]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <>
      <div className="src-panel-overlay" onClick={onClose} />
      <aside className="src-panel" role="dialog" aria-label="Sources">
        <header className="src-panel-head">
          <span className="src-panel-title">Sources ({sources.length})</span>
          <button type="button" className="src-panel-close" onClick={onClose} aria-label="Close">×</button>
        </header>
        <div className="src-panel-list">
          {sources.map((s, i) => {
            const content = (s.content ?? '').trim();
            return (
              <div
                key={i}
                ref={(el) => (cardRefs.current[i] = el)}
                className={`src-card ${i === active ? 'src-card-active' : ''}`}
              >
                <div className="src-card-head">
                  <span className="src-card-num">{i + 1}</span>
                  <span className="src-card-spec">{s.spec_id}</span>
                  {s.section && <span className="src-card-sec">§{s.section}</span>}
                  {/* chunk_id is the only identifier that resolves back to the corpus —
                      spec_id + section title is ambiguous across reprinted clauses, and
                      it is what a gold label is written against. Click to copy. */}
                  {s.chunk_id && (
                    <span
                      className="src-card-cid"
                      title="chunk_id — click to copy"
                      onClick={() => navigator.clipboard?.writeText(s.chunk_id)}
                    >
                      {s.chunk_id}
                    </span>
                  )}
                  {typeof s.score === 'number' && (
                    <span className="src-card-score">score {s.score.toFixed(2)}</span>
                  )}
                </div>
                <div className="src-card-body">{content || <em>No content.</em>}</div>
              </div>
            );
          })}
        </div>
      </aside>
    </>
  );
}

export function MessageList({ messages, streaming }: MessageListProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Scrollable source panel: which message's sources + which one is active.
  const [panel, setPanel] = React.useState<{ sources: Source[]; active: number } | null>(null);
  const openCite = (sources: Source[], number: number) =>
    setPanel({ sources, active: number - 1 });
  // On while the user is near the bottom; off once they scroll up to read older text
  const followRef = useRef(true);
  const prevMessagesLen = useRef(messages.length);

  // Detect user scroll: more than 80px from the bottom means they are reading back
  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    followRef.current = distance < 80;
  };

  // On a new user message: always reset follow=true and jump to the bottom
  useEffect(() => {
    if (messages.length > prevMessagesLen.current) {
      followRef.current = true;
      bottomRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' });
    }
    prevMessagesLen.current = messages.length;
  }, [messages.length]);

  // While streaming: auto-scroll only when follow is on
  useEffect(() => {
    if (!followRef.current) return;
    bottomRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' });
  }, [streaming?.answer, streaming?.stages.length]);

  const isEmpty = messages.length === 0 && !streaming;

  return (
    <div ref={containerRef} onScroll={handleScroll} className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto px-4 py-6">
        {isEmpty && (
          <div className="flex flex-col items-center justify-center h-[60vh] text-center">
            <div className="text-2xl font-semibold text-gray-800 mb-2">
              3GPP Knowledge Assistant
            </div>
            <div className="text-sm text-gray-500 max-w-md">
              Ask about 5G architecture, network functions (AMF, SMF, UPF…), procedures, or any
              technical specification.
            </div>
          </div>
        )}

        <div className="space-y-6">
          {messages.map((msg, i) => (
            <MessageRow key={i} msg={msg} streaming={false} onCite={openCite} />
          ))}

          {streaming && (
            <MessageRow
              msg={{
                role: 'assistant',
                content: streaming.answer,
                thinking: streaming.thinking,
                stages: streaming.stages,
                sources: streaming.sources,
                startedAt: streaming.startedAt,
              }}
              streaming
              onCite={openCite}
            />
          )}
        </div>

        <div ref={bottomRef} />
      </div>

      {panel && (
        <SourcePanel
          sources={panel.sources}
          active={panel.active}
          onClose={() => setPanel(null)}
        />
      )}
    </div>
  );
}

// Single message row — claude.ai style: user has a subtle bubble, assistant flows on background
function MessageRow({
  msg, streaming, onCite,
}: {
  msg: ChatMessage;
  streaming: boolean;
  onCite: (sources: Source[], number: number) => void;
}) {
  // The hook is always called (rules of hooks); with streaming=false it returns the
  // full text immediately
  const displayedContent = useTypewriter(msg.content, streaming);

  if (msg.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] bg-gray-100 text-gray-900 rounded-2xl rounded-tr-sm px-4 py-2.5 text-[15px] whitespace-pre-wrap leading-relaxed">
          {msg.content}
        </div>
      </div>
    );
  }

  // Assistant: pipeline trail + thinking trace above, markdown body, sources below
  return (
    <div>
      {msg.stages !== undefined && (
        <ThinkingTrail
          stages={msg.stages}
          streaming={streaming}
          startedAt={msg.startedAt ?? Date.now()}
        />
      )}
      {msg.thinking && msg.thinking.length > 0 && (
        <ReasoningPanel thinking={msg.thinking} streaming={streaming && !msg.content} />
      )}
      <div className="md-body text-[15px] leading-relaxed text-gray-900">
        <ReactMarkdown
          components={makeCitationComponents(msg.sources, (n) => onCite(msg.sources ?? [], n))}
        >
          {displayedContent}
        </ReactMarkdown>
        {streaming && (
          <span className="inline-block w-1.5 h-4 bg-gray-500 animate-pulse align-text-bottom ml-0.5" />
        )}
      </div>
      {msg.sources && msg.sources.length > 0 && <SourcesFooter sources={msg.sources} />}
    </div>
  );
}

// Collapsible "Thoughts" panel — auto-expanded while the model is still thinking,
// auto-collapsed once the answer starts streaming so the user's eye moves to it.
function ReasoningPanel({ thinking, streaming }: { thinking: string; streaming: boolean }) {
  const [open, setOpen] = React.useState(streaming);
  // While streaming-and-thinking, force-open so the user can watch tokens land
  React.useEffect(() => {
    if (streaming) setOpen(true);
  }, [streaming]);

  return (
    <div className="my-2 border border-gray-200 rounded-lg bg-gray-50 text-sm">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-gray-600 hover:bg-gray-100 rounded-lg"
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`transition-transform ${open ? 'rotate-90' : ''}`}
        >
          <path d="M9 18l6-6-6-6" />
        </svg>
        <span className="font-medium">Thoughts</span>
        {streaming && (
          <span className="text-[11px] text-gray-400 italic ml-1">thinking…</span>
        )}
        <span className="ml-auto text-[11px] text-gray-400">{thinking.length} chars</span>
      </button>
      {open && (
        <pre className="px-3 pb-3 pt-1 whitespace-pre-wrap text-[13px] leading-relaxed text-gray-700 font-sans max-h-72 overflow-y-auto">
          {thinking}
        </pre>
      )}
    </div>
  );
}
