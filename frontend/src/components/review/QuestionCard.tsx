import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { api, ApiError, GoldChunk, QuestionView, Rating, Segment } from './api';
import { C1, C2, C3, VERDICT } from './labels';
import { overlapScore, refTokenSet, splitSentences } from './highlight';

export function specLabel(c: GoldChunk): string {
  const m = /^(ts|tr)_(\d+)_(\d+(?:-\d+)?)$/i.exec(c.spec_id || '');
  const spec = m ? `${m[1].toUpperCase()} ${m[2]}.${m[3]}` : (c.spec_id || c.chunk_id);
  return `${spec}${c.section_id ? ' · ' + c.section_id : ''}`;
}

type Form = {
  c1: string; c1_note: string; c2: string; c2_note: string; irrelevant: Set<string>; own_answer: string;
  c3: string; verdict: string; comment: string; fix_text: string;
};
const fromRating = (r: Rating | null): Form => ({
  c1: r?.c1 || '', c1_note: r?.c1_note || '', c2: r?.c2 || '', c2_note: r?.c2_note || '',
  irrelevant: new Set(r?.irrelevant_chunks || []), own_answer: r?.own_answer || '',
  c3: r?.c3 || '', verdict: r?.verdict || '', comment: r?.comment || '', fix_text: r?.fix_text || '',
});

function Radio({ name, options, value, onChange, disabled }: {
  name: string; options: { v: string; t: string; d: string }[]; value: string; onChange: (v: string) => void; disabled?: boolean;
}) {
  return (
    <div className="space-y-1.5">
      {options.map(o => (
        <label key={o.v} className={`flex items-start gap-2 rounded-md px-2 py-1.5 cursor-pointer ring-1 ${
          value === o.v ? 'ring-gray-800 bg-gray-50' : 'ring-gray-200 hover:bg-gray-50'} ${disabled ? 'opacity-60' : ''}`}>
          <input type="radio" name={name} value={o.v} checked={value === o.v} disabled={disabled}
                 onChange={() => onChange(o.v)} className="mt-1" />
          <span className="text-sm"><b>{o.t}</b><span className="text-gray-500"> — {o.d}</span></span>
        </label>
      ))}
    </div>
  );
}

function ClauseText({ text, refTokens }: { text: string; refTokens: Set<string> | null }) {
  const sentences = useMemo(() => splitSentences(text), [text]);
  if (!refTokens) return <>{text}</>;
  return <>{sentences.map((s, i) => (
    <span key={i} className={overlapScore(s, refTokens) >= 0.5 ? 'bg-yellow-100' : ''}>{s}{' '}</span>
  ))}</>;
}

/** A table the chunker had flattened to one cell per line, rebuilt from the source JSON. */
function ClauseTable({ headers, rows, refTokens }: { headers: string[]; rows: string[][]; refTokens: Set<string> | null }) {
  const cols = Math.max(headers.length, ...rows.map(r => r.length));
  return (
    <div className="my-2 overflow-x-auto whitespace-normal font-sans">
      <table className="min-w-full border-collapse text-[13px] leading-snug">
        {headers.length > 0 && (
          <thead>
            <tr>{Array.from({ length: cols }, (_, i) => (
              <th key={i} className="border border-gray-300 bg-gray-100 px-2 py-1 text-left font-semibold align-top">{headers[i] ?? ''}</th>
            ))}</tr>
          </thead>
        )}
        <tbody>
          {rows.map((r, j) => {
            const hit = refTokens ? overlapScore(r.join(' '), refTokens) >= 0.5 : false;
            return (
              <tr key={j} className={hit ? 'bg-yellow-100' : j % 2 ? 'bg-gray-50/60' : ''}>
                {Array.from({ length: cols }, (_, i) => (
                  <td key={i} className="border border-gray-200 px-2 py-1 align-top whitespace-pre-wrap break-words">{r[i] ?? ''}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Chunk({ c, idx, revealed, refTokens, irrelevant, onToggle }: {
  c: GoldChunk; idx: number; revealed: boolean; refTokens: Set<string> | null;
  irrelevant: boolean; onToggle: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);
  const text = c.content || '';
  const segs: Segment[] = c.segments?.length ? c.segments : [{ type: 'text', text }];

  // Only offer expand/collapse when the clause is actually longer than the preview.
  useLayoutEffect(() => {
    const el = bodyRef.current;
    if (el && !open) setOverflows(el.scrollHeight > el.clientHeight + 2);
  }, [text, revealed, open, c.segments]);

  const toggleBtn = 'rounded-md px-2.5 py-1 text-xs font-medium ring-1 ring-gray-300 bg-white text-gray-700 hover:bg-gray-100';
  return (
    <div className={`rounded-lg ring-1 ${irrelevant ? 'ring-red-200 bg-red-50/30' : 'ring-gray-200'} overflow-hidden`}>
      <div className="flex items-center gap-2 px-3 py-2 bg-gray-50 border-b border-gray-200 text-xs">
        <span className="font-semibold text-gray-700">Gold {idx + 1}</span>
        <span className="font-mono text-gray-700">{specLabel(c)}</span>
        {c.section_title && <span className="text-gray-600 truncate">· {c.section_title}</span>}
        <span className="ml-auto text-gray-400 shrink-0">{(c.n_chars || 0).toLocaleString('en-US')} chars</span>
        {overflows && (
          <button type="button" onClick={() => setOpen(o => !o)} className={`${toggleBtn} shrink-0`}>
            {open ? '▴ Collapse' : '▾ Expand'}
          </button>
        )}
        <label className={`flex items-center gap-2 shrink-0 cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium ring-1 select-none ${
          irrelevant ? 'bg-red-600 text-white ring-red-600' : 'bg-white text-red-700 ring-red-300 hover:bg-red-50'}`}>
          <input type="checkbox" checked={irrelevant} onChange={onToggle} className="w-4 h-4 accent-red-600" />
          Not relevant
        </label>
      </div>
      {c.missing ? (
        <div className="px-3 py-2 text-xs text-red-600">The text of this clause was not found in the knowledge graph (missing chunk).</div>
      ) : (
        <div className="relative">
          <div ref={bodyRef}
               // 10 lines of preview: line-height is 1.625em (leading-relaxed) at this font size.
               style={open ? undefined : { maxHeight: 'calc(1.625em * 10)' }}
               className="px-3 py-2 overflow-hidden text-[13px] leading-relaxed whitespace-pre-wrap break-words font-[ui-monospace,Menlo,monospace] text-gray-800">
            {segs.map((seg, k) => seg.type === 'table'
              ? <ClauseTable key={k} headers={seg.headers} rows={seg.rows} refTokens={revealed ? refTokens : null} />
              : <ClauseText key={k} text={seg.text} refTokens={revealed ? refTokens : null} />)}
          </div>
          {overflows && !open && (
            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-white to-transparent" />
          )}
        </div>
      )}
      {overflows && !c.missing && (
        <div className="px-3 py-1.5 border-t border-gray-100 bg-gray-50/60">
          <button type="button" onClick={() => setOpen(o => !o)} className="text-xs font-medium text-blue-700 hover:underline">
            {open ? '▴ Show less' : `▾ Show full clause (${(c.n_chars || 0).toLocaleString('en-US')} chars)`}
          </button>
        </div>
      )}
    </div>
  );
}

/** One question, two steps. Step-1 fields can be edited after the reveal (a
 *  rater may notice something on re-reading) but the server froze the pre-reveal
 *  answers in its event log, so the blind record survives. */
export function QuestionCard({ campaign, qid, reviewer, onBack, onNext }: {
  campaign: number; qid: string; reviewer: string; onBack: () => void; onNext: (qid: string | null) => void;
}) {
  const [v, setV] = useState<QuestionView | null>(null);
  const [f, setF] = useState<Form>(fromRating(null));
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const mountedAt = useRef(Date.now());
  const revealedAt = useRef<number | null>(null);
  const dirty = useRef(false);
  // Set only by the rater's own edits: loading a question must not create an empty draft.
  const edited = useRef(false);

  useEffect(() => {
    setV(null); setErr(''); mountedAt.current = Date.now(); revealedAt.current = null; dirty.current = false; edited.current = false;
    api.question(campaign, qid).then(q => {
      setV(q); setF(fromRating(q.rating));
      if (q.revealed) revealedAt.current = Date.now();
      window.scrollTo(0, 0);
    }).catch(e => setErr(e.message || String(e)));
  }, [campaign, qid]);

  const refTokens = useMemo(() => v?.reference_answer ? refTokenSet(v.reference_answer) : null, [v?.reference_answer]);

  const step1Ok = () => {
    if (!f.c1) return 'C1 is not answered.';
    if (f.c1 !== 'pass' && !f.c1_note.trim()) return 'C1 is not Pass — a reason is required.';
    if (!f.c2) return 'C2 is not answered.';
    if (f.own_answer.trim().length < 3) return 'Please write your own answer (required).';
    return '';
  };
  const step2Ok = () => {
    if (!f.c3) return 'C3 is not answered.';
    if (!f.verdict) return 'No verdict selected.';
    if (f.verdict !== 'keep' && !f.comment.trim()) return 'Fix / Discard requires a comment.';
    return '';
  };

  const payload = (): Partial<Rating> => {
    const r = v?.rating;
    const p1 = revealedAt.current == null
      ? (r?.t_phase1_s || 0) + (Date.now() - mountedAt.current) / 1000
      : (r?.t_phase1_s ?? null);
    const p2 = revealedAt.current != null
      ? (r?.t_phase2_s || 0) + (Date.now() - revealedAt.current) / 1000
      : (r?.t_phase2_s ?? null);
    return {
      c1: f.c1 || null, c1_note: f.c1_note || null, c2: f.c2 || null, c2_note: f.c2_note || null,
      irrelevant_chunks: [...f.irrelevant], own_answer: f.own_answer || null,
      c3: f.c3 || null, verdict: f.verdict || null, comment: f.comment || null, fix_text: f.fix_text || null,
      t_phase1_s: p1 == null ? undefined : Math.round(p1), t_phase2_s: p2 == null ? undefined : Math.round(p2),
    } as Partial<Rating>;
  };

  const save = async (status: 'draft' | 'done') => {
    setBusy(true); setErr('');
    try {
      const r = await api.save(campaign, qid, { ...payload(), status });
      setV(x => x ? { ...x, rating: r.rating } : x);
      // Timers restart from this save so the next one does not count the same seconds twice.
      if (revealedAt.current == null) mountedAt.current = Date.now(); else revealedAt.current = Date.now();
      setSavedAt(Date.now()); dirty.current = false;
      return true;
    } catch (e) { setErr(e instanceof ApiError ? e.message : String(e)); return false; }
    finally { setBusy(false); }
  };

  // Autosave a draft 2 s after the last edit; nothing is lost on a dropped tab.
  useEffect(() => {
    if (!v || !edited.current) return;
    dirty.current = true;
    const t = setTimeout(() => { if (dirty.current && !busy) save('draft'); }, 2000);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [f]);

  const reveal = async () => {
    const m = step1Ok();
    if (m) { setErr(m); return; }
    if (!(await save('draft'))) return;
    setBusy(true);
    try {
      const q = await api.reveal(campaign, qid);
      revealedAt.current = Date.now();
      setV(q);
    } catch (e) { setErr(e instanceof ApiError ? e.message : String(e)); }
    finally { setBusy(false); }
  };

  const finish = async () => {
    const m = step1Ok() || step2Ok();
    if (m) { setErr(m); return; }
    if (await save('done')) onNext(v?.next_qid ?? null);
  };

  const set = (k: keyof Form, val: string) => { edited.current = true; setF(x => ({ ...x, [k]: val })); };
  const toggleIrr = (id: string) => { edited.current = true; setF(x => {
    const s = new Set(x.irrelevant); s.has(id) ? s.delete(id) : s.add(id); return { ...x, irrelevant: s };
  }); };

  if (err && !v) return <div className="p-6 text-sm text-red-600">{err} <button onClick={onBack} className="underline ml-2">Back to queue</button></div>;
  if (!v) return <div className="p-6 text-sm text-gray-500">Loading question…</div>;
  const done = v.rating?.status === 'done';

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto p-6 space-y-5">
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <button onClick={onBack} className="underline">← Queue</button>
          <span>Question <b className="text-gray-800">{v.idx}</b> / {v.total}</span>
          {done && <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800">Done — still editable</span>}
          <span className="ml-auto">{reviewer}{savedAt ? ` · saved ${new Date(savedAt).toLocaleTimeString('en-GB')}` : ''}</span>
        </div>

        <div className="rounded-xl ring-1 ring-gray-900/10 bg-gray-50 p-4">
          <div className="text-base font-semibold text-gray-900 mb-1">Question</div>
          <div className="text-[15px] leading-relaxed">{v.question}</div>
        </div>

        <div>
          <div className="mb-3">
            <div className="text-base font-semibold text-gray-900">Gold clauses ({v.gold.length})</div>
            <div className="text-sm text-gray-600 mt-0.5">Tick <b className="text-red-700">Not relevant</b> on any clause that does not help answer the question.</div>
          </div>
          <div className="space-y-3">
            {v.gold.map((c, i) => (
              <Chunk key={c.chunk_id} c={c} idx={i} revealed={v.revealed} refTokens={refTokens}
                     irrelevant={f.irrelevant.has(c.chunk_id)} onToggle={() => toggleIrr(c.chunk_id)} />
            ))}
          </div>
        </div>

        <section className="rounded-xl ring-1 ring-gray-200 p-4 space-y-4">
          <h2 className="text-sm font-semibold">Step 1 · before seeing the reference answer</h2>
          <div>
            <div className="text-sm font-medium mb-1.5">C1 · Is the question valid?</div>
            <Radio name="c1" options={C1} value={f.c1} onChange={x => set('c1', x)} />
            {f.c1 && f.c1 !== 'pass' && (
              <textarea value={f.c1_note} onChange={e => set('c1_note', e.target.value)} rows={2}
                        placeholder="Reason (required)" className="mt-2 w-full rounded-md border border-gray-300 px-3 py-2 text-sm" />
            )}
          </div>
          <div>
            <div className="text-sm font-medium mb-1.5">C2 · Do the gold clauses contain the answer?</div>
            <Radio name="c2" options={C2} value={f.c2} onChange={x => set('c2', x)} />
            <textarea value={f.c2_note} onChange={e => set('c2_note', e.target.value)} rows={2}
                      placeholder="Note (optional): which clause is missing or superfluous, what part of the answer is missing"
                      className="mt-2 w-full rounded-md border border-gray-300 px-3 py-2 text-sm" />
          </div>
          <div>
            <div className="text-sm font-medium mb-1.5">Your own answer (from the gold clauses) <span className="text-red-600">*</span></div>
            <textarea value={f.own_answer} onChange={e => set('own_answer', e.target.value)} rows={3}
                      placeholder='Answer briefly from the gold clauses. If they do not answer the question, write "cannot be answered from gold" and why.'
                      className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm" />
          </div>
          {!v.revealed && (
            <div className="flex items-center gap-3">
              <button onClick={reveal} disabled={busy}
                      className="rounded-md bg-gray-900 text-white text-sm px-4 py-2 disabled:opacity-40">
                Show reference answer →
              </button>
              <button onClick={() => save('draft')} disabled={busy} className="text-sm text-gray-600 underline">Save draft</button>
            </div>
          )}
        </section>

        {v.revealed && (
          <section className="rounded-xl ring-1 ring-gray-200 p-4 space-y-4 animate-fade-in-up">
            <h2 className="text-sm font-semibold">Step 2 · reference answer</h2>
            <div className="rounded-lg bg-yellow-50 ring-1 ring-yellow-200 p-3 text-sm leading-relaxed whitespace-pre-wrap">
              {v.reference_answer}
            </div>
            <p className="text-xs text-gray-500">Sentences in the gold clauses that share wording with the reference answer are highlighted above — a reading aid, not a judgement.</p>
            <div>
              <div className="text-sm font-medium mb-1.5">C3 · Is the reference answer correct with respect to the gold clauses?</div>
              <Radio name="c3" options={C3} value={f.c3} onChange={x => set('c3', x)} />
            </div>
            <div>
              <div className="text-sm font-medium mb-1.5">Verdict for this question</div>
              <Radio name="verdict" options={VERDICT} value={f.verdict} onChange={x => set('verdict', x)} />
            </div>
            <textarea value={f.comment} onChange={e => set('comment', e.target.value)} rows={2}
                      placeholder={f.verdict && f.verdict !== 'keep' ? 'Comment (required for Fix / Discard)' : 'Comment (optional)'}
                      className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm" />
            <textarea value={f.fix_text} onChange={e => set('fix_text', e.target.value)} rows={2}
                      placeholder="Suggested fix (optional): what the question / gold / answer should be"
                      className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm" />
            <div className="flex items-center gap-3">
              <button onClick={finish} disabled={busy}
                      className="rounded-md bg-emerald-700 text-white text-sm px-4 py-2 disabled:opacity-40">
                {done ? 'Save changes' : 'Complete'} {v.next_qid && !done ? '& next →' : ''}
              </button>
              <button onClick={() => save('draft')} disabled={busy} className="text-sm text-gray-600 underline">Save draft</button>
              {done && v.next_qid && <button onClick={() => onNext(v.next_qid)} className="text-sm text-gray-600 underline">Next →</button>}
            </div>
          </section>
        )}
        {err && <div className="text-sm text-red-600">{err}</div>}
      </div>
    </div>
  );
}
