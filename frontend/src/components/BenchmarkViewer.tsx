import React, { useEffect, useMemo, useState } from 'react';
import { QuestionMix } from './QuestionMix';

interface FileMeta {
  path: string;
  kind: 'questions' | 'run';
  n: number;
  name: string;
  size: number;
  mtime?: number;
}

// Fields rendered, by style. TEXT = multi-line prose, LIST = chips.
// `explanation` is TeleQnA's own rationale for the correct option.
const TEXT_FIELDS = ['question', 'reference_answer', 'explanation', 'answer'];
const LIST_FIELDS = ['short_answers', 'cover_targets', 'gold_chunk_ids', 'gold_spec_ids', 'chunk_ids', 'structure_tags'];

// Facets offered as filter chips. Types are mutually exclusive per question (OR
// within the group); structure tags stack, so selecting two means "has both".
// Derived from the loaded file, not hardcoded: the sets do not share one fixed
// vocabulary (kg_win_100 adds message / sbi_operation / information_element), and a
// fixed list silently hides every class it does not name.
const facetsOf = (items: any[], field: string, list: boolean) => {
  const c = new Map<string, number>();
  for (const it of items) {
    const v = list ? (it?.[field] || []) : [it?.[field]];
    for (const x of v) if (x) c.set(String(x), (c.get(String(x)) || 0) + 1);
  }
  return [...c.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([k]) => k);
};

// Structure facets offered as chips. Kept in step with RunExplorer.tsx — these three
// are the ones `scripts/tag_structure.py` derives with a rule verified against the v2
// labels, so they mean the same thing on every set.
const TAG_FACETS = ['multi_hop', 'cross_specification', 'cross_section'];

// Files arrive newest-first from /api/benchmarks (sorted by mtime, deliberately
// NOT by path -- a set authored today otherwise hides under whichever directory
// starts with an earlier letter). Nothing on screen showed that ordering, so the
// list read as arbitrary; this renders the age the sort is actually based on.
function fmtAge(mtime?: number): string {
  if (!mtime) return '';
  const mins = Math.floor((Date.now() / 1000 - mtime) / 60);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return days < 7 ? `${days}d ago` : new Date(mtime * 1000).toLocaleDateString();
}

function fmtSize(n: number) {
  return n > 1e6 ? (n / 1e6).toFixed(1) + ' MB' : (n / 1e3).toFixed(0) + ' KB';
}

// The URL is the single source of truth, mirroring the Run Explorer: every question
// is a real page — shareable, reloadable, and Back steps through the tree.
type Sel = { path: string; qid: string };
const readUrl = (): Sel => {
  const p = new URLSearchParams(location.search);
  return { path: p.get('path') || '', qid: p.get('qid') || '' };
};
const toQuery = (s: Sel) => {
  const p = new URLSearchParams();
  (['path', 'qid'] as const).forEach((k) => s[k] && p.set(k, s[k]));
  const q = p.toString();
  return location.pathname + (q ? '?' + q : '');
};

const qidOf = (it: any, i: number) => String(it?.qid ?? `#${i}`);

export function BenchmarkViewer() {
  const [files, setFiles] = useState<FileMeta[]>([]);
  const [sel, setSel] = useState<Sel>(readUrl);
  const [kind, setKind] = useState<'questions' | 'run' | ''>('');
  const [meta, setMeta] = useState<any>(null);
  const [items, setItems] = useState<any[]>([]);
  const [query, setQuery] = useState('');
  const [classes, setClasses] = useState<string[]>([]);
  const [types, setTypes] = useState<string[]>([]);
  const [tags, setTags] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  // On phones the tree is a slide-over; from md up it is a permanent column.
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    const onPop = () => setSel(readUrl());
    addEventListener('popstate', onPop);
    return () => removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    fetch('/api/benchmarks').then((r) => r.json()).then(setFiles).catch(() => setFiles([]));
  }, []);

  // Load whichever file the URL names — including on first paint, so a deep link works.
  useEffect(() => {
    if (!sel.path) { setItems([]); setMeta(null); setKind(''); return; }
    let alive = true;
    setLoading(true); setQuery('');
    fetch('/api/benchmarks/file?path=' + encodeURIComponent(sel.path))
      .then((r) => r.json())
      .then((d) => {
        if (!alive) return;
        setKind(d.kind);
        if (d.kind === 'questions') { setMeta(d.content.meta || null); setItems(d.content.questions || []); }
        else { setMeta(null); setItems(Array.isArray(d.content) ? d.content : []); }
      })
      .catch(() => { if (alive) { setItems([]); setMeta(null); } })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [sel.path]);

  function go(next: Partial<Sel>, closeNav = false) {
    setSel((prev) => {
      const s: Sel = { ...prev, ...next };
      if (next.path !== undefined && next.path !== prev.path) s.qid = '';
      const url = toQuery(s);
      if (url !== location.pathname + location.search) history.pushState(s, '', url);
      return s;
    });
    if (closeNav) setNavOpen(false);
  }

  const toggle = (v: string, xs: string[], set: (x: string[]) => void) =>
    set(xs.includes(v) ? xs.filter((x) => x !== v) : [...xs, v]);

  // `files` is already newest-first, and inserting in that order keeps each group
  // internally sorted -- but Object.entries would then order the GROUPS by whichever
  // happened to be seen first, which is only newest-first by luck. Sort the groups
  // explicitly by their newest member.
  const grouped = useMemo(() => {
    const g: Record<string, FileMeta[]> = {};
    for (const f of files) (g[f.path.split('/')[0]] ||= []).push(f);
    const newest = (fs: FileMeta[]) => Math.max(...fs.map((f) => f.mtime || 0));
    return Object.fromEntries(
      Object.entries(g).sort((a, b) => newest(b[1]) - newest(a[1]))
    );
  }, [files]);

  // Facet counts come from the whole file, not the filtered view, so a chip never
  // reads "0" merely because another chip is already on.
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const it of items) {
      if (it.question_class) c[it.question_class] = (c[it.question_class] || 0) + 1;
      if (it.question_type) c[it.question_type] = (c[it.question_type] || 0) + 1;
      if (it.question_subclass && it.question_subclass !== it.question_type) {
        c[it.question_subclass] = (c[it.question_subclass] || 0) + 1;
      }
      for (const t of it.structure_tags || []) c[t] = (c[t] || 0) + 1;
    }
    return c;
  }, [items]);

  // Which sub-axis actually splits the selected classes. `question_type` divides factoid
  // into its mechanisms but is a single constant inside procedure and requirement;
  // `question_subclass` divides those two (2 and 7 groups). Pick whichever yields more
  // groups so every class gets a usable second level rather than one dead chip. Sets that
  // carry neither field fall through to an empty list and the row hides itself.
  const sub = useMemo(() => {
    if (!classes.length) return { field: '', values: [] as string[] };
    const tally = (f: string) => {
      const c: Record<string, number> = {};
      for (const it of items) {
        const v = it?.[f];
        if (classes.includes(it?.question_class || '') && v) c[v] = (c[v] || 0) + 1;
      }
      return c;
    };
    const byType = tally('question_type');
    const bySub = tally('question_subclass');
    const field = Object.keys(bySub).length > Object.keys(byType).length
      ? 'question_subclass' : 'question_type';
    const c = field === 'question_subclass' ? bySub : byType;
    return { field, values: Object.keys(c).sort((a, b) => c[b] - c[a]) };
  }, [items, classes]);

  const shown = useMemo(() => {
    const all = items.map((it, i) => ({ it, i, qid: qidOf(it, i) }));
    const q = query.trim().toLowerCase();
    return all.filter(({ it, qid }) => {
      if (classes.length && !classes.includes(it.question_class)) return false;
      if (types.length && !types.includes(sub.field ? it[sub.field] : it.question_type)) return false;
      const st: string[] = it.structure_tags || [];
      if (tags.length && !tags.every((t) => st.includes(t))) return false;
      if (q && !(qid + ' ' + (it.question || '')).toLowerCase().includes(q)
            && !JSON.stringify(it).toLowerCase().includes(q)) return false;
      return true;
    });
  }, [items, query, classes, types, tags, sub]);

  const current = useMemo(
    () => items.map((it, i) => ({ it, qid: qidOf(it, i) })).find((x) => x.qid === sel.qid) || null,
    [items, sel.qid],
  );
  const pos = shown.findIndex((x) => x.qid === sel.qid);

  const tree = (
    <nav className="h-full overflow-y-auto overscroll-contain pb-24">
      <div className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-stone-400">
        Question sets ({files.length})
      </div>
      {Object.entries(grouped).map(([top, fs]) => (
        <div key={top} className="px-2 pb-1">
          <div className="px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-stone-400">{top}</div>
          {fs.map((f) => {
            const open = f.path === sel.path;
            return (
              <div key={f.path}>
                <button
                  onClick={() => go({ path: open ? '' : f.path })}
                  className={`w-full text-left rounded-lg px-3 py-2.5 transition
                              ${open ? 'bg-stone-200/70' : 'hover:bg-stone-100'}`}
                >
                  <span className="flex items-center gap-2">
                    <span className="text-stone-400 text-xs w-3">{open ? '▾' : '▸'}</span>
                    <span className="font-medium text-[13px] truncate">{f.path.replace(top + '/', '')}</span>
                  </span>
                  <span className="block pl-5 text-[11px] text-stone-500">
                    {f.n} questions · {fmtSize(f.size)}
                    {f.mtime ? <> · <span className="text-stone-400">{fmtAge(f.mtime)}</span></> : null}
                    {files[0]?.path === f.path ? (
                      <span className="ml-1.5 rounded bg-emerald-100 px-1 py-px text-[10px]
                                       font-medium text-emerald-700">newest</span>
                    ) : null}
                  </span>
                </button>

                {open && (
                  <div className="pl-3 pr-1 pb-2">
                    <input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="filter questions…"
                      className="w-full my-2 px-3 py-2 text-[13px] rounded-lg bg-white
                                 ring-1 ring-stone-200 focus:ring-stone-400 outline-none"
                    />
                    {/* class first, subject nested under it. A set with no
                        `question_class` (the older ones) shows no class row and falls
                        back to the flat type row below, so both vocabularies still work. */}
                    <FacetRow label="class" facets={facetsOf(items, "question_class", false)}
                              active={classes} counts={counts}
                              onToggle={(v) => {
                                // Dropping a class drops the subject chips that belonged
                                // to it — a stale one keeps filtering against a class no
                                // longer selected and the list silently empties.
                                const next = classes.includes(v)
                                  ? classes.filter(x => x !== v) : [...classes, v];
                                setClasses(next);
                                if (!next.length) setTypes([]);
                              }} />
                    {classes.length
                      ? (sub.values.length > 1 && (
                          <FacetRow label="subject" facets={sub.values} active={types} counts={counts}
                                    onToggle={(v) => toggle(v, types, setTypes)}
                                    hint="groups inside the selected class" />))
                      : (!facetsOf(items, "question_class", false).length && (
                          <FacetRow label="type" facets={facetsOf(items, "question_type", false)}
                                    active={types} counts={counts}
                                    onToggle={(v) => toggle(v, types, setTypes)} />))}
                    {/* The three derived tags, same as the Run Explorer — not everything
                        `structure_tags` happens to hold. Four other values live in there
                        (normative_requirement, procedure_flow, standardized_value,
                        multi_chunk_aggregation): they were written by the authoring pass
                        of one question set, restate a class the class row already offers,
                        and cover 17-100 questions each. Deriving the row from the data
                        surfaced all seven and buried the three that carry the structural
                        claim. */}
                    <FacetRow label="structure" facets={TAG_FACETS} active={tags} counts={counts}
                              onToggle={(v) => toggle(v, tags, setTags)} hint="all selected tags must be present" />
                    <div className="flex items-center gap-2 py-1 px-1">
                      <span className="text-[11px] text-stone-400">{shown.length}/{items.length}</span>
                      {(classes.length || types.length || tags.length || query) ? (
                        <button
                          onClick={() => { setClasses([]); setTypes([]); setTags([]); setQuery(''); }}
                          className="text-[11px] text-stone-500 underline hover:text-stone-800"
                        >clear</button>
                      ) : null}
                    </div>

                    {loading && <div className="px-1 py-2 text-[12px] text-stone-400">Loading…</div>}
                    {shown.map(({ it, qid }) => (
                      <button
                        key={qid}
                        onClick={() => go({ qid }, true)}
                        className={`block w-full text-left rounded-lg px-3 py-2 mb-0.5 transition
                                    ${qid === sel.qid ? 'bg-blue-100' : 'hover:bg-stone-100'}`}
                      >
                        <span className="flex items-center gap-2">
                          <span className="font-mono text-[11px] text-stone-500">{qid}</span>
                          {it.question_type && (
                            <span className="text-[10px] text-stone-400">{it.question_type}</span>
                          )}
                        </span>
                        <span className="block text-[12px] leading-snug text-stone-600">
                          {(it.question || '').length > 90 ? it.question.slice(0, 90) + '…' : it.question}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </nav>
  );

  return (
    <div className="h-full flex flex-col bg-[#faf9f7] text-stone-900">
      {/* Mobile top bar — the tree is a drawer here, so it needs its own opener */}
      <div className="md:hidden flex items-center gap-2 px-3 py-2
                      bg-[#faf9f7]/95 backdrop-blur border-b border-stone-200">
        <button
          onClick={() => setNavOpen(true)}
          className="px-3 py-2 rounded-lg ring-1 ring-stone-200 bg-white text-sm shrink-0"
        >☰ Questions</button>
        <div className="min-w-0 text-xs text-stone-500 truncate font-mono">
          {sel.qid || sel.path || 'pick a question set'}
        </div>
      </div>

      <div className="flex-1 flex min-h-0">
        <aside className="hidden md:block w-[19rem] shrink-0 border-r border-stone-200 bg-stone-50">
          {tree}
        </aside>
        {navOpen && (
          <div className="md:hidden fixed inset-0 z-40 flex">
            <div className="w-[85%] max-w-sm bg-stone-50 shadow-xl">{tree}</div>
            <button className="flex-1 bg-black/30" aria-label="Close menu" onClick={() => setNavOpen(false)} />
          </div>
        )}

        <main className="flex-1 min-w-0 overflow-y-auto">
          {!sel.path && (
            <div className="p-6 md:p-10 text-stone-500 text-sm max-w-prose">
              Pick a question set on the left, then a question — only the selected one is shown here.
            </div>
          )}

          {sel.path && !sel.qid && (
            <div className="max-w-3xl mx-auto px-4 md:px-8 py-5 md:py-8 space-y-3">
              <h1 className="text-lg md:text-xl font-semibold">{meta?.name || sel.path}</h1>
              <p className="text-[13px] text-stone-500">
                {kind} · {items.length} questions
                {shown.length !== items.length ? ` · ${shown.length} match the current filter` : ''}
              </p>
              {meta?.description && (
                <div className="text-[13px] text-stone-600 bg-white ring-1 ring-stone-200 rounded-lg px-3 py-2">
                  {meta.description}
                </div>
              )}
              {kind === 'questions' && <QuestionMix items={items} />}
              <p className="text-sm text-stone-400">Pick a question in the tree to read it.</p>
            </div>
          )}

          {sel.qid && !current && !loading && (
            <div className="p-6 md:p-10 text-sm text-rose-700">
              No question <span className="font-mono">{sel.qid}</span> in this file.
            </div>
          )}

          {current && (
            <article className="max-w-3xl mx-auto px-4 md:px-8 py-5 md:py-8 space-y-4">
              <header className="space-y-2">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] font-mono text-stone-500">
                  <span>{sel.path}</span>
                  <span className="text-stone-300">/</span>
                  <span className="px-1.5 py-0.5 rounded bg-stone-100">{current.qid}</span>
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  {current.it.question_type && <Badge color="emerald">{current.it.question_type}</Badge>}
                  {current.it.relation_pattern && <Badge color="gray">{current.it.relation_pattern}</Badge>}
                  {(current.it.structure_tags || []).map((t: string) => <Badge key={t} color="gray">{t}</Badge>)}
                  {current.it.origin && <Badge color="gray">{current.it.origin}</Badge>}
                  {typeof current.it.faithfulness === 'number' && (
                    <Badge color="amber">faith {current.it.faithfulness.toFixed(2)}</Badge>
                  )}
                  {current.it.error && <Badge color="red">error</Badge>}
                  <span className="ml-auto flex items-center gap-1 text-[11px] text-stone-400">
                    {pos >= 0 && <span>{pos + 1}/{shown.length}</span>}
                    <NavBtn disabled={pos <= 0} onClick={() => go({ qid: shown[pos - 1].qid })}>←</NavBtn>
                    <NavBtn disabled={pos < 0 || pos >= shown.length - 1} onClick={() => go({ qid: shown[pos + 1].qid })}>→</NavBtn>
                  </span>
                </div>
              </header>

              {orderedKeys(current.it).map((key) => (
                <Field key={key} k={key} value={current.it[key]} gold={current.it.gold_chunk_ids} />
              ))}
            </article>
          )}
        </main>
      </div>
    </div>
  );
}

function NavBtn({ disabled, onClick, children }: { disabled: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className={`px-2 py-0.5 rounded ring-1 ring-stone-200 bg-white
                  ${disabled ? 'opacity-40 cursor-default' : 'hover:bg-stone-100'}`}
    >{children}</button>
  );
}

function FacetRow({ label, facets, active, counts, onToggle, hint }: {
  label: string; facets: string[]; active: string[];
  counts: Record<string, number>; onToggle: (v: string) => void; hint?: string;
}) {
  // Hide a facet group the loaded file has no values for (e.g. short_answers sets).
  const present = facets.filter((f) => counts[f]);
  if (!present.length) return null;
  return (
    <div className="mb-1.5" title={hint}>
      <div className="px-1 text-[10px] uppercase tracking-wide text-stone-400">{label}</div>
      <div className="flex flex-wrap gap-1 mt-0.5">
        {present.map((f) => {
          const on = active.includes(f);
          return (
            <button
              key={f}
              onClick={() => onToggle(f)}
              className={`px-2 py-0.5 rounded-full text-[11px] ring-1 transition
                          ${on ? 'bg-blue-50 text-blue-700 ring-blue-300'
                               : 'bg-white text-stone-500 ring-stone-200 hover:bg-stone-100'}`}
            >
              {f} <span className="text-stone-400">{counts[f]}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Badge({ children, color }: { children: React.ReactNode; color: 'emerald' | 'amber' | 'gray' | 'red' }) {
  const c = color === 'emerald' ? 'bg-emerald-50 text-emerald-700 border-emerald-100'
    : color === 'amber' ? 'bg-amber-50 text-amber-700 border-amber-100'
    : color === 'red' ? 'bg-red-50 text-red-700 border-red-100'
    : 'bg-stone-100 text-stone-600 border-stone-200';
  return <span className={`text-[11px] px-1.5 py-0.5 rounded border ${c}`}>{children}</span>;
}

function orderedKeys(it: any): string[] {
  const pref = [...TEXT_FIELDS, ...LIST_FIELDS];
  return Object.keys(it)
    .filter((k) => (TEXT_FIELDS.includes(k) || LIST_FIELDS.includes(k)) && it[k] != null
      && !(Array.isArray(it[k]) && it[k].length === 0) && it[k] !== '')
    .sort((a, b) => pref.indexOf(a) - pref.indexOf(b));
}

function Field({ k, value, gold }: { k: string; value: any; gold?: string[] }) {
  const isList = LIST_FIELDS.includes(k);
  const goldSet = new Set(gold || []);
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-stone-400 mb-1">{k.replace(/_/g, ' ')}</div>
      {isList ? (
        <div className="flex flex-wrap gap-1">
          {(value as any[]).map((v, i) => (
            <span key={i}
              title={k === 'chunk_ids' ? 'click to copy' : undefined}
              onClick={k === 'chunk_ids' ? () => navigator.clipboard?.writeText(String(v)) : undefined}
              className={`text-[11px] font-mono px-1.5 py-0.5 rounded border ${
                k === 'chunk_ids' && goldSet.has(v)
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200 cursor-pointer'
                  : k === 'chunk_ids'
                  ? 'bg-white text-stone-600 border-stone-200 cursor-pointer hover:bg-stone-100'
                  : 'bg-white text-stone-600 border-stone-200'}`}>
              {String(v)}
            </span>
          ))}
        </div>
      ) : (
        <div className={`text-[15px] leading-relaxed text-stone-800 whitespace-pre-wrap
                         ${k === 'question' ? 'font-medium' : ''}
                         ${k === 'reference_answer' || k === 'answer' ? 'bg-white ring-1 ring-stone-200 rounded-lg p-3' : ''}`}>
          {String(value)}
        </div>
      )}
    </div>
  );
}
