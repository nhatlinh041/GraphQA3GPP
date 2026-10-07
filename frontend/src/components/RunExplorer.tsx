import React, { useEffect, useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { ThinkingTrail } from './ThinkingTrail';
import { RunOverview } from './RunOverview';

// Runs arrive newest-first (the API sorts by directory mtime, deliberately NOT by
// name -- "v10_..." sorts below "v9_..." lexically, which once buried the newest
// run at position 6 of 11). Nothing on screen showed that ordering though, so the
// list looked arbitrary; this renders the age that the sort is actually based on.
function runAge(mtime: number): string {
  if (!mtime) return '';
  const mins = Math.floor((Date.now() / 1000 - mtime) / 60);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return days < 7 ? `${days}d ago` : new Date(mtime * 1000).toLocaleDateString();
}

interface RunMeta { run: string; n_configs: number; n_questions: number; mtime: number; kind?: string }
interface ConfigMeta {
  config: string; n: number; hit5: number | null; faith: number | null;
  lat_p50: number | null; empty: number;
  inherited?: boolean; source_run?: string;
  mcq_acc?: number | null;
}
interface QItem {
  qid: string; question: string; question_type: string; question_class?: string;
  question_subclass?: string; structure_tags: string[];
  hit5: boolean; n_chunks: number; latency: number | null;
  faithfulness: number | null; error: string | null; repatched?: string;
  // TeleQnA runs only (multiple choice, no gold chunks); null on paper_bench runs.
  mcq_correct?: boolean | null; mcq_pred?: string | null; mcq_gold?: string | null;
}
interface Sibling {
  config: string; hit5: boolean; n_chunks: number;
  latency: number | null; faithfulness: number | null; answer_chars: number;
}

// The URL is the single source of truth, so every question is a real page: shareable,
// reloadable, and the browser Back button steps through the tree.
type Sel = { run: string; config: string; qid: string };
const readUrl = (): Sel => {
  const p = new URLSearchParams(location.search);
  return { run: p.get('run') || '', config: p.get('config') || '', qid: p.get('qid') || '' };
};
const toQuery = (s: Sel) => {
  const p = new URLSearchParams();
  (['run', 'config', 'qid'] as const).forEach(k => s[k] && p.set(k, s[k]));
  const q = p.toString();
  return location.pathname + (q ? '?' + q : '');
};

// Run header line. Configs of one run need not share a question count — a TeleQnA run
// mixes full-10k configs with Standards-only ones — so "N questions each" is said only
// when it is true, and the xlsx layout only for paper_bench runs, which it describes.
const runSummary = (ov: any): string => {
  const cfgs: any[] = ov.configs || [];
  const byN = new Map<number, number>();
  cfgs.forEach(c => byN.set(c.n ?? 0, (byN.get(c.n ?? 0) || 0) + 1));
  const fmt = (n: number) => n.toLocaleString('en-US');
  const counts = byN.size <= 1
    ? `${fmt(cfgs[0]?.n ?? 0)} questions each`
    : [...byN].sort((a, b) => b[0] - a[0]).map(([n, k]) => `${k} on ${fmt(n)} questions`).join(', ');
  const tele = cfgs.some(c => c.mcq_acc != null);
  return `${cfgs.length} configs · ${counts} · ` + (tele
    ? `TeleQnA multiple-choice${ov.meta?.n_questions ? ` (${fmt(ov.meta.n_questions)}-question set)` : ''}`
    : 'laid out as the five sheets of Results_RAG_3GPP.xlsx');
};

const num = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined ? '—' : v.toFixed(d);
const DEBUG = new URLSearchParams(location.search).has('debug');

// Facets offered as filter chips, same vocabulary as the Benchmark Viewer. A question
// carries exactly one type (so the group reads as OR) but any number of structure
// tags (so selecting two means "has both").
// Two axes, nested rather than side by side. `question_class` is the evaluation class
// and every question has exactly one; `question_type` is the subject, and in
// kg_bench_3000 the 1000 factoid questions carry a MECHANISM there (message,
// sbi_operation, ...) rather than the word "factoid". Offering both as flat rows made
// the class unselectable — picking `message` and picking `factoid` looked like sibling
// choices when one is inside the other. The subject row now appears only after a class
// is picked, and lists the types that class actually contains.
// TeleQnA subjects ride on the same axis (teleqna_to_run.py writes the subject as
// question_class); FacetRow hides whichever of these the loaded config does not have.
const CLASS_FACETS = ['factoid', 'procedure', 'requirement',
  'Lexicon', 'Research overview', 'Research publications',
  'Standards overview', 'Standards specifications'];
const TAG_FACETS = ['multi_hop', 'cross_specification', 'cross_section'];

function FacetRow({ label, facets, active, counts, onToggle, hint }: {
  label: string; facets: string[]; active: string[];
  counts: Record<string, number>; onToggle: (v: string) => void; hint?: string;
}) {
  // Hide a group the loaded config has no values for, rather than showing dead chips.
  const present = facets.filter(f => counts[f]);
  if (!present.length) return null;
  return (
    <div className="mb-1.5" title={hint}>
      <div className="px-1 text-[10px] uppercase tracking-wide text-stone-400">{label}</div>
      <div className="flex flex-wrap gap-1 mt-0.5">
        {present.map(f => {
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

function Tag({ tone = 'plain', children }: { tone?: 'plain' | 'ok' | 'warn' | 'bad'; children: React.ReactNode }) {
  const c = tone === 'ok' ? 'bg-emerald-50 text-emerald-700 ring-emerald-200'
    : tone === 'warn' ? 'bg-amber-50 text-amber-700 ring-amber-200'
    : tone === 'bad' ? 'bg-rose-50 text-rose-700 ring-rose-200'
    : 'bg-stone-100 text-stone-600 ring-stone-200';
  return <span className={`px-2 py-0.5 rounded-full text-[11px] ring-1 ${c}`}>{children}</span>;
}

/** One judge verdict, as produced by judge_correctness.py and served from the
 *  `<config>.judge.json` sidecar. Absent (not null) when the item is unjudged. */
type Judge = {
  judge_label?: 'CORRECT' | 'PARTIAL' | 'INCORRECT' | 'ABSTAIN';
  judge_key_facts_stated?: number;
  judge_key_facts_total?: number;
  judge_rationale?: string;
  judge_contradiction?: string | null;
  judge_error?: string | null;
  model?: string;
  /** The exact prompt the judge was given, rebuilt server-side from the rubric. */
  prompt?: string | null;
};

/** ABSTAIN is deliberately NEUTRAL, not red. kg_only declines on about a fifth of
 *  the set; colouring a refusal as an error would read as "wrong on 20%" when the
 *  system in fact asserted nothing — a materially different failure mode. */
const JUDGE_TONE: Record<string, 'ok' | 'warn' | 'bad' | 'plain'> = {
  CORRECT: 'ok', PARTIAL: 'warn', INCORRECT: 'bad', ABSTAIN: 'plain',
};


export function RunExplorer() {
  const [runs, setRuns] = useState<RunMeta[]>([]);
  const [sel, setSel] = useState<Sel>(readUrl);
  const [configs, setConfigs] = useState<ConfigMeta[]>([]);
  const [items, setItems] = useState<QItem[]>([]);
  const [detail, setDetail] = useState<any>(null);
  const [stages, setStages] = useState<any[] | null>(null);
  // Judge verdict for the open question. Null = not judged; the backend omits the
  // key entirely rather than sending nulls, so this stays falsy for unjudged items.
  const [judge, setJudge] = useState<Judge | null>(null);
  const [judgeOpen, setJudgeOpen] = useState(false);
  const [siblings, setSiblings] = useState<Sibling[]>([]);
  const [ov, setOv] = useState<any>(null);
  const [q, setQ] = useState('');
  const [onlyMiss, setOnlyMiss] = useState(false);
  const [classes, setClasses] = useState<string[]>([]);
  const [types, setTypes] = useState<string[]>([]);
  const [tags, setTags] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  // On phones the tree is a slide-over; from md up it is a permanent column.
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    const onPop = () => setSel(readUrl());
    addEventListener('popstate', onPop);
    return () => removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    // Re-poll instead of fetching once on mount. A benchmark run takes hours and
    // creates its directory at the START, so a page opened before it launched would
    // never list it until a manual reload — which reads as "the run is missing".
    // Only the run LIST is polled; the heavy per-config payloads below stay
    // fetch-on-select.
    const load = () =>
      fetch('/api/runs').then(r => r.json()).then((rs: RunMeta[]) => {
        setRuns(rs);
        setSel(s => (s.run || !rs.length ? s : { ...s, run: rs[0].run }));
      }).catch(() => setRuns([]));
    load();
    const id = setInterval(load, 30_000);
    const onFocus = () => load();
    addEventListener('focus', onFocus);
    return () => { clearInterval(id); removeEventListener('focus', onFocus); };
  }, []);

  useEffect(() => {
    if (!sel.run) { setOv(null); return; }
    fetch(`/api/runs/${encodeURIComponent(sel.run)}/overview`)
      .then(r => r.json()).then(setOv).catch(() => setOv(null));
  }, [sel.run]);

  useEffect(() => {
    if (!sel.run) { setConfigs([]); return; }
    fetch(`/api/runs/${encodeURIComponent(sel.run)}`)
      .then(r => r.json()).then(d => setConfigs(d.configs || [])).catch(() => setConfigs([]));
  }, [sel.run]);

  useEffect(() => {
    if (!sel.run || !sel.config) { setItems([]); return; }
    fetch(`/api/runs/${encodeURIComponent(sel.run)}/${encodeURIComponent(sel.config)}`)
      .then(r => r.json()).then(d => setItems(d.items || [])).catch(() => setItems([]));
  }, [sel.run, sel.config]);

  useEffect(() => {
    if (!sel.run || !sel.config || !sel.qid) {
      setDetail(null); setStages(null); setSiblings([]); setErr(null); return;
    }
    const url = `/api/runs/${encodeURIComponent(sel.run)}/${encodeURIComponent(sel.config)}/${encodeURIComponent(sel.qid)}`;
    // `alive` guards against a slow response for a question the user already left.
    let alive = true;
    setBusy(true); setErr(null);
    fetch(url)
      .then(async r => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${url}`);
        return r.json();
      })
      .then(d => {
        if (!alive) return;
        if (!d?.record) throw new Error(`response has no "record" field — ${url}`);
        setDetail(d.record); setStages(d.stages ?? null); setSiblings(d.siblings || []);
        setJudge(d.judge ?? null); setJudgeOpen(false);
      })
      .catch(e => {
        if (!alive) return;
        setDetail(null); setStages(null); setSiblings([]); setJudge(null);
        setErr(String(e?.message || e));
      })
      .finally(() => { if (alive) setBusy(false); });
    return () => { alive = false; };
  }, [sel.run, sel.config, sel.qid]);

  function go(next: Partial<Sel>, closeNav = false) {
    setSel(prev => {
      const s: Sel = { ...prev, ...next };
      // A cross-run page may jump straight to a config of another run, so an explicit
      // config/qid in the same call survives the run change.
      if (next.run !== undefined && next.run !== prev.run) { s.config = next.config ?? ''; s.qid = next.qid ?? ''; }
      if (next.config !== undefined && next.config !== prev.config) s.qid = '';
      const url = toQuery(s);
      if (url !== location.pathname + location.search) history.pushState(s, '', url);
      return s;
    });
    if (closeNav) setNavOpen(false);
  }

  const toggle = (v: string, xs: string[], set: (x: string[]) => void) =>
    set(xs.includes(v) ? xs.filter(x => x !== v) : [...xs, v]);

  // Counts come from the whole config, not the filtered view, so a chip never reads
  // "0" merely because another chip is already on.
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

  // Subject types the SELECTED classes actually contain, ordered by frequency. Derived
  // from the loaded config rather than hard-coded: the set differs per question set, and
  // a hard-coded list would show dead chips on one set and hide real types on another.
  // Which of the two sub-axes actually splits the selected classes. `question_type`
  // divides factoid into five mechanisms but is a single constant inside procedure and
  // requirement; `question_subclass` divides those two (2 and 7 groups) while being
  // near-synonymous with type inside factoid. Pick whichever yields more groups, so
  // every class gets a usable second level instead of one dead chip.
  const sub = useMemo(() => {
    if (!classes.length) return { field: '' as 'question_type' | 'question_subclass' | '', values: [] as string[] };
    const tally = (f: 'question_type' | 'question_subclass') => {
      const c: Record<string, number> = {};
      for (const it of items) {
        const v = it[f];
        if (classes.includes(it.question_class || '') && v) c[v] = (c[v] || 0) + 1;
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

  // A TeleQnA config has no gold chunks, so "hit" means the multiple-choice answer
  // was right; everywhere else it means gold@5.
  const isMcq = useMemo(() => items.some(x => x.mcq_correct != null), [items]);
  const okOf = (x: QItem) => (isMcq ? !!x.mcq_correct : x.hit5);

  const shown = useMemo(() => {
    let xs = items;
    if (onlyMiss) xs = xs.filter(x => !okOf(x));
    if (classes.length) xs = xs.filter(x => classes.includes(x.question_class || ''));
    if (types.length) xs = xs.filter(x => types.includes((sub.field ? x[sub.field] : x.question_type) || ''));
    if (tags.length) xs = xs.filter(x => tags.every(t => (x.structure_tags || []).includes(t)));
    const t = q.trim().toLowerCase();
    if (t) xs = xs.filter(x => (x.qid + ' ' + x.question).toLowerCase().includes(t));
    return xs;
  }, [items, q, onlyMiss, classes, types, tags, sub, isMcq]);

  const gold: string[] = detail?.gold_chunk_ids || [];
  const got: string[] = detail?.chunk_ids || [];
  const hit = gold.some(g => got.slice(0, 5).includes(g));

  const tree = (
    <nav className="h-full overflow-y-auto overscroll-contain pb-24">
      <div className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-stone-400">
        Runs ({runs.length}) · newest first
      </div>
      {runs.map((r, i) => {
        const open = r.run === sel.run;
        return (
          <div key={r.run} className="px-2">
            <button
              onClick={() => go({ run: open ? '' : r.run })}
              className={`w-full text-left rounded-lg px-3 py-2.5 transition
                          ${open ? 'bg-stone-200/70' : 'hover:bg-stone-100'}`}
            >
              <span className="flex items-center gap-2">
                <span className="text-stone-400 text-xs w-3">{open ? '▾' : '▸'}</span>
                <span className="font-medium text-[13px] truncate">{r.run}</span>
              </span>
              <span className="block pl-5 text-[11px] text-stone-500">
                {r.kind === 'compare' ? <>all runs on this set · {r.n_questions} questions</> : <>{r.n_configs} configs · {r.n_questions} questions</>}
                {r.mtime ? <> · <span className="text-stone-400">{runAge(r.mtime)}</span></> : null}
                {i === 0 ? (
                  <span className="ml-1.5 rounded bg-emerald-100 px-1 py-px text-[10px]
                                   font-medium text-emerald-700">latest</span>
                ) : null}
              </span>
            </button>

            {open && configs.map(c => {
              const on = c.config === sel.config;
              return (
                <div key={c.config} className="pl-3">
                  <button
                    onClick={() => go({ config: on ? '' : c.config })}
                    className={`w-full text-left rounded-lg px-3 py-2 mt-0.5 transition
                                ${on ? 'bg-blue-50 ring-1 ring-blue-200' : 'hover:bg-stone-100'}`}
                  >
                    <span className="flex items-center gap-2">
                      <span className="text-stone-400 text-xs w-3">{on ? '▾' : '▸'}</span>
                      <span className="font-mono text-[12px] truncate">{c.config}</span>
                      {/* Carried over from the parent run rather than measured here —
                          a targeted re-run only stores the configs it changed. */}
                      {c.inherited && (
                        <span className="shrink-0 rounded px-1 py-px text-[9px] uppercase
                                         bg-stone-100 text-stone-500 ring-1 ring-stone-200"
                              title={`carried over unchanged from ${c.source_run}`}>
                          inherited
                        </span>
                      )}
                    </span>
                    <span className="block pl-5 text-[11px] text-stone-500">
                      {c.mcq_acc != null
                        ? <>accuracy {num(c.mcq_acc)}</>
                        : <>hit@5 {num(c.hit5)}{c.empty ? ` · ${c.empty} empty` : ''}</>}
                    </span>
                  </button>

                  {on && (
                    <div className="pl-3 pr-1 pb-2">
                      <div className="flex gap-1.5 py-2">
                        <input
                          value={q} onChange={e => setQ(e.target.value)}
                          placeholder="filter questions…"
                          className="flex-1 min-w-0 px-3 py-2 text-[13px] rounded-lg bg-white
                                     ring-1 ring-stone-200 focus:ring-stone-400 outline-none"
                        />
                        <button
                          onClick={() => setOnlyMiss(v => !v)}
                          title={isMcq ? 'show only questions answered wrong'
                                       : 'show only questions that missed gold@5'}
                          className={`px-3 rounded-lg text-xs ring-1 transition
                                      ${onlyMiss ? 'bg-rose-50 ring-rose-300 text-rose-700'
                                                 : 'bg-white ring-stone-200 text-stone-500'}`}
                        >miss</button>
                      </div>
                      <FacetRow label="class" facets={CLASS_FACETS} active={classes} counts={counts}
                                onToggle={(v) => {
                                  // Dropping a class must drop the subject chips that
                                  // belonged to it, or a stale type keeps filtering
                                  // against a class no longer selected and the list
                                  // silently empties.
                                  const next = classes.includes(v)
                                    ? classes.filter(x => x !== v) : [...classes, v];
                                  setClasses(next);
                                  if (!next.length) setTypes([]);
                                }} />
                      {sub.values.length > 1 && (
                        <FacetRow label="subject" facets={sub.values} active={types} counts={counts}
                                  onToggle={(v) => toggle(v, types, setTypes)}
                                  hint="groups inside the selected class" />
                      )}
                      <FacetRow label="structure" facets={TAG_FACETS} active={tags} counts={counts}
                                onToggle={(v) => toggle(v, tags, setTags)}
                                hint="all selected tags must be present" />
                      <div className="flex items-center gap-2 pb-1 px-1">
                        <span className="text-[11px] text-stone-400">{shown.length}/{items.length}</span>
                        {(classes.length || types.length || tags.length || onlyMiss || q) ? (
                          <button
                            onClick={() => { setClasses([]); setTypes([]); setTags([]); setOnlyMiss(false); setQ(''); }}
                            className="text-[11px] text-stone-500 underline hover:text-stone-800"
                          >clear</button>
                        ) : null}
                      </div>
                      {shown.map(it => (
                        <button
                          key={it.qid}
                          onClick={() => go({ qid: it.qid }, true)}
                          className={`block w-full text-left rounded-lg px-3 py-2 mb-0.5 transition
                                      ${it.qid === sel.qid ? 'bg-blue-100' : 'hover:bg-stone-100'}`}
                        >
                          <span className="flex items-center gap-2">
                            <span className={`text-[10px] ${okOf(it) ? 'text-emerald-500' : 'text-rose-400'}`}>
                              {okOf(it) ? '●' : '○'}
                            </span>
                            <span className="font-mono text-[11px] text-stone-500">{it.qid}</span>
                          </span>
                          <span className="block pl-4 text-[12px] leading-snug text-stone-600">
                            {it.question.length > 90 ? it.question.slice(0, 90) + '…' : it.question}
                          </span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        );
      })}
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
        >☰ Runs</button>
        <div className="min-w-0 text-xs text-stone-500 truncate font-mono">
          {sel.qid || sel.config || sel.run || 'pick a run'}
        </div>
      </div>

      <div className="flex-1 flex min-h-0">
        <aside className="hidden md:block w-[19rem] shrink-0 border-r border-stone-200 bg-stone-50">
          {tree}
        </aside>
        {navOpen && (
          <div className="md:hidden fixed inset-0 z-40 flex">
            <div className="w-[85%] max-w-sm bg-stone-50 shadow-xl">{tree}</div>
            <button
              className="flex-1 bg-black/30"
              aria-label="Close menu"
              onClick={() => setNavOpen(false)}
            />
          </div>
        )}

        <main className="flex-1 min-w-0 overflow-y-auto">
          {DEBUG && (
            <div className="px-3 py-1 text-[11px] font-mono bg-yellow-50 border-b border-yellow-200">
              run={sel.run || '∅'} · config={sel.config || '∅'} · qid={sel.qid || '∅'} ·
              busy={String(busy)} · detail={detail ? 'yes' : 'null'} ·
              stages={stages ? stages.length : 'null'} · items={items.length} · err={err || '∅'}
            </div>
          )}

          {!sel.qid && !sel.run && (
            <div className="p-6 md:p-10 text-stone-500 text-sm max-w-prose">
              Pick a run to see its overview, then a config, then a question.
            </div>
          )}

          {/* The overview endpoint is newer than a long-lived rag-engine process may be;
              without this branch a run with no question selected renders blank. */}
          {!sel.qid && sel.run && !ov && (
            <div className="p-6 md:p-10 text-sm text-stone-500 max-w-prose">
              No overview available for <span className="font-mono">{sel.run}</span>. If the
              rag-engine was started before <span className="font-mono">/api/runs/&#123;run&#125;/overview</span>
              existed, restart it to pick the endpoint up. Meanwhile the tree still works —
              open a config and a question.
            </div>
          )}

          {!sel.qid && sel.run && ov?.compare && (
            <div className="max-w-5xl mx-auto px-4 md:px-8 py-5 md:py-8 space-y-8">
              <header>
                <h1 className="text-lg md:text-xl font-semibold">Cross-run comparison · {ov.compare.set}</h1>
                <p className="text-[13px] text-stone-500 mt-0.5">
                  {ov.compare.runs.length} runs · {ov.compare.n_questions} questions · only config files
                  holding exactly this set's questions are counted · click a cell to open that run's config
                </p>
              </header>
              <RunOverview ov={ov} onOpen={(run, config) => go({ run, config })} />
            </div>
          )}

          {!sel.qid && sel.run && ov && !ov.compare && (
            <div className="max-w-5xl mx-auto px-4 md:px-8 py-5 md:py-8 space-y-8">
              <header>
                <h1 className="text-lg md:text-xl font-semibold">{sel.run}</h1>
                <p className="text-[13px] text-stone-500 mt-0.5">
                  {runSummary(ov)}
                </p>
                {/* Run-level provenance from _agg.json. Without it a reader cannot tell
                    which model or context length produced the numbers on this page —
                    the two things that change results most between runs. */}
                {ov.meta && (
                  <dl className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-[12px]">
                    {[
                      // A TeleQnA run holds several generators, listed per config in
                      // meta.configs rather than once at the top.
                      ['Model', ov.meta.model ?? ([...new Set(Object.values(ov.meta.configs || {})
                        .map((c: any) => c.model).filter(Boolean))].sort().join(', ') || null)],
                      ['Context', ov.meta.context_length ? `${ov.meta.context_length} tok` : null],
                      ['Thinking', ov.meta.think === undefined ? null : ov.meta.think ? 'on' : 'off'],
                      ['Questions', ov.meta.eval_file ? String(ov.meta.eval_file).split('/').pop() : null],
                      // The generator model above says what wrote the answers; these say
                      // what graded them. Verdicts from two rubrics are not comparable, so
                      // the version belongs next to the model rather than buried per config.
                      ['Judge', ov.meta.judge_model],
                      ['Rubric', ov.meta.judge_rubric],
                      ['Code', ov.meta.code_sha],
                      ['Generated', ov.meta.generated ? String(ov.meta.generated).replace('T', ' ').slice(0, 16) : null],
                    ]
                      .filter(([, v]) => v)
                      .map(([k, v]) => (
                        <div key={k as string} className="flex items-baseline gap-1.5">
                          <dt className="text-stone-400">{k}</dt>
                          <dd className="font-mono text-stone-600">{v as string}</dd>
                        </div>
                      ))}
                  </dl>
                )}
              </header>
              <RunOverview ov={ov} active={sel.config} onPick={(c) => go({ config: c })} />
            </div>
          )}

          {busy && <div className="p-6 md:p-10 text-stone-400 text-sm">Loading…</div>}

          {/* Without this branch a failed fetch renders nothing at all: qid is set, so the
              hint above is hidden, but detail is still null — a silent blank page. */}
          {sel.qid && !busy && !detail && (
            <div className="p-6 md:p-10">
              <div className="text-rose-700 font-medium">Could not load this question.</div>
              <div className="mt-1 text-xs text-stone-600 font-mono break-all">{err || 'unknown error'}</div>
            </div>
          )}

          {sel.qid && detail && !busy && (
            <article className="max-w-3xl mx-auto px-4 md:px-8 py-5 md:py-8 space-y-6">
              <header>
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] font-mono text-stone-500">
                  <span>{sel.run}</span><span className="text-stone-300">/</span>
                  <span className="px-1.5 py-0.5 rounded bg-stone-100">{sel.config}</span>
                  <span className="text-stone-300">/</span><span>{detail.qid}</span>
                </div>
                <h1 className="mt-2 text-[19px] md:text-2xl font-semibold leading-snug">
                  {detail.question}
                </h1>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {detail.question_type && <Tag>{detail.question_type}</Tag>}
                  {(detail.structure_tags || []).map((t: string) => <Tag key={t}>{t}</Tag>)}
                  {detail.mcq_gold != null ? (
                    <Tag tone={detail.mcq_correct ? 'ok' : 'bad'}>
                      answered {detail.mcq_pred ?? '—'} · correct {detail.mcq_gold}
                    </Tag>
                  ) : (
                    <Tag tone={hit ? 'ok' : 'bad'}>{hit ? 'gold@5 hit' : 'gold@5 miss'}</Tag>
                  )}
                  <Tag>{num(detail.latency, 1)}s</Tag>
                  {detail.faithfulness != null && <Tag>faith {num(detail.faithfulness)}</Tag>}
                  {/* Judge label up here too, so scanning the question list shows
                      which answers were graded wrong without opening each one. */}
                  {judge?.judge_label && (
                    <Tag tone={JUDGE_TONE[judge.judge_label] ?? 'plain'}>{judge.judge_label}</Tag>
                  )}
                  {detail.repatched && <Tag>repatched</Tag>}
                </div>
              </header>

              <section>
                <h2 className="text-[11px] font-semibold uppercase tracking-wider text-stone-400 mb-2">
                  Retrieval pipeline
                </h2>
                {stages && stages.length > 0 ? (
                  <div className="rounded-xl bg-white ring-1 ring-stone-200 p-3 overflow-x-auto">
                    {/* Same component the chat uses, fed from the recorded trail;
                        streaming=false so it renders in its finished state. */}
                    <ThinkingTrail stages={stages as any} streaming={false} startedAt={0} />
                  </div>
                ) : (
                  <p className="text-[13px] text-stone-500 rounded-xl border border-dashed border-stone-300 p-4">
                    This run was recorded before the harness saved pipeline trails, so only the
                    final answer exists. Newer runs store every stage.
                  </p>
                )}
              </section>

              <section>
                <h2 className="text-[11px] font-semibold uppercase tracking-wider text-stone-400 mb-2">
                  Answer
                </h2>
                <div className="rounded-xl bg-white ring-1 ring-stone-200 p-4 md:p-5
                                prose prose-sm md:prose-base max-w-none prose-stone
                                prose-pre:overflow-x-auto break-words">
                  {detail.answer
                    ? <ReactMarkdown>{detail.answer}</ReactMarkdown>
                    : <span className="text-rose-600">(empty)</span>}
                </div>
              </section>

              {detail.reference_answer && (
                <section>
                  <h2 className="text-[11px] font-semibold uppercase tracking-wider text-stone-400 mb-2">
                    Reference answer
                  </h2>
                  <div className="rounded-xl bg-stone-100/70 ring-1 ring-stone-200 p-4 text-[13px]
                                  leading-relaxed whitespace-pre-wrap break-words">
                    {detail.reference_answer}
                  </div>
                </section>
              )}

              {/* Judge verdict, laid out like the retrieval trail above it: a one-line
                  summary always visible, and a Details toggle that reveals the exact
                  prompt the judge was given. Same reason the trail shows the Cypher and
                  answer prompts — a verdict you cannot audit is a number you have to
                  take on faith. Placed after the reference answer because that is what
                  the judge compares against, so reading order follows the grading. */}
              <section>
                <h2 className="text-[11px] font-semibold uppercase tracking-wider text-stone-400 mb-2">
                  LLM judge
                </h2>
                {judge?.judge_error ? (
                  <div className="rounded-xl bg-rose-50 ring-1 ring-rose-200 p-4 text-[13px] text-rose-700">
                    Judge failed: {judge.judge_error}
                  </div>
                ) : judge?.judge_label ? (
                  <div className="rounded-xl bg-white ring-1 ring-stone-200 p-3">
                    <div className="flex items-baseline gap-2 flex-wrap">
                      <Tag tone={JUDGE_TONE[judge.judge_label] ?? 'plain'}>{judge.judge_label}</Tag>
                      {judge.judge_key_facts_total != null && (
                        <span className="text-xs text-stone-500 tabular-nums">
                          {judge.judge_key_facts_stated ?? 0}/{judge.judge_key_facts_total} key facts
                        </span>
                      )}
                      {judge.model && (
                        <span className="text-[11px] text-stone-400">{judge.model}</span>
                      )}
                      {judge.prompt && (
                        <button
                          type="button"
                          onClick={() => setJudgeOpen(v => !v)}
                          className="ml-auto inline-flex items-center gap-1 text-[11px] text-stone-500
                                     hover:text-stone-700 transition-colors"
                        >
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none"
                               stroke="currentColor" strokeWidth="2.5"
                               className={`transition-transform ${judgeOpen ? 'rotate-90' : ''}`}>
                            <path d="M9 18l6-6-6-6" />
                          </svg>
                          {judgeOpen ? 'Hide' : 'Details'}
                        </button>
                      )}
                    </div>
                    {judge.judge_rationale && (
                      <div className="text-xs text-stone-500 mt-1 leading-relaxed break-words">
                        {judge.judge_rationale}
                      </div>
                    )}
                    {judge.judge_contradiction && (
                      <div className="mt-2 text-[12px] leading-relaxed text-rose-700 bg-rose-50
                                      ring-1 ring-rose-200 rounded-lg px-3 py-2 break-words">
                        Contradiction: {judge.judge_contradiction}
                      </div>
                    )}
                    {judgeOpen && judge.prompt && (
                      <div className="mt-2 space-y-2">
                        {/* Two panels, same shape as the retrieval trail's Details:
                            what went IN, then what came OUT. */}
                        <div>
                          <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">
                            Input prompt
                            <span className="ml-2 normal-case text-stone-400 text-[10px]">
                              rubric only · {judge.prompt.split('\nQUESTION:')[0].length.toLocaleString()} chars
                              <span className="ml-2">(question / reference / answer shown above)</span>
                            </span>
                          </div>
                          <pre className="text-[12px] leading-relaxed whitespace-pre-wrap break-words
                                          text-stone-700 bg-stone-50 border border-stone-200 rounded p-2
                                          max-h-96 overflow-auto font-mono">
                            {/* Rubric only: the prompt also embeds the question, the
                                reference and the answer verbatim, and all three already
                                have their own blocks on this page. Repeating them would
                                bury the one part visible nowhere else — the rules. */}
                            {judge.prompt.split('\nQUESTION:')[0]}
                          </pre>
                        </div>
                        <div>
                          <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">
                            Answer
                            <span className="ml-2 normal-case text-stone-400 text-[10px]">
                              JSON returned by the judge
                            </span>
                          </div>
                          <pre className="text-[12px] leading-relaxed whitespace-pre-wrap break-words
                                          text-stone-700 bg-stone-50 border border-stone-200 rounded p-2
                                          max-h-96 overflow-auto font-mono">
                            {/* Rebuilt from the stored fields in the shape the rubric asks
                                for, so what is shown here is what the model was told to
                                produce — not a paraphrase of it. */}
                            {JSON.stringify({
                              label: judge.judge_label,
                              key_facts_total: judge.judge_key_facts_total,
                              key_facts_stated: judge.judge_key_facts_stated,
                              contradictions: judge.judge_contradiction
                                ? [judge.judge_contradiction] : [],
                              rationale: judge.judge_rationale,
                            }, null, 2)}
                          </pre>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-[13px] text-stone-400 rounded-xl border border-dashed
                                border-stone-300 p-4">
                    Not judged — no verdict for this question yet.
                  </p>
                )}
              </section>

              <section className="grid gap-4 sm:grid-cols-2">
                <div>
                  <h2 className="text-[11px] font-semibold uppercase tracking-wider text-stone-400 mb-2">
                    Gold chunks ({gold.length})
                  </h2>
                  <div className="space-y-1">
                    {gold.map(g => (
                      <div key={g} className="font-mono text-[11px] flex gap-2 break-all">
                        <span className={got.includes(g) ? 'text-emerald-600' : 'text-rose-500'}>
                          {got.includes(g) ? `#${got.indexOf(g) + 1}` : '✗'}
                        </span>
                        <span>{g}</span>
                      </div>
                    ))}
                    {!gold.length && <span className="text-stone-400 text-xs">—</span>}
                  </div>
                </div>
                <div>
                  <h2 className="text-[11px] font-semibold uppercase tracking-wider text-stone-400 mb-2">
                    Retrieved ({got.length})
                  </h2>
                  <div className="space-y-0.5 max-h-56 overflow-y-auto pr-1">
                    {got.map((c, i) => (
                      <div key={c + i} className="font-mono text-[11px] break-all">
                        <span className="text-stone-400 mr-1">{i + 1}.</span>
                        <span className={gold.includes(c) ? 'text-emerald-700 font-semibold' : 'text-stone-700'}>
                          {c}
                        </span>
                      </div>
                    ))}
                    {!got.length && <span className="text-rose-500 text-xs">no chunks retrieved</span>}
                  </div>
                </div>
              </section>

              {siblings.length > 1 && (
                <section>
                  <h2 className="text-[11px] font-semibold uppercase tracking-wider text-stone-400 mb-2">
                    Same question, other configs
                  </h2>
                  <div className="-mx-4 md:mx-0 overflow-x-auto">
                    <table className="min-w-[34rem] w-full text-xs">
                      <thead className="text-stone-500">
                        <tr className="border-b border-stone-200">
                          <th className="text-left font-medium px-3 py-2">Config</th>
                          <th className="font-medium px-2 py-2">gold@5</th>
                          <th className="font-medium px-2 py-2">chunks</th>
                          <th className="font-medium px-2 py-2">latency</th>
                          <th className="font-medium px-2 py-2">faith</th>
                          <th className="font-medium px-2 py-2">chars</th>
                        </tr>
                      </thead>
                      <tbody>
                        {siblings.map(s => (
                          <tr key={s.config}
                              className={`border-b border-stone-100 ${s.config === sel.config ? 'bg-blue-50' : ''}`}>
                            <td className="px-3 py-2">
                              <button className="font-mono hover:underline"
                                      onClick={() => go({ config: s.config, qid: sel.qid })}>
                                {s.config}
                              </button>
                            </td>
                            <td className="px-2 py-2 text-center">
                              <span className={s.hit5 ? 'text-emerald-600' : 'text-rose-400'}>
                                {s.hit5 ? '●' : '○'}
                              </span>
                            </td>
                            <td className="px-2 py-2 text-center">{s.n_chunks}</td>
                            <td className="px-2 py-2 text-center">{num(s.latency, 1)}s</td>
                            <td className="px-2 py-2 text-center">{num(s.faithfulness)}</td>
                            <td className="px-2 py-2 text-center">{s.answer_chars}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}
            </article>
          )}
        </main>
      </div>
    </div>
  );
}
