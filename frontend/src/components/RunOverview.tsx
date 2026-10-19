import React from 'react';
import { CONFIGS, LADDERS, label, short } from '../benchLabels';
import teleqnaPublished from '../teleqnaPublished.json';

// Overview of one benchmark run, laid out as the five sheets of
// benchmark_paper/Results_RAG_3GPP.xlsx. Every sheet shows a chart AND the table:
// the chart ranks at a glance, the table carries the exact values the report quotes.
//
// Colour: Okabe-Ito, verified with the dataviz validator rather than assumed.
//   #0072B2 / #E69F00 / #009E73 — worst colour-blind separation dE 11.4, passes.
//   The palette shipped in the repo before this (#2E75B6/#ED7D31/#70AD47) FAILS:
//   its green-orange pair is dE 4.6 under deuteranopia, below the floor of 8.
//   #E69F00 sits at 2.25:1 against white, under the 3:1 bar, so every mark carries a
//   printed value — the number, not the fill, is what must be readable.
// At most three series per chart. More than that means another chart, never more hues.

const num = (v: number | null | undefined, d = 3) =>
  v === null || v === undefined ? '—' : v.toFixed(d);

const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`;

// Okabe-Ito, which stays distinguishable under the common colour-vision
// deficiencies. Sheet 1 draws FOUR series and this list held three, so Faith fell
// through to Bar's `color` default — SERIES[0] — and rendered in R@5's exact blue:
// two different metrics, one colour, no error anywhere. Hence the guard below.
const SERIES = ['#0072B2', '#E69F00', '#009E73', '#CC79A7'];

/** Never silently reuse a colour: an out-of-range series is a coding error, and
 *  cycling would repeat a colour just as invisibly as the bug this replaced. */
const seriesColor = (i: number) => SERIES[i] ?? '#6B7280';

// Config labels run to 42 characters, so the label column is sized to hold the
// longest one on a single line at md+ (18rem ~= 288px at the 12px used here).
// Keep these two in sync: one is a flex child, the other a grid template.
const LABEL_COL = 'w-40 md:w-72';
const LABEL_GRID = 'md:grid-cols-[18rem_1fr]';

/** Axis ceiling from the data, rounded up to the next 0.1 and never below 0.2.
 *
 * A hardcoded ceiling silently lies once the numbers grow past it: this section
 * was pinned at 0.5 back when KG-only scored ~0.3, and when the pure-graph arm
 * reached 0.536 every bar clamped to 100% — the grey track vanished and five
 * visibly different values all rendered as "full", which reads as five identical
 * results. Deriving it means the bars can saturate only if the metric itself does.
 */
function niceMax(values: (number | null | undefined)[]): number {
  const m = Math.max(0, ...values.filter((x): x is number => typeof x === 'number'));
  return Math.max(0.2, Math.min(1, Math.ceil((m + 1e-9) * 10) / 10));
}

function Bar({ v, max = 1, color = SERIES[0] }:
  { v: number | null | undefined; max?: number; color?: string }) {
  const pct = v == null ? 0 : Math.max(0, Math.min(1, v / max)) * 100;
  return (
    <span className="flex items-center gap-2 w-full">
      <span className="relative h-2 flex-1 min-w-[3rem] rounded-full bg-stone-200/70">
        {/* rounded data-end, anchored at the baseline */}
        <span className="absolute inset-y-0 left-0 rounded-full"
              style={{ width: `${pct}%`, background: color }} />
      </span>
      <span className="tabular-nums text-stone-700 text-[11px] w-11 text-right shrink-0">
        {num(v, 3)}
      </span>
    </span>
  );
}

function Legend({ names }: { names: string[] }) {
  if (names.length < 2) return null;
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 mb-3">
      {names.map((n, i) => (
        <span key={n} className="flex items-center gap-1.5 text-[11px] text-stone-600">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: seriesColor(i) }} />
          {n}
        </span>
      ))}
    </div>
  );
}

/** Grouped bars, the shape a paper figure uses: one block per category, one labelled
 *  bar per measure. Reads top-to-bottom on a phone instead of overflowing sideways. */
// `short` not `label` for the row name, and one line: the longest label is 48 chars
// ("Hybrid (Dense + KG) → rerank → gen (full system)") and wrapped to four lines in a
// 7rem column, which pushed each config's bars out of line with its neighbours. The
// registry already carries a 10-19 char `short` for exactly this. Full label stays in
// the tooltip.
function GroupedBars({ rows, series, max = 1, cat = (r: any) => short(r.config) }: {
  rows: any[];
  series: { name: string; key: string }[];
  max?: number;
  cat?: (r: any) => string;
}) {
  return (
    <div className="rounded-xl bg-white ring-1 ring-stone-200 p-4">
      <Legend names={series.map(s => s.name)} />
      <div className="space-y-2">
        {rows.map(r => (
          // Framed per config, matching the tables: eight systems whose names differ
          // only by a suffix are easy to misread across when the rows just float.
          <div key={r.config}
               className={`grid grid-cols-[9rem_1fr] ${LABEL_GRID} gap-x-3 gap-y-1 items-center
                           rounded-md border border-stone-200 px-2.5 py-2`}>
            <span className="text-[12px] text-stone-700 font-medium truncate"
                  title={label(r.config)}>{cat(r)}</span>
            <div className="space-y-1">
              {series.map((sr, i) => (
                <div key={sr.key} className="flex items-center gap-2">
                  {/* w-28 + nowrap: series names run to 19 chars ("Cover-EM (leak-free)")
                      and w-9 (36px) broke them over four lines, stretching each config
                      block to four times the height of its bars. */}
                  {series.length > 1 && (
                    <span className="w-28 shrink-0 text-[10px] text-stone-400 text-right whitespace-nowrap">
                      {sr.name}
                    </span>
                  )}
                  <Bar v={r[sr.key]} max={max} color={seriesColor(i)} />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** A "Reranker: yes/no" cell cannot be misread the way a prose label can — the old
 *  labels implied the four text baselines were unreranked when only one config is. */
function DescriptorHead() {
  return (
    <>
      <th className="font-medium px-2 py-2">Retriever</th>
      <th className="font-medium px-2 py-2">Rerank</th>
      <th className="font-medium px-2 py-2">Prov</th>
      <th className="font-medium px-2 py-2">Evidence</th>
    </>
  );
}

function DescriptorCells({ k }: { k: string }) {
  const c = CONFIGS[k];
  const yn = (b?: boolean) => (c === undefined ? '—' : b ? 'yes' : 'no');
  return (
    <>
      <td className="px-2 py-2 text-center text-[11px] text-stone-500">{c?.retriever ?? '—'}</td>
      <td className="px-2 py-2 text-center text-[11px]">{yn(c?.reranker)}</td>
      <td className="px-2 py-2 text-center text-[11px]">{yn(c?.provenance)}</td>
      <td className="px-2 py-2 text-center text-[11px] text-stone-500">{c?.evidence ?? '—'}</td>
    </>
  );
}

function Table({ rows, cols, onPick, active, descriptors }: {
  rows: any[];
  cols: { key: string; head: string; d?: number; fmt?: (v: any) => string }[];
  onPick?: (config: string) => void;
  active?: string;
  descriptors?: boolean;
}) {
  return (
    <div className="-mx-4 md:mx-0 overflow-x-auto">
      {/* border-separate + spacing gives every config its own framed row instead of a
          run of hairlines: with eight systems whose names differ only in a suffix
          ("…, no rerank" vs "…"), a continuous grid makes it easy to read a number off
          the neighbouring line. */}
      <table className="min-w-[38rem] w-full text-xs border-separate"
             style={{ borderSpacing: '0 4px' }}>
        <thead className="text-stone-500">
          <tr>
            <th className="text-left font-medium px-3 py-2">System</th>
            {descriptors && <DescriptorHead />}
            {cols.map(c => <th key={c.key} className="font-medium px-2 py-2">{c.head}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={`${r.config}${r._answeredOnly ? ':answered' : ''}:${i}`}
              onClick={() => onPick?.(r.config)}
              className={`[&>td]:bg-white [&>td]:border-y [&>td]:border-stone-200
                          [&>td:first-child]:border-l [&>td:first-child]:rounded-l-md
                          [&>td:last-child]:border-r [&>td:last-child]:rounded-r-md
                          ${onPick ? 'cursor-pointer [&:hover>td]:bg-stone-50' : ''}
                          ${r.config === active ? '[&>td]:bg-blue-50 [&>td]:border-blue-300' : ''}
                          ${r._answeredOnly ? 'text-stone-500 italic [&>td]:bg-stone-50/60' : ''}`}
            >
              <td className="px-3 py-2 whitespace-nowrap">
                {label(r.config)}
                {r._answeredOnly && <span className="text-stone-400"> — answered only</span>}
                {/* An inherited row is a real measurement, but from the parent run —
                    the reader must be able to tell which numbers this run produced. */}
                {r.inherited && (
                  <span className="ml-1.5 rounded px-1 py-px text-[9px] uppercase tracking-wide
                                   bg-stone-100 text-stone-500 ring-1 ring-stone-200"
                        title={`carried over unchanged from ${r.source_run}`}>
                    from {r.source_run?.replace(/_\d{8}$/, '')}
                  </span>
                )}
              </td>
              {descriptors && <DescriptorCells k={r.config} />}
              {cols.map(c => (
                <td key={c.key} className="px-2 py-2 text-center tabular-nums">
                  {c.fmt ? c.fmt(r[c.key]) : num(r[c.key], c.d ?? 3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Sheet({ n, title, note, children }:
  { n: number; title: React.ReactNode; note?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-[13px] font-semibold text-stone-800">
          <span className="text-stone-400 font-normal mr-1.5">Sheet {n}</span>{title}
        </h2>
        {note && <div className="text-[11px] text-stone-500 mt-0.5">{note}</div>}
      </div>
      {children}
    </section>
  );
}

/** One caveat, keyed to the column it is about.
 *
 *  Sheet 1 carries five INDEPENDENT warnings — one per metric — and as a single
 *  paragraph the reader had to finish all five to learn which applied to the column
 *  they were looking at. Keyed lines let them read only the one they need. */
function Caveat({ on, children }: { on: string; children: React.ReactNode }) {
  return (
    <li className="leading-snug">
      <span className="font-medium text-stone-600">{on}</span>
      <span className="text-stone-400"> — </span>
      {children}
    </li>
  );
}

// Published TeleQnA results the PAPER can use. The system targets LOCAL models of
// 8-31B, so rows are open models of ~7-32B (TeleMoM, GSMA Open Telco for the Gemma /
// Qwen / DeepSeek families, Telco-oRAG, Chat3GPP) plus three current closed models as
// an upper reference (status "frontier"); older large models (GPT-3.5/4, 70B+) are
// dropped. Every number was read from the primary source on 2026-09-27; the full
// ~200-row survey this was filtered from is
// .debug_context/teleqna-published-baselines/teleqna_published.json. NOT measured here:
// prompts, parsing and subsets differ. A per-subject, B overall-only, C 3GPP subsets.
const SUBJECT_KEYS = ['Lexicon', 'Research overview', 'Research publications',
                      'Standards overview', 'Standards specifications'];
const PUB: any = teleqnaPublished;

const shortPaper = (p: string) => p.replace(/\s*\(.*$/, '').replace(/:.*$/, '');

/** Hover-only detail, so long qualifiers do not stretch the table. */
function Info({ text }: { text: string }) {
  return (
    <span title={text} className="ml-1 cursor-help text-stone-400 hover:text-stone-700 select-none">ⓘ</span>
  );
}

// A system name keeps a SHORT parenthetical ("(TelecomGPT)", "(Google)") inline; a long
// one ("(ensemble: Qwen2.5-7B+Llama3+Mistral+Phi-4, …)") or a trailing ", context 2500
// tok" moves behind an ⓘ — measured on screen, those qualifiers were wider than the
// numbers the row exists to show.
function SysName({ name }: { name: string }) {
  let main = name, extra = '';
  const m = name.match(/^(.*?)\s*\((.{21,})\)\s*$/);
  if (m) { main = m[1]; extra = m[2]; }
  const c = main.match(/^([^,]+),\s*(.+)$/);
  if (c) { main = c[1]; extra = extra ? `${c[2]}; ${extra}` : c[2]; }
  return <>{main}{extra && <Info text={extra} />}</>;
}

// Subset descriptions run to 150 chars; the column shows a short label and the full
// wording sits behind the ⓘ.
const SUBSET_SHORT: [RegExp, string][] = [
  // Before ot-lite: TelecomGPT-R1 v2 describes its split as unstated while MENTIONING
  // ot-lite, and matching that word first mislabelled all 23 of its rows.
  [/GSMA Open Telco leaderboard/, 'GSMA (split unstated)'],
  [/ot-lite/, 'ot-lite ~1k'], [/ot-full/, 'ot-full ~10k'],
  [/imply ~500/, '~500 q'], [/surveys/, '10-q surveys'],
  [/held-out.*900|900 q/, '900 q held-out'], [/Zindi|2000 3GPP-standards/, 'Zindi 2000 q'],
  [/Release 18 part only/, 'Rel.18 780 q'], [/734/, 'Rel.17 734 + Rel.18 780'], [/Release 18 questions/, '3GPP 1840 q, Rel.18'],
  [/1840|3GPP Standard dataset/, '3GPP 1840 q'], [/3,500/, '3,500 q (claimed)'],
  [/^250 |250 TeleQnA/, '250 q'], [/8k/, '8k open-ended'],
  [/Standards specifications category/, 'Std. specifications'],
  [/lexicon/i, 'Lexicon'], [/full 10k/, 'full 10k'], [/^same$/, 'same as above'],
];
function Subset({ text }: { text: string }) {
  const hit = SUBSET_SHORT.find(([re]) => re.test(text));
  const short = hit ? hit[1] : (text.split(/[;(]/)[0].trim().slice(0, 24) || text);
  return <span className="whitespace-nowrap">{short}{short !== text && <Info text={text} />}</span>;
}

function Src({ r }: { r: any }) {
  return (
    <span className="whitespace-nowrap">
      <a href={r.url} target="_blank" rel="noreferrer"
         title={`${r.paper} — ${r.table}${r.bibkey ? ` — \\cite{${r.bibkey}}` : ''}`}
         className="text-stone-500 underline decoration-stone-300 hover:text-stone-800">
        {shortPaper(r.paper)}
      </a>
      {r.status === 'frontier' && (
        <span title="closed frontier model — an upper reference, not a local-model peer"
              className="ml-1 rounded px-1 py-px text-[9px] uppercase tracking-wide
                         bg-stone-100 text-stone-500 ring-1 ring-stone-200">frontier</span>
      )}
    </span>
  );
}

// Framed-row table styling, shared with <Table>. `ours` rows are this run's configs.
const ROW = `[&>td]:border-y [&>td]:border-stone-200
             [&>td:first-child]:border-l [&>td:first-child]:rounded-l-md
             [&>td:last-child]:border-r [&>td:last-child]:rounded-r-md`;
const rowTone = (ours: boolean) =>
  ours ? '[&>td]:bg-blue-50 [&>td]:border-blue-300 font-medium' : '[&>td]:bg-white';
const pctCell = (v: any) => (typeof v === 'number' ? v.toFixed(1) : '—');

function PubTable({ head, rows }: { head: string[]; rows: { ours?: boolean; sub?: boolean; cells: React.ReactNode[] }[] }) {
  return (
    <div className="-mx-4 md:mx-0 overflow-x-auto">
      <table className="w-full text-xs border-separate" style={{ borderSpacing: '0 3px' }}>
        <thead className="text-stone-500">
          <tr>{head.map((h, i) => (
            <th key={h} className={`font-medium px-2 py-2 ${i === 0 ? 'text-left px-3' : ''}`}>{h}</th>
          ))}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={r.sub
              ? `${ROW} [&>td]:bg-blue-50/40 [&>td]:border-blue-100 text-[11px] text-stone-600 italic`
              : `${ROW} ${rowTone(!!r.ours)}`}>
              {r.cells.map((c, j) => (
                <td key={j} className={j === 0 ? 'px-3 py-1.5 whitespace-nowrap'
                                               : 'px-2 py-1.5 text-center tabular-nums whitespace-nowrap'}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Overview for a TeleQnA run: multiple-choice accuracy, overall and per subject, set
 *  against published results. The paper_bench sheets (Recall, judge, KG ladders) have
 *  nothing to show here — TeleQnA has no gold chunks and is scored by the letter. */
function TeleQnAOverview({ ov, active, onPick }:
  { ov: any; active?: string; onPick?: (config: string) => void }) {
  const [showAllB, setShowAllB] = React.useState(false);
  const mine: any[] = (ov?.configs || []).filter((c: any) => c.mcq_acc != null);
  const meta = ov?.meta || {};
  // Per-config settings (teleqna_to_run.py writes meta.configs[<key>]); a run folder
  // holds several configs (fixed, llm_only, ...).
  const cm = (c: any) => meta.configs?.[c.config] || {};
  // A config run on only some subjects (e.g. text RAG on the two Standards subjects):
  // its overall accuracy is over those subjects, not the 10k set.
  const filtered = (c: any) => cm(c).n_planned != null && meta.n_questions != null
    && cm(c).n_planned < meta.n_questions && /rel18/i.test(cm(c).question_file || '');
  const partial = (c: any) => filtered(c) || (cm(c).subjects?.length ?? 5) < SUBJECT_KEYS.length;
  const scopeOf = (c: any) => filtered(c) ? `Rel.18 ${cm(c).n_planned} q`
    : partial(c) ? (cm(c).subjects || []).join(' + ') : 'full 10k';
  const nPlanned = (c: any) => cm(c).n_planned != null ? cm(c).n_planned : partial(c)
    ? (cm(c).subjects || []).reduce((a: number, s: string) =>
        a + ({ 'Lexicon': 500, 'Research overview': 2000, 'Research publications': 4500,
               'Standards overview': 1000, 'Standards specifications': 2000 } as any)[s], 0)
    : meta.n_questions;
  const runningOf = (c: any) => nPlanned(c) != null && c.n < nPlanned(c);
  const ragOf = (c: any) => cm(c).mode === 'llm_only' ? 'no' : 'yes';
  const running = mine.some(runningOf);
  const ourLabel = (c: any) => (
    <>{c.config}<Info text={`this run — ${cm(c).model ?? ''}, mode ${cm(c).mode ?? ''}${Object.keys(cm(c).overrides || {}).length ? ` (${Object.keys(cm(c).overrides).join(', ')})` : ''}${cm(c).force_answer ? ', forced answer' : ''}; questions: ${scopeOf(c)}${runningOf(c) ? `; ${c.n} of ${nPlanned(c)} so far` : ''}`} /></>
  );
  const ourSubj = (c: any, s: string) => {
    const g = c.mcq_groups?.[s];
    return g?.n ? (g.mcq_acc * 100) : null;
  };
  // A: per-subject. Ours first, then published by overall accuracy.
  const aRows = [
    ...mine.map(c => ({ ours: true, cells: [
      ourLabel(c), 'this system', <Subset text={scopeOf(c)} />, ragOf(c),
      ...SUBJECT_KEYS.map(s => pctCell(ourSubj(c, s))), partial(c) ? '—' : pctCell(c.mcq_acc * 100)] })),
    // No-RAG rows by overall, then the RAG rows (which have only the two Standards
    // subjects, so an overall-based sort would drop them to the bottom anyway).
    ...[...PUB.A]
      .sort((x: any, y: any) => Number(x.rag) - Number(y.rag)
                                || (y.acc.overall ?? 0) - (x.acc.overall ?? 0))
      .map((r: any) => ({ cells: [
        <SysName name={r.name} />, <Src r={r} />, <Subset text={r.subset} />, r.rag ? 'yes' : 'no',
        ...SUBJECT_KEYS.map(s => pctCell(r.acc[s])), pctCell(r.acc.overall)] })),
  ];

  // B: overall only. Ours inserted at its rank so the reader sees where it lands.
  const bAll = [
    ...mine.filter(c => !partial(c)).map(c => ({ ours: true, v: c.mcq_acc * 100, cells: [
      ourLabel(c), 'this system', <Subset text="full 10k" />, ragOf(c)] })),
    ...PUB.B.map((r: any) => ({ v: r.overall, cells: [
      <SysName name={r.name} />, <Src r={r} />, <Subset text={r.subset} />, r.rag ? 'yes' : 'no'] })),
  ].sort((x, y) => y.v - x.v);
  const bShown = showAllB ? bAll : bAll.filter((r, i) => i < 40 || r.ours);
  const bRows = bShown.map((r: any) => ({ ours: r.ours, cells: [...r.cells, pctCell(r.v)] }));

  // C: 3GPP-only subsets, one row per system; the column says which part of the set.
  const cRows = [
    ...mine.map(c => {
      const sub = (k: string) => {
        const g = c[`mcq_${k}`];
        return g?.n ? <>{(g.mcq_acc * 100).toFixed(1)} <span className="text-stone-400">({g.n})</span></> : '—';
      };
      return { ours: true, cells: [
        ourLabel(c), 'this system',
        <span className="whitespace-nowrap">TeleQnA, same filter<Info text={
          'Std. overview / specifications: the whole TeleQnA subject. 3GPP total: the 1,810 ' +
          'questions tagged "[3GPP Release N]" — close to, but not provably the same as, ' +
          'Telco-oRAG\'s 1,840. Rel.18: the 780 questions tagged "[3GPP Release 18]", the ' +
          'same count as Chat3GPP\'s Rel.18 split. (n) = questions answered so far.'} /></span>,
        ragOf(c),
        pctCell(ourSubj(c, 'Standards overview')), pctCell(ourSubj(c, 'Standards specifications')),
        sub('3gpp_tagged'), sub('rel18')] };
    }),
    ...PUB.C.map((r: any) => ({ cells: [
      <SysName name={r.name} />, <Src r={r} />, <Subset text={r.subset} />, r.rag ? 'yes' : 'no',
      pctCell(r.cols.overview), pctCell(r.cols.specs), pctCell(r.cols.total), pctCell(r.cols.rel18)] })),
  ];

  // Question-set filter: Sheet 1 lists every config of the run on ONE set, so the
  // configs compare row against row. Kept in the URL (`cmp`) so a comparison survives
  // a reload and can be linked.
  const CMP_KEYS: [string, string][] = [
    ['overall', 'Overall'],
    ...SUBJECT_KEYS.map(s => [s, s] as [string, string]),
    ['3gpp_tagged', '3GPP total'], ['rel18', 'Rel.18 (780 q)'],
  ];
  const [cmp, setCmpState] = React.useState<string>(() => {
    const k = new URLSearchParams(location.search).get('cmp');
    return CMP_KEYS.some(([x]) => x === k) ? k! : 'overall';
  });
  const setCmp = (k: string) => {
    setCmpState(k);
    const p = new URLSearchParams(location.search);
    k === 'overall' ? p.delete('cmp') : p.set('cmp', k);
    history.replaceState(history.state, '', location.pathname + (p.toString() ? '?' + p : ''));
  };
  const cmpLabel = CMP_KEYS.find(([k]) => k === cmp)?.[1] ?? cmp;
  const isOverall = cmp === 'overall';
  const statOf = (c: any) => {
    const g = isOverall ? { n: c.n, mcq_acc: c.mcq_acc }
      : SUBJECT_KEYS.includes(cmp) ? c.mcq_groups?.[cmp] : c[`mcq_${cmp}`];
    return g?.n ? g : null;
  };
  // Config keys are `<system>_<model-slug>` (teleqna_to_run.py); the slug never holds
  // `_`. meta.mode cannot give the system: bm25_dense and _norr share mode bm25_dense.
  const sysOf = (c: any) => c.config.replace(/_[^_]+$/, '');
  const modelOf = (c: any) => cm(c).model ?? c.config.slice(sysOf(c).length + 1);
  // Gap to the same generator without retrieval, only over the same questions: on
  // Overall a Standards-only config and a full-10k llm_only are different sets.
  const deltaOf = (c: any) => {
    if (sysOf(c) === 'llm_only') return null;
    const b = mine.find(x => sysOf(x) === 'llm_only' && modelOf(x) === modelOf(c));
    const g = statOf(c), bg = b && statOf(b);
    if (!g || !bg || (isOverall && partial(b) !== partial(c))) return null;
    return (g.mcq_acc - bg.mcq_acc) * 100;
  };
  const sortedConfigs = [...mine].sort((a, b) =>
    (statOf(b)?.mcq_acc ?? -1) - (statOf(a)?.mcq_acc ?? -1) || a.config.localeCompare(b.config));
  const judgeTip = (c: any) => c.trust ? `trusted = right option AND justification matches TeleQnA's official explanation AND nothing invented; verifiable = trusted AND the retrieved 3GPP passages prove it. Judge ${c.trust.model} (${c.trust.prompt_version}).` : '';

  return (
    <div className="space-y-10">
      <Sheet n={1} title={<>TeleQnA — multiple-choice accuracy: {cmpLabel}</>}
             note={<>
               <b>Δ no-RAG</b> = gap to the same generator's llm_only on the same questions.
               {running && <> <span className="text-rose-700">Still running: {mine.filter(runningOf)
                 .map(c => `${c.config} ${c.n}/${nPlanned(c)}`).join(', ')} — not final.</span></>}
             </>}>
        <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
          <span className="text-stone-500 mr-1">Question set</span>
          {CMP_KEYS.map(([k, label]) => (
            <button key={k} onClick={() => setCmp(k)}
                    className={`rounded-full px-2.5 py-1 ring-1 ${cmp === k
                      ? 'bg-stone-800 text-white ring-stone-800'
                      : 'bg-white text-stone-600 ring-stone-200 hover:ring-stone-400'}`}>
              {label}
            </button>
          ))}
        </div>
        {/* The judge columns are over the answers judged so far, which catches up with
            `n` once the judge finishes. No-letter and p50 are whole-config figures, so
            they appear on Overall only. */}
        <PubTable
          head={['Config', 'Generator', ...(isOverall ? ['Scope'] : []), 'n', 'Accuracy', 'Δ no-RAG',
                 'Judge: trusted', 'Judge: verifiable',
                 ...(isOverall ? ['No letter parsed', 'p50 (s)'] : [])]}
          rows={sortedConfigs.map(c => {
            const g = statOf(c), t = c.trust?.[cmp], d = deltaOf(c);
            const p = (v: any, n: any) => n ? `${(v * 100).toFixed(1)}%` : '—';
            const tip = judgeTip(c);
            return { ours: !!g, sub: !g, cells: [
              <button onClick={() => onPick?.(c.config)} className="text-left">{ourLabel(c)}{tip && <Info text={tip} />}</button>,
              modelOf(c),
              ...(isOverall ? [<Subset text={scopeOf(c)} />] : []),
              g ? num(g.n, 0) : '—', p(g?.mcq_acc, g?.n),
              d == null ? '—' : <span className={d >= 0 ? 'text-emerald-700' : 'text-rose-700'}>{d >= 0 ? '+' : ''}{d.toFixed(1)}</span>,
              p(t?.trusted, t?.n), p(t?.verifiable, t?.n),
              ...(isOverall ? [pct(c.mcq_extract_fail), num(c.lat_p50, 1)] : [])] };
          })} />
      </Sheet>

      <Sheet n={2} title="By subject — published per-subject results (%)"
             note={<>
               All five TeleQnA subjects, full 10k. Published rows are <b>not measured here</b> —
               each source uses its own prompt and answer parsing (hover the source for the table
               and bib key). The human row is 30 professionals answering 10 questions each, not
               the 10k set. <b>No published work reports RAG accuracy on all five subjects</b>
               (checked 2026-09-27); the two RAG rows are Telco-oRAG's Standards overview /
               specifications over its 1,840-question 3GPP subset — nearest available, not the
               same questions. TeleMoM (arXiv 2504.02712) evaluates open models on the full 10k.
             </>}>
        <PubTable head={['System', 'Source', 'Question set', 'RAG', ...SUBJECT_KEYS, 'Overall']} rows={aRows} />
      </Sheet>

      <Sheet n={3} title="Overall only — leaderboards and recent papers (%)"
             note={<>
               Single TeleQnA accuracy, no per-subject breakdown. Examples from the GSMA Open Telco
               leaderboard (CSV of 2026-09-15; ot-full ~10k or ot-lite ~1k — the two do not rank
               against each other): AT&amp;T's telecom-trained OTel-LLM-8.3B, the Qwen3 / Gemma3
               families this project generates with, two <b>frontier</b> models. <b>No RAG system
               reports on ot-full or ot-lite</b>, so the RAG rows are the nearest available: their
               own 3GPP subsets (see Question set) — not the same questions as the GSMA rows.
             </>}>
        <PubTable head={['System', 'Source', 'Question set', 'RAG', 'Overall']} rows={bRows} />
        {bAll.length > 40 && (
          <button onClick={() => setShowAllB(v => !v)}
                  className="mt-2 text-[11px] text-stone-500 underline hover:text-stone-800">
            {showAllB ? 'show top 40' : `show all ${bAll.length}`}
          </button>
        )}
      </Sheet>

      <Sheet n={4} title="3GPP-only subsets — RAG papers (%)"
             note={<>
               These papers evaluate a <b>different question set</b>: the 1,840 3GPP-related
               TeleQnA questions (Telco-oRAG) or the 780 Release-18 questions (Chat3GPP; its
               Release-17 half is left out because this system's corpus is Release 18 only).
               Our <b>Rel.18</b> cell uses the 780 questions tagged "[3GPP Release 18]" — the same
               count as Chat3GPP's split, so that column compares like with like; our 3GPP total
               (1,810 tagged questions) only approximates Telco-oRAG's 1,840. Numbers one paper copies from another are left out, and Telco-RAG is absent
               because it evaluated GPT-3.5 only. The Phi-2 rows ("Telecom Language Models: Must They Be
               Large?", arXiv 2403.04666) are on the whole TeleQnA Standards subjects, the same
               questions as our Std. columns — a small local model with and without RAG.
             </>}>
        <PubTable head={['System', 'Source', 'Question set', 'RAG', 'Std. overview', 'Std. specifications',
                         '3GPP total', 'Rel.18']} rows={cRows} />
      </Sheet>

      {/* Release 18 only: the one release the corpus holds, and the 780 questions
          Chat3GPP reports as its Rel.18 split (same count as the "[3GPP Release 18]"-tagged
          TeleQnA questions), so every row here is measured on the same questions. */}
      <Sheet n={5} title="Release 18 — the corpus's own release (780 questions)"
             note={<>
               The 780 TeleQnA questions tagged "[3GPP Release 18]" — all in Standards
               specifications, the release this system's corpus is built from, and the same count
               Chat3GPP reports for its Rel.18 split. Rows of this system are paired on identical
               questions; <b>trusted</b> / <b>verifiable</b> come from the trust judge (a model without
               retrieval is verifiable 0 by construction). Published rows use their own prompt and
               parsing. Rel.18 results that a paper copies from another are left out.
             </>}>
        <PubTable
          head={['System', 'Source', 'Model', 'RAG', 'Rel.18 accuracy', 'n', 'Judge: trusted', 'Judge: verifiable']}
          rows={[
            ...mine.filter(c => c.mcq_rel18?.n)
              .sort((a, b) => b.mcq_rel18.mcq_acc - a.mcq_rel18.mcq_acc)
              .map(c => {
                const t = c.trust?.rel18;
                return { ours: true, cells: [
                  ourLabel(c), 'this system', cm(c).model ?? '—', ragOf(c),
                  pctCell(c.mcq_rel18.mcq_acc * 100), num(c.mcq_rel18.n, 0),
                  t?.n ? pctCell(t.trusted * 100) : '—',
                  t?.n ? pctCell(t.verifiable * 100) : '—'] };
              }),
            ...PUB.C.filter((r: any) => r.cols?.rel18 != null).map((r: any) => ({ cells: [
              <SysName name={r.name} />, <Src r={r} />, 'Llama3-8B', r.rag ? 'yes' : 'no',
              pctCell(r.cols.rel18), '780', '—', '—'] })),
          ]} />
      </Sheet>


    </div>
  );
}

// Cross-run page: every run measured on one question set, config by config. The same
// questions, the same judge and the same rubric on every column, so a difference
// between two columns in a row is the generator (and how it was served) — nothing else.
// Sheets 1-3 are laid out as the tables the paper needs; each has a LaTeX copy button.
type CmpMetric = { key: string; head: string; lower?: boolean; group?: boolean;
                   fmt: (v: number | null | undefined) => string };
const CMP_METRICS: CmpMetric[] = [
  { key: 'recall@5', head: 'Recall@5', group: true, fmt: v => num(v) },
  { key: 'recall@1', head: 'Recall@1', fmt: v => num(v) },
  { key: 'recall@10', head: 'Recall@10', fmt: v => num(v) },
  { key: 'mrr', head: 'MRR', fmt: v => num(v) },
  { key: 'judge_score_strict', head: 'Judge strict', group: true, fmt: v => num(v) },
  { key: 'judge_score_cond', head: 'Judge cond', group: true, fmt: v => num(v) },
  { key: 'judge_abstain', head: 'Abstain', lower: true, group: true, fmt: v => pct(v) },
  { key: 'faithfulness', head: 'Faithfulness', fmt: v => num(v) },
  { key: 'lat_p50', head: 'Latency p50 (s)', lower: true, fmt: v => num(v, 1) },
];

const runTag = (r: any) => r.run.split('_')[0];
const texEsc = (x: string) => x.replace(/\\/g, '\\textbackslash{}').replace(/([&%$#_{}])/g, '\\$1')
  .replace(/→/g, '$\\rightarrow$').replace(/−/g, '$-$').replace(/…/g, '\\ldots{}').replace(/×/g, '$\\times$');

function CopyTex({ tex }: { tex: () => string }) {
  const [ok, setOk] = React.useState(false);
  return (
    <button onClick={() => { navigator.clipboard?.writeText(tex()).then(() => { setOk(true); setTimeout(() => setOk(false), 1500); }); }}
            className="rounded px-2 py-1 text-[11px] ring-1 ring-stone-200 bg-white text-stone-600 hover:bg-stone-50">
      {ok ? 'copied' : 'Copy LaTeX'}
    </button>
  );
}

function PaperBadge({ what }: { what: string }) {
  return <span className="ml-2 rounded px-1.5 py-px text-[9px] uppercase tracking-wide bg-emerald-50
                          text-emerald-700 ring-1 ring-emerald-200 align-middle">paper · {what}</span>;
}

function Chips({ items, value, onChange }:
  { items: { key: string; head: string }[]; value: string; onChange: (k: string) => void }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map(m => (
        <button key={m.key} onClick={() => onChange(m.key)}
                className={`rounded-full px-2.5 py-1 text-[11px] ring-1 transition
                            ${m.key === value ? 'bg-stone-800 text-white ring-stone-800'
                                              : 'bg-white text-stone-600 ring-stone-200 hover:bg-stone-50'}`}>
          {m.head}
        </button>
      ))}
    </div>
  );
}

function RunHead({ r }: { r: any }) {
  return (
    <th className="font-medium px-2 py-2 align-bottom">
      <div className="text-stone-800">{r.model || '?'}</div>
      <div className="text-[10px] font-normal font-mono text-stone-400">
        {runTag(r)}{r.paper && <span className="ml-1 text-emerald-600">(paper)</span>}
      </div>
    </th>
  );
}

/** Spread = max − min across generators. Near 0 for configs whose retrieval does not
 *  touch the generator; large where the generator writes the Cypher. */
function spreadOf(vals: (number | null | undefined)[]) {
  const have = vals.filter((v): v is number => typeof v === 'number');
  return have.length > 1 ? Math.max(...have) - Math.min(...have) : null;
}

function CompareMatrix({ runs, configs, metric, value, onOpen }: {
  runs: any[]; configs: string[]; metric: CmpMetric;
  value: (run: any, config: string) => number | null | undefined;
  onOpen?: (run: string, config: string) => void;
}) {
  return (
    <div className="-mx-4 md:mx-0 overflow-x-auto">
      <table className="min-w-[38rem] w-full text-xs border-separate" style={{ borderSpacing: '0 4px' }}>
        <thead className="text-stone-500">
          <tr>
            <th className="text-left font-medium px-3 py-2">System</th>
            {runs.map(r => <RunHead key={r.run} r={r} />)}
            <th className="font-medium px-2 py-2 align-bottom" title="max − min across generators">spread</th>
          </tr>
        </thead>
        <tbody>
          {configs.map(c => {
            const vals = runs.map(r => value(r, c));
            const have = vals.filter((v): v is number => typeof v === 'number');
            // No winner when every column ties (LLM-only's Recall is 0 everywhere).
            const best = have.length > 1 && Math.min(...have) !== Math.max(...have)
              ? (metric.lower ? Math.min(...have) : Math.max(...have)) : null;
            const sp = spreadOf(vals);
            return (
              <tr key={c} className={`${ROW} [&>td]:bg-white`}>
                <td className="px-3 py-2 whitespace-nowrap">{label(c)}</td>
                {runs.map((r, i) => {
                  const v = vals[i];
                  const running = r.running?.[c];
                  return (
                    <td key={r.run}
                        onClick={() => typeof v === 'number' && onOpen?.(r.run, c)}
                        className={`px-2 py-2 text-center tabular-nums whitespace-nowrap
                                    ${typeof v === 'number' && onOpen ? 'cursor-pointer hover:!bg-stone-50' : ''}
                                    ${r.paper ? '!bg-emerald-50/40' : ''}
                                    ${best !== null && v === best ? 'font-semibold text-stone-900' : 'text-stone-600'}`}>
                      {typeof v === 'number' ? metric.fmt(v)
                        : running != null ? <span className="italic text-stone-400">running {running}</span>
                        : <span className="text-stone-300">—</span>}
                    </td>
                  );
                })}
                <td className="px-2 py-2 text-center tabular-nums text-stone-400">
                  {sp === null ? '—' : metric.key === 'judge_abstain' ? pct(sp) : metric.fmt(sp)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function matrixTex(runs: any[], configs: string[], metric: CmpMetric, caption: string,
                   value: (r: any, c: string) => number | null | undefined) {
  const cols = 'l' + 'c'.repeat(runs.length);
  const head = ['System', ...runs.map(r => texEsc(r.model || runTag(r)))].join(' & ');
  const body = configs.map(c => {
    const vals = runs.map(r => value(r, c));
    const have = vals.filter((v): v is number => typeof v === 'number');
    const best = have.length > 1 && Math.min(...have) !== Math.max(...have)
      ? (metric.lower ? Math.min(...have) : Math.max(...have)) : null;
    return [texEsc(label(c)), ...vals.map(v => typeof v !== 'number' ? '--'
      : v === best ? `\\textbf{${texEsc(metric.fmt(v))}}` : texEsc(metric.fmt(v)))].join(' & ') + ' \\\\';
  });
  return ['\\begin{table}[t]', '\\centering', '\\small', `\\caption{${texEsc(caption)}}`,
          `\\begin{tabular}{${cols}}`, '\\toprule', head + ' \\\\', '\\midrule', ...body,
          '\\bottomrule', '\\end{tabular}', '\\end{table}'].join('\n');
}

const fmtP = (p: number, B: number) => (p <= 1 / B ? `<${(1 / B).toFixed(4)}` : p.toFixed(3));
const EXPECT: Record<string, string> = { pos: 'Δ > 0', neg: 'Δ < 0', zero: 'Δ ≈ 0 (CI ∋ 0)' };

function ClaimsTable({ cmp, runs }: { cmp: any; runs: any[] }) {
  const B = cmp.boot?.B || 2000;
  return (
    <div className="-mx-4 md:mx-0 overflow-x-auto">
      <table className="min-w-[44rem] w-full text-xs border-separate" style={{ borderSpacing: '0 4px' }}>
        <thead className="text-stone-500">
          <tr>
            <th className="text-left font-medium px-3 py-2">Claim in the paper draft</th>
            <th className="font-medium px-2 py-2 align-bottom">needs</th>
            {runs.map(r => <RunHead key={r.run} r={r} />)}
          </tr>
        </thead>
        <tbody>
          {cmp.claims.map((c: any) => (
            <tr key={c.key} className={`${ROW} [&>td]:bg-white`}>
              <td className="px-3 py-2">
                <div>{c.text}</div>
                <div className="text-[10px] text-stone-400">
                  {short(c.a)} − {short(c.b)}{c.class ? ` · ${c.class} only` : ''}
                </div>
              </td>
              <td className="px-2 py-2 text-center text-[11px] text-stone-500 whitespace-nowrap">{EXPECT[c.expected]}</td>
              {runs.map(r => {
                const v = r.claims?.[c.key];
                if (!v) return <td key={r.run} className="px-2 py-2 text-center text-stone-300">pending</td>;
                return (
                  <td key={r.run} className={`px-2 py-2 text-center tabular-nums whitespace-nowrap
                                               ${v.holds ? '!bg-emerald-50' : '!bg-rose-50'}`}
                      title={`${short(c.a)} ${v.a.toFixed(3)} vs ${short(c.b)} ${v.b.toFixed(3)} · n=${v.n}`}>
                    <div className={`font-semibold ${v.holds ? 'text-emerald-700' : 'text-rose-700'}`}>
                      {v.holds ? '✓' : '✗'} {v.delta >= 0 ? '+' : ''}{v.delta.toFixed(3)}
                    </div>
                    <div className="text-[10px] text-stone-500">
                      [{v.lo.toFixed(3)}, {v.hi.toFixed(3)}] · p {fmtP(v.p, B)}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function claimsTex(cmp: any, runs: any[]) {
  const B = cmp.boot?.B || 2000;
  const head = ['Claim', ...runs.map(r => texEsc(r.model || runTag(r)))].join(' & ');
  const body = cmp.claims.map((c: any) => [texEsc(c.text), ...runs.map(r => {
    const v = r.claims?.[c.key];
    if (!v) return '--';
    const d = `${v.delta >= 0 ? '+' : ''}${v.delta.toFixed(3)}`;
    return `${v.holds ? '' : '\\textit{'}${d} [${v.lo.toFixed(3)}, ${v.hi.toFixed(3)}]${v.holds ? '' : '}'}`;
  })].join(' & ') + ' \\\\');
  return ['\\begin{table*}[t]', '\\centering', '\\small',
          `\\caption{Paired differences (95\\% bootstrap CI, B=${B}) per generator on the same 3,000 questions. Italic: the claim does not hold for that generator.}`,
          `\\begin{tabular}{p{6.5cm}${'c'.repeat(runs.length)}}`, '\\toprule', head + ' \\\\', '\\midrule',
          ...body, '\\bottomrule', '\\end{tabular}', '\\end{table*}'].join('\n');
}

function CompareOverview({ ov, onOpen }:
  { ov: any; onOpen?: (run: string, config: string) => void }) {
  const cmp = ov.compare;
  const runs: any[] = cmp.runs || [];
  const [metric, setMetric] = React.useState('recall@5');
  const [gMetric, setGMetric] = React.useState('recall@5');
  const [group, setGroup] = React.useState<string>(cmp.groups?.[0] || 'factoid');
  const [showAll, setShowAll] = React.useState(false);
  const measured = (c: string) => runs.some(r => r.configs?.[c] || r.running?.[c] != null);
  const main = (cmp.config_order as string[]).filter(c => c.startsWith('A_') && measured(c));
  const extra = (cmp.config_order as string[]).filter(c => !c.startsWith('A_') && measured(c));
  const configs = showAll ? [...main, ...extra] : main;
  const m = CMP_METRICS.find(x => x.key === metric)!;
  const gm = CMP_METRICS.find(x => x.key === gMetric)!;
  const judges = [...new Set(runs.map(r => r.judge_model).filter(Boolean))];
  const rubrics = [...new Set(runs.map(r => r.judge_rubric).filter(Boolean))];
  const v1 = (r: any, c: string) => r.configs?.[c]?.[metric];
  // 'all' = the whole set, i.e. the overall value rather than a class cell.
  const v2 = (r: any, c: string) => group === 'all' ? r.configs?.[c]?.[gMetric]
                                                    : r.configs?.[c]?.groups?.[group]?.[gMetric];
  return (
    <div className="space-y-10">
      <Sheet n={1} title={<>Do the paper's claims hold for every generator?<PaperBadge what="robustness table" /></>}
             note={<>Paired difference over the same questions, 95% bootstrap CI (B={cmp.boot?.B}, seed {cmp.boot?.seed}),
               two-sided p. ✓ = the difference has the sign the claim needs and its CI excludes 0
               (for "≈ 0", the CI includes 0). Hover a cell for the two means.</>}>
        <div className="flex justify-end"><CopyTex tex={() => claimsTex(cmp, runs)} /></div>
        <ClaimsTable cmp={cmp} runs={runs} />
      </Sheet>

      <Sheet n={2} title={<>Table 1 across generators<PaperBadge what="Table 1 + generator columns" /></>}
             note="Bold = best in the row. Shaded column = the run the paper's tables come from. Spread = max − min across generators: near 0 where retrieval never touches the generator.">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Chips items={CMP_METRICS} value={metric} onChange={setMetric} />
          <div className="flex items-center gap-3">
            {extra.length > 0 && (
              <label className="flex items-center gap-1.5 text-[11px] text-stone-500 cursor-pointer">
                <input type="checkbox" checked={showAll} onChange={e => setShowAll(e.target.checked)} />
                ablation / study ({extra.length})
              </label>
            )}
            <CopyTex tex={() => matrixTex(runs, configs, m, `${m.head} per configuration and generator, ${cmp.n_questions} questions.`, v1)} />
          </div>
        </div>
        <CompareMatrix runs={runs} configs={configs} metric={m} onOpen={onOpen} value={v1} />
      </Sheet>

      <Sheet n={3} title={<>Table 2 across generators (per class)<PaperBadge what="Table 2" /></>}
             note="1,000 questions per class. KG and text retrieval swap places between factoid and requirement, so the class rows, not the overall row, carry the result.">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <Chips items={[{ key: 'all', head: 'all' }, ...(cmp.groups as string[]).map(g => ({ key: g, head: g }))]} value={group} onChange={setGroup} />
            <span className="text-stone-300">|</span>
            <Chips items={CMP_METRICS.filter(x => x.group)} value={gMetric} onChange={setGMetric} />
          </div>
          <CopyTex tex={() => matrixTex(runs, main, gm, `${gm.head} on ${group === 'all' ? 'all' : group} questions per configuration and generator.`, v2)} />
        </div>
        <CompareMatrix runs={runs} configs={main} metric={gm} onOpen={onOpen} value={v2} />
      </Sheet>

      <Sheet n={4} title="Runs on this set">
        <PubTable head={['Run', 'Generator', 'Judge', 'Rubric', 'Configs done', 'Running']}
          rows={runs.map(r => ({ ours: r.paper, cells: [
            <span className="font-mono">{r.run}</span>, r.model || '—',
            r.judge_model || <span className="text-stone-400">not judged</span>, r.judge_rubric || '—',
            Object.keys(r.configs || {}).length,
            Object.entries(r.running || {}).map(([c, n]) => `${short(c)} ${n}/${cmp.n_questions}`).join(', ') || '—',
          ] }))} />
        <ul className="text-[11px] text-stone-500 space-y-1 list-disc pl-5">
          <Caveat on="Recall">
            BM25, Dense and BM25+Dense retrieve without the generator, so their Recall should agree
            across columns (spread ≈ 0); a gap there points at a configuration fault, not at the
            model. KG, Dense+KG and the full system include the Cypher the generator writes, so their
            recall moves with the model — report them as "KG + Cypher-LLM".
          </Caveat>
          <Caveat on="Judge">
            {judges.length <= 1 && rubrics.length <= 1
              ? <>one judge ({judges[0] || '—'}) and one rubric ({rubrics[0] || '—'}) across every run, so judge scores compare directly.</>
              : <span className="text-rose-700">runs were judged by different models or rubrics ({judges.join(' / ')}; {rubrics.join(' / ')}) — judge columns do not compare until re-judged.</span>}
          </Caveat>
          <Caveat on="Pending">
            a claim shows "pending" until the run has finished both configs it compares.
          </Caveat>
        </ul>
      </Sheet>
    </div>
  );
}

export function RunOverview({ ov, active, onPick, onOpen }:
  { ov: any; active?: string; onPick?: (config: string) => void;
    onOpen?: (run: string, config: string) => void }) {
  if (ov?.compare) return <CompareOverview ov={ov} onOpen={onOpen} />;
  if ((ov?.configs || []).some((c: any) => c.mcq_acc != null)) {
    return <TeleQnAOverview ov={ov} active={active} onPick={onPick} />;
  }
  const all: any[] = ov?.configs || [];
  const byOrder = (a: any, b: any) =>
    (CONFIGS[a.config]?.order ?? 999) - (CONFIGS[b.config]?.order ?? 999);
  const inGroup = (g: string) =>
    all.filter(c => CONFIGS[c.config]?.group === g).sort(byOrder);

  const phaseA = inGroup('phaseA');
  const retrieval = phaseA.filter(c => c.config !== 'A_llm_only');
  const kgabl = inGroup('kgabl');
  // The study ladder is ordered by the registry, not by key prefix: each step must
  // change exactly one variable, and two of its rows are phaseA configs.
  const study = (LADDERS.study || [])
    .map(k => all.find(c => c.config === k))
    .filter(Boolean) as any[];
  // Sheet 2 shows only the six headline groups — the three form types plus the
  // three structure tags. `group_order` is the full BM.GROUPS list, which also
  // carries the per-mechanism subclasses (message, sbi_operation, ...) used by
  // kg_win_100; rendering all 16 turned one comparison into a wall of blocks and
  // most rows were n=0 for any given set. Empty groups are dropped, so a run whose
  // questions are all mechanism-typed (kg_win_100) falls back to whatever of the
  // six it does populate rather than showing six blank blocks.
  const MAIN_GROUPS = ['factoid', 'procedure', 'requirement',
                       'multi_hop', 'cross_specification', 'cross_section'];
  const groupOrder: string[] = ov?.group_order || [];
  const populated = (g: string) => all.some(c => (c.groups?.[g]?.n ?? 0) > 0);
  const groups = MAIN_GROUPS.filter(g => groupOrder.includes(g)).filter(populated);
  // The remaining BM.GROUPS split by AXIS rather than being one long list: a
  // mechanism (message, sbi_operation) answers "which graph structure carries this
  // question", a structure tag answers "how is the evidence laid out". Mixing them
  // into one run of blocks was what made all 16 unreadable in the first place.
  const MECHANISM = ['message', 'sbi_operation', 'information_element',
                     'standardised_value', 'multi_actor_procedure'];
  const STRUCTURE = ['standardized_value', 'multi_chunk_aggregation',
                     'procedure_flow', 'normative_requirement'];
  const extraAxes = [
    { title: 'By graph mechanism', keys: MECHANISM.filter(populated),
      note: 'Which node type the question is built around. A question belongs to at ' +
            'most one of these, but they cover only the entity-named classes.' },
    { title: 'By evidence structure', keys: STRUCTURE.filter(populated),
      note: 'How the gold is laid out. These overlap each other and the form types, ' +
            'so they do not partition the set.' },
  ].filter(a => a.keys.length > 0);

  // Render one group block. Factored out because Sheet 2 now draws it in three places:
  // the six main groups and two secondary axes inside <details> — same reading, different groups.
  const groupBlock = (g: string) => {
    // A group is judged only if some config in it has verdicts. Deciding this
    // per group, not per run, keeps the extra bars off blocks where the judge
    // has not reached yet instead of drawing a row of empty tracks.
    const judged = retrieval.some(c => c.groups?.[g]?.judge_score_strict != null);
    return (
    <div key={g} className="rounded-xl bg-white ring-1 ring-stone-200 p-4">
      <div className="text-[12px] font-medium text-stone-700 mb-2">
        {g}
        <span className="text-stone-400 font-normal">
          {' '}(n = {retrieval[0]?.groups?.[g]?.n ?? 0})
        </span>
      </div>
      {judged && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 mb-2.5">
          {['R@5', 'Judge (strict)'].map((n, i) => (
            <span key={n} className="flex items-center gap-1.5 text-[10px] text-stone-500">
              <span className="h-2 w-2 rounded-sm" style={{ background: seriesColor(i) }} />
              {n}
            </span>
          ))}
        </div>
      )}
      <div className="space-y-2">
        {retrieval.map(c => {
          const cell = c.groups?.[g] || {};
          return (
          // Each config is one framed block once it carries three bars: an
          // unframed run of 24 bars reads as one list, and it stops being
          // obvious which three belong to the same system.
          <div key={c.config}
               className={judged
                 ? `grid grid-cols-1 ${LABEL_GRID} gap-x-3 gap-y-1 items-center
                    rounded-md border border-stone-200 px-2.5 py-1.5`
                 : 'flex items-center gap-3'}>
            {/* No `truncate` here: the longest registry label is 42 chars
                ("Dense, no KG term expansion -> rerank -> gen") and the old
                w-48 (192px) cut it mid-word, so two configs that differ only
                in their suffix rendered identically. LABEL_COL fits it on one
                line at md+; below that it wraps rather than hiding. */}
            <span className={`${judged ? '' : LABEL_COL + ' shrink-0'}
                             text-[12px] leading-tight text-stone-600`}
                  title={label(c.config)}>
              {label(c.config)}
              {judged && cell.judge_abstain > 0 && (
                // Abstain is the one number that explains a strict/cond gap,
                // so it sits on the label rather than as a fourth bar — it is
                // a property of the row, not another score to compare.
                <span className="ml-1.5 text-[10px] text-stone-400 whitespace-nowrap">
                  abstain {pct(cell.judge_abstain)}
                </span>
              )}
            </span>
            {judged ? (
              <div className="space-y-1">
                {/* strict, not cond: R@5 sits beside it to compare "was the passage
                    found" with "was the answer right", and that comparison only holds
                    when both share a denominator. R@5 is over the whole group; strict
                    is too, while cond drops the questions the config declined, so its
                    denominator moves per config. Measured on v54's requirement group:
                    KG finds 9.5% of the passages yet scores cond 0.644 — next to BM25's
                    0.961 — because it abstains on 77.4%. The abstain column beside these
                    bars carries exactly that, and strict + abstain recovers cond. */}
                {[['R@5', 'recall@5'], ['Judge (strict)', 'judge_score_strict']].map(([nm, k], i) => (
                  <div key={k} className="flex items-center gap-2">
                    {/* w-24: "Judge (strict)" is 14 chars — w-9 (36px) wrapped it onto
                        two lines and pushed the bar out of alignment with the row above. */}
                    <span className="w-24 shrink-0 text-[10px] text-stone-400 text-right">
                      {nm}
                    </span>
                    <Bar v={cell[k]} color={seriesColor(i)} />
                  </div>
                ))}
              </div>
            ) : (
              <Bar v={cell['recall@5']} />
            )}
          </div>
          );
        })}
      </div>
    </div>
    );
  };


  return (
    <div className="space-y-10">
      <Sheet n={1} title="Answer quality — is the answer right?"
             note={
               <ul className="space-y-0.5 mt-1">
                 <Caveat on="Judge (strict vs cond)">
                   read both — <b>strict</b> counts an abstention as failure, <b>cond</b> only
                   grades what was answered. A dash means not judged, <b>not</b> zero.
                 </Caveat>
                 <Caveat on="Exact Match">
                   0 everywhere — answers end with a citation, so a string match never fires.
                 </Caveat>
                 <Caveat on="Cover-EM">
                   leak-free variant. Raw gives <b>0.302</b> to a config that retrieves nothing,
                   by echoing targets the question contained; leak-free leaves <b>0.009</b>.
                 </Caveat>
                 <Caveat on="Faithfulness (ctx)">
                   skips questions with no retrieved context — 0 there is arithmetic, not evidence.
                 </Caveat>
                 <li className="leading-snug text-stone-400 pt-0.5">
                   Judge calibration: 1.000 on a question's own reference, 0.920 on another's.
                 </li>
               </ul>
             }>
        {/* Recall@5 leads, even on the answer-quality sheet: the question this
            whole comparison exists to answer is whether better retrieval yields
            better answers, and that is unreadable when the retrieval number lives
            two sheets away. Seeing them on one row is also what makes the size of
            the effect visible — a 0.05 Recall gap moving the judge score by ~0.01
            is the arithmetic behind "the retrieval configs are indistinguishable
            on correctness", not a contradiction of it. */}
        <GroupedBars rows={phaseA} series={[
          { name: 'R@5', key: 'recall@5' },
          { name: 'Judge (strict)', key: 'judge_score_strict' },
          { name: 'Faith', key: 'faithfulness_ctx' },
        ]} />
        {/* TWO tables, deliberately not one. The judge answers "is it right?";
            the lexical block answers "how much surface text overlaps?" — and on
            this task the second question has no discriminating answer at all
            (EM 0.000 everywhere; F1/ROUGE-L spread under 0.02 across the text
            configs). Interleaving them in one row invites reading a 0.37 F1 as if
            it were comparable evidence to a 0.60 judge score. Separated, the
            lexical table reads as what it is: the control that shows why these
            metrics were not used to rank anything. */}
        <div className="text-[11px] font-semibold uppercase tracking-wider text-stone-400 mt-4 mb-2">
          Correctness — LLM judge
        </div>
        <Table
          // One row per system, NOT the doubled "…— answered only" listing this
          // table used to carry. That second row existed to show what a config
          // scores over the questions it did answer — which is exactly what the
          // Judge (cond) column now reports, next to the abstention rate that
          // explains the gap. Two rows per system to say what two columns say is
          // noise, and it left every judge cell of the duplicate row blank.
          rows={phaseA}
          cols={[
            // All three judge numbers are needed: strict alone misdescribes a system
            // that declines rather than guesses (KG refuses on ~30% of the set, so
            // its low strict score is silence, not error).
            //
            // The judge's own sample size is deliberately NOT a column: it is asked
            // for on a whole run, so once judging finishes it equals `n` for every
            // row and only adds a column of identical numbers. The cost is that a
            // judge run still IN PROGRESS looks finished here — `/api/runs` serves
            // sidecar verdicts as soon as they checkpoint, so a config 80 questions
            // into 2106 shows a score computed from those 80. Read n_judged from the
            // API (it is still in the payload) before quoting a number while a judge
            // pass is running.
            //
            // R@5 repeated from the retrieval sheet ON PURPOSE. Judge scores are
            // only interpretable next to how much evidence the system actually
            // found; without it a low score reads as "bad generator" when it may
            // be "nothing to generate from".
            { key: 'recall@5', head: 'R@5' },
            { key: 'judge_score_strict', head: 'Judge (strict)' },
            { key: 'judge_score_cond', head: 'Judge (cond)' },
            { key: 'judge_abstain', head: 'Abstain', fmt: pct },
            { key: 'faithfulness_ctx', head: 'Faith (ctx)' },
            { key: 'n', head: 'n', d: 0 },
          ]}
          onPick={onPick} active={active}
        />

        <div className="text-[11px] font-semibold uppercase tracking-wider text-stone-400 mt-5 mb-2">
          Lexical overlap — shown as a negative control
          <span className="ml-2 font-normal normal-case tracking-normal text-stone-400">
            none of these separates the systems
          </span>
        </div>
        <Table
          rows={phaseA}
          cols={[
            // Recall first here as well, and this is where it does the most work:
            // it is the yardstick that shows these columns are NOT measuring
            // retrieval quality. KG retrieves the fewest passages of the three
            // (R@5 0.419) yet BM25 and Dense sit within 0.007 of each other on F1
            // while their Recall differs by 0.05 — the lexical numbers track answer
            // LENGTH and phrasing, not whether the evidence was found.
            { key: 'recall@1', head: 'R@1' },
            { key: 'recall@5', head: 'R@5' },
            // EM and Cover-EM hidden (06/10/2026): EM is 0.000 for every config
            // (citation suffix), and kg_bench_3000_v5 carries no cover_targets, so
            // Cover-EM is blank on every run the paper uses.
            { key: 'f1', head: 'F1' },
            { key: 'rouge_l', head: 'ROUGE-L' },
            { key: 'n', head: 'n', d: 0 },
          ]}
          onPick={onPick} active={active}
        />
      </Sheet>

      <Sheet n={2} title="Questions — by question group"
             note={
               <ul className="space-y-0.5 mt-1">
                 <li className="leading-snug">
                   The structure groups overlap the form groups, so the six blocks do not
                   sum to the question total.
                 </li>
                 <Caveat on="Retrieval vs answer">
                   <b>R@5</b> = passages found, <b>judge</b> = answer right. Both over the
                   whole group, so the pair is comparable.
                 </Caveat>
                 <Caveat on="Abstain">
                   share the config declined. Strict scores those zero — KG on requirement
                   declines 77%, so it reads low despite being right on the rest.
                 </Caveat>
               </ul>
             }>
        <div className="space-y-4">
          {groups.map(g => groupBlock(g))}

          {extraAxes.map(axis => (
            // Collapsed by default: this is the breakdown to open WHEN an aggregate looks flat,
            // not the first thing to read. The group count in the title says what opens.
            <details key={axis.title} className="rounded-xl bg-white ring-1 ring-stone-200">
              <summary className="cursor-pointer px-4 py-2.5 text-[12px] font-medium
                                  text-stone-600 hover:text-stone-900 select-none
                                  flex items-center gap-2">
                <span className="text-stone-400">▸</span>
                {axis.title}
                <span className="text-stone-400 font-normal">
                  {' '}({axis.keys.length} groups)
                </span>
              </summary>
              <div className="px-4 pb-4 pt-1 space-y-4">
                <p className="text-[11px] leading-snug text-stone-500 max-w-prose">
                  {axis.note}
                </p>
                {axis.keys.map(g => groupBlock(g))}
              </div>
            </details>
          ))}
        </div>
      </Sheet>

      <Sheet n={3} title="Retrieval quality — did it find the right passages?"
             note="Recall@K here is COVERAGE, not hit rate: the denominator is the question's
                   own gold count, so retrieving one of three gold passages scores 0.333 even
                   when the generated answer is fully correct. It understates end-to-end
                   quality by design and is read as a retrieval diagnostic, not as a proxy
                   for usefulness — which is what the answer-quality sheet above is for.">
        <GroupedBars rows={retrieval} series={[
          { name: 'R@1', key: 'recall@1' },
          { name: 'R@5', key: 'recall@5' },
          { name: 'R@10', key: 'recall@10' },
        ]} />
        <Table
          rows={retrieval}
          cols={[
            { key: 'recall@1', head: 'R@1' },
            { key: 'recall@5', head: 'R@5' },
            { key: 'recall@10', head: 'R@10' },
            { key: 'mrr', head: 'MRR' },
            { key: 'lat_p50', head: 'p50 (s)', d: 1 },
            { key: 'empty_rate', head: 'no chunk retrieved', fmt: pct },
          ]}
          onPick={onPick} active={active} descriptors
        />
      </Sheet>

      {kgabl.length > 0 && (
        <Sheet n={4} title="KG Ablation — one layer added at a time"
               note="Run in KG-only mode so each layer is isolated.">
          <GroupedBars rows={kgabl} series={[
            { name: 'R@5', key: 'recall@5' },
            { name: 'R@10', key: 'recall@10' },
          ]} max={niceMax(kgabl.flatMap((r: any) => [r['recall@5'], r['recall@10']]))} />
          <Table
            rows={kgabl}
            cols={[
              { key: 'recall@1', head: 'R@1' },
              { key: 'recall@5', head: 'R@5' },
              { key: 'recall@10', head: 'R@10' },
              { key: 'mrr', head: 'MRR' },
              { key: 'f1', head: 'F1' },
            ]}
            onPick={onPick} active={active}
          />
        </Sheet>
      )}

      {study.length > 1 && (
        <Sheet n={5} title="Study Ablation — what evidence the model is given"
               note={
                 <ul className="space-y-0.5 mt-1">
                   <li className="leading-snug">
                     Ordered so each step changes exactly ONE variable: evidence type across
                     the first three rows, then +provenance, then +reranker.
                   </li>
                   <Caveat on="Judge, not Cover-EM">
                     the ladder's own numbers show why: <b>LLM-only</b> retrieves nothing
                     (R@5 = 0) yet scores Cover-EM 0.138 — level with triples-only at 0.147.
                     A metric that cannot separate <i>answered from evidence</i> from
                     <i>guessed</i> cannot rank evidence formats.
                   </Caveat>
                   <Caveat on="Abstain">
                     the row that explains the strict/cond gap. Triples-only declines
                     <b> 47%</b> of questions against 9% for the same retrieval delivered as
                     text — the evidence is thinner, not the model worse.
                   </Caveat>
                 </ul>
               }>
          {/* judge + abstain, not Cover-EM/F1: this sheet compares EVIDENCE FORMATS at a
              fixed retriever, so the whole signal is in the answer. F1 and ROUGE-L spread
              0.145-0.157 across the ladder — token overlap barely moves when the same
              passages arrive as triples instead of prose — while judge strict spreads
              0.237 and abstain 0.438. Both stay in the table below for reference. */}
          <GroupedBars rows={study} series={[
            { name: 'Judge (strict)', key: 'judge_score_strict' },
            { name: 'Faith (ctx)', key: 'faithfulness_ctx' },
            { name: 'Abstain', key: 'judge_abstain' },
          ]} />
          <Table
            rows={study}
            cols={[
              { key: 'judge_score_strict', head: 'Judge (strict)' },
              { key: 'judge_score_cond', head: 'Judge (cond)' },
              { key: 'judge_abstain', head: 'Abstain', fmt: pct },
              { key: 'faithfulness_ctx', head: 'Faith (ctx)' },
              { key: 'f1', head: 'F1' },
              { key: 'lat_p50', head: 'p50 (s)', d: 1 },
            ]}
            onPick={onPick} active={active} descriptors
          />
        </Sheet>
      )}
    </div>
  );
}
