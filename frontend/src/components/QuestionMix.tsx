import React, { useMemo, useState } from 'react';

// Composition of a question set. Three top-level classes on the inner ring; click one
// to expand its subclasses on the outer ring.
//
// Three levels of grouping exist in the data and only two of them belong on a pie:
//   question_class     — a partition into three, sums to n
//   question_subclass  — a partition WITHIN a class, sums to that class
//   structure_tags     — overlapping labels that do NOT sum to anything
// The first two nest, so a click-to-expand ring reads correctly. structure_tags are
// listed separately below the chart with their own denominator, because drawing an
// overlapping set as ring slices invites adding it to the partition — which is how a
// 300-question set gets read as six independent classes of ~50.

type Q = {
  question_class?: string;
  question_type?: string;
  question_subclass?: string;
  structure_tags?: string[];
};

const PALETTE = ['#0f766e', '#b45309', '#4338ca', '#a21caf', '#0369a1', '#65a30d', '#be123c'];
const SHADES = ['#0f766e', '#0d9488', '#14b8a6', '#5eead4', '#99f6e4'];

const TAU = Math.PI * 2;

function arc(cx: number, cy: number, r0: number, r1: number, a0: number, a1: number): string {
  // A full circle has coincident endpoints and cannot be one SVG arc, so a lone slice
  // is drawn as two halves rather than collapsing to nothing.
  if (a1 - a0 >= TAU - 1e-6) {
    const m = a0 + Math.PI;
    return `${arc(cx, cy, r0, r1, a0, m)} ${arc(cx, cy, r0, r1, m, a0 + TAU)}`;
  }
  const p = (r: number, a: number) => [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  const [x0, y0] = p(r1, a0), [x1, y1] = p(r1, a1);
  const [x2, y2] = p(r0, a1), [x3, y3] = p(r0, a0);
  const big = a1 - a0 > Math.PI ? 1 : 0;
  return `M${x0} ${y0} A${r1} ${r1} 0 ${big} 1 ${x1} ${y1} L${x2} ${y2} `
       + `A${r0} ${r0} 0 ${big} 0 ${x3} ${y3} Z`;
}

const pretty = (s: string) => s.replace(/_/g, ' ');
const classOf = (q: Q) => q.question_class || q.question_type || 'untyped';
const subOf = (q: Q) => q.question_subclass || q.question_type || classOf(q);

export function QuestionMix({ items }: { items: Q[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);

  const { classes, subs, tags, n, hasSubs } = useMemo(() => {
    const byClass = new Map<string, number>();
    const bySub = new Map<string, Map<string, number>>();
    const byTag = new Map<string, number>();
    for (const q of items) {
      const c = classOf(q), s = subOf(q);
      byClass.set(c, (byClass.get(c) || 0) + 1);
      const m = bySub.get(c) ?? bySub.set(c, new Map()).get(c)!;
      m.set(s, (m.get(s) || 0) + 1);
      const t = q.structure_tags || [];
      for (const x of t) byTag.set(x, (byTag.get(x) || 0) + 1);
    }
    const sort = (m: Map<string, number>) =>
      [...m.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    // Only offer expansion where a class actually splits — a single-subclass class
    // would open to a ring identical to the slice that was clicked.
    const hasSubs = new Set([...bySub].filter(([, m]) => m.size > 1).map(([c]) => c));
    return {
      classes: sort(byClass),
      subs: new Map([...bySub].map(([c, m]) => [c, sort(m)])),
      tags: sort(byTag), n: items.length, hasSubs,
    };
  }, [items]);

  if (!n) return null;

  const S = 250, C = S / 2;
  const openSubs = open ? subs.get(open) || [] : [];
  const openN = classes.find(([c]) => c === open)?.[1] ?? 0;

  let a = -Math.PI / 2;
  const slices = classes.map(([label, v], i) => {
    const a0 = a;
    a += (v / n) * TAU;
    return { label, v, a0, a1: a, color: PALETTE[i % PALETTE.length], expandable: hasSubs.has(label) };
  });

  return (
    <div className="bg-white ring-1 ring-stone-200 rounded-lg p-4">
      <div className="flex flex-col sm:flex-row gap-5 items-center sm:items-start">
        <svg width={S} height={S} viewBox={`0 0 ${S} ${S}`} className="shrink-0">
          {slices.map((s) => (
            <path
              key={s.label}
              d={arc(C, C, 40, 78, s.a0, s.a1)}
              fill={s.color}
              opacity={open && open !== s.label ? 0.3 : hover && hover !== s.label ? 0.7 : 1}
              stroke="#fff"
              strokeWidth={1.5}
              style={{ cursor: s.expandable ? 'pointer' : 'default' }}
              onClick={() => s.expandable && setOpen(open === s.label ? null : s.label)}
              onMouseEnter={() => setHover(s.label)}
              onMouseLeave={() => setHover(null)}
            />
          ))}

          {/* Outer ring spans only the clicked slice's own angles, so a subclass
              occupies visually the share of the whole set that it really is. */}
          {open && (() => {
            const parent = slices.find((s) => s.label === open)!;
            let b = parent.a0;
            return openSubs.map(([label, v], i) => {
              const b0 = b;
              b += ((v / Math.max(openN, 1)) * (parent.a1 - parent.a0));
              return (
                <path key={label} d={arc(C, C, 82, 112, b0, b)}
                      fill={SHADES[i % SHADES.length]} stroke="#fff" strokeWidth={1.5}
                      opacity={hover && hover !== label ? 0.7 : 1}
                      onMouseEnter={() => setHover(label)}
                      onMouseLeave={() => setHover(null)} />
              );
            });
          })()}

          <text x={C} y={C - 3} textAnchor="middle" className="fill-stone-800"
                style={{ fontSize: 21, fontWeight: 600 }}>{open ? openN : n}</text>
          <text x={C} y={C + 12} textAnchor="middle" className="fill-stone-400"
                style={{ fontSize: 9.5 }}>{open ? pretty(open) : 'questions'}</text>
        </svg>

        <div className="flex-1 min-w-0 space-y-3.5 w-full">
          <div>
            <h4 className="text-[11px] uppercase tracking-wide text-stone-400 mb-1.5">
              Question class · sums to {n}
            </h4>
            <ul className="space-y-1">
              {slices.map((s) => (
                <li key={s.label}>
                  <button
                    type="button"
                    disabled={!s.expandable}
                    onClick={() => setOpen(open === s.label ? null : s.label)}
                    onMouseEnter={() => setHover(s.label)}
                    onMouseLeave={() => setHover(null)}
                    className={`w-full flex items-center gap-2 text-[12px] text-left rounded px-1 -mx-1 py-0.5 ${
                      s.expandable ? 'hover:bg-stone-50 cursor-pointer' : 'cursor-default'
                    }`}
                    style={{ opacity: open && open !== s.label ? 0.5 : 1 }}
                  >
                    <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: s.color }} />
                    <span className="text-stone-700">{pretty(s.label)}</span>
                    {s.expandable && (
                      <span className="text-stone-300 text-[10px]">
                        {open === s.label ? '▾' : '▸'}
                      </span>
                    )}
                    <span className="ml-auto font-mono text-stone-400">
                      {s.v} · {Math.round((s.v / n) * 100)}%
                    </span>
                  </button>

                  {open === s.label && (
                    <ul className="mt-1 mb-1.5 ml-4 pl-2.5 border-l border-stone-200 space-y-1">
                      {openSubs.map(([label, v], i) => (
                        <li key={label}
                            className="flex items-center gap-2 text-[12px]"
                            style={{ opacity: hover && hover !== label ? 0.5 : 1 }}
                            onMouseEnter={() => setHover(label)}
                            onMouseLeave={() => setHover(null)}>
                          <span className="w-2 h-2 rounded-sm shrink-0"
                                style={{ background: SHADES[i % SHADES.length] }} />
                          <span className="text-stone-600">{pretty(label)}</span>
                          <span className="ml-auto font-mono text-stone-400">
                            {v} · {Math.round((v / Math.max(openN, 1)) * 100)}%
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              ))}
            </ul>
            {[...hasSubs].length > 0 && !open && (
              <p className="mt-1.5 text-[11px] text-stone-400">Click a class to see its subclasses.</p>
            )}
          </div>

          {tags.length > 0 && (
            <div className="pt-3 border-t border-stone-100">
              <h4 className="text-[11px] uppercase tracking-wide text-stone-400 mb-1.5">
                Evidence shape · overlapping
              </h4>
              <ul className="flex flex-wrap gap-x-4 gap-y-1">
                {tags.map(([label, v]) => (
                  <li key={label} className="text-[12px] text-stone-600">
                    {pretty(label)} <span className="font-mono text-stone-400">{v}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
