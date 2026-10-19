/** Rubric labels. Keys are what the API stores; text is what the rater reads. */
export const C1 = [
  { v: 'pass', t: 'Pass', d: 'The question is clear, self-contained, technically sound and answerable from the 3GPP specifications.' },
  { v: 'fail', t: 'Fail', d: 'Technically wrong, ambiguous, built on a false premise, or asks for something the specifications do not define.' },
  { v: 'unsure', t: 'Unsure', d: 'Not enough grounds to decide (give a reason).' },
];
export const C2 = [
  { v: 'full', t: 'Yes, fully', d: 'The gold clauses contain everything needed to answer the question completely.' },
  { v: 'partial', t: 'Partially', d: 'The gold clauses are relevant but part of the answer is missing, or some clause is superfluous.' },
  { v: 'none', t: 'No', d: 'The gold clauses do not contain the answer.' },
];
export const C3 = [
  { v: 'correct', t: 'Correct', d: 'The reference answer matches the gold clauses, with no wrong or invented detail.' },
  { v: 'partial', t: 'Partially correct', d: 'Right in substance but incomplete or with an inaccurate detail.' },
  { v: 'wrong', t: 'Wrong', d: 'The reference answer contradicts the gold clauses.' },
];
export const VERDICT = [
  { v: 'keep', t: 'Keep', d: 'Question, gold and reference answer are usable as they are.' },
  { v: 'fix', t: 'Fix', d: 'Keep the question but the question / gold / answer must be corrected (say what).' },
  { v: 'discard', t: 'Discard', d: 'Cannot be fixed, or not worth keeping (give the reason).' },
];
const map = (opts: { v: string; t: string }[]) => Object.fromEntries(opts.map(o => [o.v, o.t]));
/** Label text per criterion — C2 and C3 share the stored key "partial", so one flat map would mislabel one of them. */
export const LABEL: Record<string, Record<string, string>> = { c1: map(C1), c2: map(C2), c3: map(C3), verdict: map(VERDICT) };
/** Only for values that are unambiguous across criteria (keep/fix/discard). */
export const TEXT: Record<string, string> = map(VERDICT);
export const CRIT_NAME: Record<string, string> = {
  c1: 'C1 · Question valid', c2: 'C2 · Gold contains the answer', c3: 'C3 · Reference answer correct', verdict: 'Verdict',
};
