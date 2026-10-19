/**
 * Phase-2 helper: mark the sentences of a gold clause that overlap the
 * reference answer, so the rater can check C3 without re-reading 20 KB.
 * Lexical only (shared content tokens ≥ 4 chars), computed in the browser and
 * shown ONLY after the reveal — in the blind phase nothing may hint at the
 * answer.
 */
const TOKEN = /[A-Za-z0-9_]{4,}/g;

export function splitSentences(text: string): string[] {
  // Keep line structure: specs are mostly one clause/step per line.
  return text.split(/(?<=[.;:])\s+(?=[A-Z0-9(])|\n/).filter(s => s.length > 0);
}

export function overlapScore(sentence: string, refTokens: Set<string>): number {
  const toks = (sentence.match(TOKEN) || []).map(t => t.toLowerCase());
  if (toks.length < 4) return 0;
  const hit = toks.filter(t => refTokens.has(t)).length;
  return hit / toks.length;
}

export function refTokenSet(reference: string): Set<string> {
  return new Set((reference.match(TOKEN) || []).map(t => t.toLowerCase()));
}
