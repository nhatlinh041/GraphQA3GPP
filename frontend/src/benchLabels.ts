// GENERATED — do not edit. Source: tests/benchmark/paper_bench/configs.json
// Regenerate with:  .venv/bin/python scripts/gen_labels_ts.py
export const LABELS_VERSION = 6;

export interface BenchConfig {
  label: string;
  short: string;
  reranker: boolean;
  provenance: boolean;
  evidence: string;
  retriever: string;
  group: string;
  order: number;
}

export const CONFIGS: Record<string, BenchConfig> = {
  "A_llm_only": { label: "LLM-only (no retrieval)", short: "LLM-only", reranker: false, provenance: false, evidence: "none", retriever: "none", group: "phaseA", order: 1 },
  "A_bm25_noterm": { label: "BM25 \u2192 rerank \u2192 gen", short: "BM25", reranker: true, provenance: true, evidence: "chunks", retriever: "BM25 (sparse), no KG synonyms", group: "phaseA", order: 2 },
  "A_vector_noterm": { label: "Dense \u2192 rerank \u2192 gen", short: "Dense", reranker: true, provenance: true, evidence: "chunks", retriever: "Dense (e5-base-v2), no KG synonyms", group: "phaseA", order: 3 },
  "A_kg_notitle": { label: "KG \u2192 rerank \u2192 gen", short: "KG", reranker: true, provenance: true, evidence: "chunks", retriever: "KG (Cypher)", group: "phaseA", order: 4 },
  "A_bm25_dense_noterm_norr": { label: "Hybrid (BM25 + Dense), no rerank", short: "BM25+Dense \u2212rr", reranker: false, provenance: true, evidence: "chunks", retriever: "BM25 + Dense", group: "phaseA", order: 5 },
  "A_bm25_dense_noterm": { label: "Hybrid (BM25 + Dense) \u2192 rerank \u2192 gen", short: "BM25+Dense", reranker: true, provenance: true, evidence: "chunks", retriever: "BM25 + Dense", group: "phaseA", order: 6 },
  "A_hybrid": { label: "Hybrid (Dense + KG), no rerank", short: "Dense+KG \u2212rr", reranker: false, provenance: true, evidence: "chunks", retriever: "Dense + KG", group: "phaseA", order: 7 },
  "A_fixed": { label: "Hybrid (Dense + KG) \u2192 rerank \u2192 gen (full system)", short: "Full system", reranker: true, provenance: true, evidence: "chunks", retriever: "Dense + KG", group: "phaseA", order: 8 },
  "kgabl_entity": { label: "KG layer: entity only", short: "entity", reranker: true, provenance: true, evidence: "chunks", retriever: "KG (MENTIONS only)", group: "kgabl", order: 11 },
  "kgabl_relation": { label: "KG layer: + relation", short: "+relation", reranker: true, provenance: true, evidence: "chunks", retriever: "KG (+ Cypher traversal, anchors)", group: "kgabl", order: 12 },
  "kgabl_hierarchy": { label: "KG layer: + hierarchy", short: "+hierarchy", reranker: true, provenance: true, evidence: "chunks", retriever: "KG (+ PARENT_SECTION)", group: "kgabl", order: 13 },
  "kgabl_xref": { label: "KG layer: + cross-reference", short: "+xref", reranker: true, provenance: true, evidence: "chunks", retriever: "KG (+ REFERENCES_CHUNK hop)", group: "kgabl", order: 14 },
  "kgabl_provenance": { label: "KG layer: + provenance", short: "+provenance", reranker: true, provenance: true, evidence: "chunks", retriever: "KG (all layers)", group: "kgabl", order: 15 },
  "study_text_norr": { label: "Evidence: text only", short: "text only", reranker: false, provenance: false, evidence: "chunks", retriever: "Dense (e5-base-v2)", group: "study", order: 21 },
  "study_triples_norr": { label: "Evidence: KG triples only", short: "triples only", reranker: false, provenance: false, evidence: "triples", retriever: "KG (Cypher)", group: "study", order: 22 },
  "study_noprov": { label: "Evidence: text + KG", short: "text+KG", reranker: false, provenance: false, evidence: "chunks", retriever: "Dense + KG", group: "study", order: 23 },
  "study_triples": { label: "KG triples as evidence (reranked)", short: "triples +rr", reranker: true, provenance: true, evidence: "triples", retriever: "KG (Cypher)", group: "study", order: 24 },
  "study_text_prov": { label: "Evidence: text only + provenance", short: "text + prov", reranker: false, provenance: true, evidence: "chunks", retriever: "Dense (e5-base-v2)", group: "study", order: 25 },
  "A_bm25": { label: "BM25 (legacy: KG term expansion ON)", short: "BM25 (legacy)", reranker: true, provenance: true, evidence: "chunks", retriever: "BM25 (sparse)", group: "phaseA", order: 50 },
  "A_vector_only": { label: "Dense (legacy: KG term expansion ON)", short: "Dense (legacy)", reranker: true, provenance: true, evidence: "chunks", retriever: "Dense (e5-base-v2)", group: "phaseA", order: 51 },
  "A_bm25_dense": { label: "BM25 + Dense (legacy: KG term expansion ON)", short: "BM25+Dense (legacy)", reranker: true, provenance: true, evidence: "chunks", retriever: "BM25 + Dense", group: "phaseA", order: 52 },
  "A_kg_only": { label: "KG (legacy: includes lexical title search)", short: "KG (legacy)", reranker: true, provenance: true, evidence: "chunks", retriever: "KG (Cypher)", group: "phaseA", order: 53 },
  "A_bm25_dense_kg": { label: "BM25 + Dense + KG (RRF) \u2192 rerank \u2192 gen", short: "Sparse+Dense+KG", reranker: true, provenance: true, evidence: "chunks", retriever: "BM25 + Dense + KG", group: "phaseA", order: 54 },
  "A_kg_stepev": { label: "KG + step-scoped evidence \u2192 rerank \u2192 gen", short: "KG step-ev", reranker: true, provenance: true, evidence: "chunks", retriever: "KG (Cypher)", group: "phaseA", order: 55 },
  "A_kg_nobypass": { label: "KG, step-actor floor bypass off", short: "KG \u2212bypass", reranker: true, provenance: true, evidence: "chunks", retriever: "KG (Cypher)", group: "phaseA", order: 56 },
  "A_kg_norank": { label: "KG, step-actor rerank bias off", short: "KG \u2212bias", reranker: true, provenance: true, evidence: "chunks", retriever: "KG (Cypher)", group: "phaseA", order: 57 },
};

/** Ordered config keys per ablation ladder — each step changes exactly one variable. */
export const LADDERS: Record<string, string[]> = {
  "provenance_control": ["study_text_norr", "study_text_prov"],
  "study": ["study_text_norr", "study_triples_norr", "study_noprov", "A_hybrid", "A_fixed"],
};

/** Unknown keys fall through to the raw key so a new config still renders. */
export const label = (k: string): string => CONFIGS[k]?.label ?? k;
export const short = (k: string): string => CONFIGS[k]?.short ?? k;
export const orderOf = (k: string): number => CONFIGS[k]?.order ?? 999;
