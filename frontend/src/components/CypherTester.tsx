import React, { useEffect, useState } from 'react';

// Predefined queries to seed the editor — each one exercises a different KG slice.
// The first six reproduce the figures the paper states, so a reader can check any
// of them against the live graph instead of trusting the text: node and edge counts,
// corpus size, and the internal/external split of the citation edge. Verified against
// numbers.tex — 1,800 documents, 233,604 chunks, 652,491 MENTIONS.
const SAMPLE_QUERIES: { label: string; query: string; params?: string }[] = [
  {
    label: 'Corpus size',
    query:
      'MATCH (d:Document) WITH count(d) AS documents\n'
      + 'MATCH (c:Chunk) RETURN documents, count(c) AS chunks',
  },
  {
    label: 'Schema size',
    query:
      'MATCH (n) UNWIND labels(n) AS l WITH count(DISTINCT l) AS node_types\n'
      + 'MATCH ()-[r]->() RETURN node_types, count(DISTINCT type(r)) AS edge_types',
  },
  {
    label: 'Nodes per type',
    query:
      'MATCH (n) UNWIND labels(n) AS l\n'
      + 'RETURN l AS node_type, count(*) AS count ORDER BY count DESC',
  },
  {
    label: 'Edges per type',
    query:
      'MATCH ()-[r]->()\n'
      + 'RETURN type(r) AS edge_type, count(*) AS count ORDER BY count DESC',
  },
  {
    label: 'Citation edges: internal vs cross-spec',
    query:
      'MATCH (a:Chunk)-[r:REFERENCES_CHUNK]->(b:Chunk)\n'
      + "RETURN CASE WHEN r.is_external THEN 'cross-spec' ELSE 'within-spec' END AS kind,\n"
      + '       count(*) AS count ORDER BY count DESC',
  },
  {
    label: 'Chunks per chunk_type',
    query:
      'MATCH (c:Chunk)\n'
      + 'RETURN c.chunk_type AS chunk_type, count(*) AS count ORDER BY count DESC',
  },
  {
    label: 'List node labels',
    query: 'CALL db.labels() YIELD label RETURN label ORDER BY label',
  },
  {
    label: 'List relationship types',
    query:
      'CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType',
  },
  {
    label: 'Sample 10 chunks',
    query:
      'MATCH (c:Chunk) RETURN c.chunk_id AS id, c.spec_id AS spec, c.section_title AS section, substring(c.content, 0, 120) AS preview LIMIT 10',
  },
  {
    label: 'Count nodes per label',
    query:
      'MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC',
  },
  {
    label: 'Top 10 terms by spec spread',
    query:
      'MATCH (t:Term) WHERE t.source_specs IS NOT NULL RETURN t.abbreviation AS abbreviation, t.full_name AS full_name, size(t.source_specs) AS spec_count ORDER BY spec_count DESC LIMIT 10',
  },
  {
    label: 'Find chunks mentioning AMF (Pattern A)',
    query:
      "MATCH (t:Term {abbreviation: 'AMF'}) WITH t, t.full_name AS full_name LIMIT 1 MATCH (c:Chunk) WHERE c.spec_id IN t.source_specs OR c.section_title CONTAINS full_name RETURN c.spec_id AS spec, c.section_title AS section, substring(c.content, 0, 160) AS preview LIMIT 10",
  },
  {
    label: 'Term co-occurrence via key_terms',
    query:
      "MATCH (c:Chunk) WHERE 'AMF' IN c.key_terms UNWIND c.key_terms AS kt WITH kt WHERE kt <> 'AMF' RETURN kt AS related_term, count(*) AS co_occurrences ORDER BY co_occurrences DESC LIMIT 15",
  },
  {
    label: 'Find term by abbreviation (param)',
    query:
      'MATCH (t:Term) WHERE t.abbreviation CONTAINS $term RETURN t.abbreviation AS abbreviation, t.full_name AS full_name, t.primary_spec AS primary_spec LIMIT $top_k',
    params: '{\n  "term": "AMF",\n  "top_k": 10\n}',
  },
  {
    label: 'Chunks for interface N6 (Pattern B)',
    query:
      "MATCH (c:Chunk) WHERE c.section_title =~ '(?i).*\\\\bN6\\\\b.*' RETURN c.spec_id AS spec, c.section_title AS section, c.chunk_type AS chunk_type, substring(c.content, 0, 160) AS preview LIMIT 10",
  },
  {
    label: 'Subjects with chunk counts',
    query:
      'MATCH (s:Subject)<-[:HAS_SUBJECT]-(c:Chunk) RETURN s.name AS subject, s.priority AS priority, count(c) AS chunks ORDER BY chunks DESC',
  },
];

interface SchemaInfo {
  labels: string[];
  label_counts: Record<string, number>;
  relationship_types: string[];
  relationship_counts: Record<string, number>;
  // Optional: an older rag-engine does not send it, and the block hides itself.
  corpus?: {
    documents: number;
    specs: number;
    chunks: number;
    chunk_types: Record<string, number>;
  };
}

interface CypherResult {
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
  truncated: boolean;
  elapsed_ms: number;
}

// Render a single cell value — Neo4j nodes/rels arrive as objects with _type
function CellValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <span className="text-gray-400 italic">null</span>;
  }
  if (typeof value === 'object') {
    const obj = value as { _type?: string; labels?: string[]; type?: string; properties?: Record<string, unknown> };
    if (obj._type === 'node') {
      return (
        <span title={JSON.stringify(obj.properties, null, 2)} className="text-blue-700">
          (:{(obj.labels ?? []).join(':')}) {Object.keys(obj.properties ?? {}).length} props
        </span>
      );
    }
    if (obj._type === 'relationship') {
      return <span className="text-purple-700">[:{obj.type}]</span>;
    }
    return <code className="text-xs">{JSON.stringify(value)}</code>;
  }
  if (typeof value === 'string') {
    // Truncate very long strings inline; full text on hover via title
    return value.length > 200 ? (
      <span title={value}>{value.slice(0, 200)}…</span>
    ) : (
      <span>{value}</span>
    );
  }
  return <span>{String(value)}</span>;
}

export function CypherTester() {
  const [query, setQuery] = useState<string>(SAMPLE_QUERIES[0].query);
  const [paramsText, setParamsText] = useState<string>('{}');
  const [limit, setLimit] = useState<number>(200);
  const [result, setResult] = useState<CypherResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [schema, setSchema] = useState<SchemaInfo | null>(null);
  // All three schema blocks start collapsed. Open, they run to 37 rows and push the
  // sample queries — the thing most sessions actually come here for — off screen. The
  // header of each block still carries its headline count, so the collapsed state is
  // not empty: 1,800 docs / 10 labels / 18 relationship types.
  const [open, setOpen] = useState({ corpus: false, labels: false, rels: false });
  const toggle = (k: 'corpus' | 'labels' | 'rels') =>
    setOpen(o => ({ ...o, [k]: !o[k] }));

  // Fetch schema once on mount so the sidebar shows what's in the KG
  useEffect(() => {
    fetch('/api/cypher/schema')
      .then((r) => (r.ok ? r.json() : null))
      .then(setSchema)
      .catch(() => setSchema(null));
  }, []);

  const runQuery = async () => {
    if (!query.trim() || loading) return;
    // Parse params JSON up-front so a typo surfaces as a clear UI error
    let params: Record<string, unknown> = {};
    const trimmedParams = paramsText.trim();
    if (trimmedParams && trimmedParams !== '{}') {
      try {
        params = JSON.parse(trimmedParams);
        if (typeof params !== 'object' || params === null || Array.isArray(params)) {
          setError('Parameters must be a JSON object, e.g. {"top_k": 10}');
          return;
        }
      } catch (err) {
        setError(`Invalid parameter JSON: ${String(err)}`);
        return;
      }
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch('/api/cypher', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, limit, params }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(typeof data?.message === 'string' ? data.message : `HTTP ${res.status}`);
        return;
      }
      setResult(data as CypherResult);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  // Cmd/Ctrl+Enter to run — works from either the query or params textarea
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      runQuery();
    }
  };

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar: schema + sample queries */}
      <aside className="w-72 border-r border-gray-200 bg-gray-50 flex flex-col overflow-y-auto">
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
            Schema
          </h3>
          {schema ? (
            <div className="space-y-3 text-sm">
              {schema.corpus && (
                <div>
                  <button
                    type="button"
                    onClick={() => toggle('corpus')}
                    className="w-full flex items-baseline justify-between text-left
                               text-[11px] uppercase text-gray-400 hover:text-gray-600"
                  >
                    <span>{open.corpus ? '▾' : '▸'} Corpus</span>
                    <span className="normal-case text-gray-500 tabular-nums">
                      {schema.corpus.documents.toLocaleString()} docs
                    </span>
                  </button>
                  {open.corpus && (
                    <div className="mt-1 space-y-0.5 pl-3">
                      <div className="flex justify-between text-gray-700">
                        <span>chunks</span>
                        <span className="text-gray-500 tabular-nums">
                          {schema.corpus.chunks.toLocaleString()}
                        </span>
                      </div>
                      {Object.entries(schema.corpus.chunk_types).map(([t, n]) => (
                        <div key={t} className="flex justify-between text-gray-600">
                          <span className="font-mono text-[12px]">{t}</span>
                          <span className="text-gray-500 tabular-nums">{n.toLocaleString()}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
              <div>
                <button
                  type="button"
                  onClick={() => toggle('labels')}
                  className="w-full flex items-baseline justify-between text-left
                             text-[11px] uppercase text-gray-400 hover:text-gray-600"
                >
                  <span>{open.labels ? '▾' : '▸'} Labels</span>
                  <span className="normal-case text-gray-500 tabular-nums">
                    {schema.labels.length}
                  </span>
                </button>
                {open.labels && (
                  <div className="mt-1 space-y-0.5">
                    {schema.labels.map((label) => (
                      <div key={label} className="flex justify-between text-gray-700">
                        <span className="font-mono text-blue-700">:{label}</span>
                        <span className="text-gray-500 tabular-nums">
                          {(schema.label_counts[label] ?? 0).toLocaleString()}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <div>
                <button
                  type="button"
                  onClick={() => toggle('rels')}
                  className="w-full flex items-baseline justify-between text-left
                             text-[11px] uppercase text-gray-400 hover:text-gray-600"
                >
                  <span>{open.rels ? '▾' : '▸'} Relationships</span>
                  <span className="normal-case text-gray-500 tabular-nums">
                    {schema.relationship_types.length}
                  </span>
                </button>
                {open.rels && (
                  <div className="mt-1 space-y-0.5">
                    {schema.relationship_types.map((rel) => (
                      <div key={rel} className="flex justify-between text-gray-700">
                        <span className="font-mono text-purple-700">[:{rel}]</span>
                        <span className="text-gray-500 tabular-nums">
                          {(schema.relationship_counts[rel] ?? 0).toLocaleString()}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-xs text-gray-400">Loading schema…</div>
          )}
        </div>

        <div className="p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
            Sample queries
          </h3>
          <div className="space-y-1">
            {SAMPLE_QUERIES.map((s) => (
              <button
                key={s.label}
                type="button"
                onClick={() => {
                  setQuery(s.query);
                  setParamsText(s.params ?? '{}');
                }}
                className="w-full text-left text-sm px-2 py-1.5 rounded hover:bg-gray-200 text-gray-700"
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </aside>

      {/* Main: editor + results */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Editor — query on the left, optional parameters JSON on the right */}
        <div className="border-b border-gray-200 p-4 bg-white">
          <div className="flex gap-3">
            <div className="flex-1 min-w-0">
              <div className="text-[11px] uppercase text-gray-400 mb-1">Query</div>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                spellCheck={false}
                placeholder="MATCH (n) RETURN n LIMIT 10"
                className="w-full h-32 font-mono text-sm p-3 border border-gray-200 rounded-lg
                           outline-none focus:border-gray-400 resize-y bg-gray-50"
              />
            </div>
            <div className="w-64 shrink-0">
              <div className="text-[11px] uppercase text-gray-400 mb-1">Parameters (JSON)</div>
              <textarea
                value={paramsText}
                onChange={(e) => setParamsText(e.target.value)}
                onKeyDown={handleKeyDown}
                spellCheck={false}
                placeholder={'{\n  "top_k": 10\n}'}
                className="w-full h-32 font-mono text-xs p-3 border border-gray-200 rounded-lg
                           outline-none focus:border-gray-400 resize-y bg-gray-50"
              />
            </div>
          </div>
          <div className="flex items-center gap-3 mt-2 text-sm">
            <button
              type="button"
              onClick={runQuery}
              disabled={loading || !query.trim()}
              className="px-4 py-1.5 rounded-md bg-gray-900 text-white text-sm font-medium
                         hover:bg-gray-800 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {loading ? 'Running…' : 'Run (⌘/Ctrl+Enter)'}
            </button>
            <label className="text-gray-600 text-xs flex items-center gap-1">
              Limit:
              <input
                type="number"
                min={1}
                max={1000}
                value={limit}
                onChange={(e) => setLimit(Math.max(1, Math.min(1000, parseInt(e.target.value) || 200)))}
                className="w-16 px-2 py-0.5 border border-gray-200 rounded text-sm"
              />
            </label>
            {result && (
              <span className="text-gray-500 text-xs">
                {result.row_count} row{result.row_count === 1 ? '' : 's'}
                {result.truncated ? ' (truncated)' : ''} · {result.elapsed_ms}ms
              </span>
            )}
          </div>
          <div className="text-[11px] text-gray-400 mt-1">
            Read-only · CREATE/MERGE/DELETE/SET/REMOVE/DROP are blocked
          </div>
        </div>

        {/* Results */}
        <div className="flex-1 overflow-auto bg-white">
          {error && (
            <div className="m-4 px-3 py-2 bg-red-50 text-red-700 text-sm rounded-lg border border-red-200 whitespace-pre-wrap">
              {error}
            </div>
          )}
          {!error && !result && !loading && (
            <div className="p-8 text-center text-gray-400 text-sm">
              Pick a sample query or write your own, then press Run.
            </div>
          )}
          {result && result.rows.length === 0 && (
            <div className="p-8 text-center text-gray-400 text-sm">
              Query returned no rows.
            </div>
          )}
          {result && result.rows.length > 0 && (
            <table className="w-full text-sm border-collapse">
              <thead className="sticky top-0 bg-gray-100 border-b border-gray-200">
                <tr>
                  {result.columns.map((col) => (
                    <th
                      key={col}
                      className="text-left px-3 py-2 font-mono text-xs font-semibold text-gray-700"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row, i) => (
                  <tr key={i} className="border-b border-gray-100 hover:bg-gray-50 align-top">
                    {result.columns.map((col) => (
                      <td key={col} className="px-3 py-2 font-mono text-xs">
                        <CellValue value={row[col]} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </main>
    </div>
  );
}
