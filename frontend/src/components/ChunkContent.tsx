/**
 * Table-aware rendering of a chunk's text for the chat's source cards.
 *
 * Chunk content stores tables flattened one cell per line; the backend
 * (`POST /api/chunks/segments`) locates each table in the text using the
 * structured copy in the source JSON and returns text/table segments. Fetched
 * lazily, once per chunk_id per page load; while it loads, or if it fails,
 * the plain text is shown exactly as before.
 */
import React, { useEffect, useState } from 'react';

export type Segment =
  | { type: 'text'; text: string }
  | { type: 'table'; headers: string[]; rows: string[][]; truncated?: boolean };

// Keyed by chunk_id AND content: the same chunk can reach the chat as a 2000-char
// excerpt (adaptive-hop path) in one answer and in full in the next, and caching
// by chunk_id alone replayed the excerpt's table-less result for the full text.
const cache = new Map<string, Segment[]>();
const inflight = new Map<string, Promise<void>>();

function fingerprint(s: string): string {
  let h = 2166136261;                       // FNV-1a, 32-bit
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return `${s.length}:${(h >>> 0).toString(36)}`;
}
const keyOf = (chunkId: string, content: string) => `${chunkId}#${fingerprint(content)}`;

async function fetchSegments(items: { chunk_id: string; content: string }[]): Promise<void> {
  const keyed = items.map(i => ({ ...i, key: keyOf(i.chunk_id, i.content) }));
  const todo = keyed.filter(i => i.chunk_id && i.content && !cache.has(i.key) && !inflight.has(i.key));
  // Wait for everything this caller needs, including items another caller is already fetching.
  const others = keyed.map(i => inflight.get(i.key)).filter(Boolean) as Promise<void>[];
  if (!todo.length) {
    await Promise.all(others);
    return;
  }
  const p = fetch('/api/chunks/segments', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ items: todo.map(({ chunk_id, content }) => ({ chunk_id, content })) }),
  })
    .then(r => (r.ok ? r.json() : { segments: {} }))
    .then((j: { segments: Record<string, Segment[]> }) => {
      for (const i of todo) { const segs = j.segments?.[i.chunk_id]; if (segs) cache.set(i.key, segs); }
    })
    .catch(() => { /* fall back to plain text */ })
    .finally(() => todo.forEach(i => inflight.delete(i.key)));
  todo.forEach(i => inflight.set(i.key, p));
  await Promise.all([p, ...others]);
}

/** Fetch segments for every source of a message in ONE request; returns a version counter that bumps when they arrive. */
export function useChunkSegments(items: { chunk_id?: string; content?: string }[] | undefined): number {
  const [version, setVersion] = useState(0);
  const key = (items || []).map(i => i.chunk_id).join('|');
  useEffect(() => {
    const list = (items || [])
      .filter(i => i.chunk_id && i.content)
      .map(i => ({ chunk_id: i.chunk_id as string, content: i.content as string }));
    if (!list.length) return;
    let alive = true;
    fetchSegments(list).then(() => { if (alive) setVersion(v => v + 1); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return version;
}

export function TableBlock({ headers, rows, truncated }: { headers: string[]; rows: string[][]; truncated?: boolean }) {
  const cols = Math.max(headers.length, ...rows.map(r => r.length), 1);
  return (
    <div className="my-2 overflow-x-auto whitespace-normal">
      <table className="min-w-full border-collapse text-[12.5px] leading-snug text-gray-700">
        {headers.length > 0 && (
          <thead>
            <tr>{Array.from({ length: cols }, (_, i) => (
              <th key={i} className="border border-gray-300 bg-gray-100 px-2 py-1 text-left font-semibold align-top">{headers[i] ?? ''}</th>
            ))}</tr>
          </thead>
        )}
        <tbody>
          {rows.map((r, j) => (
            <tr key={j} className={j % 2 ? 'bg-gray-50' : 'bg-white'}>
              {Array.from({ length: cols }, (_, i) => (
                <td key={i} className="border border-gray-200 px-2 py-1 align-top whitespace-pre-wrap break-words">{r[i] ?? ''}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {truncated && (
        <div className="mt-1 text-[11px] italic text-gray-500">Table cut off in this excerpt — only the complete rows are shown.</div>
      )}
    </div>
  );
}

/** The chunk's text with its tables drawn as tables (plain text until segments arrive). */
export function ChunkContent({ chunkId, content, raw }: { chunkId?: string; content: string; raw?: string }) {
  // `raw` is the exact string that was sent for segmentation (callers may display a trimmed copy).
  const segs = chunkId ? cache.get(keyOf(chunkId, raw ?? content)) : undefined;
  if (!segs || !segs.some(s => s.type === 'table')) return <>{content}</>;
  return (
    <>
      {segs.map((s, k) => s.type === 'table'
        ? <TableBlock key={k} headers={s.headers} rows={s.rows} truncated={s.truncated} />
        : <React.Fragment key={k}>{s.text}</React.Fragment>)}
    </>
  );
}

export function hasTable(chunkId?: string, content?: string): boolean {
  return !!chunkId && content != null && !!cache.get(keyOf(chunkId, content))?.some(s => s.type === 'table');
}
