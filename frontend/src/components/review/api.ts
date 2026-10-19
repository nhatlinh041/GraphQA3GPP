/**
 * Thin fetch layer for /api/review. The rater token lives in localStorage
 * (raters come back over several days; the server token expires after 7 days),
 * the admin token in sessionStorage (12 h, gone when the browser closes).
 */
const TOKEN = 'review-token';
const NAME = 'review-name';
const ADMIN = 'review-admin';
const ADMIN_USER = 'review-admin-user';

const ls = {
  get: (k: string) => { try { return localStorage.getItem(k) || ''; } catch { return ''; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  del: (k: string) => { try { localStorage.removeItem(k); } catch { /* private mode */ } },
};

export const session = {
  get token() { return ls.get(TOKEN); },
  get name() { return ls.get(NAME); },
  get admin() { return sessionStorage.getItem(ADMIN) || ''; },
  get adminUser() { return sessionStorage.getItem(ADMIN_USER) || ''; },
  signIn(token: string, name: string) { ls.set(TOKEN, token); ls.set(NAME, name); },
  signOut() { ls.del(TOKEN); ls.del(NAME); },
  setAdmin(token: string, user: string) { sessionStorage.setItem(ADMIN, token); sessionStorage.setItem(ADMIN_USER, user); },
  clearAdmin() { sessionStorage.removeItem(ADMIN); sessionStorage.removeItem(ADMIN_USER); },
};

export class ApiError extends Error {
  status: number;
  constructor(status: number, msg: string) { super(msg); this.status = status; }
}

async function call<T>(method: string, path: string, body?: unknown, admin = false): Promise<T> {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (admin) headers['x-review-admin'] = session.admin;
  else if (session.token) headers['x-review-token'] = session.token;
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = j.detail || msg; } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, msg);
  }
  return res.json();
}

export interface GoldChunk {
  chunk_id: string; spec_id: string | null; section_id: string | null; section_title: string | null;
  content: string | null; n_chars: number | null; missing: boolean;
  segments: Segment[];
}
export type Segment = { type: 'text'; text: string } | { type: 'table'; headers: string[]; rows: string[][] };
export interface Rating {
  status: 'draft' | 'done'; c1: string | null; c1_note: string | null; c2: string | null; c2_note: string | null;
  irrelevant_chunks: string[]; own_answer: string | null; c3: string | null; verdict: string | null;
  comment: string | null; fix_text: string | null; t_phase1_s: number | null; t_phase2_s: number | null;
  revealed_at: number | null; revision: number; updated_at: number;
}
export interface QuestionView {
  qid: string; idx: number; total: number; question: string; gold: GoldChunk[];
  rating: Rating | null; revealed: boolean; reference_answer: string | null; next_qid: string | null;
}
export interface QueueItem { qid: string; idx: number; status: 'todo' | 'draft' | 'done'; revealed: boolean; updated_at: number | null }
export interface Lock { requires_campaign_id: number; requires_name: string; n_assigned: number; n_left: number }
export interface CampaignMeta {
  id: number; name: string; status: string; n_assigned: number; n_done: number;
  all_raters: number; requires_campaign_id: number | null; lock: Lock | null; is_pilot: boolean;
}
export interface Me {
  reviewer: { id: number; name: string; role: string };
  campaigns: CampaignMeta[];
}

export const api = {
  login: (name: string, password: string) =>
    call<{ token: string; expires_at: number; reviewer: Me['reviewer'] }>('POST', '/api/review/login', { name, password }),
  changePassword: (old_password: string, new_password: string) =>
    call<{ ok: boolean; token: string }>('POST', '/api/review/me/password', { old_password, new_password }),
  me: () => call<Me>('GET', '/api/review/me'),
  queue: (campaign: number) => call<{ items: QueueItem[]; n_done: number; n_total: number; lock: Lock | null }>('GET', `/api/review/queue?campaign=${campaign}`),
  question: (campaign: number, qid: string) => call<QuestionView>('GET', `/api/review/q/${encodeURIComponent(qid)}?campaign=${campaign}`),
  reveal: (campaign: number, qid: string) => call<QuestionView>('POST', `/api/review/q/${encodeURIComponent(qid)}/reveal?campaign=${campaign}`),
  save: (campaign: number, qid: string, body: Partial<Rating> & { status: 'draft' | 'done' }) =>
    call<{ rating: Rating }>('PUT', `/api/review/q/${encodeURIComponent(qid)}/rating?campaign=${campaign}`, body),
  admin: {
    login: (username: string, password: string) =>
      call<{ token: string; username: string; expires_at: number }>('POST', '/api/review/admin/login', { username, password }),
    me: () => call<{ username: string }>('GET', '/api/review/admin/me', undefined, true),
    overview: () => call<any>('GET', '/api/review/admin/overview', undefined, true),
    addReviewer: (name: string, password?: string, role = 'rater') =>
      call<any>('POST', '/api/review/admin/reviewers', { name, role, password: password || null }, true),
    resetPassword: (id: number, password?: string) =>
      call<{ id: number; password: string }>('POST', `/api/review/admin/reviewers/${id}/password`, { password: password || null }, true),
    patchReviewer: (id: number, body: { name?: string; active?: boolean; role?: string }) =>
      call<any>('PATCH', `/api/review/admin/reviewers/${id}`, body, true),
    createCampaign: (body: { name: string; sample_file: string; raters_per_q: number; seed: number; reviewer_ids: number[];
                             all_raters: boolean; requires_campaign_id: number | null }) =>
      call<any>('POST', '/api/review/admin/campaigns', body, true),
    setRequires: (cid: number, requires_campaign_id: number | null) =>
      call<any>('PATCH', `/api/review/admin/campaigns/${cid}/requires`, { requires_campaign_id }, true),
    deleteCampaign: (cid: number) => call<any>('DELETE', `/api/review/admin/campaigns/${cid}`, undefined, true),
    assign: (cid: number, reviewer_ids: number[]) => call<any>('POST', `/api/review/admin/campaigns/${cid}/assign`, { reviewer_ids }, true),
    setStatus: (cid: number, status: string) => call<any>('PATCH', `/api/review/admin/campaigns/${cid}/status`, { status }, true),
    stats: (cid: number) => call<any>('GET', `/api/review/admin/campaigns/${cid}/stats`, undefined, true),
    disagreements: (cid: number) => call<{ items: any[] }>('GET', `/api/review/admin/campaigns/${cid}/disagreements`, undefined, true),
    question: (cid: number, qid: string) => call<any>('GET', `/api/review/admin/campaigns/${cid}/q/${encodeURIComponent(qid)}`, undefined, true),
    adjudicate: (cid: number, qid: string, body: { decision: string; note?: string; decided_by?: string }) =>
      call<any>('PUT', `/api/review/admin/campaigns/${cid}/adjudicate/${encodeURIComponent(qid)}`, body, true),
    exportUrl: (cid: number, format: 'json' | 'csv') => `/api/review/admin/campaigns/${cid}/export?format=${format}`,
  },
};

/** Download an admin export through fetch so the admin header travels with it. */
export async function downloadExport(cid: number, format: 'json' | 'csv') {
  const res = await fetch(api.admin.exportUrl(cid, format), { headers: { 'x-review-admin': session.admin } });
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  const blob = await res.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `review_${cid}.${format}`;
  a.click();
  URL.revokeObjectURL(a.href);
}
