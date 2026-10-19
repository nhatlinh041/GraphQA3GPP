import React, { useEffect, useState } from 'react';
import { api, ApiError, downloadExport, session } from './api';
import { CRIT_NAME, LABEL, TEXT } from './labels';
import { specLabel } from './QuestionCard';

const pct = (x: number | null | undefined) => x == null ? '—' : (100 * x).toFixed(1) + '%';
const num = (x: number | null | undefined) => x == null ? '—' : x.toFixed(3);
const ci = (r: any) => r?.ci95 ? ` [${pct(r.ci95[0])}, ${pct(r.ci95[1])}]` : '';

function CritTable({ block }: { block: any }) {
  const rows = Object.entries(block.criteria) as [string, any][];
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-gray-500 border-b">
          <th className="py-1 pr-2">Criterion</th><th className="pr-2">Label distribution</th>
          <th className="pr-2">Pass / rating</th><th className="pr-2">Pass by both raters</th><th className="pr-2">n (≥2 ratings)</th>
          <th className="pr-2">Raw agreement</th><th className="pr-2">Fleiss κ</th><th className="pr-2">PABAK</th><th className="pr-2">Kripp. α</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([k, c]) => (
          <tr key={k} className="border-b border-gray-100 align-top">
            <td className="py-1 pr-2 font-medium">{CRIT_NAME[k] || k}</td>
            <td className="pr-2 text-gray-600">{Object.entries(c.dist).map(([l, n]) => `${LABEL[k]?.[l] || l} ${n}`).join(' · ')}</td>
            <td className="pr-2">{pct(c.pass_rating.p)}<span className="text-gray-400">{ci(c.pass_rating)}</span></td>
            <td className="pr-2">{pct(c.pass_all_raters.p)}<span className="text-gray-400">{ci(c.pass_all_raters)}</span></td>
            <td className="pr-2">{c.agreement.n_items}</td>
            <td className="pr-2">{pct(c.agreement.raw)}</td>
            <td className="pr-2">{num(c.agreement.fleiss_kappa)}</td>
            <td className="pr-2">{num(c.agreement.pabak)}</td>
            <td className="pr-2">{num(c.agreement.krippendorff_alpha)}</td>
          </tr>
        ))}
        <tr className="align-top">
          <td className="py-1 pr-2 font-medium">C1 + C2 + C3 all pass</td>
          <td className="pr-2 text-gray-400">—</td>
          <td className="pr-2">{pct(block.all_three.pass_rating.p)}<span className="text-gray-400">{ci(block.all_three.pass_rating)}</span></td>
          <td className="pr-2">{pct(block.all_three.pass_all_raters.p)}<span className="text-gray-400">{ci(block.all_three.pass_all_raters)}</span></td>
          <td className="pr-2">{block.all_three.agreement.n_items}</td>
          <td className="pr-2">{pct(block.all_three.agreement.raw)}</td>
          <td className="pr-2">{num(block.all_three.agreement.fleiss_kappa)}</td>
          <td className="pr-2">{num(block.all_three.agreement.pabak)}</td>
          <td className="pr-2">{num(block.all_three.agreement.krippendorff_alpha)}</td>
        </tr>
      </tbody>
    </table>
  );
}

function JudgeTable({ j }: { j: any }) {
  const cl = j.chunk_level, ql = j.question_level;
  return (
    <div className="grid md:grid-cols-2 gap-4 text-xs">
      <div>
        <div className="font-medium mb-1">Clause level: judge × human (n={cl.n}, raw {pct(cl.raw)}, binary κ {num(cl.cohen_kappa_binary)})</div>
        <table className="text-xs"><thead><tr className="text-gray-500"><th className="text-left pr-3">Judge \ Human</th><th className="pr-3">relevant</th><th>not relevant</th></tr></thead>
          <tbody>{Object.entries(cl.confusion).map(([l, r]: any) => (
            <tr key={l}><td className="pr-3">{l}</td><td className="pr-3 text-right">{r.relevant}</td><td className="text-right">{r.irrelevant}</td></tr>
          ))}</tbody></table>
      </div>
      <div>
        <div className="font-medium mb-1">Question level: judge's worst gold label × human C2</div>
        <table className="text-xs"><thead><tr className="text-gray-500"><th className="text-left pr-3">Judge \ C2</th><th className="pr-3">Yes</th><th className="pr-3">Partial</th><th>No</th></tr></thead>
          <tbody>{Object.entries(ql.confusion).map(([l, r]: any) => (
            <tr key={l}><td className="pr-3">{l}</td><td className="pr-3 text-right">{r.full}</td><td className="pr-3 text-right">{r.partial}</td><td className="text-right">{r.none}</td></tr>
          ))}</tbody></table>
      </div>
    </div>
  );
}

function QuestionDetail({ cid, qid, onClose, onSaved }: { cid: number; qid: string; onClose: () => void; onSaved: () => void }) {
  const [d, setD] = useState<any>(null);
  const [decision, setDecision] = useState('keep');
  const [note, setNote] = useState('');
  const [by, setBy] = useState(session.adminUser);
  const [err, setErr] = useState('');
  useEffect(() => {
    api.admin.question(cid, qid).then(x => {
      setD(x);
      if (x.adjudication) { setDecision(x.adjudication.decision); setNote(x.adjudication.note || ''); setBy(x.adjudication.decided_by || session.adminUser); }
    }).catch(e => setErr(e.message));
  }, [cid, qid]);
  if (!d) return <div className="text-xs text-gray-500">Loading… {err}</div>;
  const q = d.question;
  return (
    <div className="rounded-xl ring-1 ring-gray-300 p-4 space-y-3 bg-white">
      <div className="flex items-center gap-2 text-xs text-gray-500">
        <b className="text-gray-800">{qid}</b> · {q.question_class} / {q.question_subclass} · batch {q.batch}
        <button onClick={onClose} className="ml-auto underline">Close</button>
      </div>
      <div className="text-sm">{q.question}</div>
      <div className="rounded-lg bg-yellow-50 ring-1 ring-yellow-200 p-2 text-xs whitespace-pre-wrap">{q.reference_answer}</div>
      <details className="text-xs">
        <summary className="cursor-pointer text-gray-600">Gold ({d.gold.length}) · judge labels: {Object.entries(q.judge_gold).map(([c, l]) => `${c}=${l}`).join(', ') || '—'}</summary>
        <div className="mt-2 space-y-2">
          {d.gold.map((c: any) => (
            <div key={c.chunk_id} className="rounded ring-1 ring-gray-200 p-2">
              <div className="font-mono text-gray-700">{specLabel({ ...c, missing: false })} · {c.section_title}</div>
              <div className="whitespace-pre-wrap text-gray-700 max-h-64 overflow-y-auto mt-1">{c.content}</div>
            </div>
          ))}
        </div>
      </details>
      <table className="w-full text-xs">
        <thead><tr className="text-left text-gray-500 border-b"><th>Rater</th><th>Status</th><th>C1</th><th>C2</th><th>Not relevant</th><th>C3</th><th>Verdict</th><th>Own answer</th><th>Comment / fix</th><th>Time</th></tr></thead>
        <tbody>{d.ratings.map((r: any) => (
          <tr key={r.reviewer_id} className="border-b border-gray-100 align-top">
            <td className="py-1 pr-2">{r.reviewer}</td><td className="pr-2">{r.status}</td>
            <td className="pr-2">{LABEL.c1[r.c1] || r.c1}{r.c1_note ? ` (${r.c1_note})` : ''}</td>
            <td className="pr-2">{LABEL.c2[r.c2] || r.c2}{r.c2_note ? ` (${r.c2_note})` : ''}</td>
            <td className="pr-2">{(r.irrelevant_chunks || []).join(', ') || '—'}</td>
            <td className="pr-2">{LABEL.c3[r.c3] || r.c3 || '—'}</td><td className="pr-2">{TEXT[r.verdict] || r.verdict || '—'}</td>
            <td className="pr-2 whitespace-pre-wrap">{r.own_answer}</td>
            <td className="pr-2 whitespace-pre-wrap">{r.comment}{r.fix_text ? `\n→ ${r.fix_text}` : ''}</td>
            <td className="pr-2 tabular-nums">{Math.round(r.t_phase1_s || 0)}s + {Math.round(r.t_phase2_s || 0)}s</td>
          </tr>
        ))}</tbody>
      </table>
      <div className="flex flex-wrap items-end gap-2 text-xs">
        <label>Adjudication:
          <select value={decision} onChange={e => setDecision(e.target.value)} className="ml-1 rounded border border-gray-300 px-1 py-0.5">
            <option value="keep">Keep</option><option value="fix">Fix</option><option value="discard">Discard</option>
          </select>
        </label>
        <input value={by} onChange={e => setBy(e.target.value)} placeholder="decided by" className="rounded border border-gray-300 px-2 py-0.5" />
        <input value={note} onChange={e => setNote(e.target.value)} placeholder="note / what to fix" className="flex-1 min-w-[200px] rounded border border-gray-300 px-2 py-0.5" />
        <button onClick={async () => { try { await api.admin.adjudicate(cid, qid, { decision, note, decided_by: by }); onSaved(); } catch (e: any) { setErr(e.message); } }}
                className="rounded bg-gray-900 text-white px-3 py-1">Save adjudication</button>
        {d.adjudication && <span className="text-emerald-700">adjudicated: {TEXT[d.adjudication.decision]}</span>}
        {err && <span className="text-red-600">{err}</span>}
      </div>
    </div>
  );
}

function RaterRow({ r, campaigns, onError, onChanged, onIssued }: {
  r: any; campaigns: any[]; onError: (m: string) => void; onChanged: () => void; onIssued: (name: string, password: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState<string>(r.name);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setName(r.name); setEditing(false); }, [r.name]);
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true); onError('');
    try { await fn(); onChanged(); } catch (ex: any) { onError(ex.message || String(ex)); }
    setBusy(false);
  };
  const assigned = campaigns
    .map(c => ({ c, p: c.progress.find((p: any) => p.reviewer_id === r.id) }))
    .filter(x => x.p)
    .map(x => `${x.c.name}: ${x.p.n_done}/${x.p.n_assigned}`)
    .join(' · ');
  const btn = 'rounded-md border border-gray-300 bg-white px-2.5 py-1 text-xs hover:bg-gray-50 disabled:opacity-40';
  return (
    <tr className={`border-b border-gray-100 ${r.active ? '' : 'text-gray-400'}`}>
      <td className="py-2 pr-3 text-gray-500">{r.id}</td>
      <td className="py-2 pr-3">
        {editing ? (
          <form className="flex items-center gap-2" onSubmit={e => {
            e.preventDefault();
            if (name.trim() === r.name) { setEditing(false); return; }
            run(() => api.admin.patchReviewer(r.id, { name: name.trim() }));
          }}>
            <input value={name} onChange={e => setName(e.target.value)} disabled={busy} autoFocus
                   className="rounded-md border border-gray-300 px-2 py-1 text-sm w-56" />
            <button className="rounded-md bg-gray-900 text-white px-3 py-1 text-xs">Save</button>
            <button type="button" onClick={() => { setName(r.name); setEditing(false); }} className={btn}>Cancel</button>
          </form>
        ) : (
          <span className="text-sm font-medium">{r.name}</span>
        )}
      </td>
      <td className="py-2 pr-3">
        <select value={r.role} disabled={busy} onChange={e => run(() => api.admin.patchReviewer(r.id, { role: e.target.value }))}
                className="rounded-md border border-gray-300 px-2 py-1 text-xs bg-white">
          <option value="rater">rater</option><option value="admin">admin</option>
        </select>
      </td>
      <td className="py-2 pr-3">
        <span className={`inline-block rounded-full px-2 py-0.5 text-[11px] ${r.active ? 'bg-emerald-100 text-emerald-800' : 'bg-gray-100 text-gray-500'}`}>
          {r.active ? 'active' : 'inactive'}
        </span>
      </td>
      <td className="py-2 pr-3">
        {r.has_password ? <span className="text-gray-500">set</span> : <span className="text-red-600">not set</span>}
      </td>
      <td className="py-2 pr-3 text-gray-600">{assigned || '—'}</td>
      <td className="py-2 text-right whitespace-nowrap">
        <div className="inline-flex gap-1.5">
          {!editing && <button disabled={busy} onClick={() => setEditing(true)} className={btn}>Edit name</button>}
          <button disabled={busy} onClick={() => {
            const p = prompt(`New password for ${r.name} (leave empty to generate one)`);
            if (p === null) return;
            run(async () => { const res = await api.admin.resetPassword(r.id, p || undefined); onIssued(r.name, res.password); });
          }} className={btn}>{r.has_password ? 'Reset password' : 'Set password'}</button>
          <button disabled={busy} onClick={() => run(() => api.admin.patchReviewer(r.id, { active: !r.active }))} className={btn}>
            {r.active ? 'Deactivate' : 'Activate'}
          </button>
        </div>
      </td>
    </tr>
  );
}

export function Admin({ onExit }: { onExit: () => void }) {
  const [user, setUser] = useState('');
  const [pw, setPw] = useState('');
  const [ov, setOv] = useState<any>(null);
  const [err, setErr] = useState('');
  const [newName, setNewName] = useState('');
  const [newPw, setNewPw] = useState('');
  const [issued, setIssued] = useState<{ name: string; password: string; enrolled?: string[] } | null>(null);
  const [addToCampaign, setAddToCampaign] = useState(false);
  const [camp, setCamp] = useState<{ name: string; sample_file: string; raters_per_q: number; seed: number; reviewer_ids: number[];
                                     all_raters: boolean; requires_campaign_id: number | null }>(
    { name: '', sample_file: '', raters_per_q: 2, seed: 20261012, reviewer_ids: [], all_raters: false, requires_campaign_id: null });
  const [sel, setSel] = useState<number | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [dis, setDis] = useState<any[]>([]);
  const [openQ, setOpenQ] = useState<string | null>(null);
  const [assignIds, setAssignIds] = useState<number[]>([]);

  const load = async () => {
    setErr('');
    try {
      const o = await api.admin.overview();
      setOv(o);
      if (!camp.sample_file && o.samples[0]) setCamp(c => ({ ...c, sample_file: o.samples[0].file }));
      const first = o.campaigns[o.campaigns.length - 1];
      if ((sel == null || !o.campaigns.some((c: any) => c.id === sel)) && first) setSel(first.id);
    } catch (e) { setErr(e instanceof ApiError ? e.message : String(e)); if (e instanceof ApiError && e.status === 401) { setOv(null); } }
  };
  useEffect(() => { if (session.admin) load(); /* eslint-disable-next-line */ }, []);
  const loadCampaign = async (cid: number) => {
    try {
      const [s, d] = await Promise.all([api.admin.stats(cid), api.admin.disagreements(cid)]);
      setStats(s); setDis(d.items);
      setAssignIds(ov?.campaigns.find((c: any) => c.id === cid)?.progress.map((p: any) => p.reviewer_id) || []);
    } catch (e: any) { setErr(e.message); }
  };
  useEffect(() => { if (sel != null && ov) loadCampaign(sel); /* eslint-disable-next-line */ }, [sel, ov]);

  const signIn = async (e: React.FormEvent) => {
    e.preventDefault(); setErr('');
    try {
      const r = await api.admin.login(user.trim(), pw);
      session.setAdmin(r.token, r.username);
      setPw('');
      await load();
    } catch (ex) { setErr(ex instanceof ApiError ? ex.message : String(ex)); }
  };

  if (!ov) {
    return (
      <div className="p-8 max-w-sm">
        <h1 className="text-base font-semibold">Admin sign-in</h1>
        <p className="text-xs text-gray-500 mt-1">Administrator account (not a rater account). Sessions expire after 12 hours.</p>
        <form onSubmit={signIn} className="mt-3 space-y-2">
          <input value={user} onChange={e => setUser(e.target.value)} placeholder="Username" autoFocus autoComplete="username"
                 className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm" />
          <input type="password" value={pw} onChange={e => setPw(e.target.value)} placeholder="Password" autoComplete="current-password"
                 className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm" />
          <button disabled={!user.trim() || !pw} className="rounded-md bg-gray-900 text-white text-sm px-4 py-2 disabled:opacity-40">Sign in</button>
          <button type="button" onClick={onExit} className="ml-3 text-sm text-gray-600 underline">Back</button>
        </form>
        {err && <div className="mt-2 text-xs text-red-600">{err}</div>}
      </div>
    );
  }

  const campaign = ov.campaigns.find((c: any) => c.id === sel);
  const toggleId = (ids: number[], id: number) => ids.includes(id) ? ids.filter(x => x !== id) : [...ids, id];

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-6xl mx-auto p-6 space-y-6 text-sm">
        <div className="flex items-center gap-3">
          <h1 className="text-base font-semibold">Review administration</h1>
          <button onClick={load} className="text-xs underline text-gray-600">Reload</button>
          <span className="ml-auto text-xs text-gray-500">Signed in as <b className="text-gray-800">{session.adminUser}</b></span>
          <button onClick={() => { session.clearAdmin(); setOv(null); }} className="text-xs underline text-gray-600">Sign out</button>
          <button onClick={onExit} className="text-xs underline text-gray-600">Rater page</button>
        </div>
        {err && <div className="text-xs text-red-600">{err}</div>}

        <section className="rounded-xl ring-1 ring-gray-200 p-4">
          <h2 className="font-semibold mb-2">Raters</h2>
          <p className="text-xs text-gray-500 mb-2">The name is the username a rater signs in with. Renaming keeps the id, assignments and ratings.</p>
          <table className="w-full text-xs">
            <thead><tr className="text-left text-gray-500 border-b"><th className="py-1 pr-3">#</th><th className="pr-3">Name (username)</th><th className="pr-3">Role</th><th className="pr-3">Status</th><th className="pr-3">Password</th><th className="pr-3">Assigned (done/total)</th><th className="text-right">Actions</th></tr></thead>
            <tbody>
              {ov.reviewers.map((r: any) => (
                <RaterRow key={r.id} r={r} campaigns={ov.campaigns} onError={setErr} onChanged={load}
                          onIssued={(name, password) => setIssued({ name, password })} />
              ))}
            </tbody>
          </table>
          <form onSubmit={async e => {
                  e.preventDefault();
                  try {
                    const r = await api.admin.addReviewer(newName, newPw || undefined);
                    setIssued({ name: r.name, password: r.initial_password,
                                enrolled: (r.enrolled_campaigns || []).map((id: number) => ov.campaigns.find((c: any) => c.id === id)?.name || `#${id}`) });
                    if (addToCampaign && sel != null && campaign && !campaign.all_raters) {
                      await api.admin.assign(sel, [...campaign.progress.map((p: any) => p.reviewer_id), r.id]);
                    }
                    setNewName(''); setNewPw(''); load();
                  } catch (ex: any) { setErr(ex.message); }
                }} className="mt-3 flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-gray-700 w-full">Add a rater</span>
            <input value={newName} onChange={e => setNewName(e.target.value)} placeholder="Name (username)" className="rounded-md border border-gray-300 px-3 py-1.5 text-sm w-64" />
            <input value={newPw} onChange={e => setNewPw(e.target.value)} placeholder="Password (leave empty to generate)" className="rounded-md border border-gray-300 px-3 py-1.5 text-sm w-60" />
            <span className="text-xs text-gray-500 w-full">New raters are enrolled automatically in every open campaign where every rater rates every question (pilot + main).</span>
            {campaign && !campaign.all_raters && (
              <label className="flex items-center gap-1 text-xs text-gray-600">
                <input type="checkbox" checked={addToCampaign} onChange={e => setAddToCampaign(e.target.checked)} />
                also assign in campaign "{campaign.name}" (rebalances unrated questions)
              </label>
            )}
            <button disabled={!newName.trim()} className="rounded-md bg-gray-900 text-white text-xs px-3 py-1.5 disabled:opacity-40">Add rater</button>
          </form>
          {issued && (
            <div className="mt-3 rounded-md bg-amber-50 ring-1 ring-amber-200 p-3 text-xs flex items-center gap-3">
              <span>Password for <b>{issued.name}</b>: <code className="font-mono text-sm bg-white px-2 py-0.5 rounded ring-1 ring-amber-200 select-all">{issued.password}</code> — shown once, hand it over now.
                {issued.enrolled && issued.enrolled.length > 0 && <> Enrolled in: <b>{issued.enrolled.join(', ')}</b>.</>}</span>
              <button onClick={() => navigator.clipboard?.writeText(issued.password)} className="underline">Copy</button>
              <button onClick={() => setIssued(null)} className="ml-auto underline">Dismiss</button>
            </div>
          )}
        </section>

        <section className="rounded-xl ring-1 ring-gray-200 p-4">
          <h2 className="font-semibold mb-2">Create a campaign</h2>
          <div className="grid md:grid-cols-4 gap-2 text-xs">
            <input value={camp.name} onChange={e => setCamp({ ...camp, name: e.target.value })} placeholder="Campaign name (e.g. main_300)" className="rounded-md border border-gray-300 px-2 py-1.5" />
            <select value={camp.sample_file} onChange={e => setCamp({ ...camp, sample_file: e.target.value })} className="rounded-md border border-gray-300 px-2 py-1.5">
              {ov.samples.map((s: any) => <option key={s.file} value={s.file}>{s.file} ({s.n} questions)</option>)}
            </select>
            <label className={camp.all_raters ? 'text-gray-400' : ''}>Raters / question <input type="number" min={1} disabled={camp.all_raters} value={camp.raters_per_q} onChange={e => setCamp({ ...camp, raters_per_q: Number(e.target.value) })} className="ml-1 w-14 rounded-md border border-gray-300 px-2 py-1" /></label>
            <label>Seed <input type="number" value={camp.seed} onChange={e => setCamp({ ...camp, seed: Number(e.target.value) })} className="ml-1 w-28 rounded-md border border-gray-300 px-2 py-1" /></label>
          </div>
          <div className="mt-2 flex flex-wrap gap-3 text-xs">
            {ov.reviewers.filter((r: any) => r.active).map((r: any) => (
              <label key={r.id} className="flex items-center gap-1"><input type="checkbox" checked={camp.reviewer_ids.includes(r.id)} onChange={() => setCamp({ ...camp, reviewer_ids: toggleId(camp.reviewer_ids, r.id) })} />{r.name}</label>
            ))}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-4 text-xs">
            <label className="flex items-center gap-1"><input type="checkbox" checked={camp.all_raters} onChange={e => setCamp({ ...camp, all_raters: e.target.checked })} />
              Pilot: every rater rates every question</label>
            <label>Must finish first:
              <select value={camp.requires_campaign_id ?? ''} onChange={e => setCamp({ ...camp, requires_campaign_id: e.target.value ? Number(e.target.value) : null })}
                      className="ml-1 rounded-md border border-gray-300 px-2 py-1">
                <option value="">— none —</option>
                {ov.campaigns.map((c: any) => <option key={c.id} value={c.id}>#{c.id} {c.name}</option>)}
              </select>
            </label>
          </div>
          <button onClick={async () => { try { const c = await api.admin.createCampaign(camp); setSel(c.id); if (c.missing_chunks?.length) alert(`${c.missing_chunks.length} chunks missing from the KG: ${c.missing_chunks.slice(0, 5).join(', ')}`); load(); } catch (ex: any) { setErr(ex.message); } }}
                  className="mt-3 rounded-md bg-gray-900 text-white text-xs px-3 py-1.5">Create campaign (snapshots gold from Neo4j and assigns)</button>
        </section>

        {ov.campaigns.length > 0 && (
          <section className="rounded-xl ring-1 ring-gray-200 p-4 space-y-4">
            <div className="flex items-center gap-3 flex-wrap">
              <h2 className="font-semibold">Campaign</h2>
              <select value={sel ?? ''} onChange={e => setSel(Number(e.target.value))} className="rounded-md border border-gray-300 px-2 py-1 text-xs">
                {ov.campaigns.map((c: any) => <option key={c.id} value={c.id}>#{c.id} {c.name}{c.all_raters ? ' (pilot)' : ''} · {c.n_main}+{c.n_diag} questions · {c.status}</option>)}
              </select>
              {campaign && (
                <>
                  <button onClick={async () => { await api.admin.setStatus(campaign.id, campaign.status === 'open' ? 'closed' : 'open'); load(); }} className="text-xs underline">
                    {campaign.status === 'open' ? 'Close campaign' : 'Reopen'}
                  </button>
                  <label className="text-xs text-gray-600">Must finish first:
                    <select value={campaign.requires_campaign_id ?? ''} className="ml-1 rounded-md border border-gray-300 px-1 py-0.5"
                            onChange={async e => { try { await api.admin.setRequires(campaign.id, e.target.value ? Number(e.target.value) : null); load(); } catch (ex: any) { setErr(ex.message); } }}>
                      <option value="">— none —</option>
                      {ov.campaigns.filter((c: any) => c.id !== campaign.id).map((c: any) => <option key={c.id} value={c.id}>#{c.id} {c.name}</option>)}
                    </select>
                  </label>
                  <button onClick={async () => { if (!confirm(`Delete campaign "${campaign.name}"? Only possible while it has no ratings.`)) return; try { await api.admin.deleteCampaign(campaign.id); setSel(null); setStats(null); load(); } catch (ex: any) { setErr(ex.message); } }}
                          className="text-xs underline text-red-700">Delete</button>
                  <button onClick={() => downloadExport(campaign.id, 'json')} className="text-xs underline">Export JSON</button>
                  <button onClick={() => downloadExport(campaign.id, 'csv')} className="text-xs underline">Export CSV</button>
                  <span className="ml-auto text-xs text-gray-500">Rater link: {location.origin}/review.html</span>
                </>
              )}
            </div>
            {campaign && (
              <>
                <table className="text-xs w-full">
                  <thead><tr className="text-left text-gray-500 border-b"><th>Rater</th><th>Assigned</th><th>Done</th><th>Draft</th><th>Avg s / question</th><th>Last activity</th></tr></thead>
                  <tbody>{campaign.progress.map((p: any) => (
                    <tr key={p.reviewer_id} className="border-b border-gray-100">
                      <td className="py-1">{p.name}</td><td>{p.n_assigned}</td><td>{p.n_done}</td><td>{p.n_draft}</td>
                      <td>{p.avg_seconds ? Math.round(p.avg_seconds) : '—'}</td>
                      <td>{p.last_activity ? new Date(p.last_activity * 1000).toLocaleString('en-GB') : '—'}</td>
                    </tr>
                  ))}</tbody>
                </table>
                <div className="flex flex-wrap items-center gap-3 text-xs">
                  <span className="text-gray-500">Reassign to:</span>
                  {ov.reviewers.filter((r: any) => r.active).map((r: any) => (
                    <label key={r.id} className="flex items-center gap-1"><input type="checkbox" checked={assignIds.includes(r.id)} onChange={() => setAssignIds(toggleId(assignIds, r.id))} />{r.name}</label>
                  ))}
                  <button onClick={async () => { if (!confirm('Reassign? Pairs that already carry a rating are kept.')) return; try { await api.admin.assign(campaign.id, assignIds); load(); } catch (ex: any) { setErr(ex.message); } }}
                          className="rounded-md bg-gray-100 px-3 py-1">Assign</button>
                </div>
              </>
            )}

            {stats && Object.entries(stats.batches).map(([batch, b]: any) => (
              <div key={batch} className="space-y-3">
                <h3 className="font-semibold text-sm">{batch === 'main' ? 'Main batch' : 'Diagnostic batch (judge-flagged)'} — {b.all.n_questions} questions, {b.all.n_rated_2plus} with ≥2 completed ratings</h3>
                <CritTable block={b.all} />
                {Object.entries(b.per_class).map(([cls, blk]: any) => (
                  <details key={cls} className="rounded-lg ring-1 ring-gray-200 p-2">
                    <summary className="cursor-pointer text-xs font-medium">{cls} — {blk.n_rated_2plus}/{blk.n_questions} with 2 ratings · all three pass (both raters) {pct(blk.all_three.pass_all_raters.p)}</summary>
                    <div className="mt-2 space-y-3"><CritTable block={blk} /><JudgeTable j={blk.judge} /></div>
                  </details>
                ))}
                <details className="rounded-lg ring-1 ring-gray-200 p-2">
                  <summary className="cursor-pointer text-xs font-medium">Human vs LLM judge (whole batch)</summary>
                  <div className="mt-2"><JudgeTable j={b.judge} /></div>
                </details>
                <div className="text-xs text-gray-500">Adjudication: keep {b.all.adjudication.keep} · fix {b.all.adjudication.fix} · discard {b.all.adjudication.discard} · pending {b.all.adjudication.pending}</div>
              </div>
            ))}
            {stats && (
              <div className="text-xs text-gray-500">
                Rater pairs (Cohen κ on C1): {stats.batches.main?.all.criteria.c1.pairwise.map((p: any) => `#${p.reviewers.join('–#')} n=${p.n} κ=${num(p.cohen_kappa)}`).join(' · ') || '—'}
              </div>
            )}

            <div>
              <h3 className="font-semibold text-sm mb-2">Questions needing adjudication ({dis.filter(d => !d.adjudication).length} pending / {dis.length})</h3>
              <div className="rounded-lg ring-1 ring-gray-200 divide-y">
                {dis.map(d => (
                  <div key={d.qid} className="px-3 py-2 text-xs">
                    <div className="flex items-center gap-2">
                      <button onClick={() => setOpenQ(openQ === d.qid ? null : d.qid)} className="font-mono underline">{d.qid}</button>
                      <span className="text-gray-500">{d.question_class} · {d.batch}</span>
                      <span className="text-gray-700">{d.ratings.map((r: any) => `${r.reviewer}: ${LABEL.c1[r.c1]}/${LABEL.c2[r.c2]}/${LABEL.c3[r.c3]}/${TEXT[r.verdict]}`).join('  |  ')}</span>
                      <span className={`ml-auto ${d.adjudication ? 'text-emerald-700' : 'text-amber-700'}`}>{d.adjudication ? `adjudicated: ${TEXT[d.adjudication.decision]}` : 'pending'}</span>
                    </div>
                    {openQ === d.qid && sel != null && (
                      <div className="mt-2"><QuestionDetail cid={sel} qid={d.qid} onClose={() => setOpenQ(null)} onSaved={() => loadCampaign(sel)} /></div>
                    )}
                  </div>
                ))}
                {dis.length === 0 && <div className="px-3 py-2 text-xs text-gray-400">No question with two completed ratings disagrees yet.</div>}
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
