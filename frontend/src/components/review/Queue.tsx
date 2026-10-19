import React, { useEffect, useState } from 'react';
import { api, Lock, Me, QueueItem, session } from './api';
import { Guide } from './Guide';

const STATUS: Record<string, { t: string; cls: string }> = {
  todo: { t: 'To do', cls: 'bg-gray-100 text-gray-600' },
  draft: { t: 'In progress', cls: 'bg-amber-100 text-amber-800' },
  done: { t: 'Done', cls: 'bg-emerald-100 text-emerald-800' },
};

export function Queue({ me, campaign, onCampaign, onOpen, onSignOut, onAdmin }: {
  me: Me; campaign: number | null; onCampaign: (c: number) => void;
  onOpen: (qid: string) => void; onSignOut: () => void; onAdmin: () => void;
}) {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [lock, setLock] = useState<Lock | null>(null);
  const [filter, setFilter] = useState<'all' | 'todo' | 'draft' | 'done'>('all');
  const [guide, setGuide] = useState(false);
  const [err, setErr] = useState('');

  useEffect(() => {
    if (campaign == null) return;
    api.queue(campaign).then(r => { setItems(r.items); setLock(r.lock); }).catch(e => setErr(String(e.message || e)));
  }, [campaign]);

  const changePassword = async () => {
    const oldPw = prompt('Current password');
    if (!oldPw) return;
    const newPw = prompt('New password (at least 6 characters)');
    if (!newPw) return;
    try {
      const r = await api.changePassword(oldPw, newPw);
      session.signIn(r.token, me.reviewer.name);
      alert('Password changed.');
    } catch (e: any) { alert(e.message || String(e)); }
  };

  const nDone = items.filter(i => i.status === 'done').length;
  const next = items.find(i => i.status !== 'done');
  const shown = items.filter(i => filter === 'all' || i.status === filter);

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto p-6">
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-base font-semibold">{me.reviewer.name}'s queue</h1>
          <button onClick={() => setGuide(g => !g)} title="Rating guide"
                  className="w-6 h-6 rounded-full ring-1 ring-gray-300 text-xs text-gray-600 hover:bg-gray-100">?</button>
          <div className="ml-auto flex items-center gap-3 text-xs text-gray-500">
            {me.reviewer.role === 'admin' && <button onClick={onAdmin} className="underline">Admin</button>}
            <button onClick={changePassword} className="underline">Change password</button>
            <button onClick={onSignOut} className="underline">Sign out</button>
          </div>
        </div>
        {guide && <div className="mt-3 rounded-lg ring-1 ring-gray-200 p-4 bg-gray-50/60"><Guide compact /></div>}

        {me.campaigns.length > 1 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {me.campaigns.map(c => {
              const active = c.id === campaign;
              return (
                <button key={c.id} onClick={() => onCampaign(c.id)}
                        className={`rounded-md px-3 py-1.5 text-sm ring-1 ${active ? 'ring-gray-900 bg-gray-900 text-white' : 'ring-gray-300 text-gray-700 hover:bg-gray-50'}`}>
                  {c.is_pilot ? 'Pilot' : 'Main'} · {c.name}
                  <span className={`ml-2 text-xs ${active ? 'text-gray-300' : 'text-gray-500'}`}>
                    {c.lock ? 'locked' : `${c.n_done}/${c.n_assigned}`}
                  </span>
                </button>
              );
            })}
          </div>
        )}
        {lock && (
          <div className="mt-4 rounded-lg bg-amber-50 ring-1 ring-amber-200 p-3 text-sm text-amber-900">
            {lock.n_assigned
              ? <>This campaign unlocks after you finish the pilot <b>{lock.requires_name}</b> ({lock.n_left} question{lock.n_left === 1 ? '' : 's'} left).{' '}
                  <button onClick={() => onCampaign(lock.requires_campaign_id)} className="underline font-medium">Go to the pilot</button></>
              : <>This campaign requires the pilot <b>{lock.requires_name}</b>, which has not been assigned to you yet. Contact the administrator.</>}
          </div>
        )}
        {me.campaigns.length === 0 && (
          <div className="mt-6 text-sm text-gray-500">No questions assigned to you yet. Contact the administrator.</div>
        )}

        {items.length > 0 && (
          <>
            <div className="mt-5">
              <div className="flex items-center justify-between text-xs text-gray-600">
                <span>{nDone} / {items.length} questions done</span>
                <span>{Math.round(100 * nDone / items.length)}%</span>
              </div>
              <div className="mt-1 h-2 rounded bg-gray-100 overflow-hidden">
                <div className="h-full bg-emerald-500 transition-[width]" style={{ width: `${100 * nDone / items.length}%` }} />
              </div>
            </div>
            <div className={`mt-4 flex items-center gap-2 flex-wrap ${lock ? 'opacity-40 pointer-events-none' : ''}`}>
              {next && (
                <button onClick={() => onOpen(next.qid)}
                        className="rounded-md bg-gray-900 text-white text-sm px-4 py-2">
                  {nDone === 0 ? 'Start' : 'Continue'} · question {next.idx}
                </button>
              )}
              {!next && <span className="text-sm text-emerald-700 font-medium">All done in this campaign — thank you!</span>}
              <div className="ml-auto flex gap-1 text-xs">
                {(['all', 'todo', 'draft', 'done'] as const).map(f => (
                  <button key={f} onClick={() => setFilter(f)}
                          className={`px-2 py-1 rounded ${filter === f ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-600'}`}>
                    {f === 'all' ? 'All' : STATUS[f].t}
                  </button>
                ))}
              </div>
            </div>
            <div className={`mt-3 rounded-lg ring-1 ring-gray-200 divide-y ${lock ? 'opacity-40 pointer-events-none' : ''}`}>
              {shown.map(i => (
                <button key={i.qid} onClick={() => onOpen(i.qid)}
                        className="w-full flex items-center gap-3 px-3 py-2 text-sm hover:bg-gray-50 text-left">
                  <span className="w-12 text-gray-500 tabular-nums">#{i.idx}</span>
                  <span className={`text-[11px] px-2 py-0.5 rounded-full ${STATUS[i.status].cls}`}>{STATUS[i.status].t}</span>
                  <span className="ml-auto text-xs text-gray-400">
                    {i.updated_at ? new Date(i.updated_at * 1000).toLocaleString('en-GB') : ''}
                  </span>
                </button>
              ))}
            </div>
          </>
        )}
        {err && <div className="mt-3 text-xs text-red-600">{err}</div>}
      </div>
    </div>
  );
}
