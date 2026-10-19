import React, { useEffect, useState } from 'react';
import { api, ApiError, Me, session } from './api';
import { Login } from './Login';
import { Queue } from './Queue';
import { QuestionCard } from './QuestionCard';
import { Admin } from './Admin';

type View = { kind: 'login' } | { kind: 'queue' } | { kind: 'question'; qid: string } | { kind: 'admin' };

const readUrl = (): View => {
  const p = new URLSearchParams(location.search);
  if (p.get('admin') === '1') return { kind: 'admin' };
  const qid = p.get('qid');
  return qid ? { kind: 'question', qid } : { kind: 'queue' };
};
const writeUrl = (v: View) => {
  const p = new URLSearchParams();
  if (v.kind === 'admin') p.set('admin', '1');
  if (v.kind === 'question') p.set('qid', v.qid);
  const q = p.toString();
  history.pushState(null, '', location.pathname + (q ? '?' + q : ''));
};

/**
 * Expert review of the 3GPP-ClauseQA sample. Everything but the sign-in card
 * is behind an account; the URL carries only the view (admin / qid) so a
 * reload lands back on the same question.
 */
export function ReviewApp() {
  const [me, setMe] = useState<Me | null>(null);
  const [campaign, setCampaign] = useState<number | null>(null);
  const [view, setView] = useState<View>(() => readUrl());
  const [booting, setBooting] = useState(true);

  const go = (v: View) => { writeUrl(v); setView(v); };

  const loadMe = async () => {
    const m = await api.me();
    setMe(m);
    // Campaigns arrive pilots-first: land on the first open one that is unlocked and unfinished.
    const todo = m.campaigns.find(c => c.status === 'open' && !c.lock && c.n_done < c.n_assigned)
      || m.campaigns.find(c => c.status === 'open') || m.campaigns[0];
    setCampaign(c => {
      const cur = m.campaigns.find(x => x.id === c);
      // Stay where the rater is, unless that was a pilot they just finished.
      if (cur && !(cur.is_pilot && cur.n_done >= cur.n_assigned)) return cur.id;
      return todo ? todo.id : null;
    });
    return m;
  };

  useEffect(() => {
    (async () => {
      if (view.kind === 'admin') { setBooting(false); return; }
      if (!session.token) { setView({ kind: 'login' }); setBooting(false); return; }
      try { await loadMe(); }
      catch (e) {
        if (e instanceof ApiError && (e.status === 401 || e.status === 403)) { session.signOut(); setView({ kind: 'login' }); }
      }
      setBooting(false);
    })();
    const onPop = () => setView(readUrl());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (booting) return <div className="p-8 text-sm text-gray-500">Loading…</div>;

  if (view.kind === 'admin') {
    return <Admin onExit={() => go(session.token ? { kind: 'queue' } : { kind: 'login' })} />;
  }

  if (view.kind === 'login' || !me) {
    return (
      <Login
        onSignedIn={async () => { await loadMe(); go({ kind: 'queue' }); }}
        onAdminSignedIn={() => go({ kind: 'admin' })}
      />
    );
  }

  const signOut = () => { session.signOut(); setMe(null); go({ kind: 'login' }); };

  if (view.kind === 'question' && campaign != null) {
    return (
      <QuestionCard
        campaign={campaign}
        qid={view.qid}
        reviewer={me.reviewer.name}
        onBack={() => { loadMe().catch(() => {}); go({ kind: 'queue' }); }}
        onNext={(qid) => { if (!qid) loadMe().catch(() => {}); go(qid ? { kind: 'question', qid } : { kind: 'queue' }); }}
      />
    );
  }

  return (
    <Queue
      me={me}
      campaign={campaign}
      onCampaign={setCampaign}
      onOpen={(qid) => go({ kind: 'question', qid })}
      onSignOut={signOut}
      onAdmin={() => go({ kind: 'admin' })}
    />
  );
}
