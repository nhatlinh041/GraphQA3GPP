import React, { useState } from 'react';
import { api, ApiError, session } from './api';
import { Guide } from './Guide';

/** One ordinary username/password form for everybody: the credentials are
 *  tried as a rater account first, then as an admin account. */
export function Login({ onSignedIn, onAdminSignedIn }: { onSignedIn: () => Promise<void>; onAdminSignedIn: () => void }) {
  const [user, setUser] = useState('');
  const [pw, setPw] = useState('');
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(''); setBusy(true);
    try {
      const r = await api.login(user, pw);
      session.signIn(r.token, r.reviewer.name);
      await onSignedIn();
      return;
    } catch (ex) {
      session.signOut();
      if (!(ex instanceof ApiError && ex.status === 401)) { setErr(ex instanceof ApiError ? ex.message : String(ex)); setBusy(false); return; }
    }
    try {
      const a = await api.admin.login(user, pw);
      session.setAdmin(a.token, a.username);
      onAdminSignedIn();
    } catch (ex) { setErr('Wrong username or password.'); }
    setBusy(false);
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto p-6 lg:p-10 grid lg:grid-cols-[360px_1fr] gap-8">
        <form onSubmit={submit} className="rounded-xl ring-1 ring-gray-200 p-5 h-fit bg-gray-50/60 space-y-3">
          <div>
            <h1 className="text-base font-semibold">3GPP-ClauseQA expert review</h1>
            <p className="text-xs text-gray-500 mt-1">Sign in with the account you were given.</p>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Username</label>
            <input value={user} onChange={e => setUser(e.target.value)} autoFocus autoComplete="username"
                   className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-gray-300" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Password</label>
            <input type="password" value={pw} onChange={e => setPw(e.target.value)} autoComplete="current-password"
                   className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-gray-300" />
          </div>
          <button disabled={busy || !user.trim() || !pw} className="w-full rounded-md bg-gray-900 text-white text-sm py-2 disabled:opacity-40">
            Sign in
          </button>
          {err && <div className="text-xs text-red-600">{err}</div>}
          <p className="text-[11px] text-gray-400 pt-2 border-t border-gray-200">Raters and administrators use the same form. Ask the administrator if you have no account.</p>
        </form>
        <div>
          <h2 className="text-sm font-semibold mb-2">Rating guide</h2>
          <Guide />
        </div>
      </div>
    </div>
  );
}
