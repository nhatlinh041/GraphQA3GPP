/**
 * Shared left rail across the four pages.
 *
 * They used to carry four DIFFERENT hand-written navs — the chat had bordered
 * buttons that opened new tabs, the other three had underlined links that did
 * not, and each listed a different subset of destinations (the Cypher page
 * offered only "back to chat"). So which pages existed at all depended on which
 * page you happened to be standing on. One component, one list.
 *
 * Plain <a> rather than a router: these are four separate Vite entry points with
 * their own HTML files, not routes in one SPA.
 */
import { useEffect, useState } from 'react';

type Item = { href: string; label: string; icon: JSX.Element; match: string };

/** Collapsed state is per-browser and shared by all four pages, so the rail does
 *  not silently re-expand every time you move between them — these are separate
 *  HTML entry points, not routes, so each navigation is a full page load. */
const KEY = 'sidenav-collapsed';

const I = {
  chat: (
    <>
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </>
  ),
  cypher: (
    <>
      <circle cx="12" cy="5" r="2.5" /><circle cx="5" cy="18" r="2.5" />
      <circle cx="19" cy="18" r="2.5" />
      <path d="M10.5 7L6.5 15.7M13.5 7l4 8.7M7.5 18h9" />
    </>
  ),
  runs: (
    <>
      <path d="M3 3v18h18" /><path d="M7 15l4-5 3 3 5-7" />
    </>
  ),
  bench: (
    <>
      <path d="M4 19V5M10 19V9M16 19v-7M22 19H2" />
    </>
  ),
};

const ITEMS: Item[] = [
  { href: '/', label: 'Chat', icon: I.chat, match: 'chat' },
  { href: '/cypher.html', label: 'Cypher', icon: I.cypher, match: 'cypher' },
  { href: '/run.html', label: 'Runs', icon: I.runs, match: 'run' },
  { href: '/benchmarks.html', label: 'Benchmarks', icon: I.bench, match: 'bench' },
];

/** `current` names the active page: 'chat' | 'cypher' | 'run' | 'bench'. */
export function SideNav({ current, children }: { current: string; children?: React.ReactNode }) {
  // Read lazily so the first paint is already in the right state — initialising to
  // false and correcting in an effect makes the rail visibly snap on every load.
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try { return localStorage.getItem(KEY) === '1'; } catch { return false; }
  });
  useEffect(() => {
    try { localStorage.setItem(KEY, collapsed ? '1' : '0'); } catch { /* private mode */ }
  }, [collapsed]);

  // Below `lg` the rail is icon-only regardless: there is no room for labels, and
  // letting the toggle expand it there would cover the content it navigates to.
  const wide = collapsed ? 'w-14' : 'w-14 lg:w-44';
  const labelCls = collapsed ? 'hidden' : 'hidden lg:inline';

  return (
    <nav className={`${wide} shrink-0 border-r border-gray-200 bg-gray-50/70
                    flex flex-col py-3 gap-0.5 transition-[width] duration-150`}>
      <div className="px-3 lg:px-4 pb-3 mb-1 border-b border-gray-200 flex items-center">
        <span className={`text-sm font-semibold text-gray-700 ${labelCls}`}>3GPP QA</span>
        {/* Narrow rail keeps a mark rather than going blank, so the column still
            reads as a nav and not as an unexplained empty strip. */}
        <span className={`text-sm font-semibold text-gray-700 ${collapsed ? '' : 'lg:hidden'}`}>3G</span>
        <button
          type="button"
          onClick={() => setCollapsed(v => !v)}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className={`ml-auto p-1 -mr-1 rounded text-gray-400 hover:text-gray-700
                      hover:bg-gray-200/70 ${collapsed ? 'hidden' : 'hidden lg:block'}`}
        >
          {/* Hamburger both ways: one glyph that reads as "the menu control",
              rather than a left chevron that has to be re-read as a direction. */}
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="2" strokeLinecap="round">
            <path d="M4 7h16M4 12h16M4 17h16" />
          </svg>
        </button>
      </div>
      {collapsed && (
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          title="Expand sidebar"
          aria-label="Expand sidebar"
          className="mx-2 mb-1 px-2.5 py-1.5 rounded-md hidden lg:flex items-center justify-center
                     text-gray-400 hover:text-gray-700 hover:bg-gray-100"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="2" strokeLinecap="round">
            <path d="M4 7h16M4 12h16M4 17h16" />
          </svg>
        </button>
      )}
      {ITEMS.map(it => {
        const active = it.match === current;
        return (
          <a
            key={it.href}
            href={it.href}
            aria-current={active ? 'page' : undefined}
            title={it.label}
            className={`mx-2 px-2.5 lg:px-3 py-2 rounded-md flex items-center gap-2.5 text-[13px]
                        transition-colors ${
              active
                ? 'bg-white text-gray-900 font-medium ring-1 ring-gray-200'
                : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
            }`}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"
                 className="shrink-0">
              {it.icon}
            </svg>
            <span className={labelCls}>{it.label}</span>
          </a>
        );
      })}
      {/* Page-specific actions (e.g. Export JSON) sit at the bottom of the rail so
          the destination list above stays identical everywhere. */}
      {children && <div className="mt-auto px-2 pt-3 border-t border-gray-200">{children}</div>}
    </nav>
  );
}
