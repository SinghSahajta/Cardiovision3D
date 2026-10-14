import { useEffect, useState, type ReactNode } from 'react';

import { ThemeSwitch } from './ThemeSwitch';

export interface NavSection {
  id: string;
  label: string;
  icon: 'heart' | 'explain' | 'intake' | 'model';
  /** The explanation panel does not exist while a prediction is blocked. */
  available: boolean;
}

interface Props {
  sections: NavSection[];
  active: string | null;
  /** Primary line of the patient card, e.g. "Cohort patient 6". */
  patientName: string;
  /** Secondary line, e.g. "of 303 · in-sample". */
  patientDetail: string;
  /** One or two characters for the avatar disc. */
  patientInitial: string;
}

const ICONS: Record<NavSection['icon'], ReactNode> = {
  heart: (
    <path d="M8 14s-5.5-3.5-5.5-7A3.2 3.2 0 0 1 8 5.2 3.2 3.2 0 0 1 13.5 7c0 3.5-5.5 7-5.5 7Z" />
  ),
  explain: <path d="M2.5 13.5v-4m3.5 4v-7m3.5 7v-10m3.5 10v-5.5" />,
  intake: <path d="M8 10.5V2.5m0 0L5 5.5m3-3 3 3M2.5 10v2.5a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1V10" />,
  model: <path d="M2.5 13.5h11m-9-2.5v-4m3 4V4.5m3 6.5v-3m3 3V6" />,
};

const STORAGE_KEY = 'rail-collapsed';

/**
 * The persistent rail: identity, navigation, and the mode switch.
 *
 * Navigation is over one scrolling document, not routes - there is no router in
 * this app and adding one to move between four sections of a single page would
 * be ceremony. Items scroll to their section and the active one is derived from
 * scroll position by useScrollSpy.
 *
 * An unavailable section renders disabled rather than hidden. The explanation
 * panel disappears whenever the blocking question is unanswered, and a nav list
 * that changes length as you fill in a form is disorienting - a greyed item
 * says "there is a section here, it has nothing to show yet".
 *
 * COLLAPSING keeps the icons rather than hiding the rail outright, so
 * navigation and the mode switch stay reachable and the page does not lose its
 * left anchor. The collapsed width is 84px because the theme switch is a fixed
 * 64px wide and still has to fit inside the padding - a narrower rail would
 * have meant swapping it for a different control in one state only.
 *
 * The choice is remembered. Collapsing a sidebar is a statement about how you
 * want to work, not a per-visit whim, and having it spring back open on reload
 * is the kind of small betrayal that stops people using the control at all.
 *
 * Below `lg` this is not rendered at all; App.tsx puts up a compact top bar
 * instead, which keeps the phone layout single-column.
 */
export function Sidebar({
  sections,
  active,
  patientName,
  patientDetail,
  patientInitial,
}: Props) {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) === '1';
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0');
    } catch {
      // Blocked storage: the rail still collapses, it just will not be
      // remembered.
    }
  }, [collapsed]);

  const go = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <aside
      className={`rail sticky top-0 hidden h-screen shrink-0 flex-col py-5 lg:flex ${
        collapsed ? 'w-[84px] px-2.5' : 'w-[232px] px-4'
      }`}
    >
      <div
        className={`flex items-center gap-2.5 ${collapsed ? 'justify-center' : 'px-1'}`}
      >
        <span
          className="grid h-8 w-8 shrink-0 place-items-center rounded-[10px]"
          style={{ background: 'var(--accent)' }}
          aria-hidden="true"
        >
          <svg width="17" height="17" viewBox="0 0 16 16" fill="var(--on-accent)">
            <path d="M8 14.2S1.9 10.4 1.9 6.3A3.6 3.6 0 0 1 8 4.5a3.6 3.6 0 0 1 6.1 1.8c0 4.1-6.1 7.9-6.1 7.9Z" />
          </svg>
        </span>
        {!collapsed && (
          <h1 className="text-[13.5px] leading-[1.15] font-semibold tracking-tight">
            Coronary
            <br />
            Risk Mapping
          </h1>
        )}
      </div>

      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        aria-expanded={!collapsed}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        className={`rail-link mt-4 ${collapsed ? 'is-collapsed' : ''}`}
      >
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
          style={{ transform: collapsed ? 'rotate(180deg)' : undefined }}
        >
          <path d="M2.5 3.5v9m8-6.5L7.5 8l3 2.5" />
          <path d="M13.5 8h-6" />
        </svg>
        {!collapsed && 'Collapse'}
      </button>

      <nav className="mt-2 flex flex-col gap-0.5">
        {sections.map((s) => (
          <button
            key={s.id}
            type="button"
            className={`rail-link ${collapsed ? 'is-collapsed' : ''}`}
            aria-current={active === s.id ? 'true' : undefined}
            disabled={!s.available}
            onClick={() => go(s.id)}
            title={
              s.available
                ? collapsed
                  ? s.label
                  : undefined
                : 'Nothing to show until a prediction runs'
            }
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
              className="shrink-0"
            >
              {ICONS[s.icon]}
            </svg>
            {!collapsed && s.label}
          </button>
        ))}
      </nav>

      <div className="mt-auto">
        <div
          className={`mb-3 flex items-center gap-2.5 rounded-[var(--radius)] py-2 ${
            collapsed ? 'justify-center px-0' : 'px-2'
          }`}
          style={{ background: 'color-mix(in srgb, var(--ink) 5%, transparent)' }}
          title={collapsed ? `${patientName} — ${patientDetail}` : undefined}
        >
          <span
            className="tnum grid h-8 w-8 shrink-0 place-items-center rounded-[var(--radius-pill)] text-[11.5px] font-semibold"
            style={{ background: 'var(--surface)', color: 'var(--ink-secondary)' }}
            aria-hidden="true"
          >
            {patientInitial}
          </span>
          {!collapsed && (
            <span className="min-w-0">
              <span className="block truncate text-[12.5px] font-medium">{patientName}</span>
              <span
                className="block truncate text-[11px]"
                style={{ color: 'var(--ink-secondary)' }}
              >
                {patientDetail}
              </span>
            </span>
          )}
        </div>

        <div
          className={`flex items-center ${
            collapsed ? 'justify-center' : 'justify-between px-1'
          }`}
        >
          {!collapsed && (
            <span className="text-[11.5px]" style={{ color: 'var(--ink-secondary)' }}>
              Appearance
            </span>
          )}
          <ThemeSwitch />
        </div>
      </div>
    </aside>
  );
}
