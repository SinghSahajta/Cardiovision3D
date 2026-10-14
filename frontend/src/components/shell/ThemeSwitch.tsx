import { useState } from 'react';

/**
 * Light/dark switch.
 *
 * The visual is the Uiverse slider, styled in index.css. Two things about it
 * matter here rather than there:
 *
 * 1. The markup order is fixed. `.input:checked + .slider` is an adjacent
 *    sibling selector, so the <input> must be immediately followed by the
 *    slider or the checked state does nothing. Sun and moon come after.
 *
 * 2. It needs state, which the button this replaces did not have. That button
 *    read and wrote document.documentElement.dataset.theme directly and always
 *    read "Theme" whichever mode you were in - it had nothing to render from.
 *    A slider has a position, so it has to know the current mode on mount.
 *
 * The initial value is read from the DOM, not from localStorage, because the
 * pre-paint script in index.html has already resolved storage into the
 * data-theme attribute before React runs. Reading storage again here would
 * duplicate that logic and could disagree with what is on screen.
 */
export function ThemeSwitch() {
  const [dark, setDark] = useState(
    () => document.documentElement.dataset.theme === 'dark',
  );

  const apply = (next: boolean) => {
    document.documentElement.dataset.theme = next ? 'dark' : 'light';
    try {
      localStorage.setItem('theme', next ? 'dark' : 'light');
    } catch {
      // Private mode or blocked storage: the switch still works for this
      // session, it just will not be remembered.
    }
    setDark(next);
  };

  return (
    <label className="switch" title={dark ? 'Switch to light' : 'Switch to dark'}>
      <input
        className="input"
        type="checkbox"
        checked={dark}
        onChange={(e) => apply(e.target.checked)}
        aria-label="Dark mode"
      />
      <span className="slider" />
      <span className="sun">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="5" />
          <path d="M12 0.5a1.1 1.1 0 0 1 1.1 1.1v2.2a1.1 1.1 0 0 1-2.2 0V1.6A1.1 1.1 0 0 1 12 .5Zm0 18.6a1.1 1.1 0 0 1 1.1 1.1v2.2a1.1 1.1 0 0 1-2.2 0v-2.2a1.1 1.1 0 0 1 1.1-1.1ZM23.5 12a1.1 1.1 0 0 1-1.1 1.1h-2.2a1.1 1.1 0 0 1 0-2.2h2.2a1.1 1.1 0 0 1 1.1 1.1ZM4.9 12a1.1 1.1 0 0 1-1.1 1.1H1.6a1.1 1.1 0 0 1 0-2.2h2.2A1.1 1.1 0 0 1 4.9 12ZM20.13 3.87a1.1 1.1 0 0 1 0 1.56l-1.56 1.56a1.1 1.1 0 1 1-1.56-1.56l1.56-1.56a1.1 1.1 0 0 1 1.56 0ZM6.99 17.01a1.1 1.1 0 0 1 0 1.56l-1.56 1.56a1.1 1.1 0 0 1-1.56-1.56l1.56-1.56a1.1 1.1 0 0 1 1.56 0ZM20.13 20.13a1.1 1.1 0 0 1-1.56 0l-1.56-1.56a1.1 1.1 0 1 1 1.56-1.56l1.56 1.56a1.1 1.1 0 0 1 0 1.56ZM6.99 6.99a1.1 1.1 0 0 1-1.56 0L3.87 5.43a1.1 1.1 0 0 1 1.56-1.56l1.56 1.56a1.1 1.1 0 0 1 0 1.56Z" />
        </svg>
      </span>
      <span className="moon">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M21.3 14.73a9.07 9.07 0 0 1-11.99-11.1.86.86 0 0 0-1.1-1.06A10.1 10.1 0 1 0 22.36 15.8a.86.86 0 0 0-1.06-1.07Z" />
        </svg>
      </span>
    </label>
  );
}
