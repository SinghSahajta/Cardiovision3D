import type { ReactNode } from 'react';

interface Stat {
  tint: 'tint-a' | 'tint-b' | 'tint-c';
  label: string;
  value: string;
  /** Small trailing unit or qualifier, set at body size beside the value. */
  suffix?: string;
  icon: ReactNode;
  title?: string;
  /** Set when the value is a phrase rather than a figure, so it is not set at
      display size and is allowed to truncate. */
  compact?: boolean;
}

interface Props {
  stats: Stat[];
}

/**
 * The tinted context strip.
 *
 * Every card here is deliberately NOT a risk number. Decorative colour and
 * clinical colour cannot share a surface: if a lavender card can mean "cohort
 * position" then a red one is ambiguous between "decorative" and "high risk",
 * and the status palette stops being reliable. So tints carry context - which
 * patient, how complete the input is, what the model is - and every risk figure
 * stays on a neutral surface with a StatusPill doing the colour.
 *
 * Label text is --ink-secondary, not --ink-muted. Measured against the worst
 * pixel of the backdrop photo showing through, secondary is 4.8:1 on every tint
 * and muted only 3.7:1.
 */
export function StatStrip({ stats }: Props) {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {stats.map((s) => (
        <div
          key={s.label}
          className={`card-tint ${s.tint} flex items-center gap-3 px-4 py-3`}
          title={s.title}
        >
          <span className="chip h-9 w-9" aria-hidden="true">
            {s.icon}
          </span>
          <span className="min-w-0">
            <span
              className="block truncate text-[11.5px] font-medium"
              style={{ color: 'var(--ink-secondary)' }}
            >
              {s.label}
            </span>
            <span className="flex items-baseline gap-1">
              <span
                className={`metric leading-tight ${
                  s.compact ? 'truncate text-[14px]' : 'text-[19px]'
                }`}
                style={{ color: 'var(--ink)' }}
              >
                {s.value}
              </span>
              {s.suffix && (
                <span className="text-[11.5px]" style={{ color: 'var(--ink-secondary)' }}>
                  {s.suffix}
                </span>
              )}
            </span>
          </span>
        </div>
      ))}
    </div>
  );
}
