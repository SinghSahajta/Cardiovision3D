import { TISSUE } from '../../lib/anatomy';
import { TIER_BANDS, TIER_COLOR } from '../../lib/risk';

const TISSUE_KEYS = [
  { group: 'myocardium' as const, label: 'Myocardium' },
  { group: 'atrium' as const, label: 'Atria' },
  { group: 'great_vessel' as const, label: 'Great vessels' },
];

/**
 * Two legends in one strip, kept visually distinct because they do different
 * jobs: the status colours are a model output, the tissue colours are just what
 * a structure is. Each band carries its label and numeric range, so colour is
 * never the only channel communicating the tier.
 */
export function Legend() {
  return (
    <div
      className="rounded-lg border px-3 py-2 text-[11px]"
      style={{
        background: 'color-mix(in srgb, var(--surface) 92%, transparent)',
        borderColor: 'var(--hairline)',
        color: 'var(--ink-secondary)',
      }}
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="font-semibold" style={{ color: 'var(--ink)' }}>
          Predicted stenosis risk
        </span>
        {TIER_BANDS.map(({ tier, label }) => (
          <span key={tier} className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ background: TIER_COLOR[tier] }}
              aria-hidden
            />
            {label}
          </span>
        ))}
        <span className="inline-flex items-center gap-1">
          <span className="font-bold" style={{ color: 'var(--ink)' }}>
            !
          </span>
          past the model&rsquo;s threshold
        </span>
      </div>

      <div
        className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 border-t pt-1.5 text-[10px]"
        style={{ borderColor: 'var(--hairline)', color: 'var(--ink-muted)' }}
      >
        <span>Tissue</span>
        {TISSUE_KEYS.map(({ group, label }) => (
          <span key={group} className="inline-flex items-center gap-1">
            <span
              className="inline-block h-2 w-2 rounded-sm"
              style={{ background: TISSUE[group].color }}
              aria-hidden
            />
            {label}
          </span>
        ))}
        <span>&middot; narrower vessel = higher predicted risk, severity only</span>
      </div>
    </div>
  );
}
