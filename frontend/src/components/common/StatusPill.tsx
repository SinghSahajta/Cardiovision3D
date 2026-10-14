import type { RiskTier } from '../../lib/risk';
import { TIER_COLOR } from '../../lib/risk';

/**
 * Risk tier badge. Shows a dot AND the tier word - status colour never travels
 * alone, per the palette rules for reserved status colours.
 *
 * The dot carries `.status-dot`, which adds the inset ring. That ring is the
 * documented mitigation for the moderate amber's low contrast against a white
 * surface (see the note at the top of index.css) - it was defined but applied
 * at no call site, so the amber swatch had no edge at all until now.
 */
export function StatusPill({
  tier,
  children,
}: {
  tier: RiskTier;
  children?: React.ReactNode;
}) {
  const color = TIER_COLOR[tier];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-[var(--radius-pill)] border px-2.5 py-0.5 text-[11px] font-semibold whitespace-nowrap"
      style={{
        borderColor: color,
        color: 'var(--ink)',
        background: `color-mix(in srgb, ${color} 16%, transparent)`,
      }}
    >
      <span
        className="status-dot inline-block h-2 w-2 shrink-0"
        style={{ background: color }}
        aria-hidden
      />
      {children ?? tier}
    </span>
  );
}
