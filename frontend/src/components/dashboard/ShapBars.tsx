import type { FeatureAttribution } from '../../api/types';
import { measure } from '../../lib/risk';

/**
 * Diverging contribution chart for one target's TreeSHAP attributions.
 *
 * Form: the data's job is polarity plus magnitude per named feature, so this is a
 * diverging horizontal bar list with a zero axis down the middle - not a pie, not
 * a radar, and not a single-hue bar chart that would throw away the sign.
 *
 * Colour: the validated diverging pair (red pole = pushes risk up, blue pole =
 * pushes risk down) with a neutral zero rule. Sign is also carried by which side
 * of the axis a bar sits on and by the +/- in the value column, so the chart does
 * not depend on hue being perceivable.
 *
 * The right-hand value column doubles as the table view: every magnitude is
 * readable as a number, which is what keeps this accessible without a separate
 * "show data" toggle.
 */
export function ShapBars({
  attributions,
  baseValue,
}: {
  attributions: FeatureAttribution[];
  baseValue: number;
}) {
  if (!attributions.length) {
    return (
      <p className="text-sm" style={{ color: 'var(--ink-muted)' }}>
        No attributions available.
      </p>
    );
  }

  const maxAbs = Math.max(...attributions.map((a) => Math.abs(a.shap_value)), 1e-6);

  return (
    <div>
      <div
        className="mb-2 flex items-center justify-between text-[11px]"
        style={{ color: 'var(--ink-muted)' }}
      >
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ background: 'var(--shap-down)' }}
            aria-hidden
          />
          lowers risk
        </span>
        <span>log-odds contribution</span>
        <span className="inline-flex items-center gap-1.5">
          raises risk
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ background: 'var(--shap-up)' }}
            aria-hidden
          />
        </span>
      </div>

      <ul className="flex flex-col gap-[3px]">
        {attributions.map((a) => {
          const up = a.shap_value > 0;
          const frac = Math.abs(a.shap_value) / maxAbs;
          const widthPct = Math.max(frac * 50, 0.8);
          return (
            <li key={a.processed_feature} className="grid grid-cols-[minmax(0,9rem)_1fr_4.2rem] items-center gap-2">
              <span
                className="truncate text-[12px]"
                style={{ color: 'var(--ink-secondary)' }}
                title={`${a.display_name} = ${measure(a.feature_value)}`}
              >
                {a.display_name}
              </span>

              {/* Bar track with a centred zero rule. */}
              <span className="relative block h-[14px]">
                <span
                  className="absolute inset-y-0 left-1/2 w-px"
                  style={{ background: 'var(--axis)' }}
                  aria-hidden
                />
                <span
                  className="absolute top-1/2 h-[9px] -translate-y-1/2"
                  style={{
                    background: up ? 'var(--shap-up)' : 'var(--shap-down)',
                    width: `${widthPct}%`,
                    left: up ? '50%' : undefined,
                    right: up ? undefined : '50%',
                    borderRadius: up ? '0 4px 4px 0' : '4px 0 0 4px',
                  }}
                />
              </span>

              <span
                className="tnum text-right text-[12px]"
                style={{ color: 'var(--ink-secondary)' }}
              >
                {a.shap_value > 0 ? '+' : ''}
                {a.shap_value.toFixed(2)}
              </span>
            </li>
          );
        })}
      </ul>

      <p className="mt-3 text-[11px] leading-snug" style={{ color: 'var(--ink-muted)' }}>
        TreeSHAP values in <strong>log-odds margin space</strong>, from the exact
        LightGBM booster that produced this prediction. They sum to the model&rsquo;s
        margin (baseline {baseValue >= 0 ? '+' : ''}
        {baseValue.toFixed(2)}), <em>not</em> to the percentage above &mdash; the two are
        related by a monotonic calibration map, so the ranking is the same but the
        units differ.
      </p>
    </div>
  );
}
