import { useMemo, useState } from 'react';

import type { FeatureAttribution } from '../../api/types';
import { measure } from '../../lib/risk';

interface Props {
  attributions: FeatureAttribution[];
  baseValue: number;
  /** Total of ALL contributions, including ones outside the top-k shown. */
  shapSum: number;
}

const ROW = 22;
const LEFT = 150;
const TRACK = 300;

/**
 * Waterfall from the cohort baseline to this patient's model margin.
 *
 * This is the chart that answers "how did we get from an average patient to this
 * number", which a ranked bar list cannot show - the bar list gives magnitudes,
 * the waterfall gives the running total. Any contributions beyond the top-k are
 * folded into a single explicit "other features" step rather than dropped, so
 * the last bar genuinely lands on the model's margin.
 */
export function ShapWaterfall({ attributions, baseValue, shapSum }: Props) {
  const [hover, setHover] = useState<number | null>(null);

  const steps = useMemo(() => {
    const shown = [...attributions].sort(
      (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value),
    );
    const shownSum = shown.reduce((acc, a) => acc + a.shap_value, 0);
    const remainder = shapSum - shownSum;

    const rows = shown.map((a) => ({
      label: a.display_name,
      value: a.shap_value,
      detail: `${a.display_name} = ${measure(a.feature_value)}`,
    }));
    if (Math.abs(remainder) > 0.005) {
      rows.push({
        label: 'all other features',
        value: remainder,
        detail: 'Combined effect of every feature outside the top contributors',
      });
    }

    // Built with a plain loop rather than .map: the running total is a
    // cumulative sum, and mutating a captured accumulator from inside a
    // callback during render is exactly what react-hooks/immutability forbids.
    const out: ((typeof rows)[number] & { from: number; to: number })[] = [];
    let running = baseValue;
    for (const r of rows) {
      const from = running;
      running += r.value;
      out.push({ ...r, from, to: running });
    }
    return out;
  }, [attributions, baseValue, shapSum]);

  const final = baseValue + shapSum;
  const lo = Math.min(baseValue, final, ...steps.flatMap((s) => [s.from, s.to]));
  const hi = Math.max(baseValue, final, ...steps.flatMap((s) => [s.from, s.to]));
  const pad = (hi - lo) * 0.08 || 0.5;
  const x = (v: number) => ((v - (lo - pad)) / (hi - lo + pad * 2)) * TRACK + LEFT;

  const height = (steps.length + 2) * ROW + 16;

  return (
    <div className="overflow-x-auto">
      <svg
        width={LEFT + TRACK + 56}
        height={height}
        role="img"
        aria-label="Waterfall of SHAP contributions from the cohort baseline to this patient's model margin"
        style={{ maxWidth: '100%' }}
      >
        {/* Zero rule, only if zero is inside the visible range. */}
        {lo - pad < 0 && hi + pad > 0 && (
          <line
            x1={x(0)}
            x2={x(0)}
            y1={8}
            y2={height - 8}
            stroke="var(--grid)"
            strokeDasharray="2 3"
          />
        )}

        <WaterfallRow
          y={8}
          label="cohort baseline"
          valueText={fmt(baseValue)}
          barX1={x(Math.min(baseValue, 0))}
          barX2={x(Math.max(baseValue, 0))}
          fill="var(--axis)"
          roundRight={baseValue >= 0}
          emphasis
        />

        {steps.map((s, i) => {
          const up = s.value > 0;
          const x1 = x(Math.min(s.from, s.to));
          const x2 = x(Math.max(s.from, s.to));
          return (
            <g
              key={`${s.label}-${i}`}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            >
              {/* Invisible wide hit target: easier to hover than a 9px bar. */}
              <rect
                x={LEFT}
                y={(i + 1) * ROW + 8}
                width={TRACK}
                height={ROW}
                fill="transparent"
              />
              <WaterfallRow
                y={(i + 1) * ROW + 8}
                label={s.label}
                valueText={`${up ? '+' : ''}${s.value.toFixed(2)}`}
                barX1={x1}
                barX2={x2}
                fill={up ? 'var(--shap-up)' : 'var(--shap-down)'}
                roundRight={up}
                highlighted={hover === i}
              />
              {/* Connector to the next step's start. */}
              {i < steps.length - 1 && (
                <line
                  x1={x(s.to)}
                  x2={x(s.to)}
                  y1={(i + 1) * ROW + 8 + ROW / 2 + 5}
                  y2={(i + 2) * ROW + 8 + ROW / 2 - 5}
                  stroke="var(--grid)"
                />
              )}
            </g>
          );
        })}

        <WaterfallRow
          y={(steps.length + 1) * ROW + 8}
          label="model margin"
          valueText={fmt(final)}
          barX1={x(Math.min(final, 0))}
          barX2={x(Math.max(final, 0))}
          fill="var(--axis)"
          roundRight={final >= 0}
          emphasis
        />
      </svg>

      <p className="mt-1 min-h-[1.25rem] text-[11px]" style={{ color: 'var(--ink-muted)' }}>
        {hover !== null ? steps[hover].detail : 'Hover a step for the feature value.'}
      </p>
    </div>
  );
}

function fmt(v: number) {
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;
}

/** Horizontal bar with rounded corners on one end only. */
function barPath(
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
  roundRight: boolean,
): string {
  const rr = Math.max(Math.min(r, w, h / 2), 0);
  if (roundRight) {
    return (
      `M${x},${y} H${x + w - rr} A${rr},${rr} 0 0 1 ${x + w},${y + rr} ` +
      `V${y + h - rr} A${rr},${rr} 0 0 1 ${x + w - rr},${y + h} H${x} Z`
    );
  }
  return (
    `M${x + w},${y} H${x + rr} A${rr},${rr} 0 0 0 ${x},${y + rr} ` +
    `V${y + h - rr} A${rr},${rr} 0 0 0 ${x + rr},${y + h} H${x + w} Z`
  );
}

function WaterfallRow({
  y,
  label,
  valueText,
  barX1,
  barX2,
  fill,
  roundRight,
  emphasis,
  highlighted,
}: {
  y: number;
  label: string;
  valueText: string;
  barX1: number;
  barX2: number;
  fill: string;
  roundRight?: boolean;
  emphasis?: boolean;
  highlighted?: boolean;
}) {
  const w = Math.max(barX2 - barX1, 1.5);
  return (
    <>
      <text
        x={LEFT - 8}
        y={y + ROW / 2 + 4}
        textAnchor="end"
        fontSize={11}
        fill={emphasis ? 'var(--ink)' : 'var(--ink-secondary)'}
        fontWeight={emphasis ? 600 : 400}
      >
        {label.length > 22 ? `${label.slice(0, 21)}…` : label}
      </text>
      {/* Rounded on the data end only, square where it meets the previous step -
          the bar reads as a continuation of the running total rather than a
          free-floating pill. */}
      <path
        d={barPath(barX1, y + ROW / 2 - 4.5, w, 9, 4, roundRight ?? true)}
        fill={fill}
        opacity={highlighted ? 1 : emphasis ? 0.85 : 0.92}
        stroke={highlighted ? 'var(--ink)' : 'none'}
        strokeWidth={highlighted ? 1 : 0}
      />
      <text
        x={LEFT + TRACK + 8}
        y={y + ROW / 2 + 4}
        fontSize={11}
        fill={emphasis ? 'var(--ink)' : 'var(--ink-secondary)'}
        fontWeight={emphasis ? 600 : 400}
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {valueText}
      </text>
    </>
  );
}
