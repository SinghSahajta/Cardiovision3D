import type { PredictionResponse, VesselKey } from '../../api/types';
import { VESSELS } from '../../api/types';
import { pct, tierOf, VESSEL_CAVEAT, VESSEL_FULL_NAME } from '../../lib/risk';
import { StatusPill } from '../common/StatusPill';

interface Props {
  prediction: PredictionResponse;
  selected: VesselKey | 'CAD';
  onSelect: (key: VesselKey | 'CAD') => void;
  groundTruth?: Record<string, number> | null;
  /** Held-out probabilities, set only when a cohort patient is loaded. */
  heldOut?: Record<string, number> | null;
}

/**
 * The three per-vessel results, which double as the selector for the 3D view and
 * the SHAP panel. Each card carries the probability, the model's own threshold,
 * and - for LCX and RCA - the held-out precision caveat, so the weak signals are
 * labelled weak at the point of use.
 */
export function VesselCards({
  prediction,
  selected,
  onSelect,
  groundTruth,
  heldOut,
}: Props) {
  return (
    <div className="grid gap-2 sm:grid-cols-3">
      {VESSELS.map((key: VesselKey) => {
        const v = prediction.vessels[key];
        if (!v) return null;
        const tier = tierOf(v.risk_score);
        const active = selected === key;
        const truth = groundTruth?.[key];
        const correct = truth === undefined ? null : Number(v.is_stenotic) === truth;

        return (
          <button
            key={key}
            onClick={() => onSelect(key)}
            aria-pressed={active}
            className="card flex flex-col gap-1.5 p-4 text-left transition-[border-color,box-shadow] duration-150"
            style={{
              borderColor: active ? 'var(--accent)' : 'var(--hairline)',
              boxShadow: active ? 'var(--shadow-raised)' : 'var(--shadow-card)',
            }}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="text-[13px] font-semibold">{key}</div>
                <div
                  className="truncate text-[10.5px] leading-tight"
                  style={{ color: 'var(--ink-muted)' }}
                >
                  {VESSEL_FULL_NAME[key]}
                </div>
              </div>
              <StatusPill tier={tier} />
            </div>

            <div className="metric text-[30px] leading-none tracking-[-0.03em]">
              {pct(v.risk_score)}
            </div>

            <div
              className="h-1.5 w-full overflow-hidden rounded-full"
              style={{ background: 'var(--grid)' }}
            >
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.min(v.risk_score * 100, 100)}%`,
                  background: `var(--risk-${tier.toLowerCase()})`,
                }}
              />
            </div>

            <div className="text-[11px] font-medium">
              {v.is_stenotic ? 'Stenosis predicted' : 'No significant stenosis'}
            </div>
            <div className="tnum text-[10.5px]" style={{ color: 'var(--ink-muted)' }}>
              threshold {pct(v.threshold, 1)}
              {heldOut?.[key] != null && (
                <>
                  {' · held-out '}
                  <strong style={{ color: 'var(--ink-secondary)' }}>
                    {pct(heldOut[key])}
                  </strong>
                </>
              )}
            </div>

            {correct !== null && (
              <div
                className="mt-0.5 text-[10px] font-medium"
                style={{ color: correct ? 'var(--risk-low)' : 'var(--warn-ink)' }}
              >
                {correct ? 'matches angiography' : 'disagrees with angiography'}
              </div>
            )}

            {VESSEL_CAVEAT[key] && (
              <div
                className="mt-0.5 text-[10px] leading-snug"
                style={{ color: 'var(--ink-muted)' }}
              >
                {VESSEL_CAVEAT[key]}
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}
