import type { CadPrediction } from '../../api/types';
import { pct, tierOf } from '../../lib/risk';
import { StatusPill } from '../common/StatusPill';

/**
 * The headline CAD result.
 *
 * Form note: a single dominant number is a hero figure, not a chart - so this is
 * a stat tile with a magnitude bar, not a gauge or donut. The threshold marker on
 * the bar is the part that matters clinically: a 62% that sits below a 66%
 * threshold is a negative call, and showing the number alone would hide that.
 */
interface Props {
  cad: CadPrediction;
  /** Set only for cohort patients: the probability from their held-out fold. */
  heldOut?: number | null;
  /** Angiographic truth, where it is known. */
  groundTruth?: number | null;
  /** Share of the model's weight the inputs actually cover, 0-1. */
  coverage?: number;
}

export function CadSummary({ cad, heldOut, groundTruth, coverage }: Props) {
  const tier = tierOf(cad.risk_score);
  const derivedTier = tierOf(cad.vessel_derived_risk_score);

  return (
    <section className="card p-5 sm:p-6">
      <header className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h2 className="eyebrow">Overall CAD</h2>
          <p className="mt-0.5 text-[11.5px]" style={{ color: 'var(--ink-muted)' }}>
            Directly trained classifier
          </p>
        </div>
        <StatusPill tier={tier}>{cad.has_cad ? 'CAD positive' : 'CAD negative'}</StatusPill>
      </header>

      {/* The hero figure. One dominant number is the whole job of this panel,
          so it is set at display scale rather than merely large. */}
      <div className="flex items-end gap-2.5">
        <span className="display">{pct(cad.risk_score)}</span>
        <span className="pb-2 text-[11.5px]" style={{ color: 'var(--ink-muted)' }}>
          calibrated
          <br />
          probability
        </span>
      </div>

      {/* For a cohort patient the live number above is in-sample: the served
          model was fitted on this very row. The held-out figure beside it is
          what the model scores when it has not seen the patient, and it is the
          one that means anything. Both are shown rather than quietly picking. */}
      {heldOut != null && (
        <div
          className="tnum mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px]"
          style={{ color: 'var(--ink-secondary)' }}
        >
          <span
            className="rounded-[var(--radius-pill)] px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase"
            style={{ background: 'var(--surface-sunken)', color: 'var(--ink-muted)' }}
          >
            in-sample
          </span>
          <span>
            Held-out <strong style={{ color: 'var(--ink)' }}>{pct(heldOut)}</strong>
          </span>
          {groundTruth != null && (
            <span>
              Angiography{' '}
              <strong style={{ color: 'var(--ink)' }}>
                {groundTruth ? 'CAD positive' : 'normal'}
              </strong>
            </span>
          )}
        </div>
      )}

      {/* Magnitude bar with the decision threshold marked in place. */}
      <div className="mt-3">
        <div
          className="relative h-2.5 w-full overflow-hidden rounded-full"
          style={{ background: 'var(--grid)' }}
        >
          <div
            className="absolute inset-y-0 left-0 rounded-full"
            style={{
              width: `${Math.min(cad.risk_score * 100, 100)}%`,
              background: `var(--risk-${tier.toLowerCase()})`,
            }}
          />
          <div
            className="absolute inset-y-[-3px] w-0.5"
            style={{ left: `${cad.threshold * 100}%`, background: 'var(--ink)' }}
            aria-hidden
          />
        </div>
        <div
          className="tnum mt-1 flex justify-between text-[10px]"
          style={{ color: 'var(--ink-muted)' }}
        >
          <span>0%</span>
          <span>decision threshold {pct(cad.threshold, 1)}</span>
          <span>100%</span>
        </div>
      </div>

      {/* Cross-check against the vessel heads. In this cohort Cath is almost
          exactly OR(LAD,LCX,RCA), so a disagreement is worth surfacing rather
          than hiding behind one confident-looking number. */}
      <div
        className="mt-4 flex items-center justify-between gap-3 rounded-[var(--radius)] px-3.5 py-3"
        style={{ background: 'var(--surface-sunken)' }}
      >
        <div className="min-w-0">
          <p className="text-[12px] font-medium">Cross-check from the three vessels</p>
          <p className="text-[10.5px] leading-snug" style={{ color: 'var(--ink-muted)' }}>
            Noisy-OR of LAD/LCX/RCA, calibrated
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span className="tnum text-lg font-semibold">
            {pct(cad.vessel_derived_risk_score)}
          </span>
          <StatusPill tier={derivedTier} />
        </div>
      </div>

      <p
        className="mt-2 text-[11.5px] leading-snug"
        style={{ color: cad.agrees_with_vessels ? 'var(--ink-muted)' : 'var(--warn-ink)' }}
      >
        {cad.agrees_with_vessels
          ? 'The two routes agree on this patient.'
          : 'These two routes disagree on this patient — read the per-vessel detail before relying on the headline number.'}
      </p>

      {coverage != null && coverage < 0.999 && (
        <p className="mt-1.5 text-[11px]" style={{ color: 'var(--ink-muted)' }}>
          Inputs cover <strong className="tnum">{pct(coverage)}</strong> of the
          model&rsquo;s weight; the rest uses cohort defaults.
        </p>
      )}
    </section>
  );
}
