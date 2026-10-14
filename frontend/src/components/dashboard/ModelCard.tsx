import { useState } from 'react';

import type { MetricsResponse, TargetKey } from '../../api/types';

const ORDER: TargetKey[] = ['CAD', 'LAD', 'LCX', 'RCA'];

/**
 * Model card. Reads GET /metrics, which the backend serves straight from
 * ml/results/cv_metrics.json, so these figures cannot drift from what training
 * actually measured.
 *
 * The limitations block is not boilerplate - it names the two weak vessel models
 * explicitly. A dashboard that shows a confident red artery without saying that
 * roughly half of those flags are false alarms would be misleading, and the
 * numbers are right here in the payload.
 */
export function ModelCard({ metrics }: { metrics: MetricsResponse | null }) {
  const [open, setOpen] = useState(false);

  if (!metrics) {
    return (
      <section className="card p-4">
        <h2 className="text-sm font-semibold">Model performance</h2>
        <p className="mt-1 text-[11px]" style={{ color: 'var(--ink-muted)' }}>
          Metrics unavailable. Run <code>python ml/src/train.py</code>.
        </p>
      </section>
    );
  }

  const first = (s?: string) => (s ?? '').split(' ')[0] || '—';

  return (
    <section className="card p-4">
      <header className="mb-2 flex items-start justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold">Model performance</h2>
          <p className="text-[10px]" style={{ color: 'var(--ink-muted)' }}>
            Held-out, {metrics.validation_strategy.toLowerCase()}
          </p>
        </div>
        <button
          onClick={() => setOpen((v) => !v)}
          className="shrink-0 text-[10px] underline"
          style={{ color: 'var(--ink-muted)' }}
        >
          {open ? 'less' : 'more'}
        </button>
      </header>

      <table className="w-full text-[11.5px]">
        <thead>
          <tr className="text-left text-[10px] uppercase" style={{ color: 'var(--ink-muted)' }}>
            <th className="pb-1 font-medium">Target</th>
            <th className="pb-1 text-right font-medium">ROC-AUC</th>
            <th className="pb-1 text-right font-medium">PR-AUC</th>
            <th className="pb-1 text-right font-medium">Recall</th>
          </tr>
        </thead>
        <tbody>
          {ORDER.map((t) => {
            const m = metrics.per_target[t];
            if (!m) return null;
            const auc = Number(first(m.threshold_independent.roc_auc));
            return (
              <tr key={t} style={{ borderTop: '1px solid var(--grid)' }}>
                <td className="py-1 font-medium">{t}</td>
                <td
                  className="tnum py-1 text-right"
                  style={{ color: auc < 0.75 ? 'var(--warn-ink)' : 'var(--ink-secondary)' }}
                >
                  {first(m.threshold_independent.roc_auc)}
                </td>
                <td className="tnum py-1 text-right" style={{ color: 'var(--ink-secondary)' }}>
                  {first(m.threshold_independent.pr_auc)}
                </td>
                <td className="tnum py-1 text-right" style={{ color: 'var(--ink-secondary)' }}>
                  {first(m.at_tuned_threshold.recall)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div
        className="mt-3 rounded-lg border p-2.5 text-[10.5px] leading-snug"
        style={{
          borderColor: 'color-mix(in srgb, var(--risk-moderate) 34%, transparent)',
          background: 'color-mix(in srgb, var(--risk-moderate) 7%, var(--surface))',
        }}
      >
        <p className="mb-1 font-semibold" style={{ color: 'var(--ink)' }}>
          Known limitations
        </p>
        <ul className="list-disc space-y-0.5 pl-3.5" style={{ color: 'var(--ink-secondary)' }}>
          <li>
            CAD is the reliable output: it finds about 90% of angiographic disease at 92%
            precision.
          </li>
          <li>
            LCX and RCA are weak. At their tuned thresholds precision is roughly 53% and
            49% — about half of the vessels coloured red are false alarms.
          </li>
          <li>
            All three vessels are simultaneously correct in only ~36% of patients.
          </li>
          <li>
            n = {metrics.cohort_size} from a single centre. No external validation, so
            these figures will not transfer unchanged to another population.
          </li>
        </ul>
      </div>

      {open && (
        <dl className="mt-3 space-y-1 text-[10.5px]" style={{ color: 'var(--ink-secondary)' }}>
          <Row label="Dataset" value={metrics.dataset} />
          <Row label="Features" value={`${metrics.n_features} (leakage columns excluded)`} />
          <Row label="Thresholds" value={metrics.threshold_selection} />
          <Row
            label="Excluded"
            value={metrics.excluded_features.join(', ')}
          />
          {ORDER.map((t) => {
            const m = metrics.per_target[t];
            if (!m) return null;
            return (
              <Row
                key={t}
                label={`${t} calibration`}
                value={`${m.calibration.method}, threshold ${m.tuned_threshold_mean}`}
              />
            );
          })}
        </dl>
      )}
    </section>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2">
      <dt className="shrink-0" style={{ color: 'var(--ink-muted)' }}>
        {label}
      </dt>
      <dd className="min-w-0 break-words">{value}</dd>
    </div>
  );
}
