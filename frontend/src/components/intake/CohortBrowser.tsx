import { useMemo, useState } from 'react';

import type { CohortIndex, CohortSummary } from '../../api/types';
import { pct } from '../../lib/risk';

type Filter = 'all' | 'cad' | 'normal' | 'wrong';

interface Props {
  index: CohortIndex;
  /** Currently loaded cohort patient, or null when the form is custom. */
  activeId: string | null;
  onSelect: (patientId: string) => void;
  busy?: boolean;
}

const FILTERS: { key: Filter; label: string; describe: (n: number) => string }[] = [
  { key: 'all', label: 'All', describe: (n) => `${n} patients` },
  { key: 'cad', label: 'CAD positive', describe: (n) => `${n} with disease on angiography` },
  { key: 'normal', label: 'CAD negative', describe: (n) => `${n} with normal arteries` },
  { key: 'wrong', label: 'Model got it wrong', describe: (n) => `${n} held-out CAD errors` },
];

function matches(p: CohortSummary, filter: Filter): boolean {
  switch (filter) {
    case 'cad':
      return p.ground_truth.CAD === 1;
    case 'normal':
      return p.ground_truth.CAD === 0;
    case 'wrong':
      return !p.held_out_correct_cad;
    default:
      return true;
  }
}

/**
 * Step through the 303 patients the model was trained on.
 *
 * The figure shown beside each one is the **held-out** probability - from the
 * cross-validation fold where that patient was excluded - not the live
 * prediction. The served model was fitted on every one of these rows, so its
 * live output for them is memory as much as inference: it agrees with
 * angiography on 99.0% of them against a true held-out 90.3%. Listing the live
 * number here would make the model look near-perfect by construction.
 *
 * The "model got it wrong" filter is the interesting one, and it exists because
 * a demo that can only show its successes is not worth much.
 */
export function CohortBrowser({ index, activeId, onSelect, busy }: Props) {
  const [filter, setFilter] = useState<Filter>('all');
  const [jump, setJump] = useState('');

  const filtered = useMemo(
    () => index.patients.filter((p) => matches(p, filter)),
    [index.patients, filter],
  );

  const position = filtered.findIndex((p) => p.patient_id === activeId);
  const active = position >= 0 ? filtered[position] : null;

  const step = (delta: number) => {
    if (!filtered.length) return;
    const next = position < 0 ? 0 : (position + delta + filtered.length) % filtered.length;
    onSelect(filtered[next].patient_id);
  };

  const counts: Record<Filter, number> = {
    all: index.patients.length,
    cad: index.patients.filter((p) => p.ground_truth.CAD === 1).length,
    normal: index.patients.filter((p) => p.ground_truth.CAD === 0).length,
    wrong: index.patients.filter((p) => !p.held_out_correct_cad).length,
  };

  return (
    <div>
      <div className="mb-2.5 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">Browse the research cohort</h3>
          <p className="mt-0.5 text-[11.5px]" style={{ color: 'var(--ink-secondary)' }}>
            All {index.n_patients} patients the model was trained on. Figures below are{' '}
            <strong>held-out</strong> &mdash; scored by a model that never saw them.
          </p>
        </div>
      </div>

      <div className="mb-2.5 flex flex-wrap gap-1">
        {FILTERS.map((f) => {
          const on = filter === f.key;
          return (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              aria-pressed={on}
              title={f.describe(counts[f.key])}
              data-on={on}
              className="pill px-2.5 py-1 text-[11.5px] font-medium"
            >
              {f.label}
              <span className="tnum ml-1.5 opacity-70">{counts[f.key]}</span>
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1">
          <button
            onClick={() => step(-1)}
            disabled={busy || !filtered.length}
            aria-label="Previous patient"
            className="rounded-[var(--radius-pill)] px-2.5 py-1 text-sm disabled:opacity-40"
            style={{ border: '1px solid var(--hairline-strong)', color: 'var(--ink-secondary)' }}
          >
            &larr;
          </button>
          <button
            onClick={() => step(1)}
            disabled={busy || !filtered.length}
            aria-label="Next patient"
            className="rounded-[var(--radius-pill)] px-2.5 py-1 text-sm disabled:opacity-40"
            style={{ border: '1px solid var(--hairline-strong)', color: 'var(--ink-secondary)' }}
          >
            &rarr;
          </button>
        </div>

        <span className="tnum text-[12px]" style={{ color: 'var(--ink-secondary)' }}>
          {active
            ? `${position + 1} of ${filtered.length}`
            : `${filtered.length} match${filtered.length === 1 ? '' : 'es'}`}
        </span>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            const n = jump.trim();
            if (n) onSelect(n);
            setJump('');
          }}
          className="ml-auto flex items-center gap-1"
        >
          <input
            value={jump}
            onChange={(e) => setJump(e.target.value)}
            placeholder={`1–${index.n_patients}`}
            inputMode="numeric"
            aria-label="Jump to patient number"
            className="tnum w-20 rounded-[var(--radius)] px-2.5 py-1 text-[12px]"
            style={{
              background: 'var(--surface)',
              border: '1px solid var(--hairline-strong)',
              color: 'var(--ink)',
            }}
          />
          <button
            type="submit"
            className="rounded-[var(--radius-pill)] px-2.5 py-1 text-[11.5px] font-medium"
            style={{ border: '1px solid var(--hairline-strong)', color: 'var(--ink-secondary)' }}
          >
            Go
          </button>
        </form>
      </div>

      {active && (
        <div
          className="mt-2.5 rounded-[var(--radius)] px-3 py-2.5"
          style={{ background: 'var(--surface-sunken)' }}
        >
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <span className="text-[13px] font-semibold">
              Patient {active.index}
              <span className="ml-1.5 font-normal" style={{ color: 'var(--ink-muted)' }}>
                {active.age ? `${active.age}y` : ''} {active.sex === 'Fmale' ? 'F' : 'M'}
              </span>
            </span>
            <span className="text-[11.5px]" style={{ color: 'var(--ink-secondary)' }}>
              Angiography:{' '}
              <strong style={{ color: 'var(--ink)' }}>
                {active.ground_truth.CAD ? 'CAD positive' : 'normal arteries'}
              </strong>
              {active.vessels_involved > 0 &&
                `, ${active.vessels_involved} vessel${active.vessels_involved === 1 ? '' : 's'}`}
            </span>
          </div>

          <div
            className="tnum mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-[11.5px]"
            style={{ color: 'var(--ink-secondary)' }}
          >
            {(['CAD', 'LAD', 'LCX', 'RCA'] as const).map((t) => (
              <span key={t}>
                {t} held-out{' '}
                <strong style={{ color: 'var(--ink)' }}>{pct(active.held_out[t])}</strong>
                <span style={{ color: 'var(--ink-muted)' }}>
                  {' '}
                  / actual {active.ground_truth[t] ? 'yes' : 'no'}
                </span>
              </span>
            ))}
          </div>

          {!active.held_out_correct_cad && (
            <p className="mt-1.5 text-[11px]" style={{ color: 'var(--warn-ink)' }}>
              The held-out model called this patient wrong at its{' '}
              {pct(index.cad_threshold)} threshold.
            </p>
          )}
        </div>
      )}

      {!active && (
        <div
          className="mt-2.5 rounded-[var(--radius)] px-3.5 py-5 text-center"
          style={{ background: 'var(--surface-sunken)' }}
        >
          <p className="text-[12.5px] font-medium">No cohort patient loaded</p>
          <p
            className="mx-auto mt-1 max-w-sm text-[11.5px] leading-snug"
            style={{ color: 'var(--ink-secondary)' }}
          >
            Step with the arrows or type a number to load one. Their angiographic
            result and held-out probabilities appear here, and the form fills with
            their measurements.
          </p>
        </div>
      )}
    </div>
  );
}
