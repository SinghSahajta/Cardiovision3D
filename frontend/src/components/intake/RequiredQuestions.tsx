import {
  BLOCKING_FEATURE,
  REQUIRED_QUESTIONS,
  type RequiredQuestion,
} from '../../lib/completeness';

interface Props {
  features: Record<string, string | number | null>;
  onChange: (feature: string, value: string | number | null) => void;
}

const answered = (v: unknown) =>
  v !== null && v !== undefined && !(typeof v === 'string' && !v.trim());

/**
 * Yes/No as a pair of buttons rather than a dropdown, so the state is visible.
 *
 * The values sent are whatever that column uses - 1/0 for six of these, 'Y'/'N'
 * for the other four. See RequiredQuestion.yes/no for why that matters.
 */
function YesNo({
  value,
  yes,
  no,
  onChange,
}: {
  value: string | number | null;
  yes: string | number;
  no: string | number;
  onChange: (v: string | number) => void;
}) {
  return (
    <div className="flex gap-1">
      {[
        { v: yes, label: 'Yes' },
        { v: no, label: 'No' },
      ].map(({ v, label }) => {
        const on = value !== null && value !== undefined && String(value) === String(v);
        return (
          <button
            key={label}
            onClick={() => onChange(v)}
            aria-pressed={on}
            data-on={on}
            className="pill px-3 py-1 text-xs font-medium"
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

/**
 * The ten things no report can tell you.
 *
 * These are not "the rest of the form". They are the exact set that the models
 * split on AND that no lab, echo or ECG report contains - measured, not guessed.
 * Everything else a document cannot supply is inert: the models never split on
 * it, so asking would be busywork.
 *
 * Ordered heaviest first, so someone who answers only the top two has given the
 * most useful 18% of the model's attention rather than a random 18%.
 */
export function RequiredQuestions({ features, onChange }: Props) {
  const remaining = REQUIRED_QUESTIONS.filter((q) => !answered(features[q.feature]));
  const blocked = !answered(features[BLOCKING_FEATURE]);

  return (
    <div>
      <div className="mb-2.5 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">Questions a report cannot answer</h3>
          <p className="mt-0.5 text-[11.5px]" style={{ color: 'var(--ink-secondary)' }}>
            Symptoms and history. These ten carry 25.6% of the model&rsquo;s weight and
            appear in no lab, echo or ECG report.
          </p>
        </div>
        <span className="tnum text-[11.5px]" style={{ color: 'var(--ink-muted)' }}>
          {REQUIRED_QUESTIONS.length - remaining.length} of {REQUIRED_QUESTIONS.length}
        </span>
      </div>

      {blocked && (
        <p
          className="mb-2.5 rounded-[var(--radius)] px-3 py-2 text-[11.5px] leading-snug"
          style={{ background: 'var(--accent-soft)', color: 'var(--ink-secondary)' }}
        >
          The first question alone is <strong>15.2% of the model&rsquo;s weight</strong>,
          more than twice any other input. No prediction is shown until it is
          answered &mdash; a default would be the cohort&rsquo;s most common answer
          standing in for this patient&rsquo;s.
        </p>
      )}

      <ul className="space-y-0">
        {REQUIRED_QUESTIONS.map((q: RequiredQuestion, i) => {
          const value = features[q.feature] ?? null;
          const isBlocking = q.feature === BLOCKING_FEATURE;
          const unanswered = !answered(value);
          return (
            <li
              key={q.feature}
              className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5 py-2"
              style={{ borderTop: i === 0 ? 'none' : '1px solid var(--hairline)' }}
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-1.5">
                  <span
                    className="text-[13px] font-medium"
                    style={{ color: 'var(--ink)' }}
                  >
                    {q.question}
                  </span>
                  {isBlocking && unanswered && (
                    <span
                      className="rounded-[var(--radius-pill)] px-1.5 py-px text-[9.5px] font-semibold tracking-wide uppercase"
                      style={{ background: 'var(--accent)', color: 'var(--on-accent)' }}
                    >
                      Required
                    </span>
                  )}
                </div>
                {q.help && (
                  <p className="mt-0.5 text-[11px]" style={{ color: 'var(--ink-muted)' }}>
                    {q.help}
                  </p>
                )}
              </div>

              <div className="flex items-center gap-2">
                <span
                  className="tnum text-[10.5px] tabular-nums"
                  style={{ color: 'var(--ink-muted)' }}
                  title="Share of the model's total weight"
                >
                  {(q.weight * 100).toFixed(1)}%
                </span>
                {q.kind === 'yes-no' ? (
                  <YesNo
                    value={value}
                    yes={q.yes}
                    no={q.no}
                    onChange={(v) => onChange(q.feature, v)}
                  />
                ) : (
                  <select
                    value={value === null ? '' : String(value)}
                    onChange={(e) =>
                      onChange(q.feature, e.target.value === '' ? null : Number(e.target.value))
                    }
                    className="rounded-[var(--radius)] px-2.5 py-1 text-xs"
                    style={{
                      background: 'var(--surface-sunken)',
                      border: '1px solid var(--hairline-strong)',
                      color: 'var(--ink)',
                    }}
                  >
                    <option value="">&mdash;</option>
                    {q.options?.map((o) => (
                      <option key={String(o.value)} value={String(o.value)}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
