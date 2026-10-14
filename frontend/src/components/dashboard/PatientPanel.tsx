import { useMemo, useState } from 'react';

import { isInert } from '../../lib/completeness';

import type { FeatureSchema, PresetPatient } from '../../api/types';

interface Props {
  schema: FeatureSchema | null;
  presets: PresetPatient[];
  activePresetId: string | null;
  features: Record<string, string | number | null>;
  onPreset: (id: string) => void;
  onFeatureChange: (feature: string, value: string | number) => void;
  onReset: () => void;
  busy?: boolean;
}

/**
 * Patient selection and the live feature editor.
 *
 * The editable field list is driven by GET /schema rather than hardcoded, so the
 * form cannot drift from the model's actual inputs - add a feature in training
 * and it appears here. These few are surfaced first because they carry most of
 * the attribution weight; everything else keeps its cohort value.
 */
const PRIORITY = [
  'Age',
  'Typical Chest Pain',
  'EF-TTE',
  'Region RWMA',
  'BP',
  'Function Class',
  'LDL',
  'HDL',
  'FBS',
  'TG',
  'DM',
  'HTN',
  'Current Smoker',
  'Tinversion',
  'St Depression',
  'Q Wave',
  'BMI',
  'PR',
];

export function PatientPanel({
  schema,
  presets,
  activePresetId,
  features,
  onPreset,
  onFeatureChange,
  onReset,
  busy,
}: Props) {
  const [expanded, setExpanded] = useState(false);

  const editable = useMemo(() => {
    if (!schema) return [];
    const inSchema = (f: string) => schema.feature_order.includes(f);
    const priority = PRIORITY.filter(inSchema);
    if (!expanded) return priority;
    // Expanded shows everything the model can take - minus the inert fields,
    // which get their own section below so they are not presented as equals.
    return [
      ...priority,
      ...schema.feature_order.filter((f) => !priority.includes(f) && !isInert(f)),
    ];
  }, [schema, expanded]);

  const inertFields = useMemo(
    () => (schema ? schema.feature_order.filter(isInert) : []),
    [schema],
  );

  const isNumeric = (f: string) => schema?.numeric_features.includes(f) ?? true;

  return (
    <section className="card p-4">
      <header className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">Patient</h2>
        {busy && (
          <span className="text-[10px]" style={{ color: 'var(--ink-muted)' }}>
            predicting&hellip;
          </span>
        )}
      </header>

      <label className="block text-[11px]" style={{ color: 'var(--ink-muted)' }}>
        Demo case
      </label>
      <select
        value={activePresetId ?? ''}
        onChange={(e) => e.target.value && onPreset(e.target.value)}
        className="mt-1 w-full rounded-lg border px-2 py-1.5 text-[12px]"
        style={{
          background: 'var(--surface-sunken)',
          borderColor: 'var(--hairline)',
          color: 'var(--ink)',
        }}
      >
        {/* Without this the browser shows the first preset whenever the value
            is empty, so an edited, uploaded or cohort-browsed patient would look
            like one of the four demo cases. */}
        <option value="">&mdash; not a demo case &mdash;</option>
        {presets.map((p) => (
          <option key={p.patient_id} value={p.patient_id}>
            {p.label} ({p.vessels_involved} vessel{p.vessels_involved === 1 ? '' : 's'})
          </option>
        ))}
      </select>

      {presets.length > 0 && (
        <p className="mt-1.5 text-[10px] leading-snug" style={{ color: 'var(--ink-muted)' }}>
          Four real cohort rows, one per vessel count. Like everything in the
          research cohort their predictions are in-sample and flattering; the
          held-out figures beside the result are the honest ones.
        </p>
      )}

      <div
        className="mt-3 flex items-center justify-between border-t pt-3"
        style={{ borderColor: 'var(--grid)' }}
      >
        <span className="text-[11px] font-medium">Adjust measurements</span>
        <button
          onClick={onReset}
          className="text-[10px] underline"
          style={{ color: 'var(--ink-muted)' }}
        >
          reset to case
        </button>
      </div>

      <div className="mt-2 grid max-h-[22rem] gap-1.5 overflow-y-auto pr-1">
        {editable.map((f) => {
          const value = features[f];
          const levels = schema?.categorical_levels[f];
          return (
            <label key={f} className="grid grid-cols-[1fr_6rem] items-center gap-2">
              <span
                className="truncate text-[11px]"
                style={{ color: 'var(--ink-secondary)' }}
                title={f}
              >
                {schema?.display_names[f] ?? f}
              </span>
              {levels && levels.length > 0 ? (
                <select
                  value={String(value ?? '')}
                  onChange={(e) => onFeatureChange(f, e.target.value)}
                  className="rounded border px-1.5 py-1 text-[11px]"
                  style={{
                    background: 'var(--surface-sunken)',
                    borderColor: 'var(--hairline)',
                    color: 'var(--ink)',
                  }}
                >
                  {levels.map((lv) => (
                    <option key={String(lv)} value={String(lv)}>
                      {String(lv)}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  step="any"
                  value={value === null || value === undefined ? '' : String(value)}
                  onChange={(e) =>
                    onFeatureChange(
                      f,
                      isNumeric(f) && e.target.value !== ''
                        ? Number(e.target.value)
                        : e.target.value,
                    )
                  }
                  className="tnum rounded border px-1.5 py-1 text-right text-[11px]"
                  style={{
                    background: 'var(--surface-sunken)',
                    borderColor: 'var(--hairline)',
                    color: 'var(--ink)',
                  }}
                />
              )}
            </label>
          );
        })}
      </div>

      {schema && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="mt-2 text-[11px] font-medium"
          style={{ color: 'var(--accent)' }}
        >
          {expanded
            ? 'Show key measurements only'
            : `Show all ${schema.feature_order.length - inertFields.length} inputs that affect the result`}
        </button>
      )}

      {/* The inert fields, separated and labelled for what they are.
          LightGBM never selected any of these as a split in any of the four
          models, so their contribution is exactly zero and typing into them
          cannot move a prediction. They stay editable because they are part of
          the published dataset and someone may want to confirm that for
          themselves - but presenting them beside Typical Chest Pain as though
          both matter would be the misleading thing. */}
      {schema && inertFields.length > 0 && (
        <details className="mt-3 border-t pt-3" style={{ borderColor: 'var(--hairline)' }}>
          <summary
            className="cursor-pointer text-[11px] font-medium"
            style={{ color: 'var(--ink-secondary)' }}
          >
            {inertFields.length} inputs the models never use
          </summary>
          <p className="mt-1.5 text-[10.5px] leading-snug" style={{ color: 'var(--ink-muted)' }}>
            None of the four models splits on these, so their SHAP contribution is
            exactly zero and changing them cannot change any prediction. Included
            because they are part of the published dataset.
          </p>
          <div className="scroll-soft mt-2 grid max-h-48 gap-1.5 overflow-y-auto pr-1">
            {inertFields.map((f) => (
              <div
                key={f}
                className="grid grid-cols-[1fr_6rem] items-center gap-2 text-[11px]"
                style={{ color: 'var(--ink-muted)' }}
              >
                <span className="truncate" title={f}>
                  {schema.display_names[f] ?? f}
                </span>
                <span className="tnum text-right">{String(features[f] ?? '—')}</span>
              </div>
            ))}
          </div>
        </details>
      )}
    </section>
  );
}
