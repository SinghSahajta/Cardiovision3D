import { useMemo, useState } from 'react';

import { FIELD_BY_FEATURE, SOURCE_LABEL } from '../../lib/extract/dictionary';
import type { Candidate } from '../../lib/extract/parse';
import { measure } from '../../lib/risk';
import type { ExtractionBatch } from './UploadPanel';

interface Props {
  batch: ExtractionBatch;
  onApply: (values: Record<string, string | number>) => void;
  onDismiss: () => void;
}

const STATUS_NOTE: Record<Candidate['status'], string> = {
  ok: '',
  out_of_range: 'Outside the training range',
  derived: 'Calculated, not read',
  inferred: 'Inferred from medication',
};

/**
 * What the document said, before any of it becomes a model input.
 *
 * Nothing here is applied automatically. Lab layouts vary, OCR misreads digits,
 * and a wrong potassium that looks plausible is worse than a missing one - so
 * every row shows the verbatim text it came from and has to be accepted.
 *
 * Rows the parser is unsure about start switched OFF: anything outside the range
 * the model was trained on (usually a unit mismatch), and anything inferred from
 * a drug name rather than measured. Clean readings start switched on, because
 * making someone tick thirty correct values one at a time is its own way of
 * getting them to stop looking.
 */
export function ExtractionReview({ batch, onApply, onDismiss }: Props) {
  const [accepted, setAccepted] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(
      batch.candidates.map((c) => [c.feature, c.status === 'ok' || c.status === 'derived']),
    ),
  );
  const [edited, setEdited] = useState<Record<string, string>>({});

  const rows = useMemo(
    () =>
      [...batch.candidates].sort((a, b) => {
        // Put the things needing a decision at the top.
        const rank = (c: Candidate) =>
          c.status === 'out_of_range' ? 0 : c.status === 'inferred' ? 1 : 2;
        return rank(a) - rank(b) || a.label.localeCompare(b.label);
      }),
    [batch.candidates],
  );

  const acceptedCount = rows.filter((r) => accepted[r.feature]).length;

  const apply = () => {
    const out: Record<string, string | number> = {};
    for (const row of rows) {
      if (!accepted[row.feature]) continue;
      const override = edited[row.feature];
      if (override !== undefined && override.trim() !== '') {
        const spec = FIELD_BY_FEATURE.get(row.feature);
        out[row.feature] =
          spec?.kind === 'number' ? Number(override) : override.trim();
      } else {
        out[row.feature] = row.value;
      }
    }
    onApply(out);
  };

  if (!rows.length) {
    return (
      <div className="rounded-[var(--radius)] p-4" style={{ background: 'var(--surface-sunken)' }}>
        <p className="text-sm font-medium">Nothing recognised in {batch.fileName}</p>
        <p className="mt-1 text-xs" style={{ color: 'var(--ink-secondary)' }}>
          {batch.method === 'ocr'
            ? 'This was read by OCR, which struggles with low-resolution scans, '
              + 'photographs taken at an angle, and handwriting.'
            : 'The PDF had a text layer but none of the terms matched. The field '
              + 'names may be worded in a way the dictionary does not cover yet.'}
        </p>
        <button onClick={onDismiss} className="mt-3 text-xs font-medium"
          style={{ color: 'var(--accent)' }}>
          Dismiss
        </button>
      </div>
    );
  }

  return (
    <div className="rounded-[var(--radius)]" style={{ background: 'var(--surface-sunken)' }}>
      <div className="flex flex-wrap items-baseline justify-between gap-2 px-4 pt-3.5 pb-2">
        <div>
          <h3 className="text-sm font-semibold">Check before using</h3>
          <p className="mt-0.5 text-[11.5px]" style={{ color: 'var(--ink-secondary)' }}>
            {batch.fileName} &middot; {batch.pages} page{batch.pages === 1 ? '' : 's'}{' '}
            &middot;{' '}
            {batch.method === 'pdf-text'
              ? 'read exactly from the PDF text layer'
              : `recognised from an image${
                  batch.confidence ? ` (${Math.round(batch.confidence * 100)}% confidence)` : ''
                }`}
          </p>
        </div>
        <span className="tnum text-[11.5px]" style={{ color: 'var(--ink-muted)' }}>
          {acceptedCount} of {rows.length} selected
        </span>
        <p className="w-full text-[11px]" style={{ color: 'var(--ink-muted)' }}>
          These values start a new patient rather than being merged into the one
          currently loaded, so nothing of someone else&rsquo;s is carried in.
          Everything a report cannot supply is asked for afterwards.
        </p>
      </div>

      <div className="max-h-[22rem] overflow-y-auto scroll-soft px-2 pb-2">
        <table className="w-full text-[12px]">
          <thead>
            <tr className="eyebrow" style={{ color: 'var(--ink-muted)' }}>
              <th className="w-8 px-2 pb-1.5" />
              <th className="px-2 pb-1.5 text-left font-semibold">Field</th>
              <th className="px-2 pb-1.5 text-right font-semibold">Value</th>
              <th className="px-2 pb-1.5 text-left font-semibold">From the document</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const spec = FIELD_BY_FEATURE.get(row.feature);
              const on = accepted[row.feature];
              const flagged = row.status === 'out_of_range';
              const note = row.note ?? spec?.note;
              return (
                <tr
                  key={row.feature}
                  style={{
                    borderTop: '1px solid var(--hairline)',
                    opacity: on ? 1 : 0.55,
                  }}
                >
                  <td className="px-2 py-2 align-top">
                    <input
                      type="checkbox"
                      checked={on}
                      aria-label={`Use ${row.label}`}
                      onChange={() =>
                        setAccepted((a) => ({ ...a, [row.feature]: !a[row.feature] }))
                      }
                      style={{ accentColor: 'var(--accent)' }}
                    />
                  </td>

                  <td className="px-2 py-2 align-top">
                    <div className="font-medium" style={{ color: 'var(--ink)' }}>
                      {row.label}
                    </div>
                    <div className="text-[10.5px]" style={{ color: 'var(--ink-muted)' }}>
                      {row.source === 'derived'
                        ? 'Derived'
                        : row.source === 'medication'
                          ? 'Medication'
                          : SOURCE_LABEL[row.source]}
                      {STATUS_NOTE[row.status] && ` · ${STATUS_NOTE[row.status]}`}
                    </div>
                  </td>

                  <td className="px-2 py-2 text-right align-top whitespace-nowrap">
                    <input
                      value={edited[row.feature] ?? String(row.value)}
                      onChange={(e) =>
                        setEdited((v) => ({ ...v, [row.feature]: e.target.value }))
                      }
                      className="tnum w-24 rounded px-1.5 py-1 text-right text-[12px]"
                      style={{
                        background: 'var(--surface)',
                        border: `1px solid ${flagged ? 'var(--risk-moderate)' : 'var(--hairline)'}`,
                        color: 'var(--ink)',
                      }}
                    />
                    {spec?.unit && (
                      <div className="mt-0.5 text-[10px]" style={{ color: 'var(--ink-muted)' }}>
                        {spec.unit}
                      </div>
                    )}
                  </td>

                  <td className="px-2 py-2 align-top">
                    {/* The verbatim line. This is the whole point of the table:
                        you can check the parse against the page without
                        trusting the parser. */}
                    <code
                      className="block truncate text-[11px]"
                      style={{ color: 'var(--ink-secondary)' }}
                      title={row.snippet}
                    >
                      {row.snippet}
                    </code>
                    {/* Only when the number actually moved. The platelet rule
                        matches "x10^3/uL" with a factor of 1 purely to stop the
                        per-microlitre rule claiming it, and reporting that as
                        "268 ×10³/µL converted to 268 ×10³/µL" is noise. */}
                    {row.unitApplied && row.rawValue !== undefined
                      && row.rawValue !== row.value && (
                      <div className="text-[10.5px]" style={{ color: 'var(--ink-muted)' }}>
                        {measure(row.rawValue)} {row.unitApplied} converted to{' '}
                        {measure(row.value as number)} {spec?.unit}
                      </div>
                    )}
                    {note && (
                      <div
                        className="text-[10.5px]"
                        style={{ color: flagged ? 'var(--warn-ink)' : 'var(--ink-muted)' }}
                      >
                        {note}
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div
        className="flex items-center justify-end gap-2 px-4 py-2.5"
        style={{ borderTop: '1px solid var(--hairline)' }}
      >
        <button
          onClick={onDismiss}
          className="rounded-[var(--radius-pill)] px-3 py-1.5 text-xs font-medium"
          style={{ color: 'var(--ink-secondary)' }}
        >
          Discard
        </button>
        <button
          onClick={apply}
          disabled={acceptedCount === 0}
          className="rounded-[var(--radius-pill)] px-3.5 py-1.5 text-xs font-semibold disabled:opacity-40"
          style={{ background: 'var(--accent)', color: 'var(--on-accent)' }}
        >
          Use as a new patient ({acceptedCount})
        </button>
      </div>
    </div>
  );
}
