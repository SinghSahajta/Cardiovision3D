import type { FeatureAttribution } from '../../api/types';
import { measure } from '../../lib/risk';

/**
 * Requirement 3c: physiological measurements shown alongside their relative
 * contribution to the prediction.
 *
 * Deliberately a table rather than a chart - the job here is to let a clinician
 * read specific values ("EF 40%, is that what drove this?"), which a plot of
 * mixed units cannot do. The contribution column is a bar *inside* the table so
 * magnitude is still scannable, and `relative` is each feature's share of the
 * total absolute contribution, which is the only comparison that makes sense
 * across features measured in different units.
 */

const REFERENCE: Record<string, { range: string; flag?: (v: number) => boolean }> = {
  'EF-TTE': { range: 'normal >= 55%', flag: (v) => v < 50 },
  BP: { range: 'normal < 120 mmHg', flag: (v) => v >= 140 },
  LDL: { range: 'optimal < 100 mg/dL', flag: (v) => v >= 130 },
  HDL: { range: 'low < 40 mg/dL', flag: (v) => v < 40 },
  TG: { range: 'normal < 150 mg/dL', flag: (v) => v >= 200 },
  FBS: { range: 'normal < 100 mg/dL', flag: (v) => v >= 126 },
  CR: { range: 'normal 0.6-1.3 mg/dL', flag: (v) => v > 1.3 },
  HB: { range: 'normal 13-17 g/dL', flag: (v) => v < 12 },
  'Region RWMA': { range: '0 = none', flag: (v) => v > 0 },
  PR: { range: 'normal 60-100 bpm' },
  Age: { range: '' },
  BMI: { range: 'normal 18.5-25', flag: (v) => v >= 30 },
  ESR: { range: 'normal < 20 mm/h', flag: (v) => v >= 30 },
  BUN: { range: 'normal 7-20 mg/dL', flag: (v) => v > 20 },
  K: { range: 'normal 3.5-5.1 mmol/L' },
  Na: { range: 'normal 135-145 mmol/L' },
};

export function PhysiologyPanel({
  attributions,
}: {
  attributions: FeatureAttribution[];
}) {
  const total = attributions.reduce((acc, a) => acc + Math.abs(a.shap_value), 0) || 1;
  const maxAbs = Math.max(...attributions.map((a) => Math.abs(a.shap_value)), 1e-6);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[12px]">
        <thead>
          <tr
            className="text-left text-[10px] uppercase tracking-wide"
            style={{ color: 'var(--ink-muted)' }}
          >
            <th className="pb-1.5 font-medium">Measurement</th>
            <th className="pb-1.5 text-right font-medium">Value</th>
            <th className="pb-1.5 pl-3 font-medium">Contribution</th>
            <th className="pb-1.5 text-right font-medium">Share</th>
          </tr>
        </thead>
        <tbody>
          {attributions.map((a) => {
            const up = a.shap_value > 0;
            const share = Math.abs(a.shap_value) / total;
            const width = (Math.abs(a.shap_value) / maxAbs) * 100;
            const ref = REFERENCE[a.feature];
            const numeric = typeof a.feature_value === 'number' ? a.feature_value : NaN;
            const abnormal = ref?.flag && !Number.isNaN(numeric) && ref.flag(numeric);

            return (
              <tr key={a.processed_feature} style={{ borderTop: '1px solid var(--grid)' }}>
                <td className="py-1.5 pr-2">
                  <div className="truncate" style={{ color: 'var(--ink-secondary)' }}>
                    {a.display_name}
                  </div>
                  {ref?.range && (
                    <div className="text-[9.5px]" style={{ color: 'var(--ink-muted)' }}>
                      {ref.range}
                    </div>
                  )}
                </td>
                <td className="tnum py-1.5 text-right align-top font-medium whitespace-nowrap">
                  <span style={{ color: abnormal ? 'var(--warn-ink)' : 'var(--ink)' }}>
                    {measure(a.feature_value)}
                  </span>
                  {abnormal && (
                    <span className="ml-1 text-[9.5px]" style={{ color: 'var(--warn-ink)' }}>
                      !
                    </span>
                  )}
                </td>
                <td className="py-1.5 pl-3 align-middle">
                  <div className="flex items-center gap-1.5">
                    <span
                      className="inline-block h-[8px] shrink-0"
                      style={{
                        width: `${Math.max(width * 0.75, 2)}%`,
                        minWidth: 3,
                        background: up ? 'var(--shap-up)' : 'var(--shap-down)',
                        borderRadius: 3,
                      }}
                    />
                    <span className="text-[10px] whitespace-nowrap" style={{ color: 'var(--ink-muted)' }}>
                      {up ? 'raises' : 'lowers'}
                    </span>
                  </div>
                </td>
                <td
                  className="tnum py-1.5 text-right align-middle whitespace-nowrap"
                  style={{ color: 'var(--ink-secondary)' }}
                >
                  {(share * 100).toFixed(0)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-[10.5px] leading-snug" style={{ color: 'var(--ink-muted)' }}>
        Share is each measurement&rsquo;s portion of the total absolute contribution for
        the features shown. Reference ranges are general adult guides for orientation
        only and are not part of the model.
      </p>
    </div>
  );
}
