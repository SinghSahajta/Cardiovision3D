/**
 * Risk presentation helpers.
 *
 * The backend sends a `color_hex`, but we deliberately re-derive colour here from
 * `risk_level` instead of using it. Reason: the dashboard's colours come from a
 * validated palette (CVD separation, contrast vs surface, both themes), and the
 * 3D vessels, the cards and the legend must all agree. Trusting the backend's hex
 * would let a server-side tweak silently break that contract.
 *
 * Status colour never carries meaning alone - every call site pairs it with the
 * tier label and the model's decision threshold.
 */

import type { VesselKey } from '../api/types';

export type RiskTier = 'Low' | 'Moderate' | 'High';

export const TIER_COLOR: Record<RiskTier, string> = {
  Low: '#0ca30c',
  Moderate: '#fab219',
  High: '#d03b3b',
};

/** Mirrors CardiacInferenceEngine._risk_tier in backend/app/services.py. */
export function tierOf(probability: number): RiskTier {
  if (probability >= 0.65) return 'High';
  if (probability >= 0.35) return 'Moderate';
  return 'Low';
}

export const colorOf = (probability: number) => TIER_COLOR[tierOf(probability)];

export const TIER_BANDS: { tier: RiskTier; label: string }[] = [
  { tier: 'Low', label: 'Low  < 35%' },
  { tier: 'Moderate', label: 'Moderate  35-65%' },
  { tier: 'High', label: 'High  > 65%' },
];

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;

/**
 * Format a measured clinical value for display.
 *
 * Derived features arrive as full float64 from the model frame - BMI, for
 * instance, is weight/height^2 and prints as 24.489795918367346 - which reads
 * as false precision next to values a clinician actually recorded. Integers are
 * left exactly as they are; anything else is shown to one decimal, which is the
 * precision these measurements are taken at.
 */
export const measure = (value: string | number | null): string => {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'string') return value;
  if (!Number.isFinite(value)) return '—';
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
};

export const VESSEL_FULL_NAME: Record<VesselKey, string> = {
  LAD: 'Left anterior descending',
  LCX: 'Left circumflex',
  RCA: 'Right coronary artery',
};

/** What each vessel supplies, for the anatomical context panel. */
export const VESSEL_TERRITORY: Record<VesselKey, string> = {
  LAD: 'Anterior wall, anterior septum and the LV apex. The largest territory; proximal disease here carries the most myocardium at risk.',
  LCX: 'Lateral and posterolateral LV wall. Often the smallest territory and the most variable between patients.',
  RCA: 'Right ventricle, inferior wall, and in most people the AV node. Inferior infarcts and bradyarrhythmia originate here.',
};

/**
 * Per-vessel caveat shown beside the prediction. These are not decoration: the
 * held-out numbers for LCX and RCA are weak and the UI says so where the user is
 * actually looking, rather than only in the README.
 */
export const VESSEL_CAVEAT: Record<VesselKey, string | null> = {
  LAD: null,
  LCX: 'Held-out PPV 53% - about half of flagged LCX vessels are false alarms.',
  RCA: 'Held-out PPV 49% and ROC-AUC 0.70 - the weakest of the three. Treat as a prompt to image, not a finding.',
};
