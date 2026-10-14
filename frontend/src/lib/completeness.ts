/**
 * How much of the model's attention a given set of inputs actually covers.
 *
 * Every number here was measured, not estimated: mean |SHAP| per feature over
 * all 303 patients, summed across the four models, normalised to a share of the
 * total. Reproduce with `python ml/src/validate.py` plus the booster's
 * `pred_contrib`, or see the README's coverage table.
 *
 * WHY THIS EXISTS
 * Counting filled fields is misleading. "40 of 55 complete" sounds nearly done
 * even when the 15 missing include Typical Chest Pain, which alone carries more
 * weight than the entire lab panel's top three fields combined. So the UI
 * reports covered *weight*, and blocks on the one field that can sink a
 * prediction by itself.
 */

/** Share of total predictive weight, per feature, across all four models. */
export const FEATURE_WEIGHT: Record<string, number> = {
  'Typical Chest Pain': 0.1520,
  Age: 0.0863,
  'EF-TTE': 0.0588,
  'Region RWMA': 0.0582,
  TG: 0.0516,
  ESR: 0.0469,
  FBS: 0.0347,
  CR: 0.0333,
  DM: 0.0299,
  BMI: 0.0268,
  Tinversion: 0.0240,
  HDL: 0.0236,
  Na: 0.0235,
  HB: 0.0233,
  Lymph: 0.0230,
  PLT: 0.0226,
  Neut: 0.0224,
  BP: 0.0224,
  Length: 0.0213,
  K: 0.0206,
  LDL: 0.0205,
  WBC: 0.0199,
  Weight: 0.0189,
  HTN: 0.0163,
  PR: 0.0162,
  Sex: 0.0153,
  Dyspnea: 0.0145,
  VHD: 0.0135,
  BUN: 0.0110,
  Atypical: 0.0102,
  FH: 0.0101,
  'Function Class': 0.0069,
  'Current Smoker': 0.0060,
  Nonanginal: 0.0054,
  DLP: 0.0048,
  'St Depression': 0.0047,
  'Q Wave': 0.0002,
};

/**
 * The 31 fields a lab, echo or ECG report can supply.
 *
 * Measured coverage: 67.2% of total predictive weight. The rest is history,
 * symptoms and examination findings, which exist only in a clinician's note.
 */
export const DOCUMENT_FIELDS: readonly string[] = [
  'Age', 'Weight', 'Length', 'Sex', 'BMI', 'BP', 'PR',
  'FBS', 'CR', 'TG', 'LDL', 'HDL', 'BUN', 'ESR', 'HB', 'K', 'Na', 'WBC',
  'Lymph', 'Neut', 'PLT',
  'EF-TTE', 'Region RWMA', 'VHD',
  'Q Wave', 'St Elevation', 'St Depression', 'Tinversion', 'LVH',
  'Poor R Progression', 'BBB',
];

export interface RequiredQuestion {
  feature: string;
  /** Asked the way a clinician would ask it, not the way the column is named. */
  question: string;
  help?: string;
  kind: 'yes-no' | 'choice';
  options?: { value: string | number; label: string }[];
  /**
   * What a "yes" and a "no" actually are for this column.
   *
   * The dataset is not consistent: six of these ten are numeric 0/1 and four are
   * the strings 'Y'/'N'. Sending 1 where the encoder expects 'Y' does not throw -
   * the value simply falls outside the known categories and the one-hot encoder
   * treats it as unseen, which quietly changes the prediction. So the encoding
   * is declared per question and checked against the model's own metadata in
   * completeness.test.ts.
   */
  yes: string | number;
  no: string | number;
  weight: number;
}

/**
 * The ten fields that are both used by the models and absent from every report.
 *
 * Ordered by weight, which is also the order of usefulness if someone answers
 * only the first few. Everything else a report cannot supply is in INERT_FIELDS
 * below - the models never split on it, so asking would be busywork.
 */
export const REQUIRED_QUESTIONS: RequiredQuestion[] = [
  {
    feature: 'Typical Chest Pain',
    question: 'Typical anginal chest pain?',
    help:
      'Substernal, brought on by exertion, relieved by rest or nitrates. The single '
      + 'strongest driver in the model - 15% of its total weight.',
    kind: 'yes-no',
    yes: 1,
    no: 0,
    weight: 0.152,
  },
  {
    feature: 'DM',
    question: 'Diabetes mellitus?',
    kind: 'yes-no',
    yes: 1,
    no: 0,
    weight: 0.0299,
  },
  {
    feature: 'HTN',
    question: 'Hypertension?',
    kind: 'yes-no',
    yes: 1,
    no: 0,
    weight: 0.0163,
  },
  {
    feature: 'Dyspnea',
    question: 'Breathlessness on exertion?',
    kind: 'yes-no',
    yes: 'Y',
    no: 'N',
    weight: 0.0145,
  },
  {
    feature: 'Atypical',
    question: 'Atypical chest pain?',
    help: 'Chest pain meeting some but not all of the anginal criteria.',
    kind: 'yes-no',
    yes: 'Y',
    no: 'N',
    weight: 0.0102,
  },
  {
    feature: 'FH',
    question: 'Family history of coronary disease?',
    kind: 'yes-no',
    yes: 1,
    no: 0,
    weight: 0.0101,
  },
  {
    feature: 'Function Class',
    question: 'NYHA functional class',
    help: 'Symptom limitation: 0 none, 1 slight, 2 marked, 3 at rest.',
    kind: 'choice',
    yes: 1,
    no: 0,
    options: [
      { value: 0, label: '0 — no limitation' },
      { value: 1, label: 'I — slight' },
      { value: 2, label: 'II — marked' },
      { value: 3, label: 'III — symptoms at rest' },
    ],
    weight: 0.0069,
  },
  {
    feature: 'Current Smoker',
    question: 'Currently smokes?',
    kind: 'yes-no',
    yes: 1,
    no: 0,
    weight: 0.006,
  },
  {
    feature: 'Nonanginal',
    question: 'Non-anginal chest pain?',
    help: 'Chest pain not meeting the anginal criteria at all.',
    kind: 'yes-no',
    yes: 'Y',
    no: 'N',
    weight: 0.0054,
  },
  {
    feature: 'DLP',
    question: 'Dyslipidaemia?',
    kind: 'yes-no',
    yes: 'Y',
    no: 'N',
    weight: 0.0048,
  },
];

/**
 * Prediction is blocked without this one.
 *
 * It is 15.2% of the model's weight in a single yes/no. Defaulting it quietly
 * would mean serving a confident percentage built on the cohort's most common
 * answer to the question that matters most.
 */
export const BLOCKING_FEATURE = 'Typical Chest Pain';

/**
 * The 18 inputs none of the four models ever splits on.
 *
 * Changing any of these cannot change any prediction - LightGBM never selected
 * them as a split, so they carry exactly zero. They stay in the form for
 * completeness and because they are part of the published dataset, but they are
 * collapsed and labelled rather than presented as equals to the rest.
 *
 * Note `BBB`, `LVH`, `Poor R Progression` and `St Elevation` are in here AND in
 * DOCUMENT_FIELDS: a report can supply them, it just will not matter. BBB in
 * particular still drives the bundle-branch highlight in the 3D view, which is
 * anatomical context, not an attribution claim.
 */
export const INERT_FIELDS: readonly string[] = [
  'Airway disease', 'BBB', 'CHF', 'CRF', 'CVA', 'Diastolic Murmur', 'EX-Smoker',
  'Edema', 'Exertional CP', 'LVH', 'LowTH Ang', 'Lung rales', 'Obesity',
  'Poor R Progression', 'St Elevation', 'Systolic Murmur', 'Thyroid Disease',
  'Weak Peripheral Pulse',
];

const INERT = new Set(INERT_FIELDS);

export const isInert = (feature: string) => INERT.has(feature);

export interface Coverage {
  /** Share of total predictive weight supplied by the user, 0-1. */
  weight: number;
  /** Fields with a real value, excluding inert ones. */
  filled: number;
  /** Fields that could carry weight. */
  total: number;
  /** Weight-carrying fields still unanswered, heaviest first. */
  missing: { feature: string; weight: number }[];
  /** True when the blocking field has no answer. */
  blocked: boolean;
}

const isAnswered = (value: unknown): boolean =>
  value !== null && value !== undefined && !(typeof value === 'string' && !value.trim());

/**
 * Score a feature payload.
 *
 * Only features with non-zero measured weight count towards either side of the
 * ratio, so filling in the 18 inert fields cannot inflate the number.
 */
export function coverageOf(features: Record<string, unknown>): Coverage {
  const weighted = Object.entries(FEATURE_WEIGHT);
  const total = weighted.reduce((sum, [, w]) => sum + w, 0);

  let covered = 0;
  let filled = 0;
  const missing: { feature: string; weight: number }[] = [];

  for (const [feature, weight] of weighted) {
    if (isAnswered(features[feature])) {
      covered += weight;
      filled += 1;
    } else {
      missing.push({ feature, weight });
    }
  }
  missing.sort((a, b) => b.weight - a.weight);

  return {
    weight: total > 0 ? covered / total : 0,
    filled,
    total: weighted.length,
    missing,
    blocked: !isAnswered(features[BLOCKING_FEATURE]),
  };
}
