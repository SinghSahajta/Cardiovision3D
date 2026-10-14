/**
 * What to look for in a report, and what the number means when you find it.
 *
 * The model takes 55 inputs. Only 31 of them can come from a document at all -
 * the rest are symptoms, history and examination findings that exist only in a
 * doctor's note. Those 31 are defined here. See `lib/completeness.ts` for the
 * ten that have to be asked instead.
 *
 * UNITS ARE THE HAZARD, and this dataset is internally inconsistent about them:
 *
 *  - WBC is a RAW count (3,700-18,000 cells/uL) but PLT is THOUSANDS
 *    (25-742 x10^3/uL). A report printing both as "x10^3/uL" needs WBC
 *    multiplied by 1000 and PLT left alone. Get it backwards and you hand the
 *    model a white cell count of 7 or a platelet count of 210,000.
 *  - The dataset wants BUN, not urea. Plenty of labs report "Blood Urea"
 *    instead; urea mg/dL x 0.467 = BUN mg/dL.
 *  - mmol/L is standard outside the US and is NOT interchangeable: the
 *    conversion differs per analyte (LDL/HDL x38.67, TG x88.57, glucose x18.016).
 *
 * `range` is the span actually observed across the 303 training patients. It is
 * not a clinical reference interval - it is the sanity gate. A value outside it
 * is surfaced to the user instead of being used, which is what catches a unit
 * mistake the synonym matching could not.
 */

export type FieldKind = 'number' | 'boolean' | 'enum';

export interface UnitRule {
  /** Matched case-insensitively against the text following the value. */
  pattern: RegExp;
  /** Multiply the parsed number by this to reach the model's unit. */
  factor: number;
  label: string;
}

export interface EnumRule {
  /** Matched against the line; first hit wins, so order most-specific first. */
  pattern: RegExp;
  value: string | number;
}

export interface FieldSpec {
  feature: string;
  label: string;
  kind: FieldKind;
  /** Which document this normally comes from, shown in the review table. */
  source: 'lab' | 'echo' | 'ecg' | 'vitals' | 'demographics';
  /** Report wordings, longest/most specific first - matching stops at the first. */
  synonyms: string[];
  /** Model-side unit, for display. */
  unit?: string;
  /** Observed cohort range; values outside are flagged, never silently used. */
  range?: [number, number];
  /** Unit conversions. The first pattern that matches the trailing text wins. */
  units?: UnitRule[];
  /** For enum/boolean fields, what phrasing maps to what value. */
  rules?: EnumRule[];
  /** Extra note surfaced in the review table. */
  note?: string;
}

// Conversion constants, kept named so the arithmetic is checkable against a
// reference rather than appearing as magic numbers at the call site.
const MMOL_TO_MGDL_CHOL = 38.67; // LDL, HDL, total cholesterol
const MMOL_TO_MGDL_TRIG = 88.57; // triglycerides
const MMOL_TO_MGDL_GLUC = 18.016; // glucose
const UMOL_TO_MGDL_CREAT = 1 / 88.4;
export const UREA_TO_BUN = 0.467;

/** "x10^3/uL", "10*3/uL", "K/uL", "thousand/uL" and friends. */
const THOUSANDS = /\b(?:x?\s*10\s*[\^*]?\s*3|10e3|k)\s*\/?\s*(?:u|µ|μ)?l\b/i;
const PER_MICROLITRE = /\b(?:cells?\s*\/\s*)?(?:u|µ|μ)l\b/i;

export const FIELDS: FieldSpec[] = [
  // ---------------------------------------------------------------- demographics
  {
    feature: 'Age',
    label: 'Age',
    kind: 'number',
    source: 'demographics',
    synonyms: ['age'],
    unit: 'years',
    range: [30, 86],
  },
  {
    feature: 'Sex',
    label: 'Sex',
    kind: 'enum',
    source: 'demographics',
    synonyms: ['sex', 'gender'],
    // The dataset spells it "Fmale". Match the report's wording, emit the
    // dataset's, because the model's one-hot encoder only knows the latter.
    rules: [
      { pattern: /\bfemale\b|\bf\b/i, value: 'Fmale' },
      { pattern: /\bmale\b|\bm\b/i, value: 'Male' },
    ],
  },
  {
    feature: 'Weight',
    label: 'Weight',
    kind: 'number',
    source: 'demographics',
    synonyms: ['weight', 'body weight'],
    unit: 'kg',
    range: [48, 120],
    units: [{ pattern: /\b(lb|lbs|pounds?)\b/i, factor: 0.45359237, label: 'lb' }],
  },
  {
    feature: 'Length',
    label: 'Height',
    kind: 'number',
    source: 'demographics',
    synonyms: ['height', 'length', 'stature'],
    unit: 'cm',
    range: [140, 188],
    units: [
      { pattern: /\b(m|metres?|meters?)\b/i, factor: 100, label: 'm' },
      { pattern: /\b(in|inch|inches)\b/i, factor: 2.54, label: 'in' },
    ],
  },
  {
    feature: 'BMI',
    label: 'Body mass index',
    kind: 'number',
    source: 'demographics',
    synonyms: ['body mass index', 'bmi'],
    unit: 'kg/m²',
    range: [18.1154, 40.9007],
    note: 'Recomputed from height and weight when both are found.',
  },

  // ---------------------------------------------------------------------- vitals
  {
    feature: 'BP',
    label: 'Systolic blood pressure',
    kind: 'number',
    source: 'vitals',
    synonyms: ['systolic blood pressure', 'blood pressure', 'systolic bp', 'bp', 'sbp'],
    unit: 'mmHg',
    range: [90, 190],
    note: 'Systolic only. From "128/82" the first number is taken.',
  },
  {
    feature: 'PR',
    label: 'Pulse rate',
    kind: 'number',
    source: 'vitals',
    synonyms: ['pulse rate', 'heart rate', 'pulse', 'ventricular rate', 'hr'],
    unit: 'bpm',
    range: [50, 110],
  },

  // ------------------------------------------------------------------ lab panel
  {
    feature: 'FBS',
    label: 'Fasting blood sugar',
    kind: 'number',
    source: 'lab',
    synonyms: [
      'fasting blood sugar', 'fasting blood glucose', 'fasting plasma glucose',
      'fasting glucose', 'glucose fasting', 'fbs', 'fpg',
    ],
    unit: 'mg/dL',
    range: [62, 400],
    units: [{ pattern: /\bmmol\s*\/\s*l\b/i, factor: MMOL_TO_MGDL_GLUC, label: 'mmol/L' }],
  },
  {
    feature: 'CR',
    label: 'Serum creatinine',
    kind: 'number',
    source: 'lab',
    synonyms: ['serum creatinine', 'creatinine', 's. creatinine', 'creat'],
    unit: 'mg/dL',
    range: [0.5, 2.2],
    units: [
      { pattern: /\b(u|µ|μ)mol\s*\/\s*l\b/i, factor: UMOL_TO_MGDL_CREAT, label: 'µmol/L' },
    ],
  },
  {
    feature: 'TG',
    label: 'Triglycerides',
    kind: 'number',
    source: 'lab',
    synonyms: ['triglycerides', 'triglyceride', 'tgl', 'tg'],
    unit: 'mg/dL',
    range: [37, 1050],
    units: [{ pattern: /\bmmol\s*\/\s*l\b/i, factor: MMOL_TO_MGDL_TRIG, label: 'mmol/L' }],
  },
  {
    feature: 'LDL',
    label: 'LDL cholesterol',
    kind: 'number',
    source: 'lab',
    synonyms: ['ldl cholesterol', 'ldl-c', 'ldl chol', 'l.d.l', 'ldl'],
    unit: 'mg/dL',
    range: [18, 232],
    units: [{ pattern: /\bmmol\s*\/\s*l\b/i, factor: MMOL_TO_MGDL_CHOL, label: 'mmol/L' }],
  },
  {
    feature: 'HDL',
    label: 'HDL cholesterol',
    kind: 'number',
    source: 'lab',
    synonyms: ['hdl cholesterol', 'hdl-c', 'hdl chol', 'h.d.l', 'hdl'],
    unit: 'mg/dL',
    range: [15.9, 111],
    units: [{ pattern: /\bmmol\s*\/\s*l\b/i, factor: MMOL_TO_MGDL_CHOL, label: 'mmol/L' }],
  },
  {
    feature: 'BUN',
    label: 'Blood urea nitrogen',
    kind: 'number',
    source: 'lab',
    // "blood urea nitrogen" has to be tried before "blood urea", or every BUN
    // result gets the urea conversion applied to it as well.
    synonyms: ['blood urea nitrogen', 'urea nitrogen', 'bun', 'blood urea', 'urea'],
    unit: 'mg/dL',
    range: [6, 52],
    note: 'The model wants BUN. A result labelled "urea" is converted (x0.467).',
  },
  {
    feature: 'ESR',
    label: 'Erythrocyte sedimentation rate',
    kind: 'number',
    source: 'lab',
    synonyms: ['erythrocyte sedimentation rate', 'sedimentation rate', 'esr'],
    unit: 'mm/hr',
    range: [1, 90],
  },
  {
    feature: 'HB',
    label: 'Haemoglobin',
    kind: 'number',
    source: 'lab',
    synonyms: ['haemoglobin', 'hemoglobin', 'hgb', 'hb'],
    unit: 'g/dL',
    range: [8.9, 17.6],
    units: [{ pattern: /\bg\s*\/\s*l\b/i, factor: 0.1, label: 'g/L' }],
  },
  {
    feature: 'K',
    label: 'Potassium',
    kind: 'number',
    source: 'lab',
    synonyms: ['potassium', 'serum potassium', 'k+', 'k'],
    unit: 'mEq/L',
    range: [3, 6.6],
  },
  {
    feature: 'Na',
    label: 'Sodium',
    kind: 'number',
    source: 'lab',
    synonyms: ['sodium', 'serum sodium', 'na+', 'na'],
    unit: 'mEq/L',
    range: [128, 156],
  },
  {
    feature: 'WBC',
    label: 'White blood cell count',
    kind: 'number',
    source: 'lab',
    synonyms: [
      'white blood cell count', 'white blood cells', 'total leucocyte count',
      'total leukocyte count', 'leucocyte count', 'leukocyte count', 'wbc', 'tlc',
    ],
    unit: 'cells/µL',
    range: [3700, 18000],
    // The trap: this field wants the raw count, PLT two entries down wants
    // thousands. Both are routinely printed as "x10^3/uL".
    units: [{ pattern: THOUSANDS, factor: 1000, label: '×10³/µL' }],
    note: 'Model unit is the raw count, so a result in ×10³/µL is multiplied by 1000.',
  },
  {
    feature: 'Lymph',
    label: 'Lymphocytes',
    kind: 'number',
    source: 'lab',
    synonyms: ['lymphocytes', 'lymphocyte', 'lymph'],
    unit: '%',
    range: [7, 60],
  },
  {
    feature: 'Neut',
    label: 'Neutrophils',
    kind: 'number',
    source: 'lab',
    synonyms: ['neutrophils', 'neutrophil', 'neut', 'polymorphs'],
    unit: '%',
    range: [32, 89],
  },
  {
    feature: 'PLT',
    label: 'Platelet count',
    kind: 'number',
    source: 'lab',
    synonyms: ['platelet count', 'platelets', 'platelet', 'plt'],
    unit: '×10³/µL',
    range: [25, 742],
    // Mirror image of WBC: here the model already wants thousands, so a raw
    // per-microlitre count is the one that needs scaling.
    //
    // The thousands rule must be listed FIRST even though it changes nothing.
    // "x10^3/uL" ends in "uL", so the per-microlitre pattern matches it too, and
    // without an earlier rule to claim it a platelet count of 210 x10^3/uL was
    // being divided by a thousand down to 0.21.
    units: [
      { pattern: THOUSANDS, factor: 1, label: '×10³/µL' },
      { pattern: PER_MICROLITRE, factor: 1 / 1000, label: 'cells/µL' },
    ],
    note: 'Model unit is ×10³/µL, unlike WBC which is the raw count.',
  },

  // ---------------------------------------------------------------- echo report
  {
    feature: 'EF-TTE',
    label: 'Left ventricular ejection fraction',
    kind: 'number',
    source: 'echo',
    synonyms: [
      'left ventricular ejection fraction', 'ejection fraction', 'lvef', 'ef',
    ],
    unit: '%',
    range: [15, 60],
    note: 'The cohort tops out at 60%, so a higher reading sits outside the training range.',
  },
  {
    feature: 'Region RWMA',
    label: 'Regional wall motion abnormality',
    kind: 'number',
    source: 'echo',
    synonyms: ['regional wall motion abnormality', 'wall motion abnormality', 'rwma'],
    unit: 'regions (0-4)',
    range: [0, 4],
    note: 'A 0-4 count of affected regions. The dataset never records which region.',
  },
  {
    feature: 'VHD',
    label: 'Valvular heart disease',
    kind: 'enum',
    source: 'echo',
    synonyms: [
      'valvular heart disease', 'valve disease', 'valvular', 'regurgitation',
      'stenosis of valve', 'mitral regurgitation', 'aortic stenosis', 'vhd',
    ],
    rules: [
      { pattern: /\bsevere\b/i, value: 'Severe' },
      { pattern: /\bmoderate\b/i, value: 'Moderate' },
      { pattern: /\b(mild|trivial|trace)\b/i, value: 'mild' },
      { pattern: /\b(no|none|normal|nil|absent)\b/i, value: 'N' },
    ],
    note: 'Graded only. The dataset does not record which valve is affected.',
  },

  // ----------------------------------------------------------------- ECG report
  {
    feature: 'Tinversion',
    label: 'T-wave inversion',
    kind: 'boolean',
    source: 'ecg',
    synonyms: ['t wave inversion', 't-wave inversion', 'inverted t wave', 't inversion'],
  },
  {
    feature: 'St Depression',
    label: 'ST depression',
    kind: 'boolean',
    source: 'ecg',
    synonyms: ['st segment depression', 'st depression', 'st-segment depression'],
  },
  {
    feature: 'St Elevation',
    label: 'ST elevation',
    kind: 'boolean',
    source: 'ecg',
    synonyms: ['st segment elevation', 'st elevation', 'st-segment elevation'],
  },
  {
    feature: 'Q Wave',
    label: 'Pathological Q wave',
    kind: 'boolean',
    source: 'ecg',
    synonyms: ['pathological q wave', 'pathologic q wave', 'q wave', 'q-wave'],
  },
  {
    feature: 'LVH',
    label: 'Left ventricular hypertrophy',
    kind: 'enum',
    source: 'ecg',
    synonyms: ['left ventricular hypertrophy', 'lvh'],
    rules: [
      { pattern: /\b(no|none|absent|nil|without)\b/i, value: 'N' },
      { pattern: /.*/, value: 'Y' },
    ],
  },
  {
    feature: 'Poor R Progression',
    label: 'Poor R wave progression',
    kind: 'enum',
    source: 'ecg',
    synonyms: ['poor r wave progression', 'poor r progression', 'poor r-wave progression'],
    rules: [
      { pattern: /\b(no|none|absent|nil|normal)\b/i, value: 'N' },
      { pattern: /.*/, value: 'Y' },
    ],
  },
  {
    feature: 'BBB',
    label: 'Bundle branch block',
    kind: 'enum',
    source: 'ecg',
    synonyms: [
      'left bundle branch block', 'right bundle branch block', 'bundle branch block',
      'lbbb', 'rbbb', 'bbb',
    ],
    rules: [
      { pattern: /\b(lbbb|left bundle)\b/i, value: 'LBBB' },
      { pattern: /\b(rbbb|right bundle)\b/i, value: 'RBBB' },
      { pattern: /\b(no|none|absent|nil)\b/i, value: 'N' },
    ],
  },
];

export const FIELD_BY_FEATURE = new Map(FIELDS.map((f) => [f.feature, f]));

export const SOURCE_LABEL: Record<FieldSpec['source'], string> = {
  lab: 'Lab panel',
  echo: 'Echo report',
  ecg: 'ECG report',
  vitals: 'Vitals',
  demographics: 'Demographics',
};

/**
 * Medications that imply a diagnosis.
 *
 * This is inference, not extraction: a prescription lists drugs, not findings.
 * Each suggestion therefore names the drug that produced it and arrives
 * unaccepted in the review table, because "on ramipril" is good evidence of
 * hypertension and not a measurement of it.
 */
export interface DrugRule {
  feature: string;
  value: string | number;
  drugs: string[];
  reason: string;
}

export const DRUG_RULES: DrugRule[] = [
  {
    feature: 'DM',
    value: 1,
    drugs: [
      'metformin', 'glimepiride', 'gliclazide', 'glipizide', 'sitagliptin',
      'vildagliptin', 'linagliptin', 'empagliflozin', 'dapagliflozin',
      'pioglitazone', 'insulin', 'glibenclamide', 'teneligliptin',
    ],
    reason: 'an oral hypoglycaemic or insulin',
  },
  {
    feature: 'DLP',
    value: 'Y',
    drugs: [
      'atorvastatin', 'rosuvastatin', 'simvastatin', 'pravastatin',
      'fenofibrate', 'ezetimibe', 'gemfibrozil', 'statin',
    ],
    reason: 'a lipid-lowering drug',
  },
  {
    feature: 'HTN',
    value: 1,
    drugs: [
      'amlodipine', 'telmisartan', 'losartan', 'olmesartan', 'valsartan',
      'ramipril', 'enalapril', 'lisinopril', 'perindopril', 'metoprolol',
      'atenolol', 'bisoprolol', 'nebivolol', 'hydrochlorothiazide',
      'chlorthalidone', 'cilnidipine',
    ],
    reason: 'an antihypertensive',
  },
];
