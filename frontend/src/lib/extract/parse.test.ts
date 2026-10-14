/**
 * Parser tests.
 *
 * This is the riskiest code in the project: a misparse does not crash, it hands
 * the model a confident wrong number and the dashboard renders a confident wrong
 * prediction on top of it. The cases below are the specific ways that happens -
 * unit mismatches, negated findings read as positive, and a neighbouring row's
 * value being claimed by the wrong analyte.
 */

import { describe, expect, it } from 'vitest';

import { extractFields, isNegated, normaliseText, parseMedications } from './parse';
import { FIELD_BY_FEATURE } from './dictionary';

const valueOf = (text: string, feature: string) =>
  extractFields(text).candidates.find((c) => c.feature === feature);

describe('unit conversion', () => {
  it('converts a UK/Indian lipid panel from mmol/L to mg/dL', () => {
    // Each analyte has its own factor; using one factor for all four is the
    // classic version of this mistake.
    const report = [
      'LIPID PROFILE',
      'Total Cholesterol      5.2 mmol/L',
      'LDL Cholesterol        2.6 mmol/L',
      'HDL Cholesterol        1.1 mmol/L',
      'Triglycerides          1.7 mmol/L',
    ].join('\n');

    expect(valueOf(report, 'LDL')!.value).toBeCloseTo(2.6 * 38.67, 1);
    expect(valueOf(report, 'HDL')!.value).toBeCloseTo(1.1 * 38.67, 1);
    expect(valueOf(report, 'TG')!.value).toBeCloseTo(1.7 * 88.57, 1);
    expect(valueOf(report, 'LDL')!.unitApplied).toBe('mmol/L');
  });

  it('converts fasting glucose from mmol/L', () => {
    const got = valueOf('Fasting blood glucose 5.4 mmol/L', 'FBS')!;
    expect(got.value).toBeCloseTo(5.4 * 18.016, 1);
    expect(got.status).toBe('ok');
  });

  it('converts creatinine from micromoles', () => {
    const got = valueOf('Serum Creatinine: 88 umol/L', 'CR')!;
    expect(got.value).toBeCloseTo(88 / 88.4, 2);
    expect(got.status).toBe('ok');
  });

  it('converts urea to BUN but leaves a real BUN alone', () => {
    const urea = valueOf('Blood Urea            34 mg/dL', 'BUN')!;
    expect(urea.value).toBeCloseTo(34 * 0.467, 2);
    expect(urea.note).toMatch(/urea to BUN/i);

    const bun = valueOf('Blood Urea Nitrogen   16 mg/dL', 'BUN')!;
    expect(bun.value).toBe(16);
    expect(bun.note ?? '').not.toMatch(/urea to BUN/i);
  });
});

describe('the WBC / platelet scale trap', () => {
  // The dataset is internally inconsistent: WBC is a raw count, PLT is
  // thousands. A CBC prints both in the same column with the same unit.
  it('scales WBC up but leaves platelets alone when both are in x10^3/uL', () => {
    const cbc = [
      'COMPLETE BLOOD COUNT',
      'Total Leukocyte Count   7.1  x10^3/uL',
      'Platelet Count          210  x10^3/uL',
    ].join('\n');

    expect(valueOf(cbc, 'WBC')!.value).toBe(7100);
    expect(valueOf(cbc, 'PLT')!.value).toBe(210);
  });

  it('scales platelets down but leaves WBC alone when both are per microlitre', () => {
    const cbc = [
      'WBC          7100 cells/uL',
      'Platelets  210000 /uL',
    ].join('\n');

    expect(valueOf(cbc, 'WBC')!.value).toBe(7100);
    expect(valueOf(cbc, 'PLT')!.value).toBe(210);
  });
});

describe('the range gate', () => {
  it('flags a value whose unit was missed instead of using it', () => {
    // No recognisable unit, so no conversion fires - exactly the case the gate
    // exists for. 7.1 would otherwise be served as a white cell count.
    const got = valueOf('WBC 7.1', 'WBC')!;
    expect(got.status).toBe('out_of_range');
    expect(got.note).toMatch(/3700-18000/);
  });

  it('flags an ejection fraction above the cohort maximum', () => {
    const got = valueOf('LVEF 65 %', 'EF-TTE')!;
    expect(got.status).toBe('out_of_range');
    expect(got.note).toMatch(/outside/i);
  });

  it('accepts a value at the edge of the observed range', () => {
    expect(valueOf('Ejection Fraction 60%', 'EF-TTE')!.status).toBe('ok');
    expect(valueOf('Potassium 3.0 mEq/L', 'K')!.status).toBe('ok');
  });
});

describe('negation', () => {
  it.each([
    'No ST elevation.',
    'There is no evidence of ST elevation.',
    'ST elevation: absent',
    'Without ST elevation',
  ])('reads %j as absent', (line) => {
    expect(valueOf(line, 'St Elevation')!.value).toBe(0);
  });

  it('reads an unqualified finding as present', () => {
    expect(valueOf('T wave inversion in leads V1-V3.', 'Tinversion')!.value).toBe(1);
    expect(valueOf('ST depression noted in inferior leads.', 'St Depression')!.value).toBe(1);
  });

  it('does not let a previous line negate the next finding', () => {
    // "Normal" ends the sinus-rhythm line; the T wave finding on the next line
    // is a separate statement and must not inherit it.
    const ecg = 'Sinus rhythm. Axis normal.\nT wave inversion in V4-V6.';
    expect(valueOf(ecg, 'Tinversion')!.value).toBe(1);
  });

  it('negates a graded finding to absent rather than to its grading word', () => {
    const echo = 'No mitral regurgitation. Trivial tricuspid regurgitation.';
    expect(valueOf(echo, 'VHD')!.value).toBe('N');
  });
});

describe('field isolation', () => {
  it('does not take the next row value when a result is blank', () => {
    const report = ['Sodium', 'Potassium   4.2 mEq/L'].join('\n');
    // Sodium has no value on its line, so it must be reported as missing
    // rather than silently given potassium's number.
    expect(valueOf(report, 'Na')).toBeUndefined();
    expect(valueOf(report, 'K')!.value).toBe(4.2);
  });

  it('prefers the longest matching synonym', () => {
    // "HDL" is a substring of nothing here, but "LDL Cholesterol" and "LDL"
    // both match; the specific one must win so the value is read from the
    // right column.
    const got = valueOf('LDL Cholesterol 118 mg/dL', 'LDL')!;
    expect(got.value).toBe(118);
  });

  it('takes the systolic figure from a blood pressure pair', () => {
    expect(valueOf('Blood Pressure: 128/82 mmHg', 'BP')!.value).toBe(128);
  });
});

describe('derived and inferred values', () => {
  it('derives BMI from height and weight and says so', () => {
    const got = valueOf('Weight 82 kg\nHeight 174 cm', 'BMI')!;
    expect(got.value).toBeCloseTo(82 / 1.74 ** 2, 2);
    expect(got.status).toBe('derived');
    expect(got.snippet).toContain('82');
  });

  it('suggests diagnoses from medications, marked as inferred', () => {
    const rx = 'Tab. Metformin 500mg BD\nTab. Atorvastatin 10mg HS\nTab. Amlodipine 5mg OD';
    const found = parseMedications(rx);
    const features = found.map((c) => c.feature).sort();
    expect(features).toEqual(['DLP', 'DM', 'HTN']);
    for (const candidate of found) {
      expect(candidate.status).toBe('inferred');
      // The drug that caused the suggestion has to be named, or the user is
      // being asked to confirm something with no evidence attached.
      expect(candidate.note).toMatch(/metformin|atorvastatin|amlodipine/i);
    }
  });

  it('does not infer anything from a document with no medications', () => {
    expect(parseMedications('Haemoglobin 13.2 g/dL')).toHaveLength(0);
  });
});

describe('a whole report', () => {
  const REPORT = `
    CITY DIAGNOSTICS - BIOCHEMISTRY REPORT
    Patient: ANONYMISED        Age: 58 Years      Sex: Male
    Blood Pressure 134/86 mmHg      Pulse 78 /min

    TEST                      RESULT      UNIT
    Fasting Blood Sugar       104         mg/dL
    Serum Creatinine          1.1         mg/dL
    Blood Urea Nitrogen       18          mg/dL
    Sodium                    139         mEq/L
    Potassium                 4.4         mEq/L
    Triglycerides             168         mg/dL
    LDL Cholesterol           126         mg/dL
    HDL Cholesterol           38          mg/dL
    Haemoglobin               13.6        g/dL
    Total Leukocyte Count     8.2         x10^3/uL
    Platelet Count            244         x10^3/uL
    Neutrophils               62          %
    Lymphocytes               29          %
    ESR                       22          mm/hr
  `;

  it('reads the panel and lands every value inside the training range', () => {
    const { candidates } = extractFields(REPORT);
    const byFeature = Object.fromEntries(candidates.map((c) => [c.feature, c]));

    expect(byFeature.FBS.value).toBe(104);
    expect(byFeature.CR.value).toBe(1.1);
    expect(byFeature.BUN.value).toBe(18);
    expect(byFeature.Na.value).toBe(139);
    expect(byFeature.K.value).toBe(4.4);
    expect(byFeature.TG.value).toBe(168);
    expect(byFeature.LDL.value).toBe(126);
    expect(byFeature.HDL.value).toBe(38);
    expect(byFeature.HB.value).toBe(13.6);
    expect(byFeature.WBC.value).toBe(8200);
    expect(byFeature.PLT.value).toBe(244);
    expect(byFeature.Neut.value).toBe(62);
    expect(byFeature.Lymph.value).toBe(29);
    expect(byFeature.ESR.value).toBe(22);
    expect(byFeature.Age.value).toBe(58);
    expect(byFeature.Sex.value).toBe('Male');
    expect(byFeature.BP.value).toBe(134);
    expect(byFeature.PR.value).toBe(78);

    for (const candidate of candidates) {
      expect(candidate.status, `${candidate.feature} was flagged`).not.toBe('out_of_range');
    }
  });

  it('every candidate carries the text it came from', () => {
    for (const candidate of extractFields(REPORT).candidates) {
      expect(candidate.snippet.length).toBeGreaterThan(0);
    }
  });

  it('reports the symptom fields as missing, because no lab report has them', () => {
    const { missing } = extractFields(REPORT);
    // These are echo/ECG fields; a biochemistry panel cannot supply them.
    expect(missing).toContain('EF-TTE');
    expect(missing).toContain('Region RWMA');
    expect(missing).toContain('Tinversion');
  });
});

describe('helpers', () => {
  it('normalises whitespace without destroying line structure', () => {
    expect(normaliseText('a   b\r\n\r\n\r\n\r\nc')).toBe('a b\n\nc');
  });

  it('only looks backwards within the current line for a negator', () => {
    const text = 'normal\nfinding';
    expect(isNegated(text, text.indexOf('finding'))).toBe(false);
  });

  it('covers every extractable field with at least one synonym', () => {
    for (const [feature, spec] of FIELD_BY_FEATURE) {
      expect(spec.synonyms.length, `${feature} has no synonyms`).toBeGreaterThan(0);
    }
  });
});
