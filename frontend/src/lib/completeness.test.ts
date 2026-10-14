/**
 * Guards on the measured constants.
 *
 * `FEATURE_WEIGHT`, `REQUIRED_QUESTIONS` and `INERT_FIELDS` are transcriptions
 * of a measurement over the trained models, and a transcription is exactly the
 * kind of thing that rots quietly after a retrain. These do not re-derive the
 * numbers - that needs the booster - but they do assert the structural facts
 * that make the UI's claims true, so a bad edit fails here instead of shipping
 * a completeness meter that flatters the user.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  BLOCKING_FEATURE,
  DOCUMENT_FIELDS,
  FEATURE_WEIGHT,
  INERT_FIELDS,
  REQUIRED_QUESTIONS,
  coverageOf,
  isInert,
} from './completeness';

describe('the weight table', () => {
  it('is a normalised share, so "percent covered" means what it says', () => {
    const total = Object.values(FEATURE_WEIGHT).reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1, 2);
  });

  it('holds only the features that carry weight', () => {
    expect(Object.keys(FEATURE_WEIGHT)).toHaveLength(37);
    for (const [feature, weight] of Object.entries(FEATURE_WEIGHT)) {
      expect(weight, feature).toBeGreaterThan(0);
      expect(isInert(feature), `${feature} is both weighted and inert`).toBe(false);
    }
  });

  it('never lists an inert field as carrying weight', () => {
    expect(INERT_FIELDS).toHaveLength(18);
    for (const feature of INERT_FIELDS) {
      expect(FEATURE_WEIGHT[feature], feature).toBeUndefined();
    }
  });

  it('accounts for all 55 model inputs between the two sets', () => {
    expect(Object.keys(FEATURE_WEIGHT).length + INERT_FIELDS.length).toBe(55);
  });
});

describe('the required questions', () => {
  it('asks only for things no report can supply', () => {
    for (const q of REQUIRED_QUESTIONS) {
      expect(DOCUMENT_FIELDS, `${q.feature} is extractable, so should not be asked`)
        .not.toContain(q.feature);
    }
  });

  it('asks for everything that carries weight and is not in a report', () => {
    const asked = new Set(REQUIRED_QUESTIONS.map((q) => q.feature));
    const shouldAsk = Object.keys(FEATURE_WEIGHT).filter(
      (f) => !DOCUMENT_FIELDS.includes(f),
    );
    expect([...asked].sort()).toEqual(shouldAsk.sort());
  });

  it('is ordered heaviest first, so a partial answer is still the useful part', () => {
    const weights = REQUIRED_QUESTIONS.map((q) => q.weight);
    expect([...weights].sort((a, b) => b - a)).toEqual(weights);
  });

  it('blocks on the single heaviest feature in the whole model', () => {
    const heaviest = Object.entries(FEATURE_WEIGHT)
      .sort((a, b) => b[1] - a[1])[0][0];
    expect(BLOCKING_FEATURE).toBe(heaviest);
    expect(REQUIRED_QUESTIONS[0].feature).toBe(BLOCKING_FEATURE);
    // It is worth blocking on precisely because it dwarfs everything else.
    expect(FEATURE_WEIGHT[BLOCKING_FEATURE]).toBeGreaterThan(0.1);
  });

  it('gives every choice question its options', () => {
    for (const q of REQUIRED_QUESTIONS) {
      if (q.kind === 'choice') expect(q.options?.length ?? 0).toBeGreaterThan(1);
    }
  });

  it('encodes yes and no the way the trained model expects', () => {
    // The dataset is not consistent: six of these ten are numeric 0/1 and four
    // are the strings 'Y'/'N'. Sending 1 where the encoder learned 'Y' does not
    // throw - the one-hot encoder just treats it as an unseen category and the
    // prediction changes silently. So check against the model's own metadata
    // rather than against a second copy of the same assumption.
    const metadata = JSON.parse(
      readFileSync(
        resolve(__dirname, '../../../backend/app/models/metadata.json'),
        'utf-8',
      ),
    ) as {
      numeric_features: string[];
      categorical_levels: Record<string, (string | number)[]>;
    };
    const numeric = new Set(metadata.numeric_features);

    for (const q of REQUIRED_QUESTIONS) {
      if (q.kind !== 'yes-no') continue;
      if (numeric.has(q.feature)) {
        expect(typeof q.yes, `${q.feature} is numeric`).toBe('number');
        expect(typeof q.no, `${q.feature} is numeric`).toBe('number');
      } else {
        const levels = metadata.categorical_levels[q.feature] ?? [];
        expect(levels, `${q.feature} has no known levels`).not.toHaveLength(0);
        expect(levels, `${q.feature} yes=${q.yes}`).toContain(q.yes);
        expect(levels, `${q.feature} no=${q.no}`).toContain(q.no);
      }
      expect(q.yes).not.toBe(q.no);
    }
  });

  it('asks only for features the served model actually takes', () => {
    const metadata = JSON.parse(
      readFileSync(
        resolve(__dirname, '../../../backend/app/models/metadata.json'),
        'utf-8',
      ),
    ) as { raw_features: string[] };
    for (const q of REQUIRED_QUESTIONS) {
      expect(metadata.raw_features, q.feature).toContain(q.feature);
    }
    for (const feature of Object.keys(FEATURE_WEIGHT)) {
      expect(metadata.raw_features, feature).toContain(feature);
    }
  });
});

describe('coverage', () => {
  const everything = Object.fromEntries(
    Object.keys(FEATURE_WEIGHT).map((f) => [f, 1]),
  );

  it('is 100% when every weighted field is answered', () => {
    const c = coverageOf(everything);
    expect(c.weight).toBeCloseTo(1, 6);
    expect(c.missing).toHaveLength(0);
    expect(c.blocked).toBe(false);
  });

  it('is 0% and blocked when nothing is answered', () => {
    const c = coverageOf({});
    expect(c.weight).toBe(0);
    expect(c.blocked).toBe(true);
    expect(c.missing[0].feature).toBe(BLOCKING_FEATURE);
  });

  it('cannot be inflated by filling in the inert fields', () => {
    const inertOnly = Object.fromEntries(INERT_FIELDS.map((f) => [f, 'Y']));
    expect(coverageOf(inertOnly).weight).toBe(0);
  });

  it('reports a document-only intake as roughly two thirds covered', () => {
    // The measured figure is 67.2%: everything a lab, echo and ECG report can
    // give, with the symptoms and history still missing.
    const fromDocuments = Object.fromEntries(DOCUMENT_FIELDS.map((f) => [f, 1]));
    const c = coverageOf(fromDocuments);
    expect(c.weight).toBeGreaterThan(0.6);
    expect(c.weight).toBeLessThan(0.75);
    // And it is still blocked, because the heaviest field is a symptom.
    expect(c.blocked).toBe(true);
  });

  it('treats blank strings and nulls as unanswered', () => {
    expect(coverageOf({ [BLOCKING_FEATURE]: '' }).blocked).toBe(true);
    expect(coverageOf({ [BLOCKING_FEATURE]: null }).blocked).toBe(true);
    expect(coverageOf({ [BLOCKING_FEATURE]: 0 }).blocked).toBe(false);
  });

  it('ranks what is missing by weight, not alphabetically', () => {
    const c = coverageOf({});
    const weights = c.missing.map((m) => m.weight);
    expect([...weights].sort((a, b) => b - a)).toEqual(weights);
  });
});
