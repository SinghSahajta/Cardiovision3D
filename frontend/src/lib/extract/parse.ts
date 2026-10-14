/**
 * Turn report text into candidate field values.
 *
 * Nothing here ever reaches the model directly. Every candidate carries the
 * snippet it came from and a status, and the user accepts it in
 * `ExtractionReview` before it becomes an input. That is deliberate: OCR
 * misreads digits, lab layouts vary wildly, and a silently wrong potassium is
 * worse than no potassium at all.
 *
 * The four steps, in order:
 *   1. find the term      - synonyms, longest first, on word boundaries
 *   2. check for negation - "no ST elevation" must not set the flag
 *   3. normalise the unit - see dictionary.ts for why this is the hazard
 *   4. gate on range      - against the cohort's observed span, which is what
 *                           catches the unit mistakes step 3 missed
 */

import {
  DRUG_RULES,
  FIELDS,
  UREA_TO_BUN,
  type FieldSpec,
} from './dictionary';

export type CandidateStatus =
  /** Parsed cleanly and inside the range the model was trained on. */
  | 'ok'
  /** Parsed, but outside the cohort range - probably a unit or OCR error. */
  | 'out_of_range'
  /** Derived from another field rather than read, e.g. BMI from height+weight. */
  | 'derived'
  /** Implied by a medication rather than measured. */
  | 'inferred';

export interface Candidate {
  feature: string;
  label: string;
  value: string | number;
  /** The raw number before unit conversion, when one was applied. */
  rawValue?: number;
  unitApplied?: string;
  status: CandidateStatus;
  /** Verbatim source text, so the user can check the parse against the report. */
  snippet: string;
  source: FieldSpec['source'] | 'derived' | 'medication';
  note?: string;
}

/** Words that negate a finding when they appear just before it. */
const NEGATORS = [
  'no', 'not', 'non', 'none', 'nil', 'without', 'absent', 'negative',
  'denies', 'denied', 'free of', 'unremarkable', 'normal', 'ruled out',
  'no evidence of', 'no significant',
];

/**
 * Words that negate a finding when they FOLLOW it, which is how a structured
 * report row is written: "ST elevation: absent", "RWMA - nil".
 */
const TRAILING_NEGATORS =
  /^[\s:=\-|]*\b(absent|nil|none|negative|no|not seen|not present|not detected|normal|unremarkable|within normal limits|wnl)\b/i;

/** How far back from a term to look for a negator, in characters. */
const NEGATION_WINDOW = 42;

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Collapse OCR's ragged whitespace without losing line structure. */
export function normaliseText(raw: string): string {
  return raw
    .replace(/\r\n?/g, '\n')
    .replace(/[–—]/g, '-')
    .replace(/[^\S\n]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/**
 * Is the term at `index` negated by something just before it?
 *
 * Only looks backwards within one line. Looking across a newline would let the
 * "Normal" in a previous row's reference-range column negate the next analyte,
 * which is exactly the kind of quiet error this whole module exists to avoid.
 */
export function isNegated(text: string, index: number, end?: number): boolean {
  const lineStart = text.lastIndexOf('\n', index) + 1;
  const from = Math.max(lineStart, index - NEGATION_WINDOW);
  const before = text.slice(from, index).toLowerCase();
  if (
    NEGATORS.some((word) =>
      new RegExp(`(^|[^a-z])${escape(word)}[^a-z]*$`).test(before),
    )
  ) {
    return true;
  }
  // Then forward, because a report table negates on the right-hand side of the
  // row: "ST elevation: absent" reads as present to a backwards-only check.
  if (end !== undefined) {
    const after = text.slice(end, end + 32).split('\n')[0];
    return TRAILING_NEGATORS.test(after);
  }
  return false;
}

function snippetAround(text: string, index: number, length: number): string {
  const lineStart = text.lastIndexOf('\n', index) + 1;
  let lineEnd = text.indexOf('\n', index + length);
  if (lineEnd === -1) lineEnd = text.length;
  return text.slice(lineStart, Math.min(lineEnd, lineStart + 160)).trim();
}

/**
 * Read the number that belongs to a term, from the text following it.
 *
 * Skips the punctuation and colons labs pad their layouts with, and refuses to
 * jump a newline: in a two-column report the next line's value belongs to the
 * next analyte, not this one.
 */
function numberAfter(text: string, from: number): { value: number; end: number } | null {
  const tail = text.slice(from, from + 90);
  const stop = tail.indexOf('\n');
  const line = stop === -1 ? tail : tail.slice(0, stop);
  // Allow a separator run first, then the number. "Result" / "Value" column
  // headers are skipped so "Sodium   Result 141" still parses.
  const match = line.match(
    /^[\s:=.\-|]*(?:result|value|observed)?[\s:=.\-|]*(-?\d+(?:[.,]\d+)?)/i,
  );
  if (!match) return null;
  const numeric = Number(match[1].replace(',', '.'));
  if (!Number.isFinite(numeric)) return null;
  return { value: numeric, end: from + (match.index ?? 0) + match[0].length };
}

function applyUnit(
  spec: FieldSpec,
  value: number,
  text: string,
  end: number,
): { value: number; unitApplied?: string } {
  if (!spec.units?.length) return { value };
  // Only the remainder of the same line counts as this result's unit.
  const tail = text.slice(end, end + 40).split('\n')[0];
  for (const rule of spec.units) {
    if (rule.pattern.test(tail)) {
      return { value: value * rule.factor, unitApplied: rule.label };
    }
  }
  return { value };
}

function gate(spec: FieldSpec, value: number): CandidateStatus {
  if (!spec.range) return 'ok';
  const [min, max] = spec.range;
  return value >= min && value <= max ? 'ok' : 'out_of_range';
}

function outOfRangeNote(spec: FieldSpec, value: number): string {
  const [min, max] = spec.range!;
  return (
    `${value.toLocaleString()} is outside the ${min}-${max} ${spec.unit ?? ''} range ` +
    `seen across the 303 training patients. Usually a unit mismatch or an OCR ` +
    `misread - check it against the report before accepting.`
  ).trim();
}

/** Systolic from "128/82", which is how vitals are nearly always written. */
function systolicFrom(text: string, from: number): number | null {
  const line = text.slice(from, from + 60).split('\n')[0];
  const match = line.match(/^[\s:=.\-|]*(\d{2,3})\s*\/\s*(\d{2,3})/);
  return match ? Number(match[1]) : null;
}

function parseField(spec: FieldSpec, text: string): Candidate | null {
  const haystack = text.toLowerCase();

  for (const synonym of [...spec.synonyms].sort((a, b) => b.length - a.length)) {
    const pattern = new RegExp(`(^|[^a-z0-9])${escape(synonym)}([^a-z0-9]|$)`, 'gi');
    let hit: RegExpExecArray | null;

    while ((hit = pattern.exec(haystack)) !== null) {
      const termStart = hit.index + hit[1].length;
      const termEnd = termStart + synonym.length;
      const negated = isNegated(text, termStart, termEnd);
      const snippet = snippetAround(text, termStart, synonym.length);

      if (spec.kind === 'boolean') {
        return {
          feature: spec.feature,
          label: spec.label,
          value: negated ? 0 : 1,
          status: 'ok',
          snippet,
          source: spec.source,
        };
      }

      if (spec.kind === 'enum') {
        const line = snippet;
        let value: string | number | null = null;
        for (const rule of spec.rules ?? []) {
          if (rule.pattern.test(line)) {
            value = rule.value;
            break;
          }
        }
        // A negated mention beats whatever grading word happens to share the
        // line: "no mitral regurgitation, trivial tricuspid" is not "mild".
        if (negated) {
          const absent = spec.rules?.find((r) => r.value === 'N');
          value = absent ? absent.value : value;
        }
        if (value === null) continue;
        return {
          feature: spec.feature,
          label: spec.label,
          value,
          status: 'ok',
          snippet,
          source: spec.source,
        };
      }

      // Numeric.
      if (spec.feature === 'BP') {
        const systolic = systolicFrom(text, termEnd);
        if (systolic !== null) {
          const status = gate(spec, systolic);
          return {
            feature: spec.feature,
            label: spec.label,
            value: systolic,
            status,
            snippet,
            source: spec.source,
            note: status === 'out_of_range' ? outOfRangeNote(spec, systolic) : spec.note,
          };
        }
      }

      const found = numberAfter(text, termEnd);
      if (!found) continue;

      let { value } = found;
      const converted = applyUnit(spec, value, text, found.end);
      value = converted.value;

      // "Blood urea" reports urea, not BUN. Applied after unit conversion so a
      // urea result in mmol/L would still be handled if that rule is added.
      let note = spec.note;
      if (spec.feature === 'BUN' && /\burea\b/i.test(synonym) && !/nitrogen/i.test(synonym)) {
        value = value * UREA_TO_BUN;
        note = `Converted from urea to BUN (x${UREA_TO_BUN}).`;
      }

      const rounded = Math.round(value * 1000) / 1000;
      const status = gate(spec, rounded);
      return {
        feature: spec.feature,
        label: spec.label,
        value: rounded,
        rawValue: converted.unitApplied ? found.value : undefined,
        unitApplied: converted.unitApplied,
        status,
        snippet,
        source: spec.source,
        note: status === 'out_of_range' ? outOfRangeNote(spec, rounded) : note,
      };
    }
  }
  return null;
}

/** Drug-implied diagnoses. Suggestions, named for the drug that caused them. */
export function parseMedications(text: string): Candidate[] {
  const haystack = text.toLowerCase();
  const out: Candidate[] = [];
  for (const rule of DRUG_RULES) {
    const found = rule.drugs.find((drug) =>
      new RegExp(`(^|[^a-z])${escape(drug)}`, 'i').test(haystack),
    );
    if (!found) continue;
    const at = haystack.indexOf(found);
    out.push({
      feature: rule.feature,
      label: rule.feature,
      value: rule.value,
      status: 'inferred',
      snippet: snippetAround(text, at, found.length),
      source: 'medication',
      note:
        `Not measured - inferred because the document mentions ${found}, ` +
        `${rule.reason}. Confirm before accepting.`,
    });
  }
  return out;
}

export interface ExtractionResult {
  candidates: Candidate[];
  /** Fields this document had nothing to say about. */
  missing: string[];
  text: string;
}

/** Run every field spec over one document's text. */
export function extractFields(rawText: string): ExtractionResult {
  const text = normaliseText(rawText);
  const candidates: Candidate[] = [];

  for (const spec of FIELDS) {
    const found = parseField(spec, text);
    if (found) candidates.push(found);
  }

  // BMI is arithmetic, not a measurement: if the report gives height and weight
  // it is more trustworthy derived than read, because a printed BMI may have
  // been computed from different numbers than the ones on this page.
  const weight = candidates.find((c) => c.feature === 'Weight');
  const height = candidates.find((c) => c.feature === 'Length');
  if (weight && height && typeof weight.value === 'number' && typeof height.value === 'number'
      && height.value > 0) {
    const bmi = Math.round((weight.value / (height.value / 100) ** 2) * 1000) / 1000;
    const existing = candidates.findIndex((c) => c.feature === 'BMI');
    const derived: Candidate = {
      feature: 'BMI',
      label: 'Body mass index',
      value: bmi,
      status: 'derived',
      snippet: `${weight.value} kg / (${height.value} cm)²`,
      source: 'derived',
      note: 'Computed from the height and weight above, not read from the page.',
    };
    if (existing >= 0) candidates[existing] = derived;
    else candidates.push(derived);
  }

  candidates.push(...parseMedications(text));

  const seen = new Set(candidates.map((c) => c.feature));
  const missing = FIELDS.filter((f) => !seen.has(f.feature)).map((f) => f.feature);

  return { candidates, missing, text };
}
