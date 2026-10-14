/**
 * Anatomy presentation: tissue colours, and the map from model features to the
 * structures that explain them.
 *
 * Tissue colour is deliberately NOT from the risk palette. The three status hues
 * are reserved for predicted stenosis, so muscle, atrium, valve and conduction
 * tissue are given desaturated anatomical tones that cannot be mistaken for a
 * risk reading at a glance.
 */

import type { AnatomyGroup } from '../api/types';

export interface TissueStyle {
  color: string;
  /** Opacity in x-ray mode; solid mode is always opaque. */
  ghost: number;
  roughness: number;
}

export const TISSUE: Record<AnatomyGroup, TissueStyle> = {
  // Myocardium carries the most screen area, so it is the most muted thing in
  // the scene - the arteries have to win the eye, not the muscle. The chroma
  // here is deliberately low: a saturated muscle red sits close enough to the
  // High-risk status hue that a vessel stops reading as a separate thing, which
  // is the whole job of this view.
  myocardium: { color: '#9a7369', ghost: 0.14, roughness: 0.72 },
  atrium: { color: '#8c7a78', ghost: 0.1, roughness: 0.78 },
  great_vessel: { color: '#8d8fa6', ghost: 0.12, roughness: 0.6 },
  artery: { color: '#b84a4a', ghost: 1, roughness: 0.45 },
  valve: { color: '#d8cdb4', ghost: 0.9, roughness: 0.72 },
  conduction: { color: '#d8b24a', ghost: 0.95, roughness: 0.5 },
};

/** Colour used when a structure is highlighted because a feature points at it. */
export const HIGHLIGHT = '#3987e5';

/**
 * Map the morph target's influence from a calibrated probability.
 *
 * Flat below the lowest risk tier, then rising to full influence at certainty.
 * The narrowing is uniform along the vessel: the dataset records a vessel as
 * stenotic or not and never says where the lesion is, so the shape change
 * encodes severity only and claims nothing about position.
 */
export function stenosisInfluence(risk: number): number {
  if (risk <= 0.35) return 0;
  return Math.min((risk - 0.35) / 0.55, 1);
}

export interface FeatureLink {
  /** Node names in anatomy.glb this feature points at. */
  nodes: string[];
  /** Layer that must be switched on for those nodes to be visible. */
  layer: 'core' | 'valves' | 'conduction';
  /** One line explaining the link, shown when the highlight is offered. */
  note: string;
}

/**
 * Which anatomy explains which feature.
 *
 * Only links the dataset actually supports are here. Deliberately absent:
 * `Q Wave`, `St Elevation`, `St Depression`, `Tinversion` and
 * `Poor R Progression`. In real ECG reading those localise to a myocardial
 * territory, but this dataset records presence or absence with no lead
 * information, so lighting up a wall for them would be invention. They stay in
 * the SHAP panel, which claims nothing anatomical.
 */
export function featureLink(
  feature: string,
  value: string | number | null,
): FeatureLink | null {
  const text = typeof value === 'string' ? value.trim() : value;
  const isNormal = text === 'N' || text === 0 || text === '0' || text === null;

  switch (feature) {
    case 'EF-TTE':
      // Normal LVEF is 50-70%. Below 50 the left ventricle is the finding.
      return typeof text === 'number' && text < 50
        ? {
            nodes: ['myo_lv_free', 'myo_lv_septal', 'myo_septum'],
            layer: 'core',
            note: `Ejection fraction ${text}% is measured across the left ventricle, highlighted here.`,
          }
        : null;

    case 'Region RWMA':
      // 0-4. The dataset grades HOW MANY regions move abnormally and never says
      // WHICH, so the whole ventricle is highlighted and the count is quoted.
      // Singling out a wall would be a claim the data cannot support.
      return typeof text === 'number' && text > 0
        ? {
            nodes: ['myo_lv_free', 'myo_lv_septal'],
            layer: 'core',
            note: `${text} region${text === 1 ? '' : 's'} with wall motion abnormality. The dataset grades how many, not which, so the whole left ventricle is marked.`,
          }
        : null;

    case 'LVH':
      return text === 'Y'
        ? {
            nodes: ['myo_lv_free', 'myo_lv_septal', 'myo_septum'],
            layer: 'core',
            note: 'Left ventricular hypertrophy: thickening of the ventricular myocardium shown here.',
          }
        : null;

    case 'VHD':
      return !isNormal && typeof text === 'string'
        ? {
            nodes: ['valve_mitral', 'valve_tricuspid', 'valve_aortic', 'valve_pulmonary'],
            layer: 'valves',
            note: `Valvular heart disease graded ${text}. The dataset does not say which valve, so all four are shown.`,
          }
        : null;

    case 'Systolic Murmur':
      return text === 'Y'
        ? {
            nodes: ['valve_mitral', 'valve_tricuspid', 'valve_aortic'],
            layer: 'valves',
            note: 'A systolic murmur arises at the atrioventricular valves or across the aortic valve; the dataset does not localise it.',
          }
        : null;

    case 'Diastolic Murmur':
      return text === 'Y'
        ? {
            nodes: ['valve_aortic', 'valve_pulmonary'],
            layer: 'valves',
            note: 'A diastolic murmur arises at the semilunar valves; the dataset does not localise it.',
          }
        : null;

    case 'BBB':
      // The one exactly-localising feature in the set: LBBB and RBBB name the
      // branch, so the correct bundle is highlighted and nothing is hedged.
      if (text === 'LBBB') {
        return {
          nodes: ['cond_left_bundle'],
          layer: 'conduction',
          note: 'Left bundle branch block: conduction fails in the left branch of the atrioventricular bundle, shown here with its three divisions.',
        };
      }
      if (text === 'RBBB') {
        return {
          nodes: ['cond_right_bundle'],
          layer: 'conduction',
          note: 'Right bundle branch block: conduction fails in the right branch of the atrioventricular bundle, shown here.',
        };
      }
      return null;

    default:
      return null;
  }
}

/** Every feature that can produce a link, in the order they should be offered. */
export const LINKABLE_FEATURES = [
  'EF-TTE',
  'Region RWMA',
  'LVH',
  'BBB',
  'VHD',
  'Systolic Murmur',
  'Diastolic Murmur',
] as const;

/**
 * This donor's posterior descending artery arises from the circumflex, i.e. a
 * left-dominant circulation, which is true of roughly 10-15% of people. Shown
 * wherever the LCX territory is described so the grouping is never a surprise.
 */
export const DOMINANCE_NOTE =
  'This anatomical model is left-dominant: its posterior descending artery arises from the circumflex, so it is grouped with the LCX. In about 85% of people it arises from the RCA instead.';
