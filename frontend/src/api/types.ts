/**
 * Mirrors backend/app/schemas.py exactly.
 *
 * Vessel keys are UPPERCASE ("LAD" | "LCX" | "RCA") because that is what the API
 * emits. The previous frontend assumed lowercase and silently rendered nothing;
 * these types exist so `tsc` catches that class of drift at build time.
 */

export type VesselKey = 'LAD' | 'LCX' | 'RCA';
export type TargetKey = 'CAD' | VesselKey;

export const VESSELS: readonly VesselKey[] = ['LAD', 'LCX', 'RCA'] as const;

export interface FeatureAttribution {
  feature: string;
  display_name: string;
  processed_feature: string;
  feature_value: string | number | null;
  /** TreeSHAP value in raw log-odds margin space, NOT probability points. */
  shap_value: number;
  impact: 'Increases Risk' | 'Decreases Risk';
}

export interface VesselPrediction {
  vessel: VesselKey;
  /** Calibrated probability - what we display. */
  risk_score: number;
  /** Raw model probability - what the threshold is applied to. */
  uncalibrated_score: number;
  threshold: number;
  is_stenotic: boolean;
  status: string;
  risk_level: 'Low' | 'Moderate' | 'High';
  color_hex: string;
  shap_base_value: number;
  shap_sum: number;
  top_features: FeatureAttribution[];
}

export interface CadPrediction {
  risk_score: number;
  uncalibrated_score: number;
  threshold: number;
  has_cad: boolean;
  status: string;
  risk_level: 'Low' | 'Moderate' | 'High';
  color_hex: string;
  /** Noisy-OR over the three vessels, calibrated. */
  vessel_derived_risk_score: number;
  agrees_with_vessels: boolean;
  shap_base_value: number;
  shap_sum: number;
  top_features: FeatureAttribution[];
}

export interface PredictionResponse {
  patient_id?: string;
  cad: CadPrediction;
  vessels: Record<VesselKey, VesselPrediction>;
  /** Features the caller omitted, filled from training median/mode. */
  defaulted_features: string[];
  model_metadata: {
    dataset?: string;
    cohort_size?: number;
    n_features?: number;
    validation?: string;
    excluded_leakage_columns?: string[];
    shap_space?: string;
  };
  clinical_safety_disclaimer: string;
}

export interface PresetPatient {
  patient_id: string;
  label: string;
  vessels_involved: number;
  in_sample: boolean;
  features: Record<string, string | number | null>;
  ground_truth: Record<TargetKey, number>;
}

export interface FeatureSchema {
  feature_order: string[];
  numeric_features: string[];
  categorical_levels: Record<string, (string | number)[]>;
  defaults: Record<string, string | number>;
  display_names: Record<string, string>;
}

/** Shape of ml/results/cv_metrics.json, served by GET /metrics. */
export interface MetricsResponse {
  dataset: string;
  cohort_size: number;
  validation_strategy: string;
  threshold_selection: string;
  target_leakage_prevented: boolean;
  excluded_features: string[];
  n_features: number;
  per_target: Record<
    TargetKey,
    {
      'at_threshold_0.5': Record<string, string>;
      at_tuned_threshold: Record<string, string>;
      threshold_independent: { roc_auc: string; pr_auc: string; brier: string };
      tuned_threshold_mean: number;
      positive_prevalence: number;
      class_weight: string;
      calibration: {
        method: string;
        brier_uncalibrated: number;
        brier_calibrated_cv: number;
        note?: string;
      };
    }
  >;
  cad_coherence: Record<string, number | string>;
}

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  targets?: TargetKey[];
  n_features?: number;
  detail?: string;
}

/**
 * One row of the training cohort, as the patient browser lists them.
 *
 * `held_out` is the probability from the cross-validation fold in which this
 * patient was excluded, so it comes from a model that never saw them. The served
 * model WAS fitted on every one of these rows, so its live prediction for them is
 * in-sample. Both are shown, labelled; see tools/build_cohort.py.
 */
export interface CohortSummary {
  patient_id: string;
  index: number;
  age: number | null;
  sex: string | null;
  vessels_involved: number;
  ground_truth: Record<TargetKey, number>;
  held_out: Record<TargetKey, number>;
  held_out_correct_cad: boolean;
}

export interface CohortIndex {
  n_patients: number;
  dataset: string;
  note: string;
  cad_threshold: number;
  patients: CohortSummary[];
}

export interface CohortPatient {
  patient_id: string;
  index: number;
  features: Record<string, string | number | null>;
  ground_truth: Record<TargetKey, number>;
  vessels_involved: number;
  held_out: Record<TargetKey, number>;
  held_out_derived_cad: number;
  in_sample: boolean;
}

/** frontend/public/models/vessels.json, written by tools/build_anatomy.py. */

/** Which structural family a node belongs to; decides how it is materialised. */
export type AnatomyGroup =
  | 'myocardium'
  | 'atrium'
  | 'great_vessel'
  | 'artery'
  | 'valve'
  | 'conduction';

/** "core" is always drawn; the others are user-toggled layers. */
export type AnatomyLayer = 'core' | 'valves' | 'conduction';

/**
 * One named node inside anatomy.glb. There is no geometry here - the mesh lives
 * in the GLB and is BodyParts3D's own, not reconstructed. This is the provenance
 * and placement record for it.
 */
export interface AnatomyNode {
  label: string;
  anatomical_name: string;
  group: AnatomyGroup;
  layer: AnatomyLayer;
  /** Set only on the three arteries the model predicts. */
  vessel: VesselKey | null;
  fma_ids: string[];
  source_files: string[];
  triangles: number;
  anchor: [number, number, number];
  centroid: [number, number, number];
  radius: number;
}

export interface AnatomyManifest {
  attribution: string;
  source: string;
  note: string;
  normalisation: Record<string, unknown>;
  /** Fraction of calibre lost at full morph influence. */
  narrow_fraction: number;
  vessel_nodes: Record<VesselKey, string>;
  nodes: Record<string, AnatomyNode>;
}
