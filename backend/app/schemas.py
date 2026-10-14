"""Pydantic request/response schemas.

These mirror exactly what ``CardiacInferenceEngine.predict`` returns. The engine
is the single source of truth; if the two ever drift, FastAPI's response
validation fails loudly at request time instead of serving a silent mismatch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PatientFeatureInput(BaseModel):
    """Raw clinical features keyed exactly as the dataset spells them.

    Example: ``{"features": {"Age": 58, "EF-TTE": 45, "Typical Chest Pain": 1}}``.
    Unknown keys are ignored and omitted features fall back to the training
    median/mode, which is reported back in ``defaulted_features``.
    """

    features: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=40)


class FeatureAttribution(BaseModel):
    feature: str
    display_name: str
    processed_feature: str
    feature_value: Any = None
    shap_value: float
    impact: str


class VesselPrediction(BaseModel):
    vessel: str
    risk_score: float
    uncalibrated_score: float
    threshold: float
    is_stenotic: bool
    status: str
    risk_level: str
    color_hex: str
    shap_base_value: float
    shap_sum: float
    top_features: List[FeatureAttribution]


class CadPrediction(BaseModel):
    risk_score: float
    uncalibrated_score: float
    threshold: float
    has_cad: bool
    status: str
    risk_level: str
    color_hex: str
    vessel_derived_risk_score: float
    agrees_with_vessels: bool
    shap_base_value: float
    shap_sum: float
    top_features: List[FeatureAttribution]


class PredictionResponse(BaseModel):
    patient_id: Optional[str] = "CUSTOM_PATIENT"
    cad: CadPrediction
    vessels: Dict[str, VesselPrediction]
    defaulted_features: List[str] = Field(default_factory=list)
    model_metadata: Dict[str, Any] = Field(default_factory=dict)
    clinical_safety_disclaimer: str


class PresetPatientResponse(BaseModel):
    patient_id: str
    label: str
    vessels_involved: int
    in_sample: bool
    features: Dict[str, Any]
    ground_truth: Dict[str, int]


class CohortSummary(BaseModel):
    """One row in the patient browser's index - enough to pick with, no features."""

    patient_id: str
    index: int
    age: Optional[float] = None
    sex: Optional[str] = None
    vessels_involved: int
    ground_truth: Dict[str, int]
    held_out: Dict[str, float]
    # Whether the held-out model got the overall CAD call right for this patient,
    # precomputed so the browser can filter to the model's mistakes without the
    # client having to know the decision threshold.
    held_out_correct_cad: bool


class CohortIndexResponse(BaseModel):
    n_patients: int
    dataset: str
    note: str
    cad_threshold: float
    patients: List[CohortSummary]


class CohortPatientResponse(BaseModel):
    patient_id: str
    index: int
    features: Dict[str, Any]
    ground_truth: Dict[str, int]
    vessels_involved: int
    held_out: Dict[str, float]
    held_out_derived_cad: float
    # Always true for these rows, and stated rather than implied: the served model
    # was fitted on every one of them.
    in_sample: bool = True


class FeatureSchemaResponse(BaseModel):
    feature_order: List[str]
    numeric_features: List[str]
    categorical_levels: Dict[str, List[Any]]
    defaults: Dict[str, Any]
    display_names: Dict[str, str]
