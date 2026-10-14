"""FastAPI application for the cardiovascular risk engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .schemas import (
    CohortIndexResponse,
    CohortPatientResponse,
    FeatureSchemaResponse,
    PatientFeatureInput,
    PredictionResponse,
    PresetPatientResponse,
)
from .services import ArtifactsMissingError, get_engine

REPO_ROOT = Path(__file__).resolve().parents[2]
CV_METRICS_PATH = REPO_ROOT / "ml" / "results" / "cv_metrics.json"

app = FastAPI(
    title=settings.app_name,
    description=(
        "Predicts overall CAD status and per-vessel (LAD/LCX/RCA) stenosis "
        "probability with SHAP attributions. Decision support only."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "status": "online",
        "service": settings.app_name,
        "documentation": "/docs",
        "endpoints": ["/health", "/metrics", "/presets", "/schema", "/predict"],
    }


@app.get("/health")
def health_check() -> dict:
    """Reports whether the trained artifacts actually loaded."""
    try:
        engine = get_engine()
    except ArtifactsMissingError as exc:
        return {"status": "degraded", "models_loaded": False, "detail": str(exc)}
    return {
        "status": "healthy",
        "models_loaded": True,
        "targets": list(engine.models.keys()),
        "n_features": len(engine.feature_order),
    }


@app.get("/metrics")
def get_validation_metrics() -> dict:
    """Serves the measured CV metrics straight from disk.

    Read from ``ml/results/cv_metrics.json`` rather than hardcoded, so the API
    can never report numbers that differ from what training actually measured.
    """
    if not CV_METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="No evaluation metrics found. Run `python ml/src/train.py` first.",
        )
    return json.loads(CV_METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/presets", response_model=List[PresetPatientResponse])
def get_preset_patients() -> list:
    """Real cohort rows covering 0-, 1-, 2- and 3-vessel disease."""
    try:
        return get_engine().get_presets()
    except ArtifactsMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/cohort", response_model=CohortIndexResponse)
def get_cohort_index() -> dict:
    """Index of all 303 training-cohort patients for the patient browser.

    Each summary carries the **out-of-fold** probability - from the fold where
    that patient was held out - alongside the angiographic truth. The served
    model was fitted on every one of these rows, so its live prediction for them
    is in-sample; the held-out figure is the one that means anything.
    """
    try:
        return get_engine().get_cohort_index()
    except ArtifactsMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/cohort/{patient_id}", response_model=CohortPatientResponse)
def get_cohort_patient(patient_id: str) -> dict:
    """One cohort patient's 55 features, truth and held-out probabilities.

    Accepts either the id (``ROW006``) or the 1-based index (``6``), the same
    way ``ml/src/validate.py --patient N`` addresses rows.
    """
    try:
        patient = get_engine().get_cohort_patient(patient_id)
    except ArtifactsMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if patient is None:
        raise HTTPException(status_code=404, detail=f"No cohort patient {patient_id!r}")
    return patient


@app.get("/schema", response_model=FeatureSchemaResponse)
def get_feature_schema() -> dict:
    """Feature definitions and defaults so a client can build its form from the model."""
    try:
        return get_engine().get_feature_schema()
    except ArtifactsMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict_cardiac_risk(payload: PatientFeatureInput) -> dict:
    """CAD and per-vessel stenosis risk with real SHAP attributions."""
    try:
        engine = get_engine()
    except ArtifactsMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        return engine.predict(payload.features, top_k=payload.top_k)
    except Exception as exc:  # noqa: BLE001 - surface inference faults as 400
        raise HTTPException(status_code=400, detail=f"Inference error: {exc}") from exc
