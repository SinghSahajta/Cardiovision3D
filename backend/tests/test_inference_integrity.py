"""Integrity tests for the inference path.

These exist because an earlier version of this project served hand-written
sigmoid formulas while the trained LightGBM models sat unused on disk, and
reported SHAP values that were hardcoded multipliers. The tests below fail if
that ever happens again.

Run from the repo root::

    python -m pytest backend/tests -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services import CardiacInferenceEngine  # noqa: E402

TARGETS = ["CAD", "LAD", "LCX", "RCA"]
VESSELS = ["LAD", "LCX", "RCA"]
MODELS_DIR = REPO_ROOT / "backend" / "app" / "models"


@pytest.fixture(scope="module")
def engine() -> CardiacInferenceEngine:
    if not (MODELS_DIR / "cardiac_models.joblib").exists():
        pytest.skip("Artifacts missing - run `python ml/src/train.py` first.")
    return CardiacInferenceEngine(MODELS_DIR)


@pytest.fixture(scope="module")
def presets(engine: CardiacInferenceEngine) -> list[dict]:
    cases = engine.get_presets()
    assert cases, "demo_presets.json is empty"
    return cases


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------
def test_no_leakage_columns_in_feature_set(engine):
    """Requirement 1d: targets must never appear as model inputs."""
    forbidden = {"CAD", "Cath", "LAD", "LCX", "RCA"}
    assert not forbidden & set(engine.feature_order)
    assert len(engine.feature_order) == 55


def test_leakage_columns_are_stripped_from_payloads(engine, presets):
    """Even if a caller sends the answers, they must not reach the model."""
    case = presets[0]
    clean = engine.predict(case["features"])
    poisoned = dict(case["features"])
    poisoned.update({"Cath": "CAD", "LAD": "Stenotic", "LCX": "Stenotic", "RCA": "Stenotic"})
    assert engine.predict(poisoned)["cad"]["risk_score"] == clean["cad"]["risk_score"]


# --------------------------------------------------------------------------
# The engine actually serves the trained models
# --------------------------------------------------------------------------
def test_predictions_match_offline_pipeline(engine, presets):
    """The decisive test: served probability == pipeline.predict_proba on the same row.

    A hardcoded formula cannot pass this.
    """
    for case in presets:
        served = engine.predict(case["features"])
        frame = pd.DataFrame(
            [{c: case["features"].get(c) for c in engine.feature_order}],
            columns=engine.feature_order,
        )
        for target in TARGETS:
            offline = float(engine.models[target].predict_proba(frame)[0, 1])
            node = served["cad"] if target == "CAD" else served["vessels"][target]
            assert node["uncalibrated_score"] == pytest.approx(offline, abs=1e-4), (
                f"{case['patient_id']} / {target}: served "
                f"{node['uncalibrated_score']} != offline {offline}"
            )


def test_models_are_lightgbm_pipelines(engine):
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import Pipeline

    for target in TARGETS:
        pipe = engine.models[target]
        assert isinstance(pipe, Pipeline)
        assert isinstance(pipe.named_steps["clf"], LGBMClassifier)
        # Preprocessing must live inside the pipeline, not outside it.
        assert "prep" in pipe.named_steps


# --------------------------------------------------------------------------
# SHAP integrity
# --------------------------------------------------------------------------
def test_shap_values_are_additive(engine, presets):
    """TreeSHAP values must sum to the served booster's raw margin."""
    for case in presets:
        frame = pd.DataFrame(
            [{c: case["features"].get(c) for c in engine.feature_order}],
            columns=engine.feature_order,
        )
        for target in TARGETS:
            pipe = engine.models[target]
            transformed = pipe[:-1].transform(frame)
            booster = pipe.named_steps["clf"]
            margin = float(booster.predict(transformed, raw_score=True)[0])
            contrib = np.asarray(booster.predict(transformed, pred_contrib=True)).reshape(-1)

            assert contrib.sum() == pytest.approx(margin, abs=1e-6), (
                f"{case['patient_id']} / {target}: attributions do not reconcile"
            )


def test_engine_reports_the_same_base_value_as_the_booster(engine, presets):
    """shap_base_value must be the booster's own base value, not a hardcoded constant."""
    frame = pd.DataFrame(
        [{c: presets[0]["features"].get(c) for c in engine.feature_order}],
        columns=engine.feature_order,
    )
    served = engine.predict(presets[0]["features"])
    for target in TARGETS:
        pipe = engine.models[target]
        contrib = np.asarray(
            pipe.named_steps["clf"].predict(pipe[:-1].transform(frame), pred_contrib=True)
        ).reshape(-1)
        node = served["cad"] if target == "CAD" else served["vessels"][target]
        assert node["shap_base_value"] == pytest.approx(float(contrib[-1]), abs=1e-5)
        assert node["shap_base_value"] != 0.21, "hardcoded base value from the old engine"


def test_shap_values_are_not_constant_across_patients(engine, presets):
    """Hardcoded multipliers would produce identical attributions for every patient."""
    signatures = set()
    for case in presets:
        result = engine.predict(case["features"])
        top = result["cad"]["top_features"]
        signatures.add(tuple(round(f["shap_value"], 6) for f in top))
    assert len(signatures) > 1, "SHAP attributions identical across patients"


def test_reported_shap_sum_matches_returned_features(engine, presets):
    """shap_sum must be the real total, not just the sum of the top-k shown."""
    full = engine.predict(presets[0]["features"], top_k=40)
    partial = engine.predict(presets[0]["features"], top_k=3)
    assert full["cad"]["shap_sum"] == pytest.approx(partial["cad"]["shap_sum"], abs=1e-6)
    assert len(partial["cad"]["top_features"]) == 3


# --------------------------------------------------------------------------
# Clinical behaviour
# --------------------------------------------------------------------------
def test_three_vessel_case_scores_higher_than_zero_vessel_case(engine, presets):
    by_count = {c["vessels_involved"]: c for c in presets}
    if 0 not in by_count or 3 not in by_count:
        pytest.skip("presets do not cover both 0- and 3-vessel disease")
    low = engine.predict(by_count[0]["features"])
    high = engine.predict(by_count[3]["features"])
    assert high["cad"]["risk_score"] > low["cad"]["risk_score"]
    for vessel in VESSELS:
        assert high["vessels"][vessel]["risk_score"] > low["vessels"][vessel]["risk_score"], (
            f"{vessel}: three-vessel case did not score above the disease-free case"
        )


def test_thresholds_are_applied_consistently(engine, presets):
    for case in presets:
        result = engine.predict(case["features"])
        for vessel in VESSELS:
            node = result["vessels"][vessel]
            assert node["is_stenotic"] == (node["uncalibrated_score"] >= node["threshold"])


def test_response_shape_and_disclaimer(engine, presets):
    result = engine.predict(presets[0]["features"])
    assert set(result["vessels"]) == set(VESSELS)
    assert "decision support" in result["clinical_safety_disclaimer"].lower()
    cad = result["cad"]
    for key in ("risk_score", "vessel_derived_risk_score", "agrees_with_vessels", "threshold"):
        assert key in cad
    assert 0.0 <= cad["risk_score"] <= 1.0
    assert 0.0 <= cad["vessel_derived_risk_score"] <= 1.0


def test_missing_features_fall_back_to_training_defaults(engine):
    """An empty payload must still predict, and must say what it defaulted."""
    result = engine.predict({})
    assert len(result["defaulted_features"]) == len(engine.feature_order)
    assert 0.0 <= result["cad"]["risk_score"] <= 1.0


def test_underscored_aliases_are_accepted(engine, presets):
    """Older clients sent 'EF_TTE' instead of 'EF-TTE'; both must resolve."""
    case = presets[0]["features"]
    aliased = {k.replace(" ", "_").replace("-", "_"): v for k, v in case.items()}
    assert engine.predict(aliased)["cad"]["risk_score"] == pytest.approx(
        engine.predict(case)["cad"]["risk_score"], abs=1e-9
    )


# --------------------------------------------------------------------------
# Metrics provenance
# --------------------------------------------------------------------------
def test_served_metrics_come_from_disk():
    """The /metrics endpoint must not carry hardcoded numbers."""
    path = REPO_ROOT / "ml" / "results" / "cv_metrics.json"
    if not path.exists():
        pytest.skip("cv_metrics.json missing - run training first")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["target_leakage_prevented"] is True
    assert set(data["per_target"]) == set(TARGETS)

    source = (REPO_ROOT / "backend" / "app" / "main.py").read_text(encoding="utf-8")
    assert "roc_auc\": 0." not in source, "hardcoded metrics found in main.py"


def test_one_hot_columns_are_aggregated_to_their_source_feature(engine, presets):
    """A categorical must appear once, carrying the sum of its encoded columns.

    Reporting Nonanginal_N and Nonanginal_Y as separate rows both double-lists
    the same measurement and understates its influence, which matters because
    the dashboard ranks features by this number.
    """
    from collections import Counter

    for case in presets:
        result = engine.predict(case["features"], top_k=20)
        for name, node in [("CAD", result["cad"])] + list(result["vessels"].items()):
            feats = [f["feature"] for f in node["top_features"]]
            dupes = [f for f, n in Counter(feats).items() if n > 1]
            assert not dupes, f"{case['patient_id']} / {name}: duplicated {dupes}"
            # Every reported feature must be a real model input, not an encoded column.
            for f in feats:
                assert f in engine.feature_order, f"{f} is not a raw feature"


def test_aggregated_shap_matches_the_sum_of_its_encoded_columns(engine, presets):
    import numpy as np

    frame = pd.DataFrame(
        [{c: presets[-1]["features"].get(c) for c in engine.feature_order}],
        columns=engine.feature_order,
    )
    pipe = engine.models["CAD"]
    contrib = np.asarray(
        pipe.named_steps["clf"].predict(pipe[:-1].transform(frame), pred_contrib=True)
    ).reshape(-1)[:-1]

    expected: dict[str, float] = {}
    for i, name in enumerate(engine._processed_names[: len(contrib)]):
        expected.setdefault(engine._source_feature(name), 0.0)
        expected[engine._source_feature(name)] += float(contrib[i])

    for item in engine.predict(presets[-1]["features"], top_k=40)["cad"]["top_features"]:
        assert item["shap_value"] == pytest.approx(expected[item["feature"]], abs=1e-5)
