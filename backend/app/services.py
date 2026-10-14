"""Cardiac inference engine.

Loads the artifacts produced by ``ml/src/train.py`` and serves real predictions:
calibrated probabilities from the trained LightGBM pipelines, tuned per-vessel
decision thresholds, and genuine TreeSHAP attributions.

Attributions come from LightGBM's native ``pred_contrib=True``, the same
Lundberg et al. TreeSHAP algorithm ``shap.TreeExplainer`` wraps for LightGBM
models, computed by the booster that actually serves the prediction. They are
exactly additive: contributions plus ``shap_base_value`` equal the raw margin.

The displayed ``risk_score`` is a calibrated probability obtained by applying a
fitted monotonic map to that margin's probability, so the ranking the
attributions explain is the ranking displayed.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Never let these reach the model, whatever the caller sends (requirement 1d).
TARGET_LEAKAGE_COLUMNS = {"CAD", "Cath", "LAD", "LCX", "RCA"}

VESSELS = ("LAD", "LCX", "RCA")

DISCLAIMER = (
    "CLINICAL SAFETY NOTICE: This system provides statistical risk estimates for "
    "clinical decision support and medical education only. It is not a diagnostic "
    "device and must not replace invasive coronary angiography, diagnostic cardiac "
    "imaging, or physician judgement."
)

MODELS_DIR = Path(__file__).resolve().parent / "models"

REQUIRED_ARTIFACTS = (
    "cardiac_models.joblib",
    "calibration.joblib",
    "metadata.json",
    "thresholds.json",
)


class ArtifactsMissingError(RuntimeError):
    """Raised when the trained model artifacts are absent or incomplete."""


class CardiacInferenceEngine:
    """Serves CAD and per-vessel stenosis predictions from the trained pipelines."""

    def __init__(self, models_dir: Path | str = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self.disclaimer = DISCLAIMER

        missing = [n for n in REQUIRED_ARTIFACTS if not (self.models_dir / n).exists()]
        if missing:
            raise ArtifactsMissingError(
                f"Missing model artifacts in {self.models_dir}: {', '.join(missing)}. "
                "Run `python ml/src/train.py` before starting the API."
            )

        self.models = joblib.load(self.models_dir / "cardiac_models.joblib")
        self.calibrators = joblib.load(self.models_dir / "calibration.joblib")
        self.metadata = json.loads((self.models_dir / "metadata.json").read_text(encoding="utf-8"))
        self.thresholds = json.loads((self.models_dir / "thresholds.json").read_text(encoding="utf-8"))

        self.feature_order: list[str] = self.metadata["raw_features"]
        self.numeric_features = set(self.metadata["numeric_features"])
        self.defaults: dict[str, Any] = self.metadata["feature_defaults"]
        self.display_names: dict[str, str] = self.metadata.get("display_names", {})

        leaked = TARGET_LEAKAGE_COLUMNS.intersection(self.feature_order)
        if leaked:
            raise ArtifactsMissingError(
                f"Artifact metadata contains leakage columns {sorted(leaked)}; retrain."
            )

        self._processed_names = self._resolve_processed_names()
        self._cohort: dict | None = None

    # -- helpers ----------------------------------------------------------
    def _resolve_processed_names(self) -> list[str]:
        names = self.metadata.get("processed_feature_names") or []
        if names:
            return list(names)
        try:
            return list(self.models["CAD"][:-1].get_feature_names_out())
        except Exception:
            return []

    def _build_frame(self, raw: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
        """Coerce an arbitrary payload into one row with the training column order."""
        sanitized = {k: v for k, v in raw.items() if k not in TARGET_LEAKAGE_COLUMNS}
        # Tolerate underscored aliases ("EF_TTE" -> "EF-TTE") from older clients.
        alias = {k.replace(" ", "_").replace("-", "_"): k for k in self.feature_order}
        normalised: dict[str, Any] = {}
        for key, value in sanitized.items():
            canonical = key if key in self.feature_order else alias.get(
                key.replace(" ", "_").replace("-", "_")
            )
            if canonical:
                normalised[canonical] = value

        row: dict[str, Any] = {}
        defaulted: list[str] = []
        for col in self.feature_order:
            value = normalised.get(col, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                row[col] = self.defaults.get(col)
                defaulted.append(col)
            elif col in self.numeric_features:
                try:
                    row[col] = float(value)
                except (TypeError, ValueError):
                    row[col] = self.defaults.get(col)
                    defaulted.append(col)
            else:
                row[col] = str(value).strip()

        return pd.DataFrame([row], columns=self.feature_order), defaulted

    def _calibrate(self, target: str, prob: float) -> float:
        """Apply the fitted monotonic calibration map, if training kept one."""
        entry = self.calibrators.get(target)
        if not entry:
            return prob
        model, method = entry["model"], entry["method"]
        if method == "identity" or model is None:
            return prob
        arr = np.array([[prob]]) if method == "sigmoid" else np.array([prob])
        try:
            out = model.predict_proba(arr)[:, 1][0] if method == "sigmoid" else model.predict(arr)[0]
        except Exception:
            return prob
        return float(np.clip(out, 0.0, 1.0))

    def _attributions(self, target: str, frame: pd.DataFrame, top_k: int) -> tuple[list[dict], float, float]:
        """TreeSHAP values from the served booster, in raw log-odds margin space.

        ``pred_contrib=True`` returns one column per transformed feature plus a
        final column holding the base value, and the row sums to the raw margin.
        """
        pipe = self.models[target]
        transformed = pipe[:-1].transform(frame)

        contrib = np.asarray(
            pipe.named_steps["clf"].predict(transformed, pred_contrib=True)
        ).reshape(-1)
        contributions = contrib[:-1]
        expected = float(contrib[-1])

        names = self._processed_names
        if len(names) != len(contributions):
            names = [f"feature_{i}" for i in range(len(contributions))]

        raw_values = frame.iloc[0].to_dict()

        # Aggregate one-hot columns back onto the clinical feature they came
        # from. A categorical like "Nonanginal" is split into Nonanginal_N and
        # Nonanginal_Y, and reporting those separately both double-lists the
        # same measurement and understates its real influence. Summing is the
        # correct reduction: SHAP values are additive, so the group total is the
        # feature's true contribution to the margin.
        totals: dict[str, float] = {}
        members: dict[str, list[str]] = {}
        for i, value in enumerate(contributions):
            source = self._source_feature(names[i]) if i < len(names) else f"feature_{i}"
            totals[source] = totals.get(source, 0.0) + float(value)
            members.setdefault(source, []).append(names[i] if i < len(names) else "")

        ranked = sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]

        items = []
        for source, value in ranked:
            parts = members[source]
            items.append(
                {
                    "feature": source,
                    "display_name": self.display_names.get(source, source),
                    # Keep the underlying column(s) so a reviewer can trace the
                    # number back to the model's actual inputs.
                    "processed_feature": parts[0] if len(parts) == 1
                    else f"{source} ({len(parts)} encoded columns)",
                    "feature_value": raw_values.get(source),
                    "shap_value": round(float(value), 6),
                    "impact": "Increases Risk" if value > 0 else "Decreases Risk",
                }
            )
        return items, expected, float(contributions.sum())

    def _source_feature(self, processed: str) -> str:
        """Map a transformed column name back to its original clinical feature."""
        name = processed.split("__", 1)[-1] if "__" in processed else processed
        if name in self.feature_order:
            return name
        # One-hot columns are "<feature>_<level>"; match the longest real feature.
        candidates = [f for f in self.feature_order if name.startswith(f)]
        return max(candidates, key=len) if candidates else name

    @staticmethod
    def _risk_tier(prob: float) -> str:
        if prob >= 0.65:
            return "High"
        if prob >= 0.35:
            return "Moderate"
        return "Low"

    @staticmethod
    def _color(prob: float) -> str:
        if prob >= 0.65:
            return "#dc2626"
        if prob >= 0.35:
            return "#f59e0b"
        return "#16a34a"

    # -- public API -------------------------------------------------------
    def predict(self, raw: dict[str, Any], top_k: int = 10) -> dict[str, Any]:
        frame, defaulted = self._build_frame(raw)

        vessels: dict[str, dict] = {}
        for vessel in VESSELS:
            prob_raw = float(self.models[vessel].predict_proba(frame)[0, 1])
            prob = self._calibrate(vessel, prob_raw)
            threshold = float(self.thresholds[vessel]["tuned"])
            features, base, total = self._attributions(vessel, frame, top_k)
            vessels[vessel] = {
                "vessel": vessel,
                "risk_score": round(prob, 4),
                "uncalibrated_score": round(prob_raw, 4),
                "threshold": round(threshold, 4),
                "is_stenotic": bool(prob_raw >= threshold),
                "status": "Stenosis predicted (>=50% lumen)" if prob_raw >= threshold
                else "No significant stenosis predicted",
                "risk_level": self._risk_tier(prob),
                "color_hex": self._color(prob),
                "shap_base_value": round(base, 6),
                "shap_sum": round(total, 6),
                "top_features": features,
            }

        cad_raw = float(self.models["CAD"].predict_proba(frame)[0, 1])
        cad_prob = self._calibrate("CAD", cad_raw)
        cad_threshold = float(self.thresholds["CAD"]["tuned"])
        cad_features, cad_base, cad_total = self._attributions("CAD", frame, top_k)

        # Vessel-derived CAD probability (noisy-OR). Computed from the *uncalibrated*
        # vessel scores to match how training fit its calibrator, then calibrated --
        # the independence assumption ranks well but inflates the raw scale, since
        # real vessel stenoses are correlated through shared risk factors.
        derived_raw = 1.0 - float(
            np.prod([1.0 - vessels[v]["uncalibrated_score"] for v in VESSELS])
        )
        derived = self._calibrate("CAD_DERIVED", derived_raw)
        direct_positive = cad_raw >= cad_threshold
        derived_positive = derived >= 0.5

        return {
            "patient_id": raw.get("patient_id", "CUSTOM_PATIENT"),
            "cad": {
                "risk_score": round(cad_prob, 4),
                "uncalibrated_score": round(cad_raw, 4),
                "threshold": round(cad_threshold, 4),
                "has_cad": bool(direct_positive),
                "status": "CAD Positive" if direct_positive else "CAD Negative",
                "risk_level": self._risk_tier(cad_prob),
                "color_hex": self._color(cad_prob),
                "vessel_derived_risk_score": round(derived, 4),
                "agrees_with_vessels": bool(direct_positive == derived_positive),
                "shap_base_value": round(cad_base, 6),
                "shap_sum": round(cad_total, 6),
                "top_features": cad_features,
            },
            "vessels": vessels,
            "defaulted_features": defaulted,
            "model_metadata": {
                "dataset": self.metadata.get("dataset"),
                "cohort_size": self.metadata.get("n_patients"),
                "n_features": len(self.feature_order),
                "validation": self.metadata.get("validation"),
                "excluded_leakage_columns": self.metadata.get("excluded_leakage_columns"),
                "shap_space": "raw log-odds margin of the served LightGBM booster",
            },
            "clinical_safety_disclaimer": self.disclaimer,
        }

    def get_presets(self) -> list[dict]:
        path = self.models_dir / "demo_presets.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def cohort(self) -> dict:
        """The exported training cohort, loaded once and cached.

        Absent until ``python tools/build_cohort.py`` has run; the routes turn
        that into a 503 with that instruction rather than an empty list, because
        an empty patient browser looks like a cohort of zero patients.
        """
        if self._cohort is None:
            path = self.models_dir / "cohort.json"
            if not path.exists():
                raise ArtifactsMissingError(
                    f"Cohort export missing at {path}. "
                    "Run `python tools/build_cohort.py`."
                )
            self._cohort = json.loads(path.read_text(encoding="utf-8"))
        return self._cohort

    def get_cohort_index(self) -> dict:
        """Summaries for the patient browser, without the 55 features per row."""
        data = self.cohort
        threshold = float(self.thresholds["CAD"]["tuned"])
        patients = []
        for row in data["patients"]:
            features = row["features"]
            held = row["held_out"]
            patients.append({
                "patient_id": row["patient_id"],
                "index": row["index"],
                "age": features.get("Age"),
                "sex": features.get("Sex"),
                "vessels_involved": row["vessels_involved"],
                "ground_truth": row["ground_truth"],
                "held_out": held,
                # Compared at the model's own tuned threshold, the same one the
                # live prediction is called against, so "got it wrong" in the UI
                # means the same thing in both places.
                "held_out_correct_cad":
                    bool((held["CAD"] >= threshold) == bool(row["ground_truth"]["CAD"])),
            })
        return {
            "n_patients": data["n_patients"],
            "dataset": data["dataset"],
            "note": data["note"],
            "cad_threshold": round(threshold, 4),
            "patients": patients,
        }

    def get_cohort_patient(self, patient_id: str) -> dict | None:
        """One cohort row, by patient id or by 1-based index as a string."""
        wanted = patient_id.strip().upper()
        for row in self.cohort["patients"]:
            if row["patient_id"].upper() == wanted or str(row["index"]) == wanted:
                return {**row, "in_sample": True}
        return None

    def get_feature_schema(self) -> dict[str, Any]:
        """Field definitions so the UI can build its form from the real model."""
        return {
            "feature_order": self.feature_order,
            "numeric_features": sorted(self.numeric_features),
            "categorical_levels": self.metadata.get("categorical_levels", {}),
            "defaults": self.defaults,
            "display_names": self.display_names,
        }


@lru_cache(maxsize=1)
def get_engine() -> CardiacInferenceEngine:
    """Load the engine once per process."""
    return CardiacInferenceEngine()
