"""Train CAD and per-vessel stenosis classifiers on the Z-Alizadeh Sani dataset.

Design notes
------------
* All preprocessing lives *inside* an sklearn Pipeline, so the ColumnTransformer is
  fit separately on every CV training fold. Nothing about a test fold -- not even a
  column mean -- informs the transform applied to it.
* Decision thresholds are chosen on *inner* folds only, so the reported metrics at
  the tuned threshold are not scored on the data that picked the threshold.
* Attributions use LightGBM's native TreeSHAP (``pred_contrib=True``), which is the
  same Lundberg et al. algorithm ``shap.TreeExplainer`` wraps for LightGBM models and
  is exactly additive: contributions plus the base value equal the raw margin.
* Calibration is a separate 1-D monotonic map fit on out-of-fold margins, rather
  than a CalibratedClassifierCV wrapper. That keeps a single LightGBM estimator as
  the thing both served and explained: SHAP values sum exactly to its raw margin,
  and the calibrated probability is a documented monotonic function of that margin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGETS = ["CAD", "LAD", "LCX", "RCA"]
VESSELS = ["LAD", "LCX", "RCA"]

# Columns that must never reach the feature matrix (requirement 1d).
LEAKAGE_COLUMNS = {"CAD", "Cath", "LAD", "LCX", "RCA"}

# Class reweighting is kept only where it actually helps under nested CV. Both LCX
# and RCA have a minority positive class, but reweighting improves LCX marginally
# and makes RCA worse on every metric (see notebook 02). The differences sit well
# inside one standard deviation, so this is a weak preference, not a strong result.
CLASS_WEIGHTED_TARGETS = {"LCX"}

DISPLAY_NAMES = {
    "Age": "Patient Age",
    "Typical Chest Pain": "Typical Anginal Pain",
    "EF-TTE": "Left Ventricular EF",
    "Region RWMA": "Regional Wall Motion Abnormality",
    "BP": "Systolic Blood Pressure",
    "PR": "Pulse Rate",
    "LDL": "Serum LDL Cholesterol",
    "HDL": "Serum HDL Cholesterol",
    "TG": "Triglycerides",
    "FBS": "Fasting Blood Sugar",
    "DM": "Diabetes Mellitus",
    "HTN": "Hypertension",
    "Tinversion": "ECG T-Wave Inversion",
    "St Elevation": "ECG ST-Segment Elevation",
    "St Depression": "ECG ST-Segment Depression",
    "Q Wave": "ECG Pathological Q Wave",
    "LVH": "Left Ventricular Hypertrophy",
    "BBB": "Bundle Branch Block",
    "BMI": "Body Mass Index",
    "FH": "Family History of CAD",
    "Current Smoker": "Current Smoker",
    "EX-Smoker": "Former Smoker",
    "VHD": "Valvular Heart Disease",
    "CR": "Serum Creatinine",
    "BUN": "Blood Urea Nitrogen",
    "ESR": "Erythrocyte Sedimentation Rate",
    "HB": "Haemoglobin",
    "WBC": "White Blood Cell Count",
    "PLT": "Platelet Count",
    "Neut": "Neutrophil %",
    "Lymph": "Lymphocyte %",
    "Function Class": "NYHA Function Class",
    "CRF": "Chronic Renal Failure",
    "CVA": "Cerebrovascular Accident",
    "CHF": "Congestive Heart Failure",
    "DLP": "Dyslipidaemia",
    "Length": "Height (cm)",
}


# ---------------------------------------------------------------------------
# Data loading and target extraction
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    print(f"[*] Reading dataset: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset extension: {path.suffix}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def extract_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Binarise the four targets and strip every leakage column from the features."""
    present = {c.strip().lower(): c for c in df.columns}

    # 'Cath' is the overall-CAD target in this dataset. Fail loudly rather than
    # silently overwriting if a dataset ever carries both spellings.
    if "cath" in present and "cad" in present:
        raise ValueError(
            "Dataset contains both 'Cath' and 'CAD' columns; cannot infer the "
            "overall-CAD target unambiguously. Rename one of them."
        )
    cad_source = present.get("cath") or present.get("cad")
    if cad_source is None:
        raise KeyError(f"No overall-CAD target ('Cath'/'CAD') found. Columns: {list(df.columns)}")

    rename = {cad_source: "CAD"}
    for vessel in VESSELS:
        if vessel.lower() not in present:
            raise KeyError(f"Vessel target '{vessel}' missing. Columns: {list(df.columns)}")
        rename[present[vessel.lower()]] = vessel
    df = df.rename(columns=rename)

    positive_tokens = {"stenotic", "cad", "yes", "y", "1", "true", "positive"}
    negative_tokens = {"normal", "no", "n", "0", "false", "negative"}

    def to_binary(series: pd.Series, name: str) -> np.ndarray:
        if series.isna().any():
            raise ValueError(
                f"Target '{name}' has {int(series.isna().sum())} missing values. "
                "Refusing to guess a label; clean the dataset first."
            )
        out = []
        for raw in series:
            token = str(raw).strip().lower()
            if token in positive_tokens:
                out.append(1)
            elif token in negative_tokens:
                out.append(0)
            else:
                raise ValueError(f"Unrecognised value {raw!r} in target '{name}'.")
        return np.asarray(out, dtype=int)

    y = {t: to_binary(df[t], t) for t in TARGETS}

    feature_cols = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include=["object", "str"]).columns:
        X[col] = X[col].astype(str).str.strip()

    # Hard leakage assertion (verification step 2).
    leaked = LEAKAGE_COLUMNS.intersection(X.columns)
    if leaked:
        raise AssertionError(f"TARGET LEAKAGE: {sorted(leaked)} present in feature matrix.")

    print(f"[OK] Leakage guard passed - dropped {sorted(LEAKAGE_COLUMNS & set(df.columns))}")
    print(f"[OK] Feature matrix: {X.shape[1]} features x {X.shape[0]} patients")
    for t in TARGETS:
        print(f"     {t:4s} positives: {y[t].sum():3d}/{len(y[t])} ({y[t].mean():.1%})")
    return X, y


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_pipeline(X: pd.DataFrame, target: str, seed: int) -> Pipeline:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    classifier = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=10,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced" if target in CLASS_WEIGHTED_TARGETS else None,
        random_state=seed,
        verbosity=-1,
    )

    return Pipeline(
        [
            ("prep", preprocessor),
            # Drops constant columns such as 'Exertional CP' (all 'N' in this cohort).
            ("variance", VarianceThreshold(threshold=0.0)),
            ("clf", classifier),
        ]
    )


def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Youden's J on out-of-fold scores: maximises sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    best = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):  # roc_curve prepends +inf
        finite = thresholds[np.isfinite(thresholds)]
        threshold = float(finite[0]) if len(finite) else 0.5
    return threshold


def score_at(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------
def nested_cv(X: pd.DataFrame, y: np.ndarray, target: str, folds: int, seed: int) -> dict:
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    per_fold = {"default": [], "tuned": []}
    ranking = {"roc_auc": [], "pr_auc": [], "brier": []}
    chosen_thresholds = []
    oof_prob = np.zeros(len(y))

    for train_idx, test_idx in outer.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # --- inner loop: threshold selection on the training fold only ---
        inner = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 1)
        inner_prob = np.zeros(len(y_tr))
        for in_tr, in_va in inner.split(X_tr, y_tr):
            pipe = build_pipeline(X, target, seed)
            pipe.fit(X_tr.iloc[in_tr], y_tr[in_tr])
            inner_prob[in_va] = pipe.predict_proba(X_tr.iloc[in_va])[:, 1]
        threshold = pick_threshold(y_tr, inner_prob)
        chosen_thresholds.append(threshold)

        # --- outer evaluation ---
        pipe = build_pipeline(X, target, seed)
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        oof_prob[test_idx] = prob

        per_fold["default"].append(score_at(y_te, prob, 0.5))
        per_fold["tuned"].append(score_at(y_te, prob, threshold))
        ranking["roc_auc"].append(roc_auc_score(y_te, prob))
        ranking["pr_auc"].append(average_precision_score(y_te, prob))
        ranking["brier"].append(brier_score_loss(y_te, prob))

    def agg(rows: list[dict]) -> dict[str, str]:
        return {
            k: f"{np.mean([r[k] for r in rows]):.4f} ± {np.std([r[k] for r in rows]):.4f}"
            for k in rows[0]
        }

    summary = {
        "at_threshold_0.5": agg(per_fold["default"]),
        "at_tuned_threshold": agg(per_fold["tuned"]),
        "threshold_independent": {
            k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in ranking.items()
        },
        "tuned_threshold_mean": round(float(np.mean(chosen_thresholds)), 4),
        "tuned_threshold_per_fold": [round(t, 4) for t in chosen_thresholds],
        "positive_prevalence": round(float(y.mean()), 4),
        "class_weight": "balanced" if target in CLASS_WEIGHTED_TARGETS else "none",
    }

    print(f"\n===== {target} (prevalence {y.mean():.1%}) =====")
    for k, v in summary["threshold_independent"].items():
        print(f"  {k.upper():<10}: {v}")
    print(f"  {'-- at 0.5':<10}: " + "  ".join(f"{k}={v}" for k, v in summary["at_threshold_0.5"].items()))
    print(f"  {'-- tuned':<10}: " + "  ".join(f"{k}={v}" for k, v in summary["at_tuned_threshold"].items()))
    print(f"  threshold  : {summary['tuned_threshold_mean']:.4f} (mean of inner-fold choices)")

    return {"summary": summary, "oof_prob": oof_prob, "thresholds": chosen_thresholds}


def _new_platt() -> LogisticRegression:
    return LogisticRegression(C=1e6, solver="lbfgs")


def _new_isotonic() -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)


def _fit_1d(model: object, scores: np.ndarray, y: np.ndarray) -> object:
    if isinstance(model, IsotonicRegression):
        model.fit(scores, y)
    else:
        model.fit(scores.reshape(-1, 1), y)
    return model


def _predict_1d(model: object, scores: np.ndarray) -> np.ndarray:
    if isinstance(model, IsotonicRegression):
        return np.asarray(model.predict(scores))
    return np.asarray(model.predict_proba(scores.reshape(-1, 1))[:, 1])


def fit_calibrator(
    y: np.ndarray, oof_prob: np.ndarray, folds: int, seed: int
) -> tuple[object, str, float, float]:
    """Fit a 1-D monotonic probability calibrator on out-of-fold predictions.

    The method (Platt vs isotonic) is chosen by cross-validating *on the OOF
    scores*, so the reported calibrated Brier is not scored on the same data that
    fit the calibrator. Isotonic is flexible enough to look good in-sample at
    N=303, which would otherwise make it win every comparison spuriously.
    """
    raw_brier = brier_score_loss(y, oof_prob)

    def cv_brier(factory) -> float:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 7)
        held_out = np.zeros(len(y))
        for tr, te in splitter.split(oof_prob.reshape(-1, 1), y):
            model = _fit_1d(factory(), oof_prob[tr], y[tr])
            held_out[te] = _predict_1d(model, oof_prob[te])
        return brier_score_loss(y, np.clip(held_out, 0.0, 1.0))

    candidates = {
        "sigmoid": cv_brier(_new_platt),
        "isotonic": cv_brier(_new_isotonic),
    }
    best = min(candidates, key=candidates.get)

    # LightGBM is sometimes already well calibrated (notably for CAD, whose positive
    # class is the majority). Applying a map that does not improve the held-out Brier
    # would add a moving part for nothing, so fall back to the identity.
    if raw_brier <= candidates[best]:
        return None, "identity", raw_brier, raw_brier

    factory = _new_isotonic if best == "isotonic" else _new_platt
    return _fit_1d(factory(), oof_prob, y), best, raw_brier, candidates[best]


# ---------------------------------------------------------------------------
# Demo presets
# ---------------------------------------------------------------------------
def select_presets(X: pd.DataFrame, y: dict[str, np.ndarray]) -> list[dict]:
    """Pick one patient for each vessel-involvement count (0, 1, 2, 3)."""
    vessel_count = y["LAD"] + y["LCX"] + y["RCA"]
    labels = {
        0: "No significant vessel disease",
        1: "Single-vessel disease",
        2: "Two-vessel disease",
        3: "Three-vessel disease",
    }
    presets = []
    for count in (0, 1, 2, 3):
        matches = np.where(vessel_count == count)[0]
        if len(matches) == 0:
            continue
        idx = int(matches[0])
        row = X.iloc[idx]
        features = {}
        for col in X.columns:
            val = row[col]
            if pd.isna(val):
                features[col] = None
            elif isinstance(val, (np.integer,)):
                features[col] = int(val)
            elif isinstance(val, (np.floating,)):
                features[col] = float(val)
            elif isinstance(val, (int, float)):
                features[col] = val
            else:
                features[col] = str(val).strip()
        presets.append(
            {
                "patient_id": f"CASE_{count}V_ROW{idx + 1:03d}",
                "label": labels[count],
                "vessels_involved": count,
                "in_sample": True,
                "features": features,
                "ground_truth": {t: int(y[t][idx]) for t in TARGETS},
            }
        )
    return presets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "ml" / "data" / "z_alizadeh_sani.xlsx")
    parser.add_argument("--out-models", type=Path, default=REPO_ROOT / "backend" / "app" / "models")
    parser.add_argument("--out-results", type=Path, default=REPO_ROOT / "ml" / "results")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print(" CARDIOVASCULAR RISK MODEL TRAINING - CAD + LAD/LCX/RCA STENOSIS")
    print("=" * 70)

    df = load_dataset(args.data)
    X, y = extract_targets(df)

    cv_metrics: dict[str, dict] = {}
    models: dict[str, Pipeline] = {}
    calibrators: dict[str, dict] = {}
    expected_values: dict[str, float] = {}
    thresholds: dict[str, dict] = {}
    oof_probs: dict[str, np.ndarray] = {}

    for target in TARGETS:
        cv = nested_cv(X, y[target], target, args.folds, args.seed)
        cv_metrics[target] = cv["summary"]
        oof_probs[target] = cv["oof_prob"]

        # Final model on the full cohort -- this is the artifact that gets served.
        pipe = build_pipeline(X, target, args.seed)
        pipe.fit(X, y[target])
        models[target] = pipe

        calibrator, method, raw_brier, cal_brier = fit_calibrator(
            y[target], cv["oof_prob"], args.folds, args.seed
        )
        calibrators[target] = {"model": calibrator, "method": method}
        cv_metrics[target]["calibration"] = {
            "method": method,
            "brier_uncalibrated": round(float(raw_brier), 4),
            "brier_calibrated_cv": round(float(cal_brier), 4),
            "note": "calibrated Brier is cross-validated on the OOF scores",
        }
        note = " - already well calibrated, no map applied" if method == "identity" else ""
        print(f"  calibration: {method} (Brier {raw_brier:.4f} -> {cal_brier:.4f}){note}")

        # TreeSHAP straight from the booster that is served, so attributions always
        # reconcile with the probability shown next to them. The last column that
        # pred_contrib returns is the base value.
        transformed = pipe[:-1].transform(X)
        contrib = pipe.named_steps["clf"].predict(transformed, pred_contrib=True)
        expected_values[target] = float(np.asarray(contrib)[0, -1])
        margin = pipe.named_steps["clf"].predict(transformed, raw_score=True)
        additivity = float(np.abs(np.asarray(contrib).sum(axis=1) - margin).max())
        assert additivity < 1e-6, f"TreeSHAP not additive for {target}: {additivity}"
        print(f"  shap       : additive to {additivity:.2e} (base {expected_values[target]:+.4f})")

        thresholds[target] = {
            "tuned": round(float(np.mean(cv["thresholds"])), 4),
            "default": 0.5,
            "selection": "Youden's J on inner CV folds",
        }

    # --- CAD head vs vessel-derived agreement (structural consistency check) ---
    # Noisy-OR over the three vessels. It ranks CAD almost as well as the direct head,
    # but its *scale* is inflated: the independence assumption compounds each vessel's
    # floor probability, while real vessel stenoses are correlated through shared risk
    # factors. So calibrate it on out-of-fold data before showing it beside the direct
    # head, otherwise a disease-free patient reads ~0.09 direct against ~0.53 derived.
    derived_raw = 1.0 - np.prod([1.0 - oof_probs[v] for v in VESSELS], axis=0)
    derived_cal, derived_method, derived_brier_raw, derived_brier_cal = fit_calibrator(
        y["CAD"], derived_raw, args.folds, args.seed
    )
    calibrators["CAD_DERIVED"] = {"model": derived_cal, "method": derived_method}
    derived = (
        derived_raw if derived_cal is None else _predict_1d(derived_cal, derived_raw)
    )

    direct_pred = (oof_probs["CAD"] >= thresholds["CAD"]["tuned"]).astype(int)
    derived_pred = (derived >= 0.5).astype(int)
    observed_any = ((y["LAD"] | y["LCX"] | y["RCA"]) == 1).astype(int)
    consistency = {
        "description": (
            "Cath equals OR(LAD,LCX,RCA) on 302/303 rows in this cohort, so the direct "
            "CAD head and the vessel-derived noisy-OR should broadly agree. The derived "
            "score is calibrated on out-of-fold data because the independence assumption "
            "inflates its scale even though it ranks well."
        ),
        "direct_vs_derived_agreement": round(float((direct_pred == derived_pred).mean()), 4),
        "derived_roc_auc": round(float(roc_auc_score(y["CAD"], derived)), 4),
        "direct_roc_auc": round(float(roc_auc_score(y["CAD"], oof_probs["CAD"])), 4),
        "observed_or_rule_agreement": round(float((observed_any == y["CAD"]).mean()), 4),
        "derived_calibration": derived_method,
        "derived_brier_uncalibrated": round(float(derived_brier_raw), 4),
        "derived_brier_calibrated_cv": round(float(derived_brier_cal), 4),
        "derived_mean_before_calibration": round(float(derived_raw.mean()), 4),
        "derived_mean_after_calibration": round(float(derived.mean()), 4),
        "observed_prevalence": round(float(y["CAD"].mean()), 4),
    }
    print("\n===== CAD coherence =====")
    for k, v in consistency.items():
        if k != "description":
            print(f"  {k:<32}: {v}")

    # --- feature metadata and inference-time defaults ---
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    defaults = {}
    for col in numeric:
        defaults[col] = float(X[col].median())
    for col in categorical:
        defaults[col] = str(X[col].mode().iloc[0])

    try:
        processed_names = models["CAD"][:-1].get_feature_names_out().tolist()
    except Exception:  # pragma: no cover - name introspection is best-effort
        processed_names = []

    metadata = {
        "dataset": args.data.name,
        "n_patients": int(len(X)),
        "raw_features": list(X.columns),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "categorical_levels": {c: sorted(X[c].unique().tolist()) for c in categorical},
        "processed_feature_names": processed_names,
        "feature_defaults": defaults,
        "display_names": DISPLAY_NAMES,
        "targets": TARGETS,
        "vessels": VESSELS,
        "excluded_leakage_columns": sorted(LEAKAGE_COLUMNS),
        "shap_expected_values": {k: round(v, 6) for k, v in expected_values.items()},
        "shap_implementation": "LightGBM native TreeSHAP (pred_contrib=True)",
        "validation": f"Nested stratified {args.folds}-fold CV; thresholds from inner folds",
        "seed": args.seed,
    }

    assert len(metadata["raw_features"]) == 55, (
        f"Expected 55 features, got {len(metadata['raw_features'])}"
    )
    assert not LEAKAGE_COLUMNS.intersection(metadata["raw_features"]), "Leakage column in metadata!"

    args.out_models.mkdir(parents=True, exist_ok=True)
    args.out_results.mkdir(parents=True, exist_ok=True)

    joblib.dump(models, args.out_models / "cardiac_models.joblib")
    joblib.dump(calibrators, args.out_models / "calibration.joblib")
    (args.out_models / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.out_models / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    presets = select_presets(X, y)
    (args.out_models / "demo_presets.json").write_text(
        json.dumps(presets, indent=2), encoding="utf-8"
    )

    results = {
        "dataset": args.data.name,
        "cohort_size": int(len(X)),
        "validation_strategy": f"Nested stratified {args.folds}-fold cross-validation",
        "threshold_selection": "Youden's J on inner folds (never on the scored fold)",
        "target_leakage_prevented": True,
        "excluded_features": sorted(LEAKAGE_COLUMNS),
        "n_features": int(X.shape[1]),
        "per_target": cv_metrics,
        "cad_coherence": consistency,
    }
    (args.out_results / "cv_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez(
        args.out_results / "oof_predictions.npz",
        **{f"prob_{t}": oof_probs[t] for t in TARGETS},
        **{f"y_{t}": y[t] for t in TARGETS},
        derived_cad=derived,
    )

    print("\n" + "=" * 70)
    print(f"[OK] Models      -> {args.out_models}")
    print(f"[OK] Metrics     -> {args.out_results / 'cv_metrics.json'}")
    print(f"[OK] OOF preds   -> {args.out_results / 'oof_predictions.npz'}")
    print(f"[OK] Presets     -> {len(presets)} cases covering 0/1/2/3-vessel disease")
    print("=" * 70)


if __name__ == "__main__":
    main()
