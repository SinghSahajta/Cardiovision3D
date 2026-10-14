"""Export the whole Z-Alizadeh Sani cohort for the dashboard's patient browser.

Writes ``backend/app/models/cohort.json``: for each of the 303 patients, the 55
model inputs, the angiographic ground truth, and the **out-of-fold** predicted
probabilities produced during nested cross-validation.

WHY THE OUT-OF-FOLD NUMBERS ARE THE POINT
-----------------------------------------
The served model was fitted on all 303 of these rows, so anything it says about
them is in-sample. Live agreement with angiography is 99.0% for CAD; the honest
held-out figure is 90.3%. A browser that walked the cohort showing only the live
prediction would therefore be a demo of the model memorising its training set.

So each row carries the probability from the fold where that patient was held
out - scored by a model that had never seen them - and the viewer shows it beside
the live number. ``ml/src/validate.py --patient N`` prints the same comparison on
the command line; this is that, as data the UI can read.

Reads ``ml/results/oof_predictions.npz``, so it needs no retrain: run it any time
after ``python ml/src/train.py``.

Usage::

    python tools/build_cohort.py
    python tools/build_cohort.py --check    # verify an existing export, write nothing
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "ml" / "data" / "z_alizadeh_sani.xlsx"
OOF = REPO_ROOT / "ml" / "results" / "oof_predictions.npz"
METADATA = REPO_ROOT / "backend" / "app" / "models" / "metadata.json"
OUT = REPO_ROOT / "backend" / "app" / "models" / "cohort.json"

TARGETS = ["CAD", "LAD", "LCX", "RCA"]


def load_train_module():
    """Import ml/src/train.py for its dataset helpers, without running training."""
    path = REPO_ROOT / "ml" / "src" / "train.py"
    spec = importlib.util.spec_from_file_location("cardiac_train", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonable(value):
    """Coerce one cell to something json.dump can write, preserving blanks."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    return text if text else None


def build() -> dict:
    train = load_train_module()

    df = train.load_dataset(DATASET)
    X, y = train.extract_targets(df)

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    order = metadata["raw_features"]
    missing = [c for c in order if c not in X.columns]
    if missing:
        raise SystemExit(
            f"Dataset is missing columns the served model expects: {missing}"
        )
    # Column order must match what the API was trained on, not the sheet's order.
    X = X[order]

    oof = np.load(OOF)
    for target in TARGETS:
        if len(oof[f"prob_{target}"]) != len(X):
            raise SystemExit(
                f"oof_predictions.npz has {len(oof[f'prob_{target}'])} rows for "
                f"{target} but the dataset has {len(X)}. Re-run ml/src/train.py."
            )
        # The OOF labels must be the same labels we are about to publish, or the
        # held-out probability would be shown next to someone else's truth.
        if not np.array_equal(oof[f"y_{target}"], y[target]):
            raise SystemExit(
                f"Ground truth for {target} in oof_predictions.npz disagrees with "
                "the dataset. The export would pair a probability with the wrong "
                "patient. Re-run ml/src/train.py."
            )

    vessel_count = y["LAD"] + y["LCX"] + y["RCA"]
    patients = []
    for i in range(len(X)):
        row = X.iloc[i]
        patients.append({
            # 1-based, matching how ml/src/validate.py --patient N addresses rows.
            "patient_id": f"ROW{i + 1:03d}",
            "index": i + 1,
            "features": {col: jsonable(row[col]) for col in order},
            "ground_truth": {t: int(y[t][i]) for t in TARGETS},
            "vessels_involved": int(vessel_count[i]),
            "held_out": {t: round(float(oof[f"prob_{t}"][i]), 6) for t in TARGETS},
            "held_out_derived_cad": round(float(oof["derived_cad"][i]), 6),
        })

    return {
        "dataset": metadata.get("dataset", DATASET.name),
        "n_patients": len(patients),
        "n_features": len(order),
        "feature_order": order,
        "note": (
            "held_out probabilities come from the nested cross-validation fold in "
            "which each patient was the held-out case, so they were produced by a "
            "model that never saw that patient. The served model WAS fitted on "
            "every one of these rows, so its live prediction for them is in-sample "
            "and flattering: 99.0% live agreement with angiography for CAD against "
            "a true held-out 90.3%. Show both, labelled."
        ),
        "patients": patients,
    }


def summarise(export: dict) -> None:
    patients = export["patients"]
    print(f"    {len(patients)} patients x {export['n_features']} features")
    for target in TARGETS:
        truth = np.array([p["ground_truth"][target] for p in patients])
        prob = np.array([p["held_out"][target] for p in patients])
        print(f"    {target:<4} positives {truth.sum():>3}/{len(truth)}  "
              f"held-out prob {prob.min():.3f}-{prob.max():.3f} "
              f"(mean {prob.mean():.3f})")
    counts = np.bincount([p["vessels_involved"] for p in patients], minlength=4)
    print(f"    vessels involved: " +
          ", ".join(f"{n} in {c}" for n, c in enumerate(counts)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the existing export matches a fresh build, write nothing")
    args = ap.parse_args()

    print(f"[*] dataset  {DATASET}")
    print(f"[*] OOF      {OOF}")
    export = build()
    summarise(export)

    payload = json.dumps(export, indent=1)

    if args.check:
        if not OUT.exists():
            raise SystemExit(f"[FAIL] {OUT} does not exist")
        if OUT.read_text(encoding="utf-8") != payload:
            raise SystemExit(f"[FAIL] {OUT} is stale; re-run without --check")
        print(f"[ok] {OUT.name} is up to date")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload, encoding="utf-8")
    print(f"[+] wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
