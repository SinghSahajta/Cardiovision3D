"""Clinical validation: is the system actually detecting disease correctly?

Every number printed by the default report comes from a cross-validation fold in
which the patient was **held out**, read from ``ml/results/oof_predictions.npz``.
That is the honest view. Where a command has to use the final model (which was fit
on all 303 patients), the output says so explicitly and prints the held-out figure
beside it, because an in-sample number on a training patient is flattering and
means very little.

Usage::

    python ml/src/validate.py                  # per-target detection report
    python ml/src/validate.py --patient 6      # one patient, predicted vs actual
    python ml/src/validate.py --errors         # every patient the model gets wrong
    python ml/src/validate.py --live           # check the running API on all 303 rows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import TARGETS, VESSELS, extract_targets, load_dataset  # noqa: E402

# Windows consoles default to cp1252; keep output printable everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "ml" / "results"
MODELS = REPO_ROOT / "backend" / "app" / "models"
DATA = REPO_ROOT / "ml" / "data" / "z_alizadeh_sani.xlsx"

LABEL = {
    "CAD": "Overall CAD (angiographic)",
    "LAD": "LAD stenosis",
    "LCX": "LCX stenosis",
    "RCA": "RCA stenosis",
}
BAR = "=" * 78


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_oof() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    path = RESULTS / "oof_predictions.npz"
    if not path.exists():
        sys.exit(f"{path} not found. Run `python ml/src/train.py` first.")
    d = np.load(path)
    prob = {t: d[f"prob_{t}"] for t in TARGETS}
    y = {t: d[f"y_{t}"] for t in TARGETS}
    return prob, y, d["derived_cad"]


def load_thresholds() -> dict[str, dict]:
    path = MODELS / "thresholds.json"
    if not path.exists():
        sys.exit(f"{path} not found. Run `python ml/src/train.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def counts(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
    }


def rates(c: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    div = lambda a, b: (a / b) if b else float("nan")  # noqa: E731
    return {
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "accuracy": div(tp + tn, tp + fp + tn + fn),
    }


def pct(x: float) -> str:
    return "   n/a" if np.isnan(x) else f"{x * 100:5.1f}%"


# ---------------------------------------------------------------------------
# Default report
# ---------------------------------------------------------------------------
def detection_report(prob, y, derived, thresholds) -> None:
    print(BAR)
    print(" CLINICAL DETECTION REPORT")
    print(" Z-Alizadeh Sani cohort, n=303. All figures are OUT-OF-FOLD:")
    print(" each patient was scored by a model that never saw them in training.")
    print(BAR)

    for t in TARGETS:
        tuned = float(thresholds[t]["tuned"])
        n_pos = int(y[t].sum())
        print(f"\n{LABEL[t]}  -  {n_pos} of {len(y[t])} patients have it "
              f"({y[t].mean():.1%} prevalence)")
        print(f"  {'operating point':<22} {'caught':>14}  {'false alarms':>13}  "
              f"{'sens':>6} {'spec':>6} {'PPV':>6} {'NPV':>6}")

        for name, thr in (("default 0.50", 0.5), (f"tuned {tuned:.3f}", tuned)):
            c = counts(y[t], (prob[t] >= thr).astype(int))
            r = rates(c)
            caught = f"{c['tp']}/{n_pos}"
            print(f"  {name:<22} {caught:>14}  {c['fp']:>13}  "
                  f"{pct(r['sensitivity'])} {pct(r['specificity'])} "
                  f"{pct(r['ppv'])} {pct(r['npv'])}")

        c = counts(y[t], (prob[t] >= tuned).astype(int))
        print(f"  At the tuned threshold it MISSES {c['fn']} of {n_pos} real cases "
              f"and raises {c['fp']} false alarms.")
        if not np.isnan(rates(c)["ppv"]) and rates(c)["ppv"] < 0.6:
            print(f"  !  Only {rates(c)['ppv']:.0%} of flagged {t} vessels truly have "
                  f"stenosis - treat a positive as a prompt to image, not a finding.")

    # Whole-patient view: does the system get the complete vessel picture right?
    print(f"\n{BAR}")
    print(" WHOLE-PATIENT VESSEL PATTERN (all three vessels simultaneously correct)")
    print(BAR)
    tuned_pred = {v: (prob[v] >= float(thresholds[v]["tuned"])).astype(int) for v in VESSELS}
    all_right = np.ones(len(y["CAD"]), dtype=bool)
    for v in VESSELS:
        all_right &= tuned_pred[v] == y[v]
    n_correct = {v: int((tuned_pred[v] == y[v]).sum()) for v in VESSELS}
    print(f"  All 3 vessels correct : {all_right.sum():3d}/303 ({all_right.mean():.1%})")
    for v in VESSELS:
        print(f"  {v} correct            : {n_correct[v]:3d}/303 ({n_correct[v] / 303:.1%})")

    true_count = y["LAD"] + y["LCX"] + y["RCA"]
    pred_count = tuned_pred["LAD"] + tuned_pred["LCX"] + tuned_pred["RCA"]
    exact = int((true_count == pred_count).sum())
    print(f"  Correct NUMBER of diseased vessels: {exact}/303 ({exact / 303:.1%})")
    print(f"  Mean predicted vessel count {pred_count.mean():.2f} vs actual "
          f"{true_count.mean():.2f}")

    # CAD coherence
    print(f"\n{BAR}")
    print(" CAD HEAD vs VESSEL-DERIVED SCORE")
    print(BAR)
    direct = (prob["CAD"] >= float(thresholds["CAD"]["tuned"])).astype(int)
    der = (derived >= 0.5).astype(int)
    print(f"  Direct head and vessel-derived agree on {int((direct == der).sum())}/303 "
          f"patients ({(direct == der).mean():.1%})")
    print(f"  Mean vessel-derived probability {derived.mean():.3f} vs "
          f"observed prevalence {y['CAD'].mean():.3f}")

    print(f"\n{BAR}")
    print(" HOW TO READ THIS")
    print(BAR)
    print("  The CAD head is the trustworthy part: it finds ~90% of angiographic CAD.")
    print("  LAD is usable. LCX and RCA are weak - they rank better than chance but")
    print("  roughly half of the vessels they flag are false alarms. The 3D view")
    print("  colours those two from a genuinely uncertain signal, which is why the UI")
    print("  shows the threshold and the disclaimer next to every vessel.")
    print("  Run with --errors to see exactly which patients are missed.")


# ---------------------------------------------------------------------------
# --errors
# ---------------------------------------------------------------------------
def error_report(prob, y, thresholds, X: pd.DataFrame) -> None:
    print(BAR)
    print(" MISCLASSIFIED PATIENTS (out-of-fold, at the tuned threshold)")
    print(BAR)
    context = [c for c in ("Age", "Sex", "Typical Chest Pain", "EF-TTE", "Region RWMA")
               if c in X.columns]

    for t in TARGETS:
        tuned = float(thresholds[t]["tuned"])
        pred = (prob[t] >= tuned).astype(int)
        fn = np.where((pred == 0) & (y[t] == 1))[0]
        fp = np.where((pred == 1) & (y[t] == 0))[0]

        print(f"\n{LABEL[t]}  (threshold {tuned:.3f})")
        print(f"  MISSED DISEASE - false negatives: {len(fn)}")
        for i in fn[:12]:
            ctx = "  ".join(f"{c}={X.iloc[i][c]}" for c in context)
            print(f"    row {i + 1:3d}  p={prob[t][i]:.3f}   {ctx}")
        if len(fn) > 12:
            print(f"    ... and {len(fn) - 12} more")

        print(f"  FALSE ALARMS - false positives: {len(fp)}")
        for i in fp[:8]:
            ctx = "  ".join(f"{c}={X.iloc[i][c]}" for c in context)
            print(f"    row {i + 1:3d}  p={prob[t][i]:.3f}   {ctx}")
        if len(fp) > 8:
            print(f"    ... and {len(fp) - 8} more")

        if len(fn):
            near = int((prob[t][fn] >= tuned - 0.1).sum())
            print(f"  {near}/{len(fn)} of the misses were within 0.10 of the threshold "
                  f"(borderline, not confidently wrong).")


# ---------------------------------------------------------------------------
# --patient
# ---------------------------------------------------------------------------
def patient_report(row_number: int, prob, y, thresholds, X: pd.DataFrame) -> None:
    import joblib

    idx = row_number - 1
    if not 0 <= idx < len(X):
        sys.exit(f"--patient must be between 1 and {len(X)}")

    models = joblib.load(MODELS / "cardiac_models.joblib")
    metadata = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    order = metadata["raw_features"]
    frame = X.iloc[[idx]][order]

    print(BAR)
    print(f" PATIENT - cohort row {row_number}")
    print(BAR)
    print("  !  This patient is IN the training set. The 'final model' column below is")
    print("    in-sample and therefore optimistic. The 'held-out' column is the honest")
    print("    number: it comes from the fold where this patient was excluded.")
    print()
    shown = [c for c in ("Age", "Sex", "BP", "Typical Chest Pain", "Function Class",
                         "EF-TTE", "Region RWMA", "FBS", "LDL", "HDL", "TG", "DM", "HTN",
                         "Current Smoker", "Tinversion", "St Depression", "Q Wave")
             if c in X.columns]
    print("  Clinical picture:")
    for c in shown:
        print(f"    {c:22s} {X.iloc[idx][c]}")

    print(f"\n  {'target':6s} {'final model':>12s} {'held-out':>10s} "
          f"{'threshold':>10s} {'predicted':>10s} {'actual':>8s}   verdict")
    for t in TARGETS:
        tuned = float(thresholds[t]["tuned"])
        in_sample = float(models[t].predict_proba(frame)[0, 1])
        held = float(prob[t][idx])
        pred = held >= tuned
        actual = bool(y[t][idx])
        verdict = ("correct" if pred == actual
                   else ("MISSED disease" if actual else "false alarm"))
        print(f"  {t:6s} {in_sample:12.3f} {held:10.3f} {tuned:10.3f} "
              f"{str(pred):>10s} {str(actual):>8s}   {verdict}")

    # Top drivers from the booster that serves predictions
    print("\n  Top drivers (TreeSHAP, log-odds margin space, from the final model):")
    pnames = metadata["processed_feature_names"]

    def source(p: str) -> str:
        n = p.split("__", 1)[-1]
        if n in order:
            return n
        cands = [f for f in order if n.startswith(f)]
        return max(cands, key=len) if cands else n

    for t in TARGETS:
        pipe = models[t]
        contrib = np.asarray(
            pipe.named_steps["clf"].predict(pipe[:-1].transform(frame), pred_contrib=True)
        ).reshape(-1)[:-1]
        top = np.argsort(np.abs(contrib))[::-1][:5]
        parts = []
        for i in top:
            src = source(pnames[i]) if i < len(pnames) else f"f{i}"
            parts.append(f"{src}={X.iloc[idx][src] if src in X.columns else '?'} "
                         f"({contrib[i]:+.2f})")
        print(f"    {t:4s} " + ",  ".join(parts))


# ---------------------------------------------------------------------------
# --live
# ---------------------------------------------------------------------------
def live_check(url: str, X: pd.DataFrame, y) -> int:
    try:
        import httpx
    except ImportError:
        sys.exit("httpx required: pip install -r backend/requirements-dev.txt")
    import joblib

    models = joblib.load(MODELS / "cardiac_models.joblib")
    metadata = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    order = metadata["raw_features"]

    print(BAR)
    print(f" LIVE API CHECK - all {len(X)} cohort rows against {url}")
    print(" Confirms the DEPLOYED service returns what the offline pipeline computes.")
    print(BAR)

    try:
        health = httpx.get(f"{url}/health", timeout=30).json()
    except Exception as exc:
        sys.exit(f"Cannot reach {url}: {exc}\nStart it with:\n"
                 f"  python -m uvicorn backend.app.main:app --port 8000")
    if not health.get("models_loaded"):
        sys.exit(f"API reports models not loaded: {health}")
    print(f"  /health: {health['status']}, {health['n_features']} features\n")

    mismatches: list[str] = []
    agree = {t: 0 for t in TARGETS}
    with httpx.Client(timeout=60) as client:
        for i in range(len(X)):
            feats = {}
            for c in order:
                v = X.iloc[i][c]
                feats[c] = None if pd.isna(v) else (
                    float(v) if isinstance(v, (int, float, np.integer, np.floating))
                    else str(v)
                )
            r = client.post(f"{url}/predict", json={"features": feats})
            if r.status_code != 200:
                mismatches.append(f"row {i + 1}: HTTP {r.status_code}")
                continue
            body = r.json()
            frame = X.iloc[[i]][order]
            for t in TARGETS:
                offline = float(models[t].predict_proba(frame)[0, 1])
                node = body["cad"] if t == "CAD" else body["vessels"][t]
                if abs(node["uncalibrated_score"] - offline) > 1e-3:
                    mismatches.append(
                        f"row {i + 1} {t}: api={node['uncalibrated_score']} "
                        f"offline={offline:.6f}")
                flag = node["has_cad"] if t == "CAD" else node["is_stenotic"]
                if bool(flag) == bool(y[t][i]):
                    agree[t] += 1
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(X)} rows checked")

    print(f"\n  Served probability == offline pipeline: "
          f"{'YES on all rows' if not mismatches else f'{len(mismatches)} MISMATCHES'}")
    for m in mismatches[:10]:
        print(f"    - {m}")

    print("\n  Live API agreement with angiographic ground truth (in-sample - the final")
    print("  model was trained on these rows, so this is an upper bound, not validation):")
    for t in TARGETS:
        print(f"    {t:4s} {agree[t]:3d}/{len(X)} ({agree[t] / len(X):.1%})")
    print("\n  For honest detection performance use the default report (out-of-fold).")
    return 1 if mismatches else 0


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", type=int, metavar="N",
                    help="inspect cohort row N (1-303)")
    ap.add_argument("--errors", action="store_true",
                    help="list every misclassified patient")
    ap.add_argument("--live", action="store_true",
                    help="check a running API against the offline pipeline")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    args = ap.parse_args()

    X, y = extract_targets(load_dataset(DATA))
    thresholds = load_thresholds()

    if args.live:
        sys.exit(live_check(args.url.rstrip("/"), X, y))

    prob, y_oof, derived = load_oof()
    if args.patient:
        patient_report(args.patient, prob, y_oof, thresholds, X)
    elif args.errors:
        error_report(prob, y_oof, thresholds, X)
    else:
        detection_report(prob, y_oof, derived, thresholds)


if __name__ == "__main__":
    main()
