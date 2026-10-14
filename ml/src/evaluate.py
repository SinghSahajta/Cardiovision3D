"""Generate evaluation tables and figures from the trained artifacts.

Reads the out-of-fold predictions saved by ``train.py`` -- so every figure here
reflects held-out performance, not in-sample fit -- and writes ROC, precision-
recall, calibration and confusion-matrix plots plus a markdown metrics table
into ``ml/results/``.

Usage::

    python ml/src/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
VESSELS = ["LAD", "LCX", "RCA"]
COLORS = {"CAD": "#2563eb", "LAD": "#dc2626", "LCX": "#f59e0b", "RCA": "#16a34a"}


def load_oof(results_dir: Path) -> tuple[dict, dict, np.ndarray]:
    path = results_dir / "oof_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python ml/src/train.py` first."
        )
    data = np.load(path)
    prob = {t: data[f"prob_{t}"] for t in TARGETS}
    y = {t: data[f"y_{t}"] for t in TARGETS}
    return prob, y, data["derived_cad"]


def plot_roc(prob: dict, y: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for t in TARGETS:
        fpr, tpr, _ = roc_curve(y[t], prob[t])
        ax.plot(fpr, tpr, color=COLORS[t], lw=2,
                label=f"{t}  AUC={roc_auc_score(y[t], prob[t]):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC - out-of-fold predictions (nested 5-fold CV)")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_pr(prob: dict, y: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for t in TARGETS:
        precision, recall, _ = precision_recall_curve(y[t], prob[t])
        ax.plot(recall, precision, color=COLORS[t], lw=2,
                label=f"{t}  AP={average_precision_score(y[t], prob[t]):.3f}")
        ax.axhline(y[t].mean(), color=COLORS[t], ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall - dotted lines show class prevalence")
    ax.legend(loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=150)
    plt.close(fig)


def plot_calibration(prob: dict, y: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1, label="Perfectly calibrated")
    for t in TARGETS:
        n_bins = 8 if t == "CAD" else 6
        frac, mean_pred = calibration_curve(y[t], prob[t], n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", color=COLORS[t], lw=2, ms=5,
                label=f"{t}  Brier={brier_score_loss(y[t], prob[t]):.3f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title("Reliability - risk estimate quality")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "calibration_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(prob: dict, y: dict, thresholds: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for col, t in enumerate(TARGETS):
        for row, (label, thr) in enumerate(
            [("threshold 0.5", 0.5), (f"tuned {thresholds[t]['tuned']:.2f}", thresholds[t]["tuned"])]
        ):
            ax = axes[row][col]
            cm = confusion_matrix(y[t], (prob[t] >= thr).astype(int))
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "#1f2937", fontsize=12)
            ax.set_xticks([0, 1], ["Pred -", "Pred +"])
            ax.set_yticks([0, 1], ["True -", "True +"])
            ax.set_title(f"{t} @ {label}", fontsize=10)
    fig.suptitle("Confusion matrices - out-of-fold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_cad_coherence(prob: dict, y: dict, derived: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for label, mask, color in [("Cath = Normal", y["CAD"] == 0, "#16a34a"),
                               ("Cath = CAD", y["CAD"] == 1, "#dc2626")]:
        ax.scatter(prob["CAD"][mask], derived[mask], s=22, alpha=0.65,
                   color=color, edgecolor="none", label=label)
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1)
    ax.set_xlabel("Direct CAD head  P(CAD)")
    ax.set_ylabel(r"Vessel-derived  $1-\prod(1-p_v)$")
    ax.set_title(f"CAD coherence  (direct AUC {roc_auc_score(y['CAD'], prob['CAD']):.3f}"
                 f" vs derived {roc_auc_score(y['CAD'], derived):.3f})")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "cad_coherence.png", dpi=150)
    plt.close(fig)


def metrics_table(prob: dict, y: dict, thresholds: dict) -> str:
    lines = [
        "# Evaluation results (out-of-fold, nested stratified 5-fold CV)",
        "",
        "Threshold-independent ranking quality:",
        "",
        "| Target | Prevalence | ROC-AUC | PR-AUC | Brier |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for t in TARGETS:
        lines.append(
            f"| {t} | {y[t].mean():.1%} | {roc_auc_score(y[t], prob[t]):.4f} "
            f"| {average_precision_score(y[t], prob[t]):.4f} "
            f"| {brier_score_loss(y[t], prob[t]):.4f} |"
        )
    for label, use_tuned in [("default threshold 0.5", False), ("tuned threshold", True)]:
        lines += [
            "",
            f"Operating point - {label}:",
            "",
            "| Target | Threshold | Accuracy | Precision | Recall | F1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for t in TARGETS:
            thr = thresholds[t]["tuned"] if use_tuned else 0.5
            pred = (prob[t] >= thr).astype(int)
            lines.append(
                f"| {t} | {thr:.3f} | {accuracy_score(y[t], pred):.4f} "
                f"| {precision_score(y[t], pred, zero_division=0):.4f} "
                f"| {recall_score(y[t], pred, zero_division=0):.4f} "
                f"| {f1_score(y[t], pred, zero_division=0):.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "ml" / "results")
    parser.add_argument("--models", type=Path, default=REPO_ROOT / "backend" / "app" / "models")
    args = parser.parse_args()

    prob, y, derived = load_oof(args.results)
    thresholds = json.loads((args.models / "thresholds.json").read_text(encoding="utf-8"))

    args.results.mkdir(parents=True, exist_ok=True)
    plot_roc(prob, y, args.results)
    plot_pr(prob, y, args.results)
    plot_calibration(prob, y, args.results)
    plot_confusion(prob, y, thresholds, args.results)
    plot_cad_coherence(prob, y, derived, args.results)

    table = metrics_table(prob, y, thresholds)
    (args.results / "evaluation_report.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"[OK] Figures and evaluation_report.md written to {args.results}")


if __name__ == "__main__":
    main()
