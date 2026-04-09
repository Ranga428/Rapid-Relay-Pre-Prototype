"""
ml_comparison_report.py
========================
Flood Prediction — Model Comparison Report (XGBoost / RF / LightGBM)

Loads all three sensor model pkl files, runs them against the full
flood_dataset.csv, and produces:

  1. A printed comparison table (precision, recall, F1, balanced accuracy,
     ROC-AUC, specificity, accuracy, TP, TN, FP, FN) at CLEAR, WATCH,
     WARNING, and DANGER thresholds.
  2. A multi-panel PNG chart comparing probability distributions and
     ROC curves for all three models.
  3. A CSV report saved to ../ml_report/ml_comparison_report_{split}.csv
     — now includes TP, TN, FP, FN, and accuracy columns.
  4. A cross-reference accuracy report comparing live predictions for all
     three models (LGBM, RF, XGBoost combined sensor+context CSVs) against
     flood_dataset_test.csv labels on overlapping dates — saved to
     ../ml_report/live_prediction_accuracy.csv
  5. A detailed ROC AUC technical summary CSV (trapezoid-by-trapezoid) saved
     to ../ml_report/roc_auc_technical_summary_{split}.csv
  6. A concise ROC AUC summary CSV (day, H, W, ROC AUC, remarks) saved to
     ../ml_report/roc_auc_concise_summary_{split}.csv

ACCURACY FORMULA
----------------
  Accuracy = (TP + TN) / (TP + TN + FP + FN)

  This is reported in both the terminal and CSV output at every
  threshold level (CLEAR, WATCH, WARNING, DANGER).

LIVE CROSS-REFERENCE (--live-check)
-------------------------------------
  After model evaluation, loads the three combined sensor+context prediction
  CSVs and flood_dataset_test.csv (or a custom --label-file path), aligns
  them on overlapping dates, and computes TP/TN/FP/FN + accuracy for every
  threshold tier per model. This answers: "Of the live predictions already
  made, how many were correct compared to the known labeled events?"

  Threshold tiers used for cross-reference:
    WATCH   — risk_tier in (WATCH, WARNING, DANGER)
    WARNING — risk_tier in (WARNING, DANGER)
    DANGER  — risk_tier == DANGER

ROC AUC CSV OUTPUTS
--------------------
  roc_auc_technical_summary_{split}.csv
    Full trapezoid-by-trapezoid breakdown with 20 columns including
    cumulative ΣH, ΣW, segment_area, information_gap, WMO pass/fail.

  roc_auc_concise_summary_{split}.csv
    Compact table matching the document layout: model | day | H | W | ROC AUC | remarks
    Per-segment rows fill H, W, ROC AUC (repeated), and remarks (repeated).
    TOTAL row carries ΣH, ΣW, final AUC and verdict.

USAGE
-----
    python ml_comparison_report.py
    python ml_comparison_report.py --split test        # test set only (default)
    python ml_comparison_report.py --split val
    python ml_comparison_report.py --split full
    python ml_comparison_report.py --live-check        # also run live cross-reference
    python ml_comparison_report.py --live-check \\
        --label-file ..\\data\\flood_dataset_test.csv

NOTE: This script must be in the same directory as the train_*.py scripts
      because it imports CalibratedRF and CalibratedLGBM from them at runtime
      so joblib can unpickle the saved models.
"""

import os
import sys
import argparse
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score,
    roc_curve,
)
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)


# ===========================================================================
# MODULE-LEVEL CALIBRATED WRAPPERS
# ===========================================================================

class CalibratedRF:
    """Isotonic-calibrated wrapper — must match RF_train_flood_model.py exactly."""
    def __init__(self, base, calibrator):
        self.base       = base
        self.calibrator = calibrator
        self.estimator  = base

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])


class CalibratedLGBM:
    """Isotonic-calibrated wrapper for LGBMClassifier."""
    def __init__(self, base, calibrator):
        self.base       = base
        self.calibrator = calibrator
        self.estimator  = base

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])


# ===========================================================================
# CONFIG
# ===========================================================================

DATA_FILE  = r"..\data\flood_dataset.csv"
MODEL_DIR  = r"..\model"
OUTPUT_DIR = r"..\ml_report"

MODELS = {
    "XGBoost":       os.path.join(MODEL_DIR, "flood_xgb_sensor.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "flood_rf_sensor.pkl"),
    "LightGBM":      os.path.join(MODEL_DIR, "flood_lgbm_sensor.pkl"),
}

MODEL_COLORS = {
    "XGBoost":       "#e74c3c",
    "Random Forest": "#2ecc71",
    "LightGBM":      "#3498db",
}

TRAIN_END = "2024-06-30"
VAL_END   = "2025-06-30"

DANGER_FALLBACK_OFFSET = 0.10

PREDICTIONS_DIR = r"D:\Rapid-Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\predictions"

LIVE_PREDICTIONS = {
    "LightGBM":      os.path.join(PREDICTIONS_DIR, "flood_lgbm_sensor_predictions.csv"),
    "Random Forest": os.path.join(PREDICTIONS_DIR, "flood_rf_sensor_predictions.csv"),
    "XGBoost":       os.path.join(PREDICTIONS_DIR, "flood_xgb_sensor_predictions.csv"),
}

DEFAULT_LABEL_FILE = os.path.join(
    SCRIPT_DIR, r"..\data\flood_dataset_test.csv"
)

# WMO minimum AUC standard for early-warning system discrimination
_WMO_THRESHOLD = 0.80

# Deployment labels and final verdict remarks per model
_DEPLOYMENT_MAP = {
    "XGBoost":       "Sensor-Only Deployment",
    "Random Forest": "Sensor-Only (Redundancy)",
    "LightGBM":      "Sensor-Only (Experimental)",
}

_REMARKS_MAP = {
    "XGBoost": (
        "Excellent; Meets Industry Standard for live deployment. "
        "Primary production model running on real-time gauge data."
    ),
    "Random Forest": (
        "Strong; Retained as a reliable backup model. "
        "Alternative evaluated for operational redundancy."
    ),
    "LightGBM": (
        "Weak; Excluded from deployment consideration. "
        "Experimental architecture — does not clear WMO safety margin."
    ),
}


# ===========================================================================
# HELPERS
# ===========================================================================

def separator(title=""):
    line = "=" * 65
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "train":
        return df[df.index <= TRAIN_END]
    elif split == "val":
        return df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    elif split == "test":
        return df[df.index > VAL_END]
    else:
        return df


def _confusion_counts(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


def _accuracy(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    bal  = balanced_accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc  = _accuracy(tp, tn, fp, fn)

    return {
        "threshold":   threshold,
        "precision":   prec,
        "recall":      rec,
        "f1":          f1,
        "bal_acc":     bal,
        "roc_auc":     auc,
        "specificity": spec,
        "accuracy":    acc,
        "alerts":      int(y_pred.sum()),
        "tp":          tp,
        "tn":          tn,
        "fp":          fp,
        "fn":          fn,
    }


def compute_clear_metrics(y_true, y_prob, watch_threshold):
    y_pred_watch = (y_prob >= watch_threshold).astype(int)
    y_clear      = (y_pred_watch == 0).astype(int)
    y_true_neg   = (y_true == 0).astype(int)

    prec = precision_score(y_true_neg, y_clear, zero_division=0)
    rec  = recall_score(y_true_neg, y_clear, zero_division=0)
    f1   = f1_score(y_true_neg, y_clear, zero_division=0)
    bal  = balanced_accuracy_score(y_true, y_pred_watch)
    auc  = roc_auc_score(y_true, y_prob)

    tp, tn, fp, fn = _confusion_counts(y_true, y_pred_watch)
    acc       = _accuracy(tp, tn, fp, fn)
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    n_clear   = int(y_clear.sum())

    return {
        "threshold":   watch_threshold,
        "precision":   prec,
        "recall":      rec,
        "f1":          f1,
        "bal_acc":     bal,
        "roc_auc":     auc,
        "specificity": rec,
        "accuracy":    acc,
        "alerts":      n_clear,
        "miss_rate":   miss_rate,
        "n_missed":    fn,
        "tp":          tp,
        "tn":          tn,
        "fp":          fp,
        "fn":          fn,
    }


def _print_split_context_note(split: str) -> None:
    separator("Evaluation Split — Context & Caveats")
    notes = {
        "test": (
            "  Split : TEST  (post-2025-06-30)\n"
            "  ✅  Recommended for honest, unbiased performance reporting.\n"
            "      These rows were never seen during training or validation.\n"
            "      Metrics here are the best estimate of real-world performance."
        ),
        "val": (
            "  Split : VALIDATION  (2024-07-01 → 2025-06-30)\n"
            "  ⚠️   Used during hyper-parameter selection — treat results as\n"
            "       slightly optimistic. Prefer the TEST split for final reporting."
        ),
        "train": (
            "  Split : TRAIN  (pre-2024-06-30)\n"
            "  ❌  Models were fitted on this data. Metrics are HEAVILY biased\n"
            "      upward and do NOT reflect generalisation ability."
        ),
        "full": (
            "  Split : FULL DATASET\n"
            "  ⚠️   Includes the training period. Sensor-only models trained on\n"
            "       pre-2024-06-30 data will appear artificially strong on those\n"
            "       rows. Use for distribution plots only, NOT performance reporting.\n\n"
            "  NOTE ON SENSOR-ONLY MODELS vs. FULL-FEATURE MODELS:\n"
            "       The pkl files loaded here (flood_*_sensor.pkl) use ONLY\n"
            "       physical sensor readings and their derived features."
        ),
    }
    print(notes.get(split, f"  Split: {split.upper()}"))


# ===========================================================================
# ROC AUC CSV OUTPUT 1 — DETAILED TECHNICAL SUMMARY
# ===========================================================================

def save_roc_auc_technical_summary(
    results: dict,
    y_true: np.ndarray,
    split: str,
    output_dir: str,
    period_str: str = "",
) -> str:
    """
    Compute a trapezoid-by-trapezoid ROC AUC breakdown for every model and
    write a detailed technical summary CSV.

    CSV schema (20 columns):
      section, model, split, deployment, period,
      i, tpr_prev, tpr_curr, H, fpr_prev, fpr_curr, W,
      segment_area, sum_H, sum_W,
      roc_auc, wmo_threshold, meets_wmo, remarks, information_gap

    Sections:
      MODEL_HEADER  — one row per model with AUC, WMO verdict, deployment label
      TRAPEZOID     — one row per ROC curve point (i, H, W, segment_area, running sums)
      TOTAL         — ΣH, ΣW, Σ(H×W) = AUC, information_gap vs best model
      GLOBAL_SUMMARY — one footer row across all models
    """
    # Pre-compute AUC per model for information-gap column
    model_aucs: dict = {}
    for name, r in results.items():
        model_aucs[name] = float(roc_auc_score(y_true, r["probs"]))

    best_auc  = max(model_aucs.values())
    best_name = max(model_aucs, key=model_aucs.get)

    COLUMNS = [
        "section", "model", "split", "deployment", "period",
        "i", "tpr_prev", "tpr_curr", "H",
        "fpr_prev", "fpr_curr", "W",
        "segment_area", "sum_H", "sum_W",
        "roc_auc", "wmo_threshold", "meets_wmo",
        "remarks", "information_gap",
    ]

    def _blank():
        return {c: "" for c in COLUMNS}

    rows = []

    for name, r in results.items():
        probs      = r["probs"]
        auc_val    = model_aucs[name]
        meets_wmo  = auc_val >= _WMO_THRESHOLD
        deployment = _DEPLOYMENT_MAP.get(name, "Sensor-Only")
        remarks    = _REMARKS_MAP.get(name, "")
        info_gap   = round(best_auc - auc_val, 4)

        # Model header row
        header = _blank()
        header.update({
            "section":         "MODEL_HEADER",
            "model":           name,
            "split":           split,
            "deployment":      deployment,
            "period":          period_str,
            "roc_auc":         round(auc_val, 4),
            "wmo_threshold":   _WMO_THRESHOLD,
            "meets_wmo":       "TRUE" if meets_wmo else "FALSE",
            "remarks":         remarks,
            "information_gap": info_gap,
        })
        rows.append(header)

        # Trapezoid segments
        fpr_arr, tpr_arr, _ = roc_curve(y_true, probs)
        cum_H = cum_W = cum_area = 0.0

        for i in range(1, len(fpr_arr)):
            tpr_prev = float(tpr_arr[i - 1])
            tpr_curr = float(tpr_arr[i])
            fpr_prev = float(fpr_arr[i - 1])
            fpr_curr = float(fpr_arr[i])

            H    = (tpr_prev + tpr_curr) / 2.0
            W    = fpr_curr - fpr_prev
            area = H * W

            cum_H    += H
            cum_W    += W
            cum_area += area

            seg = _blank()
            seg.update({
                "section":       "TRAPEZOID",
                "model":         name,
                "split":         split,
                "deployment":    deployment,
                "period":        period_str,
                "i":             i,
                "tpr_prev":      round(tpr_prev, 6),
                "tpr_curr":      round(tpr_curr, 6),
                "H":             round(H,        6),
                "fpr_prev":      round(fpr_prev, 6),
                "fpr_curr":      round(fpr_curr, 6),
                "W":             round(W,        6),
                "segment_area":  round(area,     8),
                "sum_H":         round(cum_H,    6),
                "sum_W":         round(cum_W,    6),
                "roc_auc":       round(auc_val,  4),
                "wmo_threshold": _WMO_THRESHOLD,
                "meets_wmo":     "TRUE" if meets_wmo else "FALSE",
            })
            rows.append(seg)

        # TOTAL row
        total = _blank()
        total.update({
            "section":         "TOTAL",
            "model":           name,
            "split":           split,
            "deployment":      deployment,
            "period":          period_str,
            "i":               "TOTAL",
            "H":               round(cum_H,    6),
            "W":               round(cum_W,    6),
            "segment_area":    round(cum_area, 6),
            "sum_H":           round(cum_H,    6),
            "sum_W":           round(cum_W,    6),
            "roc_auc":         round(auc_val,  4),
            "wmo_threshold":   _WMO_THRESHOLD,
            "meets_wmo":       "TRUE" if meets_wmo else "FALSE",
            "remarks":         remarks,
            "information_gap": info_gap,
        })
        rows.append(total)
        rows.append(_blank())

    # Global summary footer
    n_pass = sum(1 for a in model_aucs.values() if a >= _WMO_THRESHOLD)
    footer = _blank()
    footer.update({
        "section":         "GLOBAL_SUMMARY",
        "model":           "ALL MODELS",
        "split":           split,
        "period":          period_str,
        "roc_auc":         round(best_auc, 4),
        "wmo_threshold":   _WMO_THRESHOLD,
        "meets_wmo":       "TRUE" if best_auc >= _WMO_THRESHOLD else "FALSE",
        "remarks":         (
            f"Best model: {best_name} (AUC={best_auc:.4f}). "
            f"{n_pass}/{len(model_aucs)} models meet WMO ≥{_WMO_THRESHOLD} threshold. "
            f"AUC = Σ(H×W) by trapezoidal rule. "
            f"H = (TPR_i + TPR_{{i-1}})/2  |  W = FPR_i − FPR_{{i-1}}."
        ),
        "information_gap": 0.0,
    })
    rows.append(footer)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"roc_auc_technical_summary_{split}.csv")
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return out_path


# ===========================================================================
# ROC AUC CSV OUTPUT 2 — CONCISE SUMMARY (row-level AUC accumulation)
# ===========================================================================

def save_roc_auc_concise_summary(
    results: dict,
    y_true: np.ndarray,
    split: str,
    output_dir: str,
    period_str: str = "Test Period",
    split_index=None,
) -> str:
    """
    Write a compact ROC AUC summary CSV showing how AUC accumulates row by row.

    Schema: model | day | H | W | ROC AUC | remarks

    One row per test observation, sorted by descending predicted probability
    (the exact order the threshold sweep traverses). For each row:

      day     = date of the observation
      H       = trapezoid HEIGHT contributed by this row
                = (TPR_prev + TPR_curr) / 2
                  where TPR is recomputed after admitting this row
      W       = trapezoid WIDTH contributed by this row
                = FPR_curr − FPR_prev
                  where FPR is recomputed after admitting this row
      ROC AUC = cumulative AUC so far  (= Σ H×W up to and including this row)
                → at the very last row this equals the final model AUC exactly
      remarks = CLEAR / WATCH / WARNING / DANGER tier  |  actual label  |
                running TP / FP counts

    TOTAL row: H = ΣH, W = ΣW (= 1.0), ROC AUC = final AUC, remarks = verdict.

    HOW THE ACCUMULATION WORKS
    --------------------------
    n_pos = total actual flood rows in the split
    n_neg = total actual non-flood rows in the split

    After admitting row k (sorted by descending prob):
        tp_k = number of actual flood rows admitted so far
        fp_k = number of actual non-flood rows admitted so far
        TPR_k = tp_k / n_pos
        FPR_k = fp_k / n_neg
        H_k   = (TPR_{k-1} + TPR_k) / 2
        W_k   = FPR_k − FPR_{k-1}
        area_k = H_k × W_k          ← this row's contribution to AUC
        cumAUC_k = cumAUC_{k-1} + area_k

    At k = n_rows: cumAUC = sklearn roc_auc_score (to floating-point precision).
    """
    COLS = ["model", "day", "H", "W", "ROC AUC", "remarks"]

    def _blank_row():
        return {c: "" for c in COLS}

    rows = []
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())

    for name, r in results.items():
        probs    = r["probs"]
        auc_val  = float(roc_auc_score(y_true, probs))
        remarks  = _REMARKS_MAP.get(name, "")
        watch_t  = r["watch_t"]
        warn_t   = r["warn_t"]
        danger_t = r["danger_t"]

        # Sort all test rows by descending predicted probability —
        # this is the threshold-sweep traversal order.
        sort_idx      = np.argsort(probs)[::-1]
        sorted_probs  = probs[sort_idx]
        sorted_labels = y_true[sort_idx]

        if split_index is not None:
            sorted_dates = np.asarray(split_index)[sort_idx]
        else:
            sorted_dates = sort_idx          # fallback: integer position

        # Column header sub-row
        rows.append({
            "model":   f"── {name} ──",
            "day":     "date",
            "H":       "H = (TPR_prev+TPR_curr)/2",
            "W":       "W = FPR_curr−FPR_prev",
            "ROC AUC": "cumulative AUC (= final AUC at last row)",
            "remarks": "tier | actual | running TP | running FP",
        })

        # Walk through every row, accumulating TPR, FPR, and AUC
        tpr_prev = 0.0
        fpr_prev = 0.0
        cum_auc  = 0.0
        cum_H    = 0.0
        cum_W    = 0.0
        tp       = 0
        fp       = 0

        for k, (date, label, prob) in enumerate(
            zip(sorted_dates, sorted_labels, sorted_probs), start=1
        ):
            # Admit this row — update TP or FP
            if label == 1:
                tp += 1
            else:
                fp += 1

            tpr_curr = tp / n_pos if n_pos > 0 else 0.0
            fpr_curr = fp / n_neg if n_neg > 0 else 0.0

            H       = (tpr_prev + tpr_curr) / 2.0
            W       = fpr_curr - fpr_prev
            area    = H * W
            cum_auc += area
            cum_H   += H
            cum_W   += W

            # Risk tier based on predicted probability
            if prob >= danger_t:
                tier = "DANGER"
            elif prob >= warn_t:
                tier = "WARNING"
            elif prob >= watch_t:
                tier = "WATCH"
            else:
                tier = "CLEAR"

            date_str = (str(date.date()) if hasattr(date, "date") else str(date))
            actual_str = "FLOOD" if label == 1 else "clear"

            rows.append({
                "model":   name,
                "day":     date_str,
                "H":       round(H,       8),
                "W":       round(W,       8),
                "ROC AUC": round(cum_auc, 8),
                "remarks": f"{tier} | {actual_str} | TP={tp} | FP={fp}",
            })

            tpr_prev = tpr_curr
            fpr_prev = fpr_curr

        # TOTAL row — cumulative AUC at the last row = final model AUC
        meets = "Meets" if auc_val >= _WMO_THRESHOLD else "Does NOT meet"
        rows.append({
            "model":   name,
            "day":     period_str,
            "H":       round(cum_H,  6),
            "W":       round(cum_W,  6),   # should be 1.0
            "ROC AUC": round(cum_auc, 6),  # should equal auc_val exactly
            "remarks": (
                f"{remarks} | "
                f"sklearn AUC={auc_val:.6f} | "
                f"row-accumulation AUC={cum_auc:.6f} | "
                f"{meets} WMO ≥{_WMO_THRESHOLD}"
            ),
        })

        # Blank separator between models
        rows.append(_blank_row())

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"roc_auc_concise_summary_{split}.csv")
    pd.DataFrame(rows, columns=COLS).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return out_path


# ===========================================================================
# ROC AUC FIGURE
# ===========================================================================

def save_roc_auc_figure(
    results: dict,
    y_true: np.ndarray,
    split: str,
    output_dir: str,
    period_str: str = "",
) -> str:
    """
    Produce a dedicated ROC AUC figure: roc_auc_figure_{split}.png

    Layout — top row: combined overlay of all three ROC curves on one axes.
             bottom rows: one panel per model showing its individual ROC curve
             with the trapezoid area shaded segment-by-segment and the WMO
             threshold line marked.

    Each panel shows:
      • The ROC curve (coloured line)
      • Shaded trapezoid fill under the curve (same colour, low alpha)
      • Random-chance diagonal (black dashed)
      • WMO 0.80 horizontal reference line (red dashed)
      • Annotated AUC value, n_test, n_flood, meets_wmo verdict
      • Threshold markers (WATCH, WARNING, DANGER) on the curve
    """
    n_models  = len(results)
    fig_rows  = 1 + n_models          # row 0 = overlay, rows 1..n = per-model
    fig, axes = plt.subplots(
        fig_rows, 1,
        figsize=(9, 4 + 4 * n_models),
        constrained_layout=True,
    )
    if fig_rows == 1:
        axes = [axes]

    n_pos = int(y_true.sum())
    n_tot = len(y_true)

    # ── Row 0: combined overlay ──────────────────────────────────────────────
    ax0 = axes[0]
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_true, r["probs"])
        auc_val     = float(roc_auc_score(y_true, r["probs"]))
        color       = MODEL_COLORS[name]
        meets       = auc_val >= _WMO_THRESHOLD
        linestyle   = "-" if meets else "--"
        ax0.plot(fpr, tpr, color=color, linewidth=2.2, linestyle=linestyle,
                 label=f"{name}  AUC={auc_val:.4f}  "
                       f"{'✓ WMO' if meets else '✗ WMO'}")
        ax0.fill_between(fpr, tpr, alpha=0.07, color=color)

    ax0.axline((0, 0), slope=1, color="black", linewidth=0.8,
               linestyle="--", alpha=0.5, label="Random chance")
    ax0.axhline(_WMO_THRESHOLD, color="red", linewidth=1.0, linestyle=":",
                alpha=0.7, label=f"WMO min AUC = {_WMO_THRESHOLD}")
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    ax0.set_xlabel("False Positive Rate (FPR)", fontsize=10)
    ax0.set_ylabel("True Positive Rate (TPR)", fontsize=10)
    ax0.set_title(
        f"ROC Curves — All Models — {split.upper()} Set\n"
        f"n={n_tot:,}  flood events={n_pos}  period: {period_str}",
        fontsize=11, fontweight="bold",
    )
    ax0.legend(loc="lower right", fontsize=9)
    ax0.grid(alpha=0.25)

    # ── Rows 1..n: per-model panels ──────────────────────────────────────────
    for panel_idx, (name, r) in enumerate(results.items(), start=1):
        ax   = axes[panel_idx]
        probs    = r["probs"]
        auc_val  = float(roc_auc_score(y_true, probs))
        color    = MODEL_COLORS[name]
        meets    = auc_val >= _WMO_THRESHOLD
        watch_t  = r["watch_t"]
        warn_t   = r["warn_t"]
        danger_t = r["danger_t"]

        fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, probs)

        # Shaded trapezoid fill — drawn segment by segment so the fill is
        # visible as a solid block under the stepped curve.
        for i in range(1, len(fpr_arr)):
            ax.fill_between(
                [fpr_arr[i - 1], fpr_arr[i]],
                [tpr_arr[i - 1], tpr_arr[i]],
                alpha=0.18, color=color,
            )

        ax.plot(fpr_arr, tpr_arr, color=color, linewidth=2.2,
                label=f"ROC curve  AUC = {auc_val:.4f}")
        ax.axline((0, 0), slope=1, color="black", linewidth=0.8,
                  linestyle="--", alpha=0.5, label="Random chance")
        ax.axhline(_WMO_THRESHOLD, color="red", linewidth=1.0, linestyle=":",
                   alpha=0.8, label=f"WMO min = {_WMO_THRESHOLD}")

        # Mark threshold operating points on the curve
        # Find the ROC point closest to each threshold value
        for t_val, t_label, t_color in [
            (watch_t,  f"WATCH ({watch_t:.2f})",   "gold"),
            (warn_t,   f"WARN ({warn_t:.2f})",     "orange"),
            (danger_t, f"DANGER ({danger_t:.2f})", "darkred"),
        ]:
            # thresh_arr is decreasing; find idx where threshold crosses t_val
            idx = np.searchsorted(-thresh_arr, -t_val)
            idx = min(idx, len(fpr_arr) - 1)
            ax.scatter(fpr_arr[idx], tpr_arr[idx], s=60, color=t_color,
                       zorder=5, edgecolors="black", linewidths=0.6)
            ax.annotate(
                t_label,
                xy=(fpr_arr[idx], tpr_arr[idx]),
                xytext=(fpr_arr[idx] + 0.03, tpr_arr[idx] - 0.06),
                fontsize=7, color=t_color,
                arrowprops=dict(arrowstyle="-", color=t_color, lw=0.8),
            )

        # AUC verdict box
        verdict_txt = (
            f"AUC = {auc_val:.4f}\n"
            f"WMO ≥ {_WMO_THRESHOLD} : {'PASS ✓' if meets else 'FAIL ✗'}\n"
            f"n_test = {n_tot:,}  flood = {n_pos}"
        )
        ax.text(
            0.97, 0.05, verdict_txt,
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=color, alpha=0.85),
        )

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=9)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=9)
        ax.set_title(
            f"{name} — ROC Curve with Trapezoid Fill — {split.upper()} Set",
            fontsize=10, fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Flood Early Warning — ROC AUC Analysis  [{split.upper()} Set]\n"
        f"Trapezoidal rule: AUC = Σ(H×W)  |  H=(TPR_i+TPR_{{i-1}})/2  "
        f"W=FPR_i−FPR_{{i-1}}",
        fontsize=12, fontweight="bold",
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"roc_auc_figure_{split}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ===========================================================================
# LIVE CROSS-REFERENCE
# ===========================================================================

def run_live_cross_reference(
    label_file: str,
    output_dir: str,
) -> None:
    separator("Live Prediction Cross-Reference (All 3 Models vs Labeled Ground Truth)")

    if not os.path.exists(label_file):
        print(f"  ⚠️  Label file not found: {label_file}")
        print("      Provide flood_dataset_test.csv or pass --label-file <path>.")
        return

    labels = pd.read_csv(label_file, parse_dates=["timestamp"],
                         index_col="timestamp")
    labels.index = labels.index.tz_localize(None).normalize()
    print(f"  Label file       : {label_file}")
    print(f"  Label rows       : {len(labels):,}")
    print(f"  Label date range : {labels.index.min().date()} → {labels.index.max().date()}")
    print(f"  Flood=1 in labels: {int(labels['flood_label'].sum())}  "
          f"({labels['flood_label'].mean()*100:.1f}%)")

    summary_rows = []
    detail_rows  = []

    for model_name, predictions_csv in LIVE_PREDICTIONS.items():
        print(f"\n  {'─'*60}")
        print(f"  Model : {model_name}")
        print(f"  CSV   : {predictions_csv}")

        if not os.path.exists(predictions_csv):
            print(f"  ⚠️  Predictions CSV not found — skipping.")
            print(f"      Run the corresponding predict script first.")
            continue

        preds = pd.read_csv(predictions_csv, parse_dates=["timestamp"],
                            index_col="timestamp")
        preds.index = preds.index.tz_localize(None).normalize()
        print(f"  Predictions rows : {len(preds):,}")
        print(f"  Pred date range  : {preds.index.min().date()} → {preds.index.max().date()}")

        overlap = preds.index.intersection(labels.index)
        if len(overlap) == 0:
            print(f"  ⚠️  No overlapping dates found between predictions and labels.")
            print(f"      Label range : {labels.index.min().date()} → {labels.index.max().date()}")
            print(f"      Pred  range : {preds.index.min().date()} → {preds.index.max().date()}")
            continue

        preds_aligned  = preds.loc[overlap]
        labels_aligned = labels.loc[overlap, "flood_label"].astype(int)

        print(f"  Overlapping dates: {len(overlap):,}")
        print(f"  Overlap range    : {overlap.min().date()} → {overlap.max().date()}")
        print(f"  Flood=1 in window: {int(labels_aligned.sum())}")

        tier_col = "risk_tier"
        if tier_col not in preds_aligned.columns:
            print(f"  ⚠️  Column '{tier_col}' not found in predictions CSV.")
            print(f"      Available columns: {list(preds_aligned.columns)}")
            continue

        tiers = preds_aligned[tier_col]

        tier_levels = {
            "CLEAR":   (tiers == "CLEAR").astype(int),
            "WATCH":   tiers.isin(["WATCH", "WARNING", "DANGER"]).astype(int),
            "WARNING": tiers.isin(["WARNING", "DANGER"]).astype(int),
            "DANGER":  (tiers == "DANGER").astype(int),
        }

        has_prob = "flood_probability" in preds_aligned.columns
        y_prob   = preds_aligned["flood_probability"].values if has_prob else None
        y_true   = labels_aligned.values

        print(f"\n  Accuracy = (TP + TN) / (TP + TN + FP + FN)\n")
        print(f"  {'Tier':<10} {'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}  "
              f"{'Accuracy':>9}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  "
              f"{'BalAcc':>8}  {'AUC':>8}")
        print(f"  {'-'*10} {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  "
              f"{'-'*9}  {'-'*10}  {'-'*8}  {'-'*6}  "
              f"{'-'*8}  {'-'*8}")

        for tier_name, y_pred in tier_levels.items():
            tp, tn, fp, fn = _confusion_counts(y_true, y_pred.values)
            acc  = _accuracy(tp, tn, fp, fn)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            f1   = f1_score(y_true, y_pred, zero_division=0)
            bal  = balanced_accuracy_score(y_true, y_pred)
            auc  = roc_auc_score(y_true, y_prob) if has_prob else float("nan")

            acc_flag = "✅" if acc >= 0.80 else "⚠️ "
            rec_flag = "✅" if rec >= 0.90 else "⚠️ "
            bal_flag = "✅" if bal >= 0.75 else "⚠️ "

            auc_str = f"{auc:>8.4f}" if has_prob else f"{'N/A':>8}"
            print(f"  {tier_name:<10} {tp:>5}  {tn:>5}  {fp:>5}  {fn:>5}  "
                  f"  {acc_flag}{acc:>6.3f}  {prec:>10.3f}  "
                  f"{rec_flag}{rec:>6.3f}  {f1:>6.3f}  "
                  f"{bal_flag}{bal:>6.3f}  {auc_str}")

            summary_rows.append({
                "model":             model_name,
                "predictions_csv":   predictions_csv,
                "tier_level":        tier_name,
                "overlapping_dates": len(overlap),
                "flood_events":      int(labels_aligned.sum()),
                "tp":                tp,
                "tn":                tn,
                "fp":                fp,
                "fn":                fn,
                "accuracy":          round(acc,  4),
                "precision":         round(prec, 4),
                "recall":            round(rec,  4),
                "f1":                round(f1,   4),
                "balanced_acc":      round(bal,  4),
                "roc_auc":           round(auc,  4) if has_prob else None,
                "label_file":        label_file,
                "overlap_start":     str(overlap.min().date()),
                "overlap_end":       str(overlap.max().date()),
            })

        print(f"\n  Targets : accuracy >= 0.80  recall >= 0.90  bal_acc >= 0.75")
        print(f"  Note    : CLEAR   tier counts predicting CLEAR as positive.")
        print(f"            WATCH   tier counts any non-CLEAR prediction as positive.")
        print(f"            WARNING tier counts WARNING or DANGER as positive.")
        print(f"            DANGER  tier counts only DANGER as positive.")

        print(f"\n  Per-date breakdown (overlapping dates only):")
        print(f"  {'Date':<14} {'Actual':>8}  {'Tier':<10} {'Prob':>7}  {'Match?':>8}")
        print(f"  {'-'*14} {'-'*8}  {'-'*10} {'-'*7}  {'-'*8}")
        for ts in overlap:
            actual  = int(labels_aligned.loc[ts])
            tier    = tiers.loc[ts]
            prob_v  = (preds_aligned.loc[ts, "flood_probability"]
                       if has_prob else None)
            prob_s  = f"{prob_v:.3f}" if prob_v is not None else "N/A"
            alerted = tier in ("WATCH", "WARNING", "DANGER")
            match   = (actual == 1 and alerted) or (actual == 0 and not alerted)
            match_s = "✅ correct" if match else "❌ wrong"
            print(f"  {str(ts.date()):<14} {'FLOOD' if actual==1 else 'clear':>8}  "
                  f"{tier:<10} {prob_s:>7}  {match_s:>8}")

            detail_rows.append({
                "model":            model_name,
                "date":             str(ts.date()),
                "actual_label":     actual,
                "actual_label_str": "FLOOD" if actual == 1 else "clear",
                "risk_tier":        tier,
                "flood_probability": round(prob_v, 4) if prob_v is not None else None,
                "alerted":          int(alerted),
                "match_watch":      int(match),
                "match_warning":    int(
                    (actual == 1 and tier in ("WARNING", "DANGER")) or
                    (actual == 0 and tier not in ("WARNING", "DANGER"))
                ),
                "match_danger":     int(
                    (actual == 1 and tier == "DANGER") or
                    (actual == 0 and tier != "DANGER")
                ),
                "predictions_csv":  predictions_csv,
                "label_file":       label_file,
            })

    if not summary_rows:
        print("\n  ⚠️  No live prediction data could be loaded for any model.")
        return

    os.makedirs(output_dir, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)

    live_numeric = [
        "overlapping_dates", "flood_events",
        "tp", "tn", "fp", "fn",
        "accuracy", "precision", "recall", "f1", "balanced_acc", "roc_auc",
    ]
    live_avg = {"model": "AVERAGE", "tier_level": "ALL"}
    for col in live_numeric:
        if col in summary_df.columns:
            live_avg[col] = round(
                pd.to_numeric(summary_df[col], errors="coerce").mean(), 4
            )
    summary_df = pd.concat(
        [summary_df, pd.DataFrame([live_avg])], ignore_index=True
    )

    summary_csv = os.path.join(output_dir, "live_prediction_accuracy_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Summary CSV (totals)   saved → {summary_csv}")

    detail_csv = os.path.join(output_dir, "live_prediction_accuracy_allrows.csv")
    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False)
    print(f"  Detail  CSV (all rows) saved → {detail_csv}")


# ===========================================================================
# MAIN
# ===========================================================================

def main(split: str = "test", live_check: bool = False,
         label_file: str = DEFAULT_LABEL_FILE):

    separator(f"Flood Prediction — ML Comparison Report  [{split.upper()} SET]")
    print(f"  Data file  : {DATA_FILE}")
    print(f"  Models     : {list(MODELS.keys())}  ")
    print(f"  Split      : {split}")
    print(f"  Accuracy   : (TP + TN) / (TP + TN + FP + FN)")

    _print_split_context_note(split)

    separator("Loading Dataset")
    if not os.path.exists(DATA_FILE):
        sys.exit(f"\n  ERROR: Dataset not found.\n  Expected: {DATA_FILE}\n"
                 f"  Run prepare_dataset.py first.")

    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()
    print(f"  Total rows      : {len(df):,}")
    print(f"  Date range      : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Flood=1         : {int(df['flood_label'].sum())}  "
          f"({df['flood_label'].mean()*100:.1f}%)")

    split_df = get_split(df, split)
    y_true   = split_df["flood_label"].values.astype(int)
    print(f"\n  {split.upper()} split rows : {len(split_df):,}")
    print(f"  {split.upper()} flood=1    : {int(y_true.sum())}  "
          f"({y_true.mean()*100:.1f}%)")
    print(f"  Date range      : {split_df.index.min().date()} -> "
          f"{split_df.index.max().date()}")

    if y_true.sum() == 0:
        sys.exit(f"\n  ERROR: No flood events in the {split} split.")

    period_str = (f"{split_df.index.min().date()} "
                  f"→ {split_df.index.max().date()}")

    separator("Loading Models & Running Predictions")

    results      = {}
    missing_pkls = []

    for name, pkl_path in MODELS.items():
        if not os.path.exists(pkl_path):
            print(f"  ⚠️  {name:<18} : NOT FOUND — {pkl_path}")
            missing_pkls.append(name)
            continue

        artifact     = joblib.load(pkl_path)
        model        = artifact["model"]
        feature_cols = artifact["feature_columns"]
        watch_t      = artifact.get("watch_threshold",   0.03)
        warn_t       = artifact.get("warning_threshold", 0.26)
        version      = artifact.get("version", "?")
        flood_weight = artifact.get("flood_weight", "?")

        if "danger_threshold" in artifact:
            danger_t      = artifact["danger_threshold"]
            danger_source = "pkl"
        else:
            danger_t      = round(warn_t + DANGER_FALLBACK_OFFSET, 2)
            danger_source = f"fallback (warn+{DANGER_FALLBACK_OFFSET})"
            print(f"  ⚠️  {name}: danger_threshold not in pkl — "
                  f"using fallback {danger_t:.2f}")

        avail   = [c for c in feature_cols if c in split_df.columns]
        missing = [c for c in feature_cols if c not in split_df.columns]
        if missing:
            print(f"  ⚠️  {name}: {len(missing)} features missing in dataset.")

        X     = split_df[avail].values
        probs = model.predict_proba(X)[:, 1]

        clear_metrics   = compute_clear_metrics(y_true, probs, watch_t)
        watch_metrics   = compute_metrics(y_true, probs, watch_t)
        warning_metrics = compute_metrics(y_true, probs, warn_t)
        danger_metrics  = compute_metrics(y_true, probs, danger_t)

        watch_warn_gap  = round(warn_t   - watch_t,  2)
        warn_danger_gap = round(danger_t - warn_t,   2)
        tier_gaps_ok    = watch_warn_gap >= 0.08 and warn_danger_gap >= 0.08

        results[name] = {
            "clear":          clear_metrics,
            "watch":          watch_metrics,
            "warning":        warning_metrics,
            "danger":         danger_metrics,
            "probs":          probs,
            "watch_t":        watch_t,
            "warn_t":         warn_t,
            "danger_t":       danger_t,
            "danger_source":  danger_source,
            "version":        version,
            "flood_weight":   flood_weight,
            "val_balanced_acc":  artifact.get("val_balanced_acc",  None),
            "test_balanced_acc": artifact.get("test_balanced_acc", None),
            "val_auc":           artifact.get("val_auc",           None),
            "test_auc":          artifact.get("test_auc",          None),
            "feature_count":  len(avail),
            "calibration_fold":   artifact.get("calibration_fold",   "val_set"),
            "calibration_method": artifact.get("calibration_method", "isotonic"),
            "tier_gaps_ok":   tier_gaps_ok,
            "watch_warn_gap": watch_warn_gap,
            "warn_danger_gap":warn_danger_gap,
        }

        rec_status  = "✅" if warning_metrics["recall"]    >= 0.90  else "⚠️ "
        bal_status  = "✅" if warning_metrics["bal_acc"]   >= 0.85  else "⚠️ "
        acc_status  = "✅" if warning_metrics["accuracy"]  >= 0.80  else "⚠️ "
        gap_status  = "✅" if tier_gaps_ok                           else "⚠️ "
        miss_flag   = "✅" if clear_metrics["miss_rate"]   <= 0.05  else "⚠️ "
        print(f"  {rec_status}  {name:<18} v{version}  "
              f"weight={flood_weight}  features={len(avail)}  "
              f"watch={watch_t:.2f}  warn={warn_t:.2f}  danger={danger_t:.2f} ({danger_source})  "
              f"gaps={watch_warn_gap:.2f}/{warn_danger_gap:.2f}{gap_status}  "
              f"Recall(warn)={warning_metrics['recall']:.4f}  "
              f"BalAcc(warn)={warning_metrics['bal_acc']:.4f}  "
              f"Acc(warn)={warning_metrics['accuracy']:.4f}{acc_status}  "
              f"MissRate(clear)={clear_metrics['miss_rate']:.4f}{miss_flag}")

    if not results:
        sys.exit("\n  ERROR: No models loaded. Run all three train scripts first.")

    if missing_pkls:
        print(f"\n  ⚠️  Missing model files: {missing_pkls}")

    for level_name, key in [
        ("CLEAR",   "clear"),
        ("WATCH",   "watch"),
        ("WARNING", "warning"),
        ("DANGER",  "danger"),
    ]:
        separator(f"Metric Comparison — {split.upper()} Set — {level_name} Threshold")
        _print_metric_table(results, threshold_key=key, y_true=y_true, level=level_name)

    separator(f"All-Threshold Summary — {split.upper()} Set")
    print(f"\n  {'Model':<20} {'Threshold':>10}  {'Level':<8}  "
          f"{'Accuracy':>9}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  "
          f"{'BalAcc':>8}  {'ROC-AUC':>8}  "
          f"{'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}  {'Note':<30}")
    print(f"  {'-'*20} {'-'*10}  {'-'*8}  "
          f"{'-'*9}  {'-'*10}  {'-'*8}  {'-'*6}  "
          f"{'-'*8}  {'-'*8}  "
          f"{'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*30}")

    for name, r in results.items():
        for level_name, key, t_key in [
            ("CLEAR",   "clear",   "watch_t"),
            ("WATCH",   "watch",   "watch_t"),
            ("WARNING", "warning", "warn_t"),
            ("DANGER",  "danger",  "danger_t"),
        ]:
            m    = r[key]
            t    = r[t_key]
            note = ""
            if level_name == "CLEAR":
                note = (f"miss_rate={m['miss_rate']:.3f}  "
                        f"missed={m['n_missed']}")
            print(f"  {name:<20} {t:>10.2f}  {level_name:<8}  "
                  f"{m['accuracy']:>9.3f}  {m['precision']:>10.3f}  "
                  f"{m['recall']:>8.3f}  {m['f1']:>6.3f}  "
                  f"{m['bal_acc']:>8.3f}  {m['roc_auc']:>8.4f}  "
                  f"{m['tp']:>5}  {m['tn']:>5}  {m['fp']:>5}  {m['fn']:>5}  "
                  f"{note:<30}")
        print()

    separator("Tier Gap Summary")
    print(f"\n  {'Model':<20} {'WATCH':>7}  {'WARNING':>8}  {'DANGER':>7}  "
          f"{'W→WN gap':>9}  {'WN→D gap':>9}  {'OK?':>5}  {'Danger Source':<25}")
    print(f"  {'-'*20} {'-'*7}  {'-'*8}  {'-'*7}  "
          f"{'-'*9}  {'-'*9}  {'-'*5}  {'-'*25}")
    for name, r in results.items():
        gap_flag = "✅" if r["tier_gaps_ok"] else "⚠️ "
        print(f"  {name:<20} {r['watch_t']:>7.2f}  {r['warn_t']:>8.2f}  "
              f"{r['danger_t']:>7.2f}  {r['watch_warn_gap']:>9.2f}  "
              f"{r['warn_danger_gap']:>9.2f}  {gap_flag:>5}  {r['danger_source']:<25}")
    print(f"\n  Minimum required gap : 0.08 (MIN_TIER_GAP)")

    separator("Probability Distribution Summary")
    print(f"  {'Model':<20} {'Min':>6}  {'Mean':>6}  {'Max':>6}  {'Std':>6}  "
          f"{'CLEAR':>6}  {'WATCH':>6}  {'WARN':>6}  {'DANGER':>6}")
    print(f"  {'-'*20} {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  "
          f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for name, r in results.items():
        probs    = r["probs"]
        watch_t  = r["watch_t"]
        warn_t   = r["warn_t"]
        danger_t = r["danger_t"]
        n_clear  = (probs < watch_t).sum()
        n_watch  = ((probs >= watch_t) & (probs < warn_t)).sum()
        n_warn   = ((probs >= warn_t)  & (probs < danger_t)).sum()
        n_danger = (probs >= danger_t).sum()
        std_flag = "✅" if probs.std() > 0.10 else "⚠️ "
        print(f"  {name:<20} {probs.min():>6.3f}  {probs.mean():>6.3f}  "
              f"{probs.max():>6.3f}  {probs.std():>6.3f}{std_flag}  "
              f"{n_clear:>6}  {n_watch:>6}  {n_warn:>6}  {n_danger:>6}")

    separator("Stored Training Metrics (from pkl artifacts)")
    print(f"  {'Model':<20} {'Cal Fold':<20} {'Cal Method':<16} "
          f"{'Val AUC':>8}  {'Val BalAcc':>10}  {'Test AUC':>8}  {'Test BalAcc':>11}")
    print(f"  {'-'*20} {'-'*20} {'-'*16} {'-'*8}  {'-'*10}  {'-'*8}  {'-'*11}")
    for name, r in results.items():
        va  = f"{r['val_auc']:.4f}"          if r["val_auc"]           is not None else "N/A"
        vb  = f"{r['val_balanced_acc']:.4f}" if r["val_balanced_acc"]  is not None else "N/A"
        ta  = f"{r['test_auc']:.4f}"         if r["test_auc"]          is not None else "N/A"
        tb  = f"{r['test_balanced_acc']:.4f}"if r["test_balanced_acc"] is not None else "N/A"
        cf  = r.get("calibration_fold",   "val_set")
        cm  = r.get("calibration_method", "isotonic")
        print(f"  {name:<20} {cf:<20} {cm:<16} {va:>8}  {vb:>10}  {ta:>8}  {tb:>11}")

    separator("Best Model Recommendation")
    scored = sorted(
        results.items(),
        key=lambda x: (
            x[1]["warning"]["bal_acc"] * 0.4 +
            x[1]["warning"]["recall"]  * 0.4 +
            x[1]["warning"]["roc_auc"] * 0.2
        ),
        reverse=True,
    )
    print(f"  Ranking (0.4×BalAcc + 0.4×Recall + 0.2×AUC at WARNING — recall-weighted):\n")
    for rank, (name, r) in enumerate(scored, 1):
        w        = r["warning"]
        c        = r["clear"]
        score    = w["bal_acc"]*0.4 + w["recall"]*0.4 + w["roc_auc"]*0.2
        rec_ok   = "✅" if w["recall"]       >= 0.90 else "⚠️ "
        bal_ok   = "✅" if w["bal_acc"]      >= 0.85 else "⚠️ "
        acc_ok   = "✅" if w["accuracy"]     >= 0.80 else "⚠️ "
        gap_ok   = "✅" if r["tier_gaps_ok"]          else "⚠️ "
        miss_ok  = "✅" if c["miss_rate"]    <= 0.05  else "⚠️ "
        status   = "  ← RECOMMENDED" if rank == 1 else ""
        print(f"  #{rank}  {name:<20}  score={score:.4f}  "
              f"Recall={w['recall']:.4f}{rec_ok}  "
              f"BalAcc={w['bal_acc']:.4f}{bal_ok}  "
              f"Accuracy={w['accuracy']:.4f}{acc_ok}  "
              f"AUC={w['roc_auc']:.4f}  "
              f"TierGaps={gap_ok}  "
              f"MissRate={c['miss_rate']:.4f}{miss_ok}  "
              f"TP={w['tp']}  TN={w['tn']}  FP={w['fp']}  FN={w['fn']}"
              f"{status}")

    separator("Saving CSV Report")
    rows = []
    for name, r in results.items():
        for thresh_name, m in [
            ("CLEAR",   r["clear"]),
            ("WATCH",   r["watch"]),
            ("WARNING", r["warning"]),
            ("DANGER",  r["danger"]),
        ]:
            row = {
                "model":              name,
                "split":              split,
                "threshold_level":    thresh_name,
                "threshold":          round(m["threshold"], 4),
                "accuracy":           round(m["accuracy"],    4),
                "precision":          round(m["precision"],   4),
                "recall":             round(m["recall"],      4),
                "f1":                 round(m["f1"],          4),
                "balanced_acc":       round(m["bal_acc"],     4),
                "roc_auc":            round(m["roc_auc"],     4),
                "specificity":        round(m["specificity"], 4),
                "tp":                 m["tp"],
                "tn":                 m["tn"],
                "fp":                 m["fp"],
                "fn":                 m["fn"],
                "alerts":             m["alerts"],
                "val_auc":            r["val_auc"],
                "val_bal_acc":        r["val_balanced_acc"],
                "test_auc":           r["test_auc"],
                "test_bal_acc":       r["test_balanced_acc"],
                "feature_count":      r["feature_count"],
                "flood_weight":       r["flood_weight"],
                "version":            r["version"],
                "calibration_fold":   r.get("calibration_fold",   "val_set"),
                "calibration_method": r.get("calibration_method", "isotonic"),
                "danger_source":      r["danger_source"],
                "watch_warn_gap":     r["watch_warn_gap"],
                "warn_danger_gap":    r["warn_danger_gap"],
                "tier_gaps_ok":       r["tier_gaps_ok"],
            }
            if thresh_name == "CLEAR":
                row["clear_miss_rate"] = round(m["miss_rate"], 4)
                row["clear_n_missed"]  = m["n_missed"]
            else:
                row["clear_miss_rate"] = ""
                row["clear_n_missed"]  = ""
            rows.append(row)

    report_df = pd.DataFrame(rows)

    numeric_cols = [
        "accuracy", "precision", "recall", "f1", "balanced_acc",
        "roc_auc", "specificity", "tp", "tn", "fp", "fn", "alerts",
        "val_auc", "val_bal_acc", "test_auc", "test_bal_acc",
        "watch_warn_gap", "warn_danger_gap", "clear_miss_rate",
    ]
    avg_row = {"model": "AVERAGE", "split": split, "threshold_level": "ALL"}
    for col in numeric_cols:
        if col in report_df.columns:
            avg_row[col] = round(
                pd.to_numeric(report_df[col], errors="coerce").mean(), 4
            )
    report_df = pd.concat(
        [report_df, pd.DataFrame([avg_row])], ignore_index=True
    )

    report_csv = os.path.join(OUTPUT_DIR, f"ml_comparison_report_{split}.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_df.to_csv(report_csv, index=False)
    print(f"  Saved → {report_csv}")

    separator("Saving Charts")
    plot_path = os.path.join(OUTPUT_DIR, f"ml_comparison_report_{split}.png")
    _save_comparison_plot(results, y_true, split, split_df.index, plot_path)

    separator("Saving ROC AUC Technical Summary CSV")
    roc_technical_csv = save_roc_auc_technical_summary(
        results    = results,
        y_true     = y_true,
        split      = split,
        output_dir = OUTPUT_DIR,
        period_str = period_str,
    )

    separator("Saving ROC AUC Concise Summary CSV")
    roc_concise_csv = save_roc_auc_concise_summary(
        results     = results,
        y_true      = y_true,
        split       = split,
        output_dir  = OUTPUT_DIR,
        period_str  = period_str,
        split_index = split_df.index,
    )

    separator("Saving ROC AUC Figure")
    roc_figure_path = save_roc_auc_figure(
        results    = results,
        y_true     = y_true,
        split      = split,
        output_dir = OUTPUT_DIR,
        period_str = period_str,
    )

    if live_check:
        run_live_cross_reference(
            label_file=label_file,
            output_dir=OUTPUT_DIR,
        )

    separator("DONE")
    print(f"  CSV report             : {report_csv}")
    print(f"  Chart                  : {plot_path}")
    print(f"  ROC AUC technical CSV  : {roc_technical_csv}")
    print(f"  ROC AUC concise CSV    : {roc_concise_csv}")
    print(f"  ROC AUC figure         : {roc_figure_path}")
    if live_check:
        print(f"  Live summary CSV  : {os.path.join(OUTPUT_DIR, 'live_prediction_accuracy_summary.csv')}")
        print(f"  Live detail  CSV  : {os.path.join(OUTPUT_DIR, 'live_prediction_accuracy_allrows.csv')}")
    separator()


# ===========================================================================
# PRINT TABLE HELPER
# ===========================================================================

def _print_metric_table(results: dict, threshold_key: str, y_true, level: str = ""):
    col_w = 20

    if level == "CLEAR":
        threshold_vals = {r["watch_t"] for r in results.values()}
        thresh_str = ", ".join(f"{t:.2f}" for t in sorted(threshold_vals))
        print(f"\n  Threshold level : CLEAR  (prob < watch_t = {thresh_str})")
        print(f"  Metrics are on the NEGATIVE CLASS (no-flood = label 0).")
        print(f"  precision = TN/(TN+FN)  — purity of CLEAR calls")
        print(f"  recall    = TN/(TN+FP)  — coverage of true safe rows")
        print(f"  miss_rate = FN/(FN+TP)  — floods incorrectly called CLEAR")
        print(f"  accuracy  = (TP+TN)/(TP+TN+FP+FN)\n")
        print(f"  {'Model':<{col_w}} {'Accuracy':>9}  {'Precision':>10}  {'Recall':>8}  "
              f"{'F1':>6}  {'BalAcc':>8}  {'ROC-AUC':>8}  "
              f"{'MissRate':>10}  {'Missed':>7}  {'N_Clear':>8}  "
              f"{'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}")
        print(f"  {'-'*col_w} {'-'*9}  {'-'*10}  {'-'*8}  "
              f"{'-'*6}  {'-'*8}  {'-'*8}  "
              f"{'-'*10}  {'-'*7}  {'-'*8}  "
              f"{'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")
        for name, r in results.items():
            m       = r[threshold_key]
            p_flag  = "✅" if m["precision"] >= 0.90        else "⚠️ "
            r_flag  = "✅" if m["recall"]    >= 0.90        else "⚠️ "
            mr_flag = "✅" if m["miss_rate"] <= 0.05        else "⚠️ "
            ac_flag = "✅" if m["accuracy"]  >= 0.80        else "⚠️ "
            print(f"  {name:<{col_w}} {ac_flag}{m['accuracy']:>7.3f}  "
                  f"{p_flag}{m['precision']:>8.3f}  "
                  f"{r_flag}{m['recall']:>6.3f}  {m['f1']:>6.3f}  "
                  f"{m['bal_acc']:>8.3f}  {m['roc_auc']:>8.4f}  "
                  f"{mr_flag}{m['miss_rate']:>8.3f}  {m['n_missed']:>7}  "
                  f"{m['alerts']:>8}  "
                  f"{m['tp']:>5}  {m['tn']:>5}  {m['fp']:>5}  {m['fn']:>5}")
        print(f"\n  Targets  : accuracy >= 0.80  precision >= 0.90  "
              f"recall >= 0.90  miss_rate <= 0.05")
        print(f"  Flood events in split : {int(y_true.sum())}")
        return

    threshold_vals = {r[threshold_key]["threshold"] for r in results.values()}
    thresh_str = ", ".join(f"{t:.2f}" for t in sorted(threshold_vals))
    print(f"\n  Threshold level : {level}  (values: {thresh_str})")
    print(f"  Accuracy = (TP + TN) / (TP + TN + FP + FN)\n")
    print(f"  {'Model':<{col_w}} {'Accuracy':>9}  {'Precision':>10}  {'Recall':>8}  "
          f"{'F1':>6}  {'BalAcc':>8}  {'ROC-AUC':>8}  {'Spec':>8}  {'Alerts':>7}  "
          f"{'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}")
    print(f"  {'-'*col_w} {'-'*9}  {'-'*10}  {'-'*8}  "
          f"{'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  "
          f"{'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")

    if level == "WATCH":
        target_met = {"precision": 0.30, "recall": 0.98, "bal_acc": 0.75, "accuracy": 0.70}
    elif level == "WARNING":
        target_met = {"precision": 0.55, "recall": 0.90, "bal_acc": 0.85, "accuracy": 0.80}
    else:
        target_met = {"precision": 0.70, "recall": 0.75, "bal_acc": 0.80, "accuracy": 0.80}

    for name, r in results.items():
        m      = r[threshold_key]
        p_flag = "✅" if m["precision"] >= target_met["precision"] else "⚠️ "
        r_flag = "✅" if m["recall"]    >= target_met["recall"]    else "⚠️ "
        b_flag = "✅" if m["bal_acc"]   >= target_met["bal_acc"]   else "⚠️ "
        a_flag = "✅" if m["accuracy"]  >= target_met["accuracy"]  else "⚠️ "
        print(f"  {name:<{col_w}} {a_flag}{m['accuracy']:>7.3f}  "
              f"{p_flag}{m['precision']:>8.3f}  "
              f"{r_flag}{m['recall']:>6.3f}  {m['f1']:>6.3f}  "
              f"{b_flag}{m['bal_acc']:>6.3f}  {m['roc_auc']:>8.4f}  "
              f"{m['specificity']:>8.3f}  {m['alerts']:>7}  "
              f"{m['tp']:>5}  {m['tn']:>5}  {m['fp']:>5}  {m['fn']:>5}")

    print(f"\n  Targets  : accuracy >= {target_met['accuracy']:.2f}  "
          f"precision >= {target_met['precision']:.2f}  "
          f"recall >= {target_met['recall']:.2f}  "
          f"bal_acc >= {target_met['bal_acc']:.2f}")
    print(f"  Flood events in split : {int(y_true.sum())}")


# ===========================================================================
# PLOT HELPER
# ===========================================================================

def _save_comparison_plot(results: dict, y_true, split: str,
                          index, plot_path: str) -> None:
    n_models = len(results)
    fig = plt.figure(figsize=(18, 4 * (n_models + 1)))
    gs  = fig.add_gridspec(n_models + 1, 3, hspace=0.45, wspace=0.35)

    ax_roc = fig.add_subplot(gs[0, :])
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_true, r["probs"])
        auc          = r["warning"]["roc_auc"]
        acc          = r["warning"]["accuracy"]
        ax_roc.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2,
                    label=f"{name}  AUC={auc:.4f}  Acc={acc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curves — {split.upper()} Set  (models)")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(alpha=0.3)

    metric_labels = ["Precision", "Recall", "F1", "BalAcc", "AUC", "Accuracy"]
    metric_keys   = ["precision", "recall", "f1", "bal_acc", "roc_auc", "accuracy"]

    for row_idx, (name, r) in enumerate(results.items(), start=1):
        probs    = r["probs"]
        watch_t  = r["watch_t"]
        warn_t   = r["warn_t"]
        danger_t = r["danger_t"]
        color    = MODEL_COLORS[name]

        ax_dist = fig.add_subplot(gs[row_idx, 0])
        ax_dist.hist(probs[y_true == 0], bins=30, alpha=0.6,
                     color="steelblue", label="No Flood", density=True)
        ax_dist.hist(probs[y_true == 1], bins=30, alpha=0.6,
                     color="red", label="Flood", density=True)
        ax_dist.axvline(watch_t,  color="gold",    linestyle="--", linewidth=1.2,
                        label=f"WATCH ({watch_t:.2f})")
        ax_dist.axvline(warn_t,   color="orange",  linestyle="--", linewidth=1.2,
                        label=f"WARN ({warn_t:.2f})")
        ax_dist.axvline(danger_t, color="darkred", linestyle="--", linewidth=1.2,
                        label=f"DANGER ({danger_t:.2f})")
        acc_w = r["warning"]["accuracy"]
        ax_dist.set_title(f"{name} — Prob Distribution  (Acc@WARN={acc_w:.3f})")
        ax_dist.set_xlabel("Flood Probability")
        ax_dist.set_ylabel("Density")
        ax_dist.legend(fontsize=7)
        ax_dist.grid(alpha=0.3)

        ax_bar = fig.add_subplot(gs[row_idx, 1])
        x     = np.arange(len(metric_labels))
        width = 0.2
        c_m        = r["clear"]
        clear_vals = [c_m[k] for k in metric_keys]
        w_vals     = [r["watch"][k]   for k in metric_keys]
        warn_vals  = [r["warning"][k] for k in metric_keys]
        danger_vals= [r["danger"][k]  for k in metric_keys]
        ax_bar.bar(x - 1.5*width, clear_vals,  width, label="CLEAR",   color="steelblue", alpha=0.50)
        ax_bar.bar(x - 0.5*width, w_vals,      width, label="WATCH",   color=color, alpha=0.40)
        ax_bar.bar(x + 0.5*width, warn_vals,   width, label="WARNING", color=color, alpha=0.70)
        ax_bar.bar(x + 1.5*width, danger_vals, width, label="DANGER",  color=color, alpha=1.00)
        ax_bar.axhline(0.90, color="red",   linestyle=":", linewidth=1.2, alpha=0.7,
                       label="Recall/Acc target (0.90)")
        ax_bar.axhline(0.85, color="black", linestyle=":", linewidth=1.2, alpha=0.7,
                       label="BalAcc target (0.85)")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=8)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_title(f"{name} — Metrics by Tier")
        ax_bar.legend(fontsize=7)
        ax_bar.grid(axis="y", alpha=0.3)

        ax_ts = fig.add_subplot(gs[row_idx, 2])
        ax_ts.plot(index, probs, color=color, linewidth=1, alpha=0.8)
        ax_ts.axhline(watch_t,  color="gold",    linestyle="--", linewidth=0.8,
                      label=f"WATCH ({watch_t:.2f})")
        ax_ts.axhline(warn_t,   color="orange",  linestyle="--", linewidth=0.8,
                      label=f"WARN ({warn_t:.2f})")
        ax_ts.axhline(danger_t, color="darkred", linestyle="--", linewidth=0.8,
                      label=f"DANGER ({danger_t:.2f})")
        flood_idx = index[y_true == 1]
        if len(flood_idx) > 0:
            ax_ts.vlines(flood_idx, 0, 1, color="red", alpha=0.12, linewidth=1)
        ax_ts.set_ylim(0, 1)
        ax_ts.set_xlabel("Date")
        ax_ts.set_ylabel("Probability")
        ax_ts.set_title(f"{name} — Probability Over Time")
        ax_ts.legend(fontsize=7, loc="upper left")
        ax_ts.grid(alpha=0.3)
        import matplotlib.dates as mdates
        ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.suptitle(
        f"Flood Prediction — Model Comparison Report  [{split.upper()} Set]",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {plot_path}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flood Prediction — ML Model Comparison Report"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "full"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--live-check",
        action="store_true",
        help=(
            "Cross-reference all three combined sensor+context prediction CSVs "
            "against labeled ground truth on overlapping dates."
        ),
    )
    parser.add_argument(
        "--label-file",
        type=str,
        default=DEFAULT_LABEL_FILE,
        help="Path to labeled CSV with flood_label column (default: ../data/flood_dataset_test.csv)",
    )
    args = parser.parse_args()
    main(
        split=args.split,
        live_check=args.live_check,
        label_file=args.label_file,
    )