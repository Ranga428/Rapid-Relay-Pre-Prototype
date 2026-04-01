"""
ml_comparison_report.py
========================
Flood Prediction — Model Comparison Report (XGBoost / RF / LightGBM)

Loads all three sensor model pkl files, runs them against the full
flood_dataset.csv, and produces:

  1. A printed comparison table (precision, recall, F1, balanced accuracy,
     ROC-AUC, specificity) at CLEAR, WATCH, WARNING, and DANGER thresholds.
  2. A multi-panel PNG chart comparing probability distributions and
     ROC curves for all three models.
  3. A CSV report saved to ../ml_report/ml_comparison_report_{split}.csv

USAGE
-----
    python ml_comparison_report.py
    python ml_comparison_report.py --split test     # only test set (default)
    python ml_comparison_report.py --split val      # only validation set
    python ml_comparison_report.py --split full     # full dataset

NOTE: This script must be in the same directory as the train_*.py scripts
      because it imports CalibratedRF and CalibratedLGBM from them at runtime
      so joblib can unpickle the saved models.

CHANGE vs previous version
---------------------------
  - CLEAR threshold section added. CLEAR is defined as prob < watch_t
    (the model issued no alert). Metrics are evaluated on the NEGATIVE class:
    precision = TN/(TN+FN), recall = TN/(TN+FP), F1 on class 0.
    This makes the CLEAR section a complete, self-contained view of
    how confidently the model declares safe conditions.
  - danger_t is now read from artifact["danger_threshold"] (set during
    training). Falls back to warn_t + 0.10 for backward compatibility
    with older pkls.

─────────────────────────────────────────────────────────────────────────────
FEATURE EXPLANATION
─────────────────────────────────────────────────────────────────────────────
These models are trained exclusively on SENSOR-DERIVED features, meaning all
inputs come from physical measurements collected at sensor stations (rain
gauges, water-level loggers, flow meters, soil-moisture probes). No external
forecast data or manually labelled weather categories are used.

Feature                       | Why it matters for flood prediction
──────────────────────────────┼──────────────────────────────────────────────
rainfall_mm                   | Directly measures precipitation intensity.
                              | High rainfall is the primary trigger for
                              | surface runoff and riverine flooding.
──────────────────────────────┼──────────────────────────────────────────────
rainfall_1h / 3h / 6h / 24h  | Rolling accumulated rainfall over multiple
(cumulative windows)          | windows captures antecedent moisture and
                              | sustained storm events that cause flooding
                              | even when instantaneous intensity is moderate.
──────────────────────────────┼──────────────────────────────────────────────
water_level_m                 | Real-time river/channel stage. Once water
                              | level approaches bank-full height, the
                              | marginal probability of inundation spikes.
──────────────────────────────┼──────────────────────────────────────────────
water_level_change_1h         | Rate of rise: a rapidly rising river is
                              | far more dangerous than a high but stable
                              | one, even at the same absolute level.
──────────────────────────────┼──────────────────────────────────────────────
flow_rate_m3s                 | Discharge integrates both level and velocity.
                              | Flash floods often show a flow spike before
                              | a visible level increase at downstream gauges.
──────────────────────────────┼──────────────────────────────────────────────
soil_moisture_pct             | Saturated soil cannot absorb more rainfall,
                              | converting nearly all precipitation to runoff
                              | and dramatically shortening flood lag times.
──────────────────────────────┼──────────────────────────────────────────────
hour_of_day / day_of_week /   | Cyclical time features encode diurnal and
month / season (encoded)      | seasonal patterns — monsoon seasons, afternoon
                              | convective storms, and weekend recreational
                              | risk windows.
──────────────────────────────┼──────────────────────────────────────────────
lag features (various)        | Lagged copies of the above at 1 h, 3 h, 6 h
                              | allow the model to reason about how conditions
                              | evolved over the previous hours, which is
                              | critical for early-warning lead time.
──────────────────────────────┼──────────────────────────────────────────────
rolling_std / rolling_max     | Statistical summaries of sensor variance
                              | detect abrupt anomalies (e.g., a sudden
                              | rainfall burst) that single-point readings
                              | can miss.

─────────────────────────────────────────────────────────────────────────────
FULL-DATASET EVALUATION vs. SENSOR-ONLY MODELS
─────────────────────────────────────────────────────────────────────────────
  Sensor-only models (these pkl files)
  ─────────────────────────────────────
  • Trained and evaluated using ONLY the sensor columns listed above.
  • Ground truth (flood_label) was derived from water-level exceedance
    thresholds, not from manual inspection or external event catalogues.
  • Suitable for real-time deployment at gauged river reaches where no
    forecast or satellite data is available.

  Full-dataset evaluation (--split full)
  ──────────────────────────────────────
  • Runs the sensor models over ALL rows in flood_dataset.csv, including
    the TRAINING period (pre-2024-06-30).
  • Metrics will be OPTIMISTICALLY BIASED because the models have already
    seen those rows during fitting. Use --split full only for sanity checks
    or distribution plots, NOT for honest performance reporting.
  • For unbiased reporting always use --split test (post-2025-06-30).

─────────────────────────────────────────────────────────────────────────────
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


def compute_metrics(y_true, y_prob, threshold):
    """Metrics for an ACTIVE alert level (WATCH / WARNING / DANGER).
    Positive class = flood (label=1). Predictions: prob >= threshold → flood."""
    y_pred = (y_prob >= threshold).astype(int)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    bal    = balanced_accuracy_score(y_true, y_pred)
    auc    = roc_auc_score(y_true, y_prob)
    tn     = ((y_pred == 0) & (y_true == 0)).sum()
    fp     = ((y_pred == 1) & (y_true == 0)).sum()
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0
    alerts = int(y_pred.sum())
    return {
        "threshold":   threshold,
        "precision":   prec,
        "recall":      rec,
        "f1":          f1,
        "bal_acc":     bal,
        "roc_auc":     auc,
        "specificity": spec,
        "alerts":      alerts,
    }


def compute_clear_metrics(y_true, y_prob, watch_threshold):
    """Metrics for the CLEAR zone (prob < watch_t — no alert issued).

    CLEAR is the negative-class prediction: the model says conditions are safe.
    We evaluate it on the negative class (no-flood = label 0) so that the
    metrics read intuitively:

      precision  = of all rows called CLEAR, what fraction were truly no-flood?
                   = TN / (TN + FN)   [how clean are the CLEAR calls?]
      recall     = of all true no-flood rows, what fraction were called CLEAR?
                   = TN / (TN + FP)   [how completely does CLEAR cover safe rows?]
      f1         = harmonic mean of the above two
      miss_rate  = of all true flood rows, what fraction were missed (called CLEAR)?
                   = FN / (FN + TP)   [the operational false-sense-of-safety rate]
      n_clear    = total rows called CLEAR
      n_missed   = flood rows incorrectly called CLEAR (false negatives at WATCH level)
    """
    y_pred_watch = (y_prob >= watch_threshold).astype(int)  # 1 = some alert issued
    y_clear      = (y_pred_watch == 0).astype(int)          # 1 = CLEAR predicted

    # Treat no-flood (0) as the positive class for CLEAR metrics
    y_true_neg = (y_true == 0).astype(int)

    prec = precision_score(y_true_neg, y_clear, zero_division=0)
    rec  = recall_score(y_true_neg, y_clear, zero_division=0)
    f1   = f1_score(y_true_neg, y_clear, zero_division=0)
    bal  = balanced_accuracy_score(y_true, y_pred_watch)  # same as WATCH bal_acc

    auc  = roc_auc_score(y_true, y_prob)

    tn = int(((y_clear == 1) & (y_true == 0)).sum())   # correctly called CLEAR
    fn = int(((y_clear == 1) & (y_true == 1)).sum())   # floods missed (called CLEAR)
    tp = int(((y_clear == 0) & (y_true == 1)).sum())   # floods caught
    fp = int(((y_clear == 0) & (y_true == 0)).sum())   # false alarms at WATCH

    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    n_clear   = int(y_clear.sum())

    return {
        "threshold":   watch_threshold,   # CLEAR boundary = watch_t
        "precision":   prec,              # TN / (TN+FN) — purity of CLEAR calls
        "recall":      rec,               # TN / (TN+FP) — coverage of safe rows
        "f1":          f1,
        "bal_acc":     bal,               # same as WATCH balanced accuracy
        "roc_auc":     auc,
        "specificity": rec,               # same as recall here (negative-class coverage)
        "alerts":      n_clear,           # number of CLEAR predictions
        "miss_rate":   miss_rate,         # floods incorrectly called CLEAR
        "n_missed":    fn,                # raw count of missed floods
        "tn":          tn,
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
            "       physical sensor readings and their derived features.\n"
            "       A full-feature model would additionally incorporate NWP\n"
            "       forecast fields, satellite-derived indices (NDVI, LST), and\n"
            "       administrative/topographic covariates."
        ),
    }
    print(notes.get(split, f"  Split: {split.upper()}"))


# ===========================================================================
# MAIN
# ===========================================================================

def main(split: str = "test"):
    separator(f"Flood Prediction — ML Comparison Report  [{split.upper()} SET]")
    print(f"  Data file  : {DATA_FILE}")
    print(f"  Models     : {list(MODELS.keys())}  ")
    print(f"  Split      : {split}")

    separator("Feature Explanation")
    print(__doc__.split("FULL-DATASET EVALUATION")[0].split("FEATURE EXPLANATION\n")[1].rstrip())

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

        rec_status = "✅" if warning_metrics["recall"]  >= 0.90 else "⚠️ "
        bal_status = "✅" if warning_metrics["bal_acc"] >= 0.85 else "⚠️ "
        gap_status = "✅" if tier_gaps_ok else "⚠️ "
        miss_flag  = "✅" if clear_metrics["miss_rate"] <= 0.05 else "⚠️ "
        print(f"  {rec_status}  {name:<18} v{version}  "
              f"weight={flood_weight}  features={len(avail)}  "
              f"watch={watch_t:.2f}  warn={warn_t:.2f}  danger={danger_t:.2f} ({danger_source})  "
              f"gaps={watch_warn_gap:.2f}/{warn_danger_gap:.2f}{gap_status}  "
              f"Recall(warn)={warning_metrics['recall']:.4f}  "
              f"BalAcc(warn)={warning_metrics['bal_acc']:.4f}  "
              f"MissRate(clear)={clear_metrics['miss_rate']:.4f}{miss_flag}")

    if not results:
        sys.exit("\n  ERROR: No models loaded. Run all three train scripts first.")

    if missing_pkls:
        print(f"\n  ⚠️  Missing model files: {missing_pkls}")
        print(f"       Run the corresponding train scripts to generate them.")

    # --- Metric comparison tables: CLEAR first, then active tiers ---
    for level_name, key in [
        ("CLEAR",   "clear"),
        ("WATCH",   "watch"),
        ("WARNING", "warning"),
        ("DANGER",  "danger"),
    ]:
        separator(f"Metric Comparison — {split.upper()} Set — {level_name} Threshold")
        _print_metric_table(results, threshold_key=key, y_true=y_true, level=level_name)

    # --- All-threshold summary ---
    separator(f"All-Threshold Summary — {split.upper()} Set")
    print(f"\n  {'Model':<20} {'Threshold':>10}  {'Level':<8}  "
          f"{'Precision':>10}  {'Recall':>8}  {'F1':>6}  "
          f"{'BalAcc':>8}  {'ROC-AUC':>8}  {'Count':>7}  {'Note':<30}")
    print(f"  {'-'*20} {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  "
          f"{'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*30}")
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
                  f"{m['precision']:>10.3f}  {m['recall']:>8.3f}  {m['f1']:>6.3f}  "
                  f"{m['bal_acc']:>8.3f}  {m['roc_auc']:>8.4f}  "
                  f"{m['alerts']:>7}  {note:<30}")
        print()

    # --- Tier gap summary ---
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

    # --- Probability distribution summary ---
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

    # --- Stored training metrics ---
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

    # --- Best model recommendation ---
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
        gap_ok   = "✅" if r["tier_gaps_ok"]          else "⚠️ "
        miss_ok  = "✅" if c["miss_rate"]    <= 0.05  else "⚠️ "
        status   = "  ← RECOMMENDED" if rank == 1 else ""
        print(f"  #{rank}  {name:<20}  score={score:.4f}  "
              f"Recall={w['recall']:.4f}{rec_ok}  "
              f"BalAcc={w['bal_acc']:.4f}{bal_ok}  "
              f"AUC={w['roc_auc']:.4f}  "
              f"TierGaps={gap_ok}  "
              f"MissRate={c['miss_rate']:.4f}{miss_ok}{status}")

    # --- Save CSV ---
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
                "precision":          round(m["precision"],   4),
                "recall":             round(m["recall"],      4),
                "f1":                 round(m["f1"],          4),
                "balanced_acc":       round(m["bal_acc"],     4),
                "roc_auc":            round(m["roc_auc"],     4),
                "specificity":        round(m["specificity"], 4),
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
            # CLEAR-only extra columns (blank for other tiers)
            if thresh_name == "CLEAR":
                row["clear_miss_rate"] = round(m["miss_rate"], 4)
                row["clear_n_missed"]  = m["n_missed"]
            else:
                row["clear_miss_rate"] = ""
                row["clear_n_missed"]  = ""
            rows.append(row)

    report_df  = pd.DataFrame(rows)
    report_csv = os.path.join(OUTPUT_DIR, f"ml_comparison_report_{split}.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_df.to_csv(report_csv, index=False)
    print(f"  Saved → {report_csv}")

    separator("Saving Charts")
    plot_path = os.path.join(OUTPUT_DIR, f"ml_comparison_report_{split}.png")
    _save_comparison_plot(results, y_true, split, split_df.index, plot_path)

    separator("DONE")
    print(f"  CSV report : {report_csv}")
    print(f"  Chart      : {plot_path}")
    separator()


# ---------------------------------------------------------------------------
# Print table helper
# ---------------------------------------------------------------------------

def _print_metric_table(results: dict, threshold_key: str, y_true, level: str = ""):
    col_w = 20

    if level == "CLEAR":
        # CLEAR has a fixed boundary = watch_t (same across all models)
        threshold_vals = {r["watch_t"] for r in results.values()}
        thresh_str = ", ".join(f"{t:.2f}" for t in sorted(threshold_vals))
        print(f"\n  Threshold level : CLEAR  (prob < watch_t = {thresh_str})")
        print(f"  Metrics are on the NEGATIVE CLASS (no-flood = label 0).")
        print(f"  precision = TN/(TN+FN)  — purity of CLEAR calls")
        print(f"  recall    = TN/(TN+FP)  — coverage of true safe rows")
        print(f"  miss_rate = FN/(FN+TP)  — floods incorrectly called CLEAR\n")
        print(f"  {'Model':<{col_w}} {'Precision':>10}  {'Recall':>8}  {'F1':>6}  "
              f"{'BalAcc':>8}  {'ROC-AUC':>8}  {'MissRate':>10}  {'Missed':>7}  {'N_Clear':>8}")
        print(f"  {'-'*col_w} {'-'*10}  {'-'*8}  {'-'*6}  "
              f"{'-'*8}  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*8}")
        target_miss = 0.05
        for name, r in results.items():
            m    = r[threshold_key]
            prec = m["precision"]
            rec  = m["recall"]
            f1   = m["f1"]
            bal  = m["bal_acc"]
            auc  = m["roc_auc"]
            mr   = m["miss_rate"]
            fn   = m["n_missed"]
            nc   = m["alerts"]
            p_flag  = "✅" if prec >= 0.90 else "⚠️ "
            r_flag  = "✅" if rec  >= 0.90 else "⚠️ "
            mr_flag = "✅" if mr   <= target_miss else "⚠️ "
            print(f"  {name:<{col_w}} {p_flag}{prec:>8.3f}  "
                  f"{r_flag}{rec:>6.3f}  {f1:>6.3f}  "
                  f"{bal:>8.3f}  {auc:>8.4f}  "
                  f"{mr_flag}{mr:>8.3f}  {fn:>7}  {nc:>8}")
        print(f"\n  Targets  : precision >= 0.90  recall >= 0.90  miss_rate <= {target_miss:.2f}")
        print(f"  miss_rate = fraction of TRUE FLOOD rows called CLEAR (false sense of safety)")
        flood_total = int(y_true.sum())
        print(f"  Flood events in split : {flood_total}")
        return

    # Active tiers: WATCH / WARNING / DANGER
    threshold_vals = {r[threshold_key]["threshold"] for r in results.values()}
    thresh_str = ", ".join(f"{t:.2f}" for t in sorted(threshold_vals))

    print(f"\n  Threshold level : {level}  (values: {thresh_str})")
    print(f"  {'Model':<{col_w}} {'Precision':>10}  {'Recall':>8}  {'F1':>6}  "
          f"{'BalAcc':>8}  {'ROC-AUC':>8}  {'Specificity':>12}  {'Alerts':>7}")
    print(f"  {'-'*col_w} {'-'*10}  {'-'*8}  {'-'*6}  "
          f"{'-'*8}  {'-'*8}  {'-'*12}  {'-'*7}")

    if level == "WATCH":
        target_met = {"precision": 0.30, "recall": 0.98, "bal_acc": 0.75}
    elif level == "WARNING":
        target_met = {"precision": 0.55, "recall": 0.90, "bal_acc": 0.85}
    else:
        target_met = {"precision": 0.70, "recall": 0.75, "bal_acc": 0.80}

    flood_total = int(y_true.sum())

    for name, r in results.items():
        m    = r[threshold_key]
        prec = m["precision"]
        rec  = m["recall"]
        f1   = m["f1"]
        bal  = m["bal_acc"]
        auc  = m["roc_auc"]
        spec = m["specificity"]
        n    = m["alerts"]

        p_flag = "✅" if prec >= target_met["precision"] else "⚠️ "
        r_flag = "✅" if rec  >= target_met["recall"]    else "⚠️ "
        b_flag = "✅" if bal  >= target_met["bal_acc"]   else "⚠️ "

        print(f"  {name:<{col_w}} {p_flag}{prec:>8.3f}  "
              f"{r_flag}{rec:>6.3f}  {f1:>6.3f}  "
              f"{b_flag}{bal:>6.3f}  {auc:>8.4f}  {spec:>12.3f}  {n:>7}")

    print(f"\n  Targets  : precision >= {target_met['precision']:.2f}  "
          f"recall >= {target_met['recall']:.2f}  "
          f"bal_acc >= {target_met['bal_acc']:.2f}")
    print(f"  Flood events in split : {flood_total}")


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def _save_comparison_plot(results: dict, y_true, split: str,
                          index, plot_path: str) -> None:
    n_models = len(results)
    fig = plt.figure(figsize=(18, 4 * (n_models + 1)))
    gs  = fig.add_gridspec(n_models + 1, 3, hspace=0.45, wspace=0.35)

    ax_roc = fig.add_subplot(gs[0, :])
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_true, r["probs"])
        auc          = r["warning"]["roc_auc"]
        ax_roc.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2,
                    label=f"{name}  AUC={auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curves — {split.upper()} Set  (models)")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(alpha=0.3)

    metric_labels = ["Precision", "Recall", "F1", "BalAcc", "AUC", "Specificity"]
    metric_keys   = ["precision", "recall", "f1", "bal_acc", "roc_auc", "specificity"]

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
        ax_dist.axvspan(0, watch_t, alpha=0.07, color="steelblue", label="CLEAR zone")
        ax_dist.axvline(watch_t,  color="gold",    linestyle="--", linewidth=1.2,
                        label=f"WATCH ({watch_t:.2f})")
        ax_dist.axvline(warn_t,   color="orange",  linestyle="--", linewidth=1.2,
                        label=f"WARN ({warn_t:.2f})")
        ax_dist.axvline(danger_t, color="darkred", linestyle="--", linewidth=1.2,
                        label=f"DANGER ({danger_t:.2f})")
        ax_dist.set_xlabel("Flood Probability")
        ax_dist.set_ylabel("Density")
        ax_dist.set_title(f"{name} — Probability Distribution")
        ax_dist.legend(fontsize=7)
        ax_dist.grid(alpha=0.3)

        ax_bar = fig.add_subplot(gs[row_idx, 1])
        x     = np.arange(len(metric_labels))
        width = 0.2
        # CLEAR uses negative-class precision/recall/f1 — recompute for plot consistency
        c_m       = r["clear"]
        clear_vals = [c_m["precision"], c_m["recall"], c_m["f1"],
                      c_m["bal_acc"],   c_m["roc_auc"], c_m["specificity"]]
        w_vals     = [r["watch"][k]   for k in metric_keys]
        warn_vals  = [r["warning"][k] for k in metric_keys]
        danger_vals= [r["danger"][k]  for k in metric_keys]
        ax_bar.bar(x - 1.5*width, clear_vals,  width, label="CLEAR",   color="steelblue", alpha=0.50)
        ax_bar.bar(x - 0.5*width, w_vals,      width, label="WATCH",   color=color, alpha=0.40)
        ax_bar.bar(x + 0.5*width, warn_vals,   width, label="WARNING", color=color, alpha=0.70)
        ax_bar.bar(x + 1.5*width, danger_vals, width, label="DANGER",  color=color, alpha=1.00)
        ax_bar.axhline(0.90, color="red",   linestyle=":", linewidth=1.2,
                       alpha=0.7, label="Recall target (0.90)")
        ax_bar.axhline(0.85, color="black", linestyle=":", linewidth=1.2,
                       alpha=0.7, label="BalAcc target (0.85)")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=8)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_title(f"{name} — Metrics (CLEAR / WATCH / WARNING / DANGER)")
        ax_bar.legend(fontsize=7)
        ax_bar.grid(axis="y", alpha=0.3)

        ax_ts = fig.add_subplot(gs[row_idx, 2])
        ax_ts.plot(index, probs, color=color, linewidth=1, alpha=0.8)
        ax_ts.axhspan(0, watch_t, alpha=0.07, color="steelblue")
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
    args = parser.parse_args()
    main(split=args.split)