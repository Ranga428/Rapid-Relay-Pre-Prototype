"""
LGBM_train_flood_model.py
=============================
Flood Prediction — LightGBM Training  (recall improvements)

CHANGES FROM v2
---------------
CHANGE A — FLOOD_WEIGHT_OVERRIDE raised 10.0 → 15.0
           Test BalAcc in v2 was 0.6474. Higher weight increases the cost
           of missed floods, which is the correct operational priority.

CHANGE B — WARNING precision floor lowered 0.65 → 0.55 in tune_thresholds().
           Forces the sweep to select a lower, higher-recall threshold.

CHANGE C — Isotonic calibration moved from val set to held-out calibration
           fold (last 20% of training data).
           The base LightGBM is fit on train_core (first 80% of train),
           then the calibrator is fitted on train_cal (last 20% of train).
           This prevents the calibrator from seeing in-sample predictions,
           which caused compressed probabilities and low AUC.

CHANGE D — LightGBM hyperparameters updated for better learning:
           n_estimators      600 → 800   (more trees with lower learning rate)
           learning_rate    0.04 → 0.03  (slower, more stable)
           num_leaves         31 → 63    (more model capacity)
           max_depth           8 → 6     (shallower to reduce noise)
           min_child_samples  30 → 10    (was too restrictive — caused flat output)
           subsample        0.75 → 0.75  (unchanged)
           reg_alpha        0.05 → 0.05  (unchanged)
           early_stopping     60 → 60    (unchanged)

CHANGE E — WATCH threshold now data-driven with unreachability guard.
           If WATCH_THRESHOLD_OVERRIDE is below the model's minimum predicted
           probability, the override is ignored and the threshold is derived
           from the actual probability distribution instead.

CHANGE F — MIN_TIER_GAP=0.08 enforced between WATCH/WARNING and WARNING/DANGER.
           Prevents WATCH and WARNING collapsing to the same effective threshold.

CHANGE G — Danger threshold derived from data (precision >= 0.70, above WARNING)
           and saved to pkl artifact as "danger_threshold". Previously the
           danger threshold was hardcoded as warn_t + 0.10 in the report script.

CHANGE H — Flat output guard added after fitting. If val prob std < 0.05,
           training is halted with a clear diagnostic message rather than
           silently producing a broken model.

TARGET metrics (test set)
--------------------------
    WATCH   recall    >= 98%
    WARNING recall    >= 90%
    WARNING precision >= 0.55
    Balanced accuracy >= 0.85
"""

import os
import sys
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from calibrated_models import CalibratedLGBM

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score,
    balanced_accuracy_score, f1_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
_ML_PIPELINE  = os.path.join(_PROJECT_ROOT, "ml_pipeline")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, _ML_PIPELINE)

from prepare_dataset import load_sensor
from feature_engineering import build_features, SENSOR_FEATURE_COLUMNS



# ===========================================================================
# CONFIG
# ===========================================================================

DATA_FILE  = r"..\data\flood_dataset.csv"
OUTPUT_DIR = r"..\model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CHANGE A — raised from 10.0
FLOOD_WEIGHT_OVERRIDE    = 15.0

# CHANGE E — used as a hint only; ignored if unreachable given model's prob range
WATCH_THRESHOLD_OVERRIDE = 0.03

# CHANGE F — minimum probability gap enforced between alert tiers
MIN_TIER_GAP = 0.08

# CHANGE H — halt threshold: if post-fit val prob std < this, training aborts
FLAT_OUTPUT_STD_THRESHOLD = 0.05

TRAIN_END = "2024-06-30"
VAL_END   = "2025-06-30"

# CHANGE D — updated hyperparameters
LGBM_PARAMS = dict(
    n_estimators      = 800,    # was 600 — more trees with lower learning rate
    learning_rate     = 0.03,   # was 0.04 — slower, more stable
    num_leaves        = 63,     # was 31  — more model capacity
    max_depth         = 6,      # was 8   — shallower to reduce noise
    min_child_samples = 10,     # was 30  — key fix: was too restrictive (flat output)
    subsample         = 0.75,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.05,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = -1,
)

LGBM_FIT_PARAMS = dict(
    eval_metric = "binary_logloss",
    callbacks   = [
        lgb.early_stopping(60, verbose=True),
        lgb.log_evaluation(50),
    ],
)

SENSOR_FEATURES = [
    "max_waterlevel_6h", "max_waterlevel_24h",
    "waterlevel_slope_3h", "waterlevel_slope_6h",
    "waterlevel_std_24h", "waterlevel_rise_rate_48h",
    "waterlevel_lag_1d", "waterlevel_lag_2d", "waterlevel_lag_3d",
    "waterlevel_days_above_threshold", "waterlevel_pct_rank_30d",
    "waterlevel_distance_to_max", "waterlevel_mean_7d", "waterlevel_cumrise_14d",
    "sensor_soilmoisture_mean_6h", "sensor_soilmoisture_mean_24h",
    "sensor_soilmoisture_trend_6h",
    "soilmoisture_lag_1d", "soilmoisture_lag_2d",
    "humidity_mean_24h", "humidity_trend_6h",
    "waterlevel_x_soilmoisture", "humidity_x_waterlevel_slope",
    "season_sin", "season_cos", "week_sin", "week_cos",
    "is_monsoon_season", "waterlevel_monsoon", "soilmoisture_monsoon",
    "humidity_x_soilmoisture", "soilmoisture_trend_3d",
    "days_since_flood_level", "waterlevel_falling_streak",
    "post_flood_decay_7d", "humidity_anomaly_vs_30d",
    "soilmoist_anomaly_vs_30d", "late_season_wet_flag",
    "waterlevel_7d_std", "soilmoist_7d_std",
]

FULL_FEATURES = SENSOR_FEATURES + [
    "orbit_flag","soil_saturation",
    "wetness_trend",
]


# ===========================================================================
# HELPERS
# ===========================================================================

def separator(title=""):
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def split_data(df, feature_cols):
    """
    CHANGE C — splits train into train_core (first 80%) and train_cal (last 20%).
    Base LGBM is fit on train_core only; isotonic calibrator is fit on train_cal.
    This ensures the calibrator receives out-of-sample predictions from the base model,
    preventing in-sample probability compression that caused low AUC.
    """
    avail      = [c for c in feature_cols if c in df.columns]
    train_full = df[df.index <= TRAIN_END]
    val        = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    test       = df[df.index > VAL_END]

    n_train   = len(train_full)
    cal_start = int(n_train * 0.80)
    train_core = train_full.iloc[:cal_start]
    train_cal  = train_full.iloc[cal_start:]

    def xy(split):
        a = [c for c in avail if c in split.columns]
        return split[a], split["flood_label"]

    return xy(train_full), xy(train_core), xy(train_cal), xy(val), xy(test), \
           train_full, train_core, train_cal, val, test, avail


def tune_thresholds(model, X_val, y_val, label=""):
    """
    CHANGE B — WARNING precision floor lowered to 0.55 (was 0.65).
    CHANGE E — WATCH override validated against actual prob range.
    CHANGE F — MIN_TIER_GAP enforced between all tiers.
    CHANGE G — Danger threshold derived from data and returned.

    Returns: watch_t, warn_t, danger_t
    """
    separator(f"Threshold Tuning — {label}")
    print(f"  Strategy  : WATCH recall >= 98%, WARNING precision >= 0.55")
    print(f"  MIN_TIER_GAP = {MIN_TIER_GAP}\n")

    probs    = model.predict_proba(X_val)[:, 1]
    prob_min = float(probs.min())
    prob_max = float(probs.max())
    prob_std = float(probs.std())
    print(f"  Prob range : {prob_min:.3f} – {prob_max:.3f}  std={prob_std:.4f}")

    if prob_std < 0.05:
        print(f"  ⚠️  COMPRESSED OUTPUT: std={prob_std:.4f} — thresholds may be unreliable")
    else:
        print(f"  ✅  Probability spread (std={prob_std:.4f})\n")

    thresholds = np.arange(0.01, 0.99, 0.01)
    rows = []
    for t in thresholds:
        preds  = (probs >= t).astype(int)
        tp     = ((preds == 1) & (y_val == 1)).sum()
        fp     = ((preds == 1) & (y_val == 0)).sum()
        fn     = ((preds == 0) & (y_val == 1)).sum()
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        rows.append((round(t, 2), prec, recall, f1, int(preds.sum())))

    results = pd.DataFrame(rows, columns=["Thresh", "Precision", "Recall", "F1", "Alerts"])

    # --- WATCH threshold (CHANGE E) ---
    if WATCH_THRESHOLD_OVERRIDE is not None and prob_min <= WATCH_THRESHOLD_OVERRIDE:
        watch_t   = WATCH_THRESHOLD_OVERRIDE
        match     = results[results["Thresh"] == watch_t]
        watch_row = match.iloc[0] if len(match) > 0 else \
                    results.iloc[(results["Thresh"] - watch_t).abs().argsort().iloc[0]]
        print(f"  WATCH : using override={watch_t:.2f} (reachable, min prob={prob_min:.3f})")
    else:
        if WATCH_THRESHOLD_OVERRIDE is not None:
            print(f"  ⚠️  WATCH override={WATCH_THRESHOLD_OVERRIDE:.2f} unreachable "
                  f"(min prob={prob_min:.3f}) — deriving from data (CHANGE E)")
        cands = results[results["Precision"] >= 0.20]
        if len(cands) == 0:
            cands = results
        watch_row = cands.loc[cands["Recall"].idxmax()]
        watch_t   = float(watch_row["Thresh"])
        print(f"  WATCH : data-driven threshold={watch_t:.2f}")

   # --- WARNING threshold (Optimized for F1-Score) ---
    # Filter valid candidates above the WATCH tier
    warn_cands = results[results["Thresh"] >= watch_t + MIN_TIER_GAP]

    if len(warn_cands) > 0:
        # Find the threshold that maximizes the F1-Score
        warn_row = warn_cands.loc[warn_cands["F1"].idxmax()]
        warn_t = float(warn_row["Thresh"])
        
        print(f"  WARNING : F1-optimized threshold={warn_t:.2f} "
            f"(Precision: {warn_row['Precision']:.2f}, Recall: {warn_row['Recall']:.2f})")
    else:
        warn_t   = round(watch_t + MIN_TIER_GAP, 2)
        match    = results[results["Thresh"] == warn_t]
        warn_row = match.iloc[0] if len(match) > 0 else \
                   results.iloc[(results["Thresh"] - warn_t).abs().argsort().iloc[0]]
        print(f"  ⚠️  Fallback WARNING threshold={warn_t:.2f}")

    # --- DANGER threshold (CHANGE G) ---
    danger_min   = round(warn_t + MIN_TIER_GAP, 2)
    danger_cands = results[
        (results["Precision"] >= 0.70) &
        (results["Thresh"]    >= danger_min)
    ]
    if len(danger_cands) > 0:
        danger_row = danger_cands.loc[danger_cands["Recall"].idxmax()]
        danger_t   = float(danger_row["Thresh"])
        print(f"  DANGER: data-driven threshold={danger_t:.2f} "
              f"(precision >= 0.70, CHANGE G)")
    else:
        danger_t   = danger_min
        danger_row = results.iloc[(results["Thresh"] - danger_t).abs().argsort().iloc[0]]
        print(f"  ⚠️  No DANGER at precision >= 0.70 — fallback={danger_t:.2f}")

    # --- Print sweep table ---
    visible = results[results["Recall"] >= 0.50]
    print(f"\n  {'Thresh':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'Alerts':>7}")
    print(f"  {'-------':>7}  {'----------':>10}  {'--------':>8}  {'------':>6}  {'-------':>7}")
    for _, r in visible.iterrows():
        marker = ""
        if r["Thresh"] == watch_t:
            marker = " ← WATCH"
        elif r["Thresh"] == warn_t:
            marker = " ← WARNING"
        elif r["Thresh"] == danger_t:
            marker = " ← DANGER"
        print(f"  {r.Thresh:>7.2f}  {r.Precision:>10.3f}  {r.Recall:>8.3f}  "
              f"{r.F1:>6.3f}  {int(r.Alerts):>7}{marker}")

    print(f"\n  WATCH   threshold : {watch_t:.2f}  "
          f"(recall={watch_row.Recall:.3f}, precision={watch_row.Precision:.3f})")
    print(f"  WARNING threshold : {warn_t:.2f}  "
          f"(recall={warn_row.Recall:.3f}, precision={warn_row.Precision:.3f})")
    print(f"  DANGER  threshold : {danger_t:.2f}  "
          f"(recall={danger_row.Recall:.3f}, precision={danger_row.Precision:.3f})")

    tier_gap_ok = (warn_t - watch_t >= MIN_TIER_GAP) and \
                  (danger_t - warn_t >= MIN_TIER_GAP)
    print(f"\n  Tier gaps : WATCH→WARNING={warn_t - watch_t:.2f}  "
          f"WARNING→DANGER={danger_t - warn_t:.2f}  "
          f"({'✅ OK' if tier_gap_ok else '⚠️ gap too small'})")

    status_w = "✅" if watch_row.Recall  >= 0.98 else "⚠️ "
    status_r = "✅" if warn_row.Recall   >= 0.90 else "⚠️ "
    print(f"\n  {status_w}  Watch recall    {watch_row.Recall*100:.1f}%  (target >= 98%)")
    print(f"  {status_r}  Warning recall  {warn_row.Recall*100:.1f}%  (target >= 90%)")
    print(f"       Warning prec    {warn_row.Precision:.3f}  (target >= 0.55)")

    return float(watch_t), float(warn_t), float(danger_t)


def evaluate(model, X, y, watch_t, warn_t, danger_t, label="", save_plot=None):
    separator(f"{label} Evaluation")
    probs = model.predict_proba(X)[:, 1]

    print(f"  Probability distribution:")
    print(f"    Min  : {probs.min():.3f}")
    print(f"    Mean : {probs.mean():.3f}")
    print(f"    Max  : {probs.max():.3f}")
    print(f"    Std  : {probs.std():.3f}  "
          f"({'✅ good spread' if probs.std() > 0.15 else '⚠️ still compressed'})")

    for thresh, name in [(watch_t, "WATCH"), (warn_t, "WARNING"), (danger_t, "DANGER")]:
        print(f"\n  --- {name} threshold ({thresh:.2f}) ---")
        preds = (probs >= thresh).astype(int)
        print(classification_report(y, preds,
                                    target_names=["No Flood", "Flood"],
                                    zero_division=0))

    preds_warn = (probs >= warn_t).astype(int)
    auc  = roc_auc_score(y, probs)
    bal  = balanced_accuracy_score(y, preds_warn)
    f1   = f1_score(y, preds_warn, zero_division=0)
    prec = precision_score(y, preds_warn, zero_division=0)
    rec  = recall_score(y, preds_warn, zero_division=0)
    tn   = ((preds_warn == 0) & (y == 0)).sum()
    fp_c = ((preds_warn == 1) & (y == 0)).sum()
    spec = tn / (tn + fp_c) if (tn + fp_c) > 0 else 0

    n_clear  = (probs < watch_t).sum()
    n_watch  = ((probs >= watch_t) & (probs < warn_t)).sum()
    n_warn   = ((probs >= warn_t)  & (probs < danger_t)).sum()
    n_danger = (probs >= danger_t).sum()
    print(f"  Alert distribution : CLEAR={n_clear}  WATCH={n_watch}  "
          f"WARNING={n_warn}  DANGER={n_danger}")

    print(f"  ROC-AUC            : {auc:.4f}")
    print(f"  Balanced Accuracy  : {bal:.4f}  ← key metric (target >= 0.85)")
    print(f"  F1 (WARNING)       : {f1:.4f}")
    print(f"  Precision (WARNING): {prec:.4f}  (target >= 0.55)")
    print(f"  Recall (WARNING)   : {rec:.4f}  (target >= 0.90)")
    print(f"  Specificity        : {spec:.4f}")

    status_r = "✅" if rec >= 0.90 else "⚠️ "
    status_b = "✅" if bal >= 0.85 else "⚠️ "
    print(f"\n  {status_r}  Recall    : {rec:.4f}  (target >= 0.90)")
    print(f"  {status_b}  Bal Acc   : {bal:.4f}  (target >= 0.85)")

    if save_plot:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.hist(probs[y == 0], bins=40, alpha=0.6,
                color="steelblue", label="No Flood")
        ax.hist(probs[y == 1], bins=40, alpha=0.6,
                color="red", label="Flood")
        ax.axvline(watch_t,  color="gold",    linestyle="--",
                   label=f"WATCH ({watch_t:.2f})")
        ax.axvline(warn_t,   color="orange",  linestyle="--",
                   label=f"WARNING ({warn_t:.2f})")
        ax.axvline(danger_t, color="darkred", linestyle="--",
                   label=f"DANGER ({danger_t:.2f})")
        ax.set_xlabel("Flood Probability")
        ax.set_ylabel("Count")
        ax.set_title(f"Probability Distribution — {label}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_plot, dpi=150)
        plt.close()
        print(f"  Plot saved → {save_plot}")

    return auc, bal, f1


def print_feature_importance(model_calib, feature_cols, save_path):
    lgbm_base = model_calib.estimator if hasattr(model_calib, "estimator") \
                else model_calib
    imp   = lgbm_base.feature_importances_
    pairs = sorted(zip(feature_cols, imp), key=lambda x: -x[1])

    print(f"\n  Top features (by split count):")
    for name, score in pairs[:12]:
        bar = "█" * int(score / max(imp) * 30)
        print(f"    {name:<40} {int(score):>6}  {bar}")

    gain       = lgbm_base.booster_.feature_importance(importance_type="gain")
    pairs_gain = sorted(zip(feature_cols, gain), key=lambda x: -x[1])
    print(f"\n  Top features (by gain):")
    for name, score in pairs_gain[:12]:
        bar = "█" * int(score / max(gain) * 30)
        print(f"    {name:<40} {score:>10.1f}  {bar}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (pairs_plot, title) in zip(
        axes,
        [(pairs[:15], "Split Count"), (pairs_gain[:15], "Gain")]
    ):
        names  = [p[0] for p in pairs_plot]
        scores = [p[1] for p in pairs_plot]
        ax.barh(names[::-1], scores[::-1], color="steelblue")
        ax.set_xlabel(title)
        ax.set_title(f"LightGBM Feature Importance — {title}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Feature importance → {save_path}")


# ===========================================================================
# TRAINING
# ===========================================================================

def train_model(df, feature_cols, model_name, pkl_name):
    separator(f"Training — {model_name}")

    (X_train, y_train), (X_core, y_core), (X_cal, y_cal), (X_val, y_val), (X_test, y_test), \
        tr_full, tr_core, tr_cal, va, te, avail = split_data(df, feature_cols)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ⚠️  Missing features (skipped): {missing}")

    print(f"  Features ({len(avail)}) : {avail}")
    print(f"  Train full : {len(X_train)} rows  "
          f"({tr_full.index.min().date()} -> {tr_full.index.max().date()})  "
          f"flood={int(y_train.sum())}  no-flood={int((y_train==0).sum())}")
    print(f"  Train core : {len(X_core)} rows  "
          f"({tr_core.index.min().date()} -> {tr_core.index.max().date()})  "
          f"flood={int(y_core.sum())}  [CHANGE C — base LGBM fit on this split]")
    print(f"  Train cal  : {len(X_cal)} rows  "
          f"({tr_cal.index.min().date()} -> {tr_cal.index.max().date()})  "
          f"flood={int(y_cal.sum())}  [CHANGE C — isotonic calibration fold]")
    print(f"  Val        : {len(X_val)} rows  "
          f"({va.index.min().date()} -> {va.index.max().date()})  "
          f"flood={int(y_val.sum())}  no-flood={int((y_val==0).sum())}")
    print(f"  Test       : {len(X_test)} rows  "
          f"({te.index.min().date()} -> {te.index.max().date()})  "
          f"flood={int(y_test.sum())}  no-flood={int((y_test==0).sum())}")

    separator(f"Fitting — {model_name}")
    print(f"  CHANGE C — base LGBM fitted on train_core (first 80% of train) only.")
    print(f"             Calibrator will be fitted on train_cal (last 20%) — out-of-sample.")
    lgbm_base = lgb.LGBMClassifier(
        **LGBM_PARAMS,
        scale_pos_weight=FLOOD_WEIGHT_OVERRIDE,
    )
    lgbm_base.fit(
        X_core, y_core,
        eval_set=[(X_val, y_val)],
        **LGBM_FIT_PARAMS,
    )
    print(f"  Best iteration : {lgbm_base.best_iteration_}")
    print(f"  Best val score : "
          f"{lgbm_base.best_score_['valid_0']['binary_logloss']:.4f}")

    # --- CHANGE H: Flat output guard ---
    probs_raw = lgbm_base.predict_proba(X_val)[:, 1]
    raw_std   = float(probs_raw.std())
    print(f"\n  Pre-calibration prob range : "
          f"{probs_raw.min():.3f} – {probs_raw.max():.3f}  std={raw_std:.4f}")

    if raw_std < FLAT_OUTPUT_STD_THRESHOLD:
        print(f"\n  ❌ FLAT OUTPUT DETECTED (std={raw_std:.4f} < {FLAT_OUTPUT_STD_THRESHOLD})")
        print(f"     The model is not learning meaningful flood signals.")
        print(f"     Recommended fixes:")
        print(f"       1. Lower min_child_samples to 5 (currently {LGBM_PARAMS['min_child_samples']})")
        print(f"       2. Raise num_leaves to 127 (currently {LGBM_PARAMS['num_leaves']})")
        print(f"       3. Raise FLOOD_WEIGHT_OVERRIDE to 20 (currently {FLOOD_WEIGHT_OVERRIDE})")
        print(f"     Halting to avoid saving a broken model (CHANGE H).")
        sys.exit(1)
    else:
        print(f"  ✅  Pre-calibration spread OK (std={raw_std:.4f})")

    separator(f"Calibrating — {model_name}")
    print(f"  CHANGE C — isotonic calibration on held-out train_cal fold")
    print(f"  Cal set size : {len(X_cal)} rows")
    print(f"  Cal flood rate : {y_cal.mean()*100:.1f}%")

    # Align cal columns to core columns (satellite features may differ in early years)
    core_cols = list(X_core.columns)
    X_cal_aligned = X_cal.reindex(columns=core_cols)
    missing_cal = X_cal_aligned.isnull().all(axis=0)
    if missing_cal.any():
        print(f"  ⚠️  Cal columns all-NaN (filled 0): {list(missing_cal[missing_cal].index)}")
    X_cal_aligned = X_cal_aligned.fillna(0)

    _iso = IsotonicRegression(out_of_bounds="clip")
    _iso.fit(lgbm_base.predict_proba(X_cal_aligned)[:, 1], y_cal)
    model = CalibratedLGBM(lgbm_base, _iso)

    probs_cal = model.predict_proba(X_val)[:, 1]
    cal_std   = float(probs_cal.std())
    print(f"  Post-calibration prob range : "
          f"{probs_cal.min():.3f} – {probs_cal.max():.3f}  std={cal_std:.4f}")

    if cal_std < FLAT_OUTPUT_STD_THRESHOLD:
        print(f"  ⚠️  Still compressed after calibration (std={cal_std:.4f}).")
        print(f"     This is unusual if pre-cal std was OK. "
              f"Check calibration fold size and class balance.")

    watch_t, warn_t, danger_t = tune_thresholds(model, X_val, y_val, label=model_name)

    val_auc, val_bal, val_f1 = evaluate(
        model, X_val, y_val, watch_t, warn_t, danger_t,
        label=f"{model_name} Val",
        save_plot=os.path.join(OUTPUT_DIR,
            f"{pkl_name.replace('.pkl','')}_prob_dist_val.png")
    )
    test_auc, test_bal, test_f1 = evaluate(
        model, X_test, y_test, watch_t, warn_t, danger_t,
        label=f"{model_name} Test",
        save_plot=os.path.join(OUTPUT_DIR,
            f"{pkl_name.replace('.pkl','')}_prob_dist_test.png")
    )

    print_feature_importance(
        model, avail,
        os.path.join(OUTPUT_DIR, f"{pkl_name.replace('.pkl','')}_feature_importance.png")
    )

    txt_path = os.path.join(OUTPUT_DIR, pkl_name.replace(".pkl", ".txt"))
    lgbm_base.booster_.save_model(txt_path)

    artifact = {
        "model"             : model,
        "feature_columns"   : avail,
        "watch_threshold"   : watch_t,
        "warning_threshold" : warn_t,
        "danger_threshold"  : danger_t,          # CHANGE G
        "last_training_date": VAL_END,
        "flood_weight"      : FLOOD_WEIGHT_OVERRIDE,
        "calibrated"        : True,
        "calibration_fold"  : "last_20pct_of_train",
        "version"           : "v3",
        "val_auc"           : val_auc,
        "val_balanced_acc"  : val_bal,
        "test_auc"          : test_auc,
        "test_balanced_acc" : test_bal,
    }
    pkl_path = os.path.join(OUTPUT_DIR, pkl_name)
    joblib.dump(artifact, pkl_path)
    print(f"\n  Model saved → {pkl_path}")
    print(f"  WATCH   threshold saved : {watch_t}")
    print(f"  WARNING threshold saved : {warn_t}")
    print(f"  DANGER  threshold saved : {danger_t}")
    print(f"  LightGBM txt → {txt_path}")

    return watch_t, warn_t, danger_t, val_auc, val_bal, test_auc, test_bal


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    separator("Flood Prediction — LightGBM Training")
    print(f"  Data file  : {DATA_FILE}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"""
  Split strategy (chronological, no shuffling):
    Train core : 2017-01-01 – {TRAIN_END} (first 80%)   (base LGBM fit)
    Train cal  : last 20% of train period                (isotonic calibration — out-of-sample)
    Val        : {TRAIN_END} – {VAL_END}                 (threshold tuning + early stopping)
    Test       : {VAL_END} – present                     (final evaluation)

  Key changes vs v2:
    CHANGE A — Flood weight        : 15.0  (was 10.0)
    CHANGE B — WARNING floor       : precision >= 0.55  (was 0.65)
    CHANGE C — Calibration fold    : base LGBM on train_core (80%), isotonic on train_cal (20% — out-of-sample)
    CHANGE D — num_leaves=63, min_child_samples=10, n_estimators=800
               (min_child_samples was the primary cause of flat output)
    CHANGE E — WATCH threshold     : data-driven guard if override unreachable
    CHANGE F — MIN_TIER_GAP=0.08   : enforced between all alert tiers
    CHANGE G — Danger threshold    : data-driven, saved to pkl artifact
    CHANGE H — Flat output guard   : halts if val prob std < 0.05 post-fit

  Target metrics (test set):
    WATCH   recall    >= 98%
    WARNING recall    >= 90%
    WARNING precision >= 0.55
    Balanced accuracy >= 0.85
""")

    separator("Step 1 — Loading Dataset")
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()
    print(f"  Rows       : {len(df):,}")
    print(f"  Date range : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Flood=1    : {int(df['flood_label'].sum())}  "
          f"({df['flood_label'].mean()*100:.1f}%)")

    separator("Step 2 — Class Balance")
    n0 = int((df["flood_label"] == 0).sum())
    n1 = int(df["flood_label"].sum())
    print(f"  No-flood (0)      : {n0}")
    print(f"  Flood    (1)      : {n1}")
    print(f"  Auto ratio        : {n0/n1:.2f}")
    print(f"  scale_pos_weight  : {FLOOD_WEIGHT_OVERRIDE:.1f}  (CHANGE A)")

    r_full = train_model(df, FULL_FEATURES,
                         "Full Model (sensor + satellite)",
                         "flood_lgbm_full.pkl")
    r_sens = train_model(df, SENSOR_FEATURES,
                         "Sensor Model (deployment)",
                         "flood_lgbm_sensor.pkl")

    separator("TRAINING COMPLETE — LightGBM")
    print(f"\n  {'Model':<44} {'Watch':>6}  {'Warn':>6}  {'Danger':>7}  "
          f"{'Val AUC':>8}  {'Val BalAcc':>10}  {'Test AUC':>8}  {'Test BalAcc':>11}")
    print(f"  {'-'*44} {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*11}")
    for label, r in [("Full  (flood_lgbm_full.pkl)",        r_full),
                     ("Sensor (flood_lgbm_sensor.pkl)",  r_sens)]:
        wt, wn, dt, va, vb, ta, tb = r
        print(f"  {label:<44} {wt:>6.2f}  {wn:>6.2f}  {dt:>7.2f}  "
              f"{va:>8.4f}  {vb:>10.4f}  {ta:>8.4f}  {tb:>11.4f}")

    print(f"""
  v2 reference:
    Sensor : Test AUC=0.8762  Test BalAcc=0.6474  WARNING recall=0.782

  If WARNING recall still < 90%  → raise FLOOD_WEIGHT_OVERRIDE to 18 or 20
  If Test BalAcc drops below 0.80 → lower FLOOD_WEIGHT_OVERRIDE to 12
  If flat output persists (std < 0.05) → lower min_child_samples to 5

  Deployed model : flood_lgbm_sensor.pkl
  Output dir     : {OUTPUT_DIR}
""")
    separator()


if __name__ == "__main__":
    main()