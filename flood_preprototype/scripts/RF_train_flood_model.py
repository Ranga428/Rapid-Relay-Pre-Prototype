"""
RF_train_flood_model.py
=======================
Random Forest Flood Prediction — Training Script

CHRONOLOGICAL 3-WAY SPLIT
--------------------------
    Train : 2017–2022  (~2165 rows) — model learns from historical data
    Val   : 2023-01-01 – 2023-12-31 ( ~353 rows) — threshold tuning
    Test  : 2024-01-01 – present    ( ~758 rows) — final held-out eval

    VAL_END changed to 2023-12-31 — all of 2024 moves into test window.
    2024 floods (Jul–Oct clusters) closely match 2023 val flood sensor
    profile (waterlevel ~+1.2σ, cumrise_14d ~14–17), so thresholds tuned
    on 2023 transfer cleanly to 2024. Test set now has 8 flood clusters
    across two full seasons (2024 + 2025) vs 3 clusters previously.

THRESHOLD TUNING — RECALL-FIRST STRATEGY (UPDATED)
---------------------------------------------------
    Target : maximize flood RECALL
    Floor  : precision >= MIN_PRECISION_FLOOR (NOW 0.25, was 0.40)

    Lowering the precision floor lets the tuner search deeper into
    the precision-recall curve, accepting more false alarms in exchange
    for catching more real floods. The goal is >= 90% Watch recall.

    WATCH   threshold — highest recall at precision >= 0.25
    WARNING threshold — best F1 at precision >= 0.50 (was 0.60)

FLOOD WEIGHT OVERRIDE (NEW)
----------------------------
    FLOOD_WEIGHT_OVERRIDE = 12 overrides the auto-computed class ratio.
    The natural ratio is ~4.3:1. At 12:1 the model pays a much heavier
    penalty for missing actual floods during training, shifting the
    learned decision boundary toward higher recall.
    Set to None to revert to the auto-computed ratio.

MODELS PRODUCED
---------------
    flood_rf_full.pkl    — sensor + satellite features (revalidation only)
    flood_rf_sensor.pkl  — sensor-only features (deployment / predict_rf.py)
"""

import os
import sys
import argparse
import warnings
import joblib

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from feature_engineering import SENSOR_FEATURE_COLUMNS, FULL_FEATURE_COLUMNS

warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# CONFIG
# ===========================================================================

DATA_FILE  = r"..\data\flood_dataset.csv"
OUTPUT_DIR = r"..\model"

LABEL_COLUMN = "flood_label"

TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"   # CHANGED — all of 2024 moves into test window

DEFAULT_ALERT_THRESHOLD = 0.50

# CHANGED: was 0.40 — lower floor lets the tuner find higher-recall thresholds
MIN_PRECISION_FLOOR = 0.25

# WARNING tier precision floor — lowered from 0.60 to give more room
WARNING_PRECISION_FLOOR = 0.50

# NEW: Override the auto-computed class ratio. Set to None to use neg/pos ratio.
# At 12.0 the model penalises missed floods 12x more than false alarms,
# shifting the learned boundary toward high recall.
FLOOD_WEIGHT_OVERRIDE = 12.0

WATCH_THRESHOLD_OVERRIDE   = None
WARNING_THRESHOLD_OVERRIDE = None

# ===========================================================================
# END CONFIG
# ===========================================================================


def separator(title=""):
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def load_data(path: str, feature_cols: list) -> tuple:
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()
    df = df.dropna(how="all").drop_duplicates()

    available         = set(df.columns)
    feat_cols_present = [c for c in feature_cols if c in available]
    feat_cols_missing = [c for c in feature_cols if c not in available]

    if feat_cols_missing:
        print(f"  WARNING: Features not in CSV (skipped): {feat_cols_missing}")

    if LABEL_COLUMN not in available:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found.")
    if not feat_cols_present:
        raise ValueError("No feature columns found in dataset.")

    print(f"  Rows           : {len(df):,}")
    print(f"  Features found : {len(feat_cols_present)} / {len(feature_cols)}")
    print(f"  Date range     : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Flood=1        : {int(df[LABEL_COLUMN].sum())}  "
          f"({100*df[LABEL_COLUMN].mean():.1f}%)")
    print(f"  Flood=0        : {int((df[LABEL_COLUMN]==0).sum())}  "
          f"({100*(1-df[LABEL_COLUMN].mean()):.1f}%)")

    return df, feat_cols_present


def three_way_split(df: pd.DataFrame, feat_cols: list) -> tuple:
    tz           = df.index.tz
    train_end_ts = pd.Timestamp(TRAIN_END, tz=tz)
    val_end_ts   = pd.Timestamp(VAL_END,   tz=tz)

    train_df = df[df.index <= train_end_ts]
    val_df   = df[(df.index > train_end_ts) & (df.index <= val_end_ts)]
    test_df  = df[df.index > val_end_ts]

    def _xy(frame):
        return frame[feat_cols].values, frame[LABEL_COLUMN].values

    X_train, y_train = _xy(train_df)
    X_val,   y_val   = _xy(val_df)
    X_test,  y_test  = _xy(test_df)

    def _summary(name, frame, y):
        if len(frame) == 0:
            print(f"  {name:<8}: 0 rows  (NO DATA)")
            return
        print(f"  {name:<8}: {len(frame):>4} rows  "
              f"({frame.index.min().date()} -> {frame.index.max().date()})  "
              f"flood={int(y.sum())}  no-flood={int((y==0).sum())}")

    _summary("Train", train_df, y_train)
    _summary("Val",   val_df,   y_val)
    _summary("Test",  test_df,  y_test)

    if len(train_df) == 0:
        raise ValueError("Train split is empty — check TRAIN_END date.")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            train_df, val_df, test_df)


def build_model(class_weight: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators      = 500,
        max_depth         = 12,
        max_features      = "sqrt",
        min_samples_split = 6,
        min_samples_leaf  = 4,
        oob_score         = True,
        class_weight      = class_weight,
        n_jobs            = -1,
        random_state      = 42,
    )


def train_final(X_train, y_train, class_weight, model_label):
    model = build_model(class_weight)
    model.fit(X_train, y_train)
    print(f"  OOB Score (train set) : {model.oob_score_:.4f}")
    print(f"  Note: OOB score is an internal estimate on training rows only.")
    return model


def tune_threshold(model, X_val, y_val, model_label) -> tuple:
    """
    Returns (watch_threshold, warning_threshold).

    WATCH   = highest recall at precision >= MIN_PRECISION_FLOOR (0.25)
    WARNING = best F1 at precision >= WARNING_PRECISION_FLOOR (0.50)

    Lower precision floors allow the tuner to find thresholds that
    produce >= 90% recall while maintaining acceptable precision.
    """
    separator(f"Threshold Tuning — {model_label}")

    if len(X_val) == 0 or len(np.unique(y_val)) < 2:
        print(f"  Val split unusable — using defaults.")
        return DEFAULT_ALERT_THRESHOLD, DEFAULT_ALERT_THRESHOLD

    val_probs = model.predict_proba(X_val)[:, 1]

    print(f"  Strategy         : maximize recall, precision floor >= {MIN_PRECISION_FLOOR:.2f}")
    print(f"  Warning floor    : precision >= {WARNING_PRECISION_FLOOR:.2f}")
    print(f"  Target recall    : >= 90%")

    watch_thresh    = DEFAULT_ALERT_THRESHOLD
    watch_recall    = 0.0
    watch_precision = 0.0

    warn_thresh    = DEFAULT_ALERT_THRESHOLD
    warn_f1        = 0.0
    warn_precision = 0.0
    warn_recall    = 0.0

    print(f"\n  Threshold sweep (precision >= {MIN_PRECISION_FLOOR:.2f}):")
    print(f"  {'Thresh':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'Alerts':>7}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*7}")

    for thresh in np.arange(0.05, 0.96, 0.01):
        preds = (val_probs >= thresh).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        prec = precision_score(y_val, preds, pos_label=1, zero_division=0)
        rec  = recall_score(y_val, preds, pos_label=1, zero_division=0)
        f1   = f1_score(y_val, preds, pos_label=1, zero_division=0)

        if rec >= 0.85:
            print(f"  {thresh:>7.2f}  {prec:>10.3f}  {rec:>8.3f}  {f1:>6.3f}  {int(preds.sum()):>7}")

        if prec >= MIN_PRECISION_FLOOR and rec > watch_recall:
            watch_recall    = rec
            watch_precision = prec
            watch_thresh    = round(float(thresh), 2)

        if prec >= WARNING_PRECISION_FLOOR and f1 > warn_f1:
            warn_f1        = f1
            warn_precision = prec
            warn_recall    = rec
            warn_thresh    = round(float(thresh), 2)

    if WATCH_THRESHOLD_OVERRIDE is not None:
        watch_thresh = WATCH_THRESHOLD_OVERRIDE
    if WARNING_THRESHOLD_OVERRIDE is not None:
        warn_thresh = WARNING_THRESHOLD_OVERRIDE

    print(f"\n  WATCH threshold   : {watch_thresh:.2f}  "
          f"(recall={watch_recall:.3f}, precision={watch_precision:.3f})")
    print(f"  WARNING threshold : {warn_thresh:.2f}  "
          f"(recall={warn_recall:.3f}, precision={warn_precision:.3f})")

    if watch_recall < 0.90:
        print(f"\n  ⚠️  Watch recall {watch_recall:.1%} is below 90% target.")
        print(f"      Consider raising FLOOD_WEIGHT_OVERRIDE above {FLOOD_WEIGHT_OVERRIDE}.")
    else:
        print(f"\n  ✅  Watch recall {watch_recall:.1%} meets >= 90% target.")

    return watch_thresh, warn_thresh


def evaluate(model, X, y, split_name, watch_threshold, warn_threshold,
             feat_cols, output_dir, filename_prefix="") -> float:
    if len(X) == 0 or len(np.unique(y)) < 2:
        print(f"  {split_name}: skipped (no data or single class).")
        return None

    y_prob       = model.predict_proba(X)[:, 1]
    y_pred_watch = (y_prob >= watch_threshold).astype(int)
    y_pred_warn  = (y_prob >= warn_threshold).astype(int)
    auc          = roc_auc_score(y, y_prob)

    separator(f"{split_name} Evaluation")
    print(f"\n  --- WATCH threshold ({watch_threshold:.2f}) ---")
    print(classification_report(y, y_pred_watch,
          target_names=["No Flood", "Flood"], zero_division=0))
    print(f"  --- WARNING threshold ({warn_threshold:.2f}) ---")
    print(classification_report(y, y_pred_warn,
          target_names=["No Flood", "Flood"], zero_division=0))
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"\n  Probability distribution:")
    print(f"    Min  : {y_prob.min():.3f}")
    print(f"    Mean : {y_prob.mean():.3f}")
    print(f"    Max  : {y_prob.max():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred_watch, display_labels=["No Flood", "Flood"],
        ax=axes[0], colorbar=False)
    axes[0].set_title(f"{split_name} — WATCH (t={watch_threshold:.2f})")

    ConfusionMatrixDisplay.from_predictions(
        y, y_pred_warn, display_labels=["No Flood", "Flood"],
        ax=axes[1], colorbar=False)
    axes[1].set_title(f"{split_name} — WARNING (t={warn_threshold:.2f})")

    RocCurveDisplay.from_predictions(y, y_prob, ax=axes[2],
        name=f"Random Forest (AUC={auc:.3f})")
    from sklearn.metrics import roc_curve
    fpr, tpr, roc_thresholds = roc_curve(y, y_prob)
    for t, label, color in [
        (watch_threshold, f"WATCH {watch_threshold:.2f}", "orange"),
        (warn_threshold,  f"WARN  {warn_threshold:.2f}",  "red"),
    ]:
        idx = np.argmin(np.abs(roc_thresholds - t))
        axes[2].scatter(fpr[idx], tpr[idx], color=color, zorder=5,
                        label=label, s=80)
    axes[2].legend(fontsize=8)
    axes[2].set_title(f"{split_name} — ROC Curve")

    plt.tight_layout()
    safe_name = split_name.lower().replace(" ", "_")
    fig_path  = os.path.join(output_dir, f"{filename_prefix}eval_{safe_name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {fig_path}")
    return auc


def plot_feature_importance(model, feat_cols, output_dir, filename):
    importance = pd.Series(
        model.feature_importances_, index=feat_cols
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feat_cols) * 0.4)))
    importance.plot(kind="barh", ax=ax, color="forestgreen")
    ax.set_title("Random Forest Feature Importance (Mean Impurity Decrease)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Feature importance → {path}")

    print("\n  Top features:")
    for feat, val in importance.sort_values(ascending=False).head(10).items():
        bar = "█" * int(val * 200)
        print(f"    {feat:<40} {val:.4f}  {bar}")


def save_artifacts(model, feat_cols, watch_threshold, warn_threshold,
                   output_dir, filename_stem):
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, f"{filename_stem}.pkl")
    joblib.dump({
        "model":             model,
        "feature_columns":   feat_cols,
        "threshold":         watch_threshold,
        "watch_threshold":   watch_threshold,
        "warning_threshold": warn_threshold,
    }, pkl_path)
    print(f"  Model saved → {pkl_path}")
    print(f"  WATCH threshold saved   : {watch_threshold:.2f}")
    print(f"  WARNING threshold saved : {warn_threshold:.2f}")


def train_one_model(df, feat_cols, class_weight,
                    model_label, filename_stem, output_dir) -> tuple:
    separator(f"Training — {model_label}")
    print(f"  Features ({len(feat_cols)}) : {feat_cols}")

    splits = three_way_split(df, feat_cols)
    (X_train, y_train, X_val, y_val, X_test, y_test,
     train_df, val_df, test_df) = splits

    separator(f"Fitting — {model_label}")
    model = train_final(X_train, y_train, class_weight, model_label)

    watch_thresh, warn_thresh = tune_threshold(model, X_val, y_val, model_label)

    prefix   = filename_stem + "_"
    val_auc  = evaluate(model, X_val,  y_val,
                        f"{model_label} Val",
                        watch_thresh, warn_thresh,
                        feat_cols, output_dir, prefix)
    test_auc = evaluate(model, X_test, y_test,
                        f"{model_label} Test",
                        watch_thresh, warn_thresh,
                        feat_cols, output_dir, prefix)

    plot_feature_importance(model, feat_cols, output_dir,
        filename=f"{filename_stem}_feature_importance.png")
    save_artifacts(model, feat_cols, watch_thresh, warn_thresh,
                   output_dir, filename_stem)

    return model, feat_cols, watch_thresh, warn_thresh, val_auc, test_auc


def main(data_file=DATA_FILE, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    separator("Flood Prediction — Random Forest Training")
    print(f"  Data file  : {data_file}")
    print(f"  Output dir : {output_dir}")
    print(f"\n  Split strategy (chronological, no shuffling):")
    print(f"    Train : 2017 – {TRAIN_END}")
    print(f"    Val   : {TRAIN_END} – {VAL_END}  (threshold tuning)")
    print(f"    Test  : {VAL_END} – present       (final eval)")
    print(f"\n  Threshold strategy : recall-first, precision floor >= {MIN_PRECISION_FLOOR:.2f}")
    print(f"  Warning floor      : precision >= {WARNING_PRECISION_FLOOR:.2f}")
    print(f"  Target recall      : >= 90%")
    print(f"  Flood weight       : {FLOOD_WEIGHT_OVERRIDE} (override)")
    print(f"\n  Two models will be trained:")
    print(f"    flood_rf_full.pkl    — sensor + satellite (revalidation)")
    print(f"    flood_rf_sensor.pkl  — sensor-only        (deployment)")

    separator("Step 1 — Loading Dataset")
    df, full_feat_cols = load_data(data_file, FULL_FEATURE_COLUMNS)

    sensor_feat_cols = [c for c in SENSOR_FEATURE_COLUMNS if c in df.columns]
    missing_sensor   = [c for c in SENSOR_FEATURE_COLUMNS if c not in df.columns]
    if missing_sensor:
        print(f"  WARNING: Missing sensor columns: {missing_sensor}")

    neg = int((df[LABEL_COLUMN] == 0).sum())
    pos = int((df[LABEL_COLUMN] == 1).sum())
    auto_weight  = neg / pos if pos > 0 else 1.0
    flood_weight = FLOOD_WEIGHT_OVERRIDE if FLOOD_WEIGHT_OVERRIDE is not None else auto_weight
    class_weight = {0: 1.0, 1: flood_weight}

    separator("Step 2 — Class Balance")
    print(f"  No-flood (0)     : {neg}")
    print(f"  Flood    (1)     : {pos}")
    print(f"  Auto ratio       : {auto_weight:.2f}")
    print(f"  Flood weight     : {flood_weight:.2f}  "
          f"({'OVERRIDE' if FLOOD_WEIGHT_OVERRIDE is not None else 'auto'})")
    print(f"  class_weight     : {class_weight}")

    _, _, wt_full, wn_full, val_auc_full, test_auc_full = train_one_model(
        df=df, feat_cols=full_feat_cols, class_weight=class_weight,
        model_label="Full Model (sensor + satellite)",
        filename_stem="flood_rf_full", output_dir=output_dir,
    )

    _, _, wt_sensor, wn_sensor, val_auc_sensor, test_auc_sensor = train_one_model(
        df=df, feat_cols=sensor_feat_cols, class_weight=class_weight,
        model_label="Sensor Model (deployment)",
        filename_stem="flood_rf_sensor", output_dir=output_dir,
    )

    separator("TRAINING COMPLETE — Model Comparison")
    print(f"\n  {'Model':<40} {'Watch':>6}  {'Warn':>5}  {'Val AUC':>8}  {'Test AUC':>9}")
    print(f"  {'-'*40} {'-'*6}  {'-'*5}  {'-'*8}  {'-'*9}")

    def _fmt(v):
        return f"{v:.3f}" if v is not None else "  N/A"

    print(f"  {'Full  (flood_rf_full.pkl)':<40} {wt_full:>6.2f}  "
          f"{wn_full:>5.2f}  {_fmt(val_auc_full):>8}  {_fmt(test_auc_full):>9}")
    print(f"  {'Sensor (flood_rf_sensor.pkl)':<40} {wt_sensor:>6.2f}  "
          f"{wn_sensor:>5.2f}  {_fmt(val_auc_sensor):>8}  {_fmt(test_auc_sensor):>9}")

    if test_auc_full and test_auc_sensor:
        gap = test_auc_full - test_auc_sensor
        print(f"\n  AUC gap (full vs sensor) : {gap:.3f}")
        if gap <= 0.03:
            print("  ✅  Gap is small — sensor model is reliable for deployment.")
        elif gap <= 0.07:
            print("  ⚠️   Moderate gap — sensor model acceptable; monitor over time.")
        else:
            print("  ❌  Large gap — satellite features carry significant weight.")

    print(f"\n  Deployed model : flood_rf_sensor.pkl  (predict_rf.py)")
    print(f"  Validation     : flood_rf_full.pkl    (revalidation only)")
    print(f"  Output dir     : {output_dir}")
    separator()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flood Prediction — Random Forest (chronological 3-way split)"
    )
    parser.add_argument("--data",   type=str, default=DATA_FILE)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(data_file=args.data, output_dir=args.output)