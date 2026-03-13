"""
train_flood_model_rf.py
=======================
Random Forest Flood Prediction — Training Script

CHRONOLOGICAL 3-WAY SPLIT
--------------------------
    Train : 2017–2022  (~326 rows)  — model learns from historical data
    Val   : 2023–2024  ( ~57 rows)  — threshold tuning & OOB validation
    Test  : 2025–2026  ( ~75 rows)  — final held-out evaluation (never seen)

WHY RANDOM FOREST?
------------------
Random Forest is a strong baseline and complementary to XGBoost:

    - Built-in OOB (out-of-bag) score avoids needing a separate val set
      for training — val is used here for threshold tuning and comparison.
    - More robust to small datasets. With ~326 training rows, RF tends to
      overfit less than gradient boosting when trees are capped.
    - Naturally provides feature importance (mean impurity decrease).
    - Probability calibration tends to be more reliable out-of-the-box.
    - No learning rate, no early stopping — fewer hyperparameters to tune.

MODELS PRODUCED
---------------
    flood_rf_full.pkl    — sensor + satellite features (revalidation only)
    flood_rf_sensor.pkl  — sensor-only features (deployment / predict.py)

Both are drop-in replacements for the XGBoost models and use the same
artifact format: {"model": ..., "feature_columns": [...]}.

Usage
-----
    python train_flood_model_rf.py
    python train_flood_model_rf.py --data path/to/flood_dataset.csv
    python train_flood_model_rf.py --data flood_dataset.csv --output model/
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
VAL_END   = "2024-12-31"

ALERT_THRESHOLD = 0.50

# ===========================================================================
# END CONFIG
# ===========================================================================


def separator(title=""):
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ---------------------------------------------------------------------------
# 1. Load & validate dataset
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Chronological 3-way split
# ---------------------------------------------------------------------------

def three_way_split(df: pd.DataFrame, feat_cols: list) -> tuple:
    tz = df.index.tz
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

    return (X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            train_df, val_df, test_df)


# ---------------------------------------------------------------------------
# 3. Build Random Forest model
# ---------------------------------------------------------------------------

def build_model(class_weight: dict) -> RandomForestClassifier:
    """
    Hyperparameter notes:
        n_estimators = 500    — more trees = more stable OOB estimate
        max_depth    = 12     — shallow trees reduce overfitting on small data
        max_features = "sqrt" — standard for classification RF
        min_samples_leaf = 4  — prevents leaves with 1-2 samples
        oob_score    = True   — free internal validation from bootstrap
        class_weight          — handles flood/no-flood imbalance
    """
    return RandomForestClassifier(
        n_estimators     = 500,
        max_depth        = 12,
        max_features     = "sqrt",
        min_samples_split= 6,
        min_samples_leaf = 4,
        oob_score        = True,
        class_weight     = class_weight,
        n_jobs           = -1,
        random_state     = 42,
    )


# ---------------------------------------------------------------------------
# 4. Train final model on training set
# ---------------------------------------------------------------------------

def train_final(
    X_train, y_train,
    class_weight: dict,
    model_label: str,
) -> RandomForestClassifier:
    model = build_model(class_weight)
    model.fit(X_train, y_train)

    print(f"  OOB Score (train set) : {model.oob_score_:.4f}")
    print(f"  Note: OOB score is an internal estimate on training rows only.")
    print(f"        It is NOT the val or test score.")

    return model


# ---------------------------------------------------------------------------
# 5. Find optimal threshold on validation set
# ---------------------------------------------------------------------------

def tune_threshold(model, X_val, y_val) -> float:
    """
    Sweep thresholds on the val set and pick the one that maximises
    flood F1-score. Falls back to ALERT_THRESHOLD if val is unusable.
    """
    if len(X_val) == 0 or len(np.unique(y_val)) < 2:
        print(f"  Threshold tuning skipped — using default {ALERT_THRESHOLD}")
        return ALERT_THRESHOLD

    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.20, 0.80, 61)
    best_thresh, best_f1 = ALERT_THRESHOLD, 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        f1 = report.get("1", {}).get("f1-score", 0.0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"  Optimal threshold (val Flood-F1 = {best_f1:.3f}) : {best_thresh:.2f}")
    return best_thresh


# ---------------------------------------------------------------------------
# 6. Evaluate model on a data split
# ---------------------------------------------------------------------------

def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    feat_cols: list,
    output_dir: str,
    filename_prefix: str = "",
    threshold: float = ALERT_THRESHOLD,
) -> float:
    if len(X) == 0 or len(np.unique(y)) < 2:
        print(f"  {split_name}: skipped (no data or single class).")
        return None

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    separator(f"{split_name} Evaluation  (threshold={threshold:.2f})")
    print(classification_report(
        y, y_pred,
        target_names=["No Flood", "Flood"],
        zero_division=0,
    ))

    auc = roc_auc_score(y, y_prob)
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"\n  Probability distribution:")
    print(f"    Min  : {y_prob.min():.3f}")
    print(f"    Mean : {y_prob.mean():.3f}")
    print(f"    Max  : {y_prob.max():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred,
        display_labels=["No Flood", "Flood"],
        ax=axes[0], colorbar=False,
    )
    axes[0].set_title(f"{split_name} — Confusion Matrix")
    RocCurveDisplay.from_predictions(
        y, y_prob, ax=axes[1],
        name=f"Random Forest (AUC={auc:.3f})",
    )
    axes[1].set_title(f"{split_name} — ROC Curve")
    plt.tight_layout()

    safe_name = split_name.lower().replace(" ", "_")
    fig_path  = os.path.join(output_dir, f"{filename_prefix}eval_{safe_name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {fig_path}")

    return auc


# ---------------------------------------------------------------------------
# 7. Feature importance plot
# ---------------------------------------------------------------------------

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
    for feat, val in importance.sort_values(ascending=False).head(5).items():
        bar = "█" * int(val * 200)
        print(f"    {feat:<36} {val:.4f}  {bar}")


# ---------------------------------------------------------------------------
# 8. Save model artifact
# ---------------------------------------------------------------------------

def save_artifacts(model, feat_cols, output_dir, filename_stem, threshold):
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, f"{filename_stem}.pkl")
    joblib.dump({
        "model":           model,
        "feature_columns": feat_cols,
        "threshold":       threshold,   # val-tuned threshold saved alongside model
    }, pkl_path)
    print(f"  Model saved → {pkl_path}")
    print(f"  Threshold saved alongside model : {threshold:.2f}")


# ---------------------------------------------------------------------------
# 9. Train one complete model variant
# ---------------------------------------------------------------------------

def train_one_model(
    df, feat_cols, class_weight,
    model_label, filename_stem, output_dir,
) -> tuple:
    separator(f"Training — {model_label}")
    print(f"  Features ({len(feat_cols)}) : {feat_cols}")

    splits = three_way_split(df, feat_cols)
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     train_df, val_df, test_df) = splits

    separator(f"Fitting — {model_label}")
    model = train_final(X_train, y_train, class_weight, model_label)

    # Tune threshold on val set
    separator(f"Threshold Tuning — {model_label}")
    best_thresh = tune_threshold(model, X_val, y_val)

    prefix = filename_stem + "_"

    val_auc  = evaluate(model, X_val,  y_val,  f"{model_label} Val",
                        feat_cols, output_dir, prefix, best_thresh)
    test_auc = evaluate(model, X_test, y_test, f"{model_label} Test",
                        feat_cols, output_dir, prefix, best_thresh)

    plot_feature_importance(
        model, feat_cols, output_dir,
        filename=f"{filename_stem}_feature_importance.png",
    )
    save_artifacts(model, feat_cols, output_dir, filename_stem, best_thresh)

    return model, feat_cols, val_auc, test_auc, best_thresh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_file=DATA_FILE, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    separator("Flood Prediction — Random Forest Training")
    print(f"  Data file  : {data_file}")
    print(f"  Output dir : {output_dir}")
    print(f"\n  Split strategy (chronological, no shuffling):")
    print(f"    Train : 2017 – {TRAIN_END}   (~326 rows)")
    print(f"    Val   : {TRAIN_END} – {VAL_END}   (~57 rows, threshold tuning)")
    print(f"    Test  : {VAL_END} – present        (~75 rows, final eval)")
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
    class_weight = {0: 1.0, 1: neg / pos if pos > 0 else 1.0}

    separator("Step 2 — Class Balance")
    print(f"  No-flood (0) : {neg}")
    print(f"  Flood    (1) : {pos}")
    print(f"  class_weight : {class_weight}  (shared by both models)")

    # Train Model 1 — Full
    _, _, val_auc_full, test_auc_full, thresh_full = train_one_model(
        df=df, feat_cols=full_feat_cols, class_weight=class_weight,
        model_label="Full Model (sensor + satellite)",
        filename_stem="flood_rf_full", output_dir=output_dir,
    )

    # Train Model 2 — Sensor-only
    _, _, val_auc_sensor, test_auc_sensor, thresh_sensor = train_one_model(
        df=df, feat_cols=sensor_feat_cols, class_weight=class_weight,
        model_label="Sensor Model (deployment)",
        filename_stem="flood_rf_sensor", output_dir=output_dir,
    )

    # Summary
    separator("TRAINING COMPLETE — Model Comparison")
    print(f"\n  {'Model':<40} {'Thresh':>7}  {'Val AUC':>8}  {'Test AUC':>9}")
    print(f"  {'-'*40} {'-'*7}  {'-'*8}  {'-'*9}")

    def _fmt(v):
        return f"{v:.3f}" if v is not None else "  N/A"

    print(f"  {'Full  (flood_rf_full.pkl)':<40} {thresh_full:.2f}    "
          f"{_fmt(val_auc_full):>8}  {_fmt(test_auc_full):>9}")
    print(f"  {'Sensor (flood_rf_sensor.pkl)':<40} {thresh_sensor:.2f}    "
          f"{_fmt(val_auc_sensor):>8}  {_fmt(test_auc_sensor):>9}")

    if test_auc_full and test_auc_sensor:
        gap = test_auc_full - test_auc_sensor
        print(f"\n  AUC gap (full vs sensor) : {gap:.3f}")
        if gap <= 0.03:
            print("  ✅  Gap is small — sensor model is reliable for deployment.")
        elif gap <= 0.07:
            print("  ⚠️   Moderate gap — sensor model is acceptable; monitor over time.")
        else:
            print("  ❌  Large gap — satellite features carry significant weight.")

    print(f"\n  Deployed model : flood_rf_sensor.pkl  (predict.py)")
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