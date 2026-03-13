"""
train_flood_model.py
====================
XGBoost Flood Prediction — Training Script

CHRONOLOGICAL 3-WAY SPLIT
--------------------------
    Train : 2017–2022  (~326 rows)  — model learns from historical data
    Val   : 2023–2024  ( ~57 rows)  — hyperparameter tuning & early stopping
    Test  : 2025–2026  ( ~75 rows)  — final held-out evaluation (never seen)

All splits are chronological (no shuffling). This mirrors real deployment
where the model only ever sees past data when making future predictions.

TWO MODELS ARE PRODUCED
-----------------------
    flood_xgb_full.pkl
        Trained on sensor + satellite features.
        Used for revalidation and benchmarking only. Never deployed.

    flood_xgb_sensor.pkl
        Trained on sensor-only features.
        This is the DEPLOYED model used by predict.py.
        No satellite data required at inference time.

Both models share identical flood labels derived from satellite data.
The performance gap between them shows the cost of dropping satellite
features from the deployed model.

Usage
-----
    python train_flood_model.py
    python train_flood_model.py --data path/to/flood_dataset.csv
    python train_flood_model.py --data flood_dataset.csv --output model/
"""

import os
import sys
import argparse
import warnings
import joblib

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Chronological split boundaries (inclusive start of each period)
TRAIN_END = "2022-12-31"   # Train: rows with timestamp <= this date
VAL_END   = "2024-12-31"   # Val:   rows with timestamp in (TRAIN_END, VAL_END]
#                            Test:   rows with timestamp > VAL_END

# Alert threshold: probability >= this triggers a flood warning
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
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found in dataset.")

    if not feat_cols_present:
        raise ValueError("No feature columns found. Check flood_dataset.csv.")

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
    """
    Splits dataset chronologically into train / val / test.

        Train : timestamp <= TRAIN_END           (~2017–2022)
        Val   : TRAIN_END < timestamp <= VAL_END (~2023–2024)
        Test  : timestamp > VAL_END              (~2025–2026)
    """
    train_end_ts = pd.Timestamp(TRAIN_END, tz="UTC") if df.index.tz else pd.Timestamp(TRAIN_END)
    val_end_ts   = pd.Timestamp(VAL_END,   tz="UTC") if df.index.tz else pd.Timestamp(VAL_END)

    train_df = df[df.index <= train_end_ts]
    val_df   = df[(df.index > train_end_ts) & (df.index <= val_end_ts)]
    test_df  = df[df.index > val_end_ts]

    def _split_xy(frame):
        return frame[feat_cols].values, frame[LABEL_COLUMN].values

    X_train, y_train = _split_xy(train_df)
    X_val,   y_val   = _split_xy(val_df)
    X_test,  y_test  = _split_xy(test_df)

    def _summary(name, frame, y):
        if len(frame) == 0:
            print(f"  {name:<8}: 0 rows  (NO DATA)")
            return
        print(f"  {name:<8}: {len(frame):>4} rows  "
              f"({frame.index.min().date()} -> {frame.index.max().date()})  "
              f"flood={int(y.sum())}  no-flood={int((y==0).sum())}")

    _summary("Train",  train_df, y_train)
    _summary("Val",    val_df,   y_val)
    _summary("Test",   test_df,  y_test)

    if len(train_df) == 0:
        raise ValueError("Train split is empty — check TRAIN_END date.")
    if len(val_df) == 0:
        print("  WARNING: Val split is empty. Early stopping disabled.")
    if len(test_df) == 0:
        print("  WARNING: Test split is empty. No final evaluation will run.")

    return (X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            train_df, val_df, test_df)


# ---------------------------------------------------------------------------
# 3. Build XGBoost model
# ---------------------------------------------------------------------------

def build_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators          = 500,
        max_depth             = 4,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 3,
        scale_pos_weight      = scale_pos_weight,
        eval_metric           = "logloss",
        early_stopping_rounds = 30,
        random_state          = 42,
        verbosity             = 0,
    )


# ---------------------------------------------------------------------------
# 4. Train final model (train set → early stop on val set)
# ---------------------------------------------------------------------------

def train_final(
    X_train, y_train,
    X_val,   y_val,
    scale_pos_weight: float,
) -> xgb.XGBClassifier:
    model = build_model(scale_pos_weight)

    has_val = len(X_val) > 0 and len(np.unique(y_val)) >= 2

    if has_val:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        print(f"  Best iteration   : {model.best_iteration}")
        print(f"  Best val logloss : {model.best_score:.4f}")
    else:
        model.set_params(early_stopping_rounds=None)
        model.fit(X_train, y_train, verbose=50)
        print("  Early stopping disabled (no usable val split).")

    return model


# ---------------------------------------------------------------------------
# 5. Evaluate model on a data split
# ---------------------------------------------------------------------------

def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    feat_cols: list,
    output_dir: str,
    filename_prefix: str = "",
) -> float:
    if len(X) == 0 or len(np.unique(y)) < 2:
        print(f"  {split_name}: skipped (no data or single class).")
        return None

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= ALERT_THRESHOLD).astype(int)

    separator(f"{split_name} Evaluation  (threshold={ALERT_THRESHOLD})")
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
        y, y_prob,
        ax=axes[1],
        name=f"XGBoost (AUC={auc:.3f})",
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
# 6. Feature importance plot
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model, feat_cols: list, output_dir: str, filename: str
) -> None:
    importance = pd.Series(
        model.feature_importances_, index=feat_cols
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feat_cols) * 0.4)))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("XGBoost Feature Importance")
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
# 7. Save model artifact
# ---------------------------------------------------------------------------

def save_artifacts(model, feat_cols, output_dir, filename_stem):
    os.makedirs(output_dir, exist_ok=True)
    pkl_path  = os.path.join(output_dir, f"{filename_stem}.pkl")
    json_path = os.path.join(output_dir, f"{filename_stem}.json")
    joblib.dump({"model": model, "feature_columns": feat_cols}, pkl_path)
    model.save_model(json_path)
    print(f"  Model saved  → {pkl_path}")
    print(f"  XGBoost JSON → {json_path}")


# ---------------------------------------------------------------------------
# 8. Train one complete model variant
# ---------------------------------------------------------------------------

def train_one_model(
    df, feat_cols, scale_pos_weight,
    model_label, filename_stem, output_dir,
) -> tuple:
    separator(f"Training — {model_label}")
    print(f"  Features ({len(feat_cols)}) : {feat_cols}")

    splits = three_way_split(df, feat_cols)
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     train_df, val_df, test_df) = splits

    separator(f"Final Model — {model_label}")
    model = train_final(X_train, y_train, X_val, y_val, scale_pos_weight)

    prefix = filename_stem + "_"

    val_auc  = evaluate(model, X_val,   y_val,   f"{model_label} Val",
                        feat_cols, output_dir, prefix)
    test_auc = evaluate(model, X_test,  y_test,  f"{model_label} Test",
                        feat_cols, output_dir, prefix)

    plot_feature_importance(
        model, feat_cols, output_dir,
        filename=f"{filename_stem}_feature_importance.png",
    )
    save_artifacts(model, feat_cols, output_dir, filename_stem)

    return model, feat_cols, val_auc, test_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_file=DATA_FILE, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    separator("Flood Prediction — XGBoost Training")
    print(f"  Data file  : {data_file}")
    print(f"  Output dir : {output_dir}")
    print(f"\n  Split strategy (chronological, no shuffling):")
    print(f"    Train : 2017 – {TRAIN_END}")
    print(f"    Val   : {TRAIN_END} – {VAL_END}  (early stopping)")
    print(f"    Test  : {VAL_END} – present       (final evaluation)")
    print(f"\n  Two models will be trained:")
    print(f"    flood_xgb_full.pkl   — sensor + satellite (revalidation)")
    print(f"    flood_xgb_sensor.pkl — sensor-only        (deployment)")

    separator("Step 1 — Loading Dataset")
    df, full_feat_cols = load_data(data_file, FULL_FEATURE_COLUMNS)

    sensor_feat_cols = [c for c in SENSOR_FEATURE_COLUMNS if c in df.columns]
    missing_sensor   = [c for c in SENSOR_FEATURE_COLUMNS if c not in df.columns]
    if missing_sensor:
        print(f"  WARNING: Missing sensor columns: {missing_sensor}")

    neg = int((df[LABEL_COLUMN] == 0).sum())
    pos = int((df[LABEL_COLUMN] == 1).sum())
    spw = neg / pos if pos > 0 else 1.0

    separator("Step 2 — Class Balance")
    print(f"  No-flood (0) : {neg}")
    print(f"  Flood    (1) : {pos}")
    print(f"  scale_pos_weight : {spw:.2f}  (shared by both models)")

    # Train Model 1 — Full
    _, _, val_auc_full, test_auc_full = train_one_model(
        df=df, feat_cols=full_feat_cols, scale_pos_weight=spw,
        model_label="Full Model (sensor + satellite)",
        filename_stem="flood_xgb_full", output_dir=output_dir,
    )

    # Train Model 2 — Sensor-only
    _, _, val_auc_sensor, test_auc_sensor = train_one_model(
        df=df, feat_cols=sensor_feat_cols, scale_pos_weight=spw,
        model_label="Sensor Model (deployment)",
        filename_stem="flood_xgb_sensor", output_dir=output_dir,
    )

    # Summary
    separator("TRAINING COMPLETE — Model Comparison")
    print(f"\n  {'Model':<38} {'Val AUC':>8}  {'Test AUC':>9}")
    print(f"  {'-'*38} {'-'*8}  {'-'*9}")

    def _fmt(v):
        return f"{v:.3f}" if v is not None else "  N/A"

    print(f"  {'Full  (flood_xgb_full.pkl)':<38} {_fmt(val_auc_full):>8}  {_fmt(test_auc_full):>9}")
    print(f"  {'Sensor (flood_xgb_sensor.pkl)':<38} {_fmt(val_auc_sensor):>8}  {_fmt(test_auc_sensor):>9}")

    if test_auc_full and test_auc_sensor:
        gap = test_auc_full - test_auc_sensor
        print(f"\n  AUC gap (full vs sensor) : {gap:.3f}")
        if gap <= 0.03:
            print("  ✅  Gap is small — sensor model is reliable for deployment.")
        elif gap <= 0.07:
            print("  ⚠️   Moderate gap — sensor model is acceptable; monitor over time.")
        else:
            print("  ❌  Large gap — satellite features carry significant weight.")
            print("      Consider more sensor data or reviewing feature engineering.")

    print(f"\n  Deployed model : flood_xgb_sensor.pkl  (predict.py)")
    print(f"  Validation     : flood_xgb_full.pkl    (revalidation only)")
    print(f"  Output dir     : {output_dir}")
    separator()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flood Prediction — XGBoost (chronological 3-way split)"
    )
    parser.add_argument("--data",   type=str, default=DATA_FILE)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(data_file=args.data, output_dir=args.output)