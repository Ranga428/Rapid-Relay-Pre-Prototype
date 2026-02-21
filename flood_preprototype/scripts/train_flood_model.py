"""
train_flood_model.py
====================
End-to-end training script for the XGBoost Flood Prediction Model.
Reads a pre-built feature + label CSV, performs time-based splitting,
trains XGBoost, evaluates, and saves the model.

Usage
-----
    python train_flood_model.py --data flood_dataset.csv --output model/

Required CSV columns
--------------------
    timestamp (index), all feature columns, flood_label (0/1)
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "max_water_level_6h",
    "max_water_level_24h",
    "water_level_slope_3h",
    "water_level_slope_6h",
    "water_level_std_24h",
    "rainfall_sum_1h",
    "rainfall_sum_6h",
    "rainfall_sum_24h",
    "rainfall_max_intensity",
    "soil_moisture_mean_24h",
    "soil_moisture_trend_48h",
    "tidal_height_current",
    "tidal_height_max_24h",
    "tidal_phase",
]

LABEL_COLUMN = "flood_label"

# Time-based split boundaries (adjust to your dataset)
TRAIN_END  = "2022-12-31"
VAL_END    = "2023-12-31"
# Everything after VAL_END → test set

ALERT_THRESHOLD = 0.60   # Flood warning if predicted probability ≥ this


# ---------------------------------------------------------------------------
# 1. Load & split
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()

    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    return df


def time_split(df: pd.DataFrame):
    train = df.loc[:TRAIN_END]
    val   = df.loc[TRAIN_END:VAL_END].iloc[1:]   # avoid boundary overlap
    test  = df.loc[VAL_END:].iloc[1:]

    print(f"Train : {train.index[0].date()} → {train.index[-1].date()}  ({len(train):,} rows)")
    print(f"Val   : {val.index[0].date()}   → {val.index[-1].date()}    ({len(val):,} rows)")
    print(f"Test  : {test.index[0].date()}  → {test.index[-1].date()}   ({len(test):,} rows)")

    X_train, y_train = train[FEATURE_COLUMNS], train[LABEL_COLUMN]
    X_val,   y_val   = val[FEATURE_COLUMNS],   val[LABEL_COLUMN]
    X_test,  y_test  = test[FEATURE_COLUMNS],  test[LABEL_COLUMN]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# 2. Class imbalance weight
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y: pd.Series) -> float:
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    ratio = neg / pos if pos > 0 else 1.0
    print(f"Class balance — Flood: {pos:,}  No-Flood: {neg:,}  "
          f"scale_pos_weight: {ratio:.2f}")
    return ratio


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------

def build_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        eval_metric       = "logloss",
        early_stopping_rounds = 20,
        use_label_encoder = False,
        random_state      = 42,
    )


def train(model, X_train, y_train, X_val, y_val):
    model.fit(
        X_train, y_train,
        eval_set          = [(X_val, y_val)],
        verbose           = 50,
    )
    return model


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, split_name: str = "Test", output_dir: str = "."):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\n{'='*50}")
    print(f"  {split_name} Evaluation")
    print(f"{'='*50}")
    print(classification_report(y, y_pred, target_names=["No Flood", "Flood"]))
    auc = roc_auc_score(y, y_prob)
    print(f"  ROC-AUC : {auc:.4f}")

    # --- Confusion matrix ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ConfusionMatrixDisplay.from_predictions(
        y, y_pred,
        display_labels=["No Flood", "Flood"],
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title(f"{split_name} — Confusion Matrix")

    RocCurveDisplay.from_predictions(y, y_prob, ax=axes[1], name=f"XGBoost ({split_name})")
    axes[1].set_title(f"{split_name} — ROC Curve (AUC={auc:.3f})")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"eval_{split_name.lower()}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Plots saved → {fig_path}")

    return auc


# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(model, output_dir: str = "."):
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("XGBoost Feature Importance (gain)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved → {path}")


# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------

def save_artifacts(model, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "flood_xgb_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved → {model_path}")

    # Also export native XGBoost format
    xgb_path = os.path.join(output_dir, "flood_xgb_model.json")
    model.save_model(xgb_path)
    print(f"XGBoost JSON saved → {xgb_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost Flood Prediction Model")
    parser.add_argument("--data",   required=True, help="Path to feature+label CSV")
    parser.add_argument("--output", default="model/", help="Output directory for model & plots")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("\n📥  Loading data...")
    df = load_data(args.data)

    print("\n✂️   Splitting data (time-based)...")
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(df)

    print("\n⚖️   Computing class weights...")
    spw = compute_scale_pos_weight(y_train)

    print("\n🏗️   Building model...")
    model = build_model(scale_pos_weight=spw)

    print("\n🚀  Training...")
    model = train(model, X_train, y_train, X_val, y_val)

    print("\n📊  Evaluating on validation set...")
    evaluate(model, X_val, y_val, split_name="Validation", output_dir=args.output)

    print("\n📊  Evaluating on test set...")
    evaluate(model, X_test, y_test, split_name="Test", output_dir=args.output)

    print("\n📈  Plotting feature importance...")
    plot_feature_importance(model, output_dir=args.output)

    print("\n💾  Saving artifacts...")
    save_artifacts(model, output_dir=args.output)

    print("\n✅  Training complete.")


if __name__ == "__main__":
    main()
