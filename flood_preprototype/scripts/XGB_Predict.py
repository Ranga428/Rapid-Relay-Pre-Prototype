"""
predict_xgb.py
==============
Flood Prediction — XGBoost Real-Time Inference

Runs the deployed XGBoost sensor model (flood_xgb_sensor.pkl).
No satellite data required at inference time.

Outputs (written to ../predictions/):
    flood_xgb_sensor_predictions.csv
    flood_xgb_sensor_predictions.png

Usage
-----
    python predict_xgb.py

    # Run on a schedule (e.g. every 12 hours)
    python predict_xgb.py --schedule --interval 12

To switch models, use the corresponding predictor script:
    predict_rf.py   — Random Forest
    predict_lgbm.py — LightGBM
"""

import os
import sys
import time
import argparse
import joblib
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from prepare_dataset import load_sensor
from feature_engineering import build_features


# ===========================================================================
# CONFIG — edit these if paths change or after retraining
# ===========================================================================

MODEL_FILE  = r"..\model\flood_xgb_sensor.pkl"
OUTPUT_DIR  = r"..\predictions"
SENSOR_FILE = os.path.join(
    SCRIPT_DIR, r"..\data\sensor\obando_environmental_data.csv"
)

PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "flood_xgb_sensor_predictions.csv")
PLOT_FILE       = os.path.join(OUTPUT_DIR, "flood_xgb_sensor_predictions.png")

# Last date included in training. Predictions generated ONLY for rows after this.
# Update after retraining on newer data.
LAST_TRAINING_DATE = "2024-12-31"

# Fallback threshold — XGBoost artifacts do not store a val-tuned threshold,
# so this CONFIG value is always used for XGBoost.
DEFAULT_ALERT_THRESHOLD = 0.50

# Risk tier offsets from the active alert threshold
WATCH_OFFSET   = -0.07   # alert - 0.07
WARNING_OFFSET =  0.00   # alert
DANGER_OFFSET  = +0.10   # alert + 0.10

# ===========================================================================
# END CONFIG
# ===========================================================================

MODEL_LABEL = "XGBoost"

RISK_TIERS = {
    "CLEAR":   {"emoji": "🟢", "color": "green"},
    "WATCH":   {"emoji": "🟡", "color": "gold"},
    "WARNING": {"emoji": "🟠", "color": "orange"},
    "DANGER":  {"emoji": "🔴", "color": "red"},
}


def separator(title=""):
    line = "=" * 55
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------

def load_model() -> tuple:
    separator("Loading Model")
    if not os.path.exists(MODEL_FILE):
        sys.exit(
            f"\n  ERROR: Model file not found.\n"
            f"  Expected : {MODEL_FILE}\n"
            f"  Fix      : Run XGB_train_flood_model.py first.\n"
        )

    artifact     = joblib.load(MODEL_FILE)
    model        = artifact["model"]
    feature_cols = artifact["feature_columns"]
    threshold    = artifact.get("threshold", DEFAULT_ALERT_THRESHOLD)

    print(f"  Model loaded     : {MODEL_FILE}")
    print(f"  Model type       : {type(model).__name__}")
    print(f"  Feature count    : {len(feature_cols)}")
    print(f"  Alert threshold  : {threshold:.2f}"
          + ("  (val-tuned, from artifact)"
             if "threshold" in artifact else "  (default)"))

    return model, feature_cols, threshold


# ---------------------------------------------------------------------------
# 2. Build sensor features
# ---------------------------------------------------------------------------

def build_sensor_features() -> pd.DataFrame:
    separator("Loading Sensor Data")
    sensor_df, freq = load_sensor(sensor_path=SENSOR_FILE)

    separator("Building Sensor Features")
    print(f"  Frequency : {freq}")
    print(f"  Mode      : sensor-only (no satellite columns used)")

    features = build_features(sensor_df, freq=freq, mode="sensor")
    print(f"  Total rows built (full history) : {len(features):,}")
    return features


# ---------------------------------------------------------------------------
# 3. Filter to new (unseen) rows only
# ---------------------------------------------------------------------------

def filter_new_rows(features: pd.DataFrame) -> pd.DataFrame:
    separator("Filtering to New Data Only")
    cutoff = pd.Timestamp(LAST_TRAINING_DATE, tz="UTC")
    new_df = features[features.index > cutoff]

    print(f"  Training cutoff  : {LAST_TRAINING_DATE}")
    print(f"  Full history     : {len(features):,} rows")
    print(f"  New rows         : {len(new_df):,}")

    if len(new_df) == 0:
        sys.exit(
            f"\n  No new data found after {LAST_TRAINING_DATE}.\n"
            f"  Append new sensor rows to the sensor CSV and re-run.\n"
            f"  If you have retrained on newer data, update\n"
            f"  LAST_TRAINING_DATE in predict_xgb.py."
        )

    print(f"  New date range   : {new_df.index[0].date()}  ->  {new_df.index[-1].date()}")
    return new_df


# ---------------------------------------------------------------------------
# 4. Risk tier classifier
# ---------------------------------------------------------------------------

def make_risk_classifier(threshold: float):
    watch_thresh   = threshold + WATCH_OFFSET
    warning_thresh = threshold + WARNING_OFFSET
    danger_thresh  = threshold + DANGER_OFFSET

    def classify(prob: float) -> str:
        if prob >= danger_thresh:
            return "DANGER"
        elif prob >= warning_thresh:
            return "WARNING"
        elif prob >= watch_thresh:
            return "WATCH"
        else:
            return "CLEAR"

    return classify, watch_thresh, warning_thresh, danger_thresh


# ---------------------------------------------------------------------------
# 5. Run predictions
# ---------------------------------------------------------------------------

def run_predictions(model, feature_cols, new_features, threshold) -> tuple:
    separator("Running Predictions")

    missing = [c for c in feature_cols if c not in new_features.columns]
    if missing:
        sys.exit(
            f"\n  ERROR: {len(missing)} feature column(s) missing from sensor data:\n"
            f"    {missing}\n"
            f"  One or more sensors are offline or the CSV is missing expected columns.\n"
            f"  Restore sensor data for the missing columns and re-run."
        )

    X     = new_features[feature_cols].values
    probs = model.predict_proba(X)[:, 1]

    classify, watch_t, warning_t, danger_t = make_risk_classifier(threshold)

    results = pd.DataFrame({
        "flood_probability": probs.round(4),
        "risk_tier":         [classify(p) for p in probs],
        "watch_threshold":   watch_t,
        "warning_threshold": warning_t,
        "danger_threshold":  danger_t,
        "alert_threshold":   threshold,
    }, index=new_features.index)

    for col in feature_cols:
        results[col] = new_features[col].values

    print(f"  Predictions made : {len(results):,}")
    return results, watch_t, warning_t, danger_t


# ---------------------------------------------------------------------------
# 6. Print results
# ---------------------------------------------------------------------------

def print_results(results: pd.DataFrame) -> None:
    separator("Prediction Results")

    tier_counts = results["risk_tier"].value_counts()
    print("  Risk tier summary:")
    for tier, info in RISK_TIERS.items():
        print(f"    {info['emoji']}  {tier:<8} : {tier_counts.get(tier, 0):>4} rows")

    latest = results.iloc[-1]
    tier   = latest["risk_tier"]
    print(f"\n  Latest prediction:")
    print(f"    Timestamp   : {results.index[-1].date()}")
    print(f"    Probability : {latest['flood_probability']:.1%}")
    print(f"    Risk tier   : {RISK_TIERS[tier]['emoji']}  {tier}")

    print(f"\n  All predictions:")
    print(f"  {'Timestamp':<32} {'Probability':>12}  {'Risk':<10}")
    print(f"  {'-'*32} {'-'*12}  {'-'*10}")
    for ts, row in results.iterrows():
        t = row["risk_tier"]
        print(f"  {str(ts.date()):<32} {row['flood_probability']:>11.1%}  "
              f"{RISK_TIERS[t]['emoji']} {t}")

    alerts = results[results["risk_tier"].isin(["WARNING", "DANGER"])]
    if len(alerts) > 0:
        print(f"\n  ⚠️  WARNING/DANGER events detected: {len(alerts)}")
        for ts, row in alerts.iterrows():
            t = row["risk_tier"]
            print(f"    {RISK_TIERS[t]['emoji']}  {ts.date()}  →  "
                  f"{row['flood_probability']:.1%}  {t}")
    else:
        print(f"\n  ✅  No WARNING or DANGER events in this prediction window.")


# ---------------------------------------------------------------------------
# 7. Save CSV (appends, deduplicates)
# ---------------------------------------------------------------------------

def save_csv(results: pd.DataFrame) -> None:
    separator("Saving CSV")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(PREDICTIONS_CSV):
        existing = pd.read_csv(
            PREDICTIONS_CSV, parse_dates=["timestamp"], index_col="timestamp"
        )
        combined = pd.concat([existing, results])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = results

    combined.index.name = "timestamp"
    combined.to_csv(PREDICTIONS_CSV)
    print(f"  Saved → {PREDICTIONS_CSV}")
    print(f"  Total rows in file : {len(combined):,}")


# ---------------------------------------------------------------------------
# 8. Save plot
# ---------------------------------------------------------------------------

def save_plot(results, watch_t, warning_t, danger_t) -> None:
    separator("Saving Plot")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(results.index, results["flood_probability"],
            color="steelblue", linewidth=1.5, label="Flood Probability", zorder=3)

    ax.axhline(watch_t,   color="gold",   linestyle="--", linewidth=1,
               label=f"WATCH ({watch_t:.0%})")
    ax.axhline(warning_t, color="orange", linestyle="--", linewidth=1,
               label=f"WARNING ({warning_t:.0%})")
    ax.axhline(danger_t,  color="red",    linestyle="--", linewidth=1,
               label=f"DANGER ({danger_t:.0%})")

    ax.axhspan(watch_t,   warning_t, alpha=0.04, color="gold")
    ax.axhspan(warning_t, danger_t,  alpha=0.06, color="orange")
    ax.axhspan(danger_t,  1.0,       alpha=0.08, color="red")

    for tier, info in RISK_TIERS.items():
        mask = results["risk_tier"] == tier
        if mask.sum() > 0:
            ax.scatter(results.index[mask], results["flood_probability"][mask],
                       color=info["color"], s=50, zorder=4,
                       label=f"{info['emoji']} {tier} ({mask.sum()})")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Flood Probability")
    ax.set_xlabel("Date")
    ax.set_title(f"Flood Risk Prediction — {MODEL_LABEL} Sensor Model")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()
    print(f"  Saved → {PLOT_FILE}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    separator("Flood Prediction — XGBoost Sensor Inference Pipeline")
    print(f"  Model           : {MODEL_FILE}")
    print(f"  Sensor data     : {SENSOR_FILE}")
    print(f"  Output CSV      : {PREDICTIONS_CSV}")
    print(f"  Output plot     : {PLOT_FILE}")
    print(f"  Training cutoff : {LAST_TRAINING_DATE}")
    print(f"  Satellite data  : NOT REQUIRED (sensor-only model)")

    model, feature_cols, threshold = load_model()
    all_features                   = build_sensor_features()
    new_features                   = filter_new_rows(all_features)
    results, watch_t, warning_t, danger_t = run_predictions(
        model, feature_cols, new_features, threshold
    )

    print_results(results)
    save_csv(results)
    save_plot(results, watch_t, warning_t, danger_t)

    separator("DONE")
    print(f"  Predictions CSV : {PREDICTIONS_CSV}")
    print(f"  Plot            : {PLOT_FILE}")
    separator()


# ---------------------------------------------------------------------------
# Scheduled mode
# ---------------------------------------------------------------------------

def run_scheduled(interval_hours: int) -> None:
    print(f"\n  Scheduled mode — running every {interval_hours}h")
    print(f"  Press Ctrl+C to stop.\n")
    while True:
        try:
            run_pipeline()
            next_run = pd.Timestamp.now() + pd.Timedelta(hours=interval_hours)
            print(f"\n  Next run : {next_run.strftime('%Y-%m-%d %H:%M')}")
            time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            print("\n  Stopped by user.")
            break
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            print(f"  Retrying in {interval_hours}h...")
            time.sleep(interval_hours * 3600)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flood Prediction — XGBoost Sensor Model Inference"
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a repeating schedule instead of once.",
    )
    parser.add_argument(
        "--interval", type=int, default=12,
        help="Hours between scheduled runs (default: 12).",
    )
    args = parser.parse_args()

    if args.schedule:
        run_scheduled(args.interval)
    else:
        run_pipeline()