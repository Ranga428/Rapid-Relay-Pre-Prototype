"""
showcase_predict.py
===================
SHOWCASE VERSION of XGB_Predict.py

Changes from original:
  - Live sensor input : showcase_merge.csv  (instead of combined_sensor_context.csv)
  - Predictions CSV   : showcase_predict.csv
  - Plot PNG          : showcase_predict.png
  - All script references updated to showcase/ folder.
  - All model/inference logic is identical to the original.
  - Predicts on every single new row — no training cutoff filter.

Usage
-----
    python showcase_predict.py                 # live mode inference
    python showcase_predict.py --data file.csv # batch mode
    python showcase_predict.py --schedule      # scheduled mode
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

# ---------------------------------------------------------------------------
# Paths — resolve relative to showcase/ folder
# ---------------------------------------------------------------------------

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))   # showcase/
_PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                   # flood_preprototype/
_ML_PIPELINE  = os.path.join(_PROJECT_ROOT, "ml_pipeline")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, _ML_PIPELINE)

from prepare_dataset import load_sensor
from feature_engineering import build_features, SENSOR_FEATURE_COLUMNS

# ===========================================================================
# CONFIG
# ===========================================================================

MODEL_FILE = os.path.join(_PROJECT_ROOT, "model", "flood_xgb_sensor.pkl")

# Showcase outputs — same predictions/ folder as the originals
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "predictions")

LIVE_SENSOR_FILE = os.path.join(
    _PROJECT_ROOT, "data", "sensor", "showcase_merge.csv"
)

FLOOD_LOG_PATH = os.path.join(_PROJECT_ROOT, "data", "flood_event_log.csv")

PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "showcase_predict.csv")
PLOT_FILE       = os.path.join(OUTPUT_DIR, "showcase_predict.png")

LAST_TRAINING_DATE      = "2024-12-31"
DEFAULT_ALERT_THRESHOLD = 0.50
MIN_CONSECUTIVE_DAYS    = 2

SUSPECTED_MISLABELED_DATES = [
    "2025-07-09", "2025-07-10", "2025-07-11",
    "2025-11-09", "2025-11-10",
]

ROLLING_MEAN_WINDOW = 7
ROLLING_SUM_WINDOW  = 14

# ===========================================================================
# END CONFIG
# ===========================================================================

MODEL_LABEL = "XGBoost (Showcase)"

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


def append_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    if "max_waterlevel_24h" not in df.columns:
        return df
    df = df.copy()
    if "waterlevel_mean_7d" not in df.columns:
        df["waterlevel_mean_7d"] = (
            df["max_waterlevel_24h"]
            .rolling(ROLLING_MEAN_WINDOW, min_periods=1).mean()
        )
    if "waterlevel_cumrise_14d" not in df.columns:
        df["waterlevel_cumrise_14d"] = (
            df["max_waterlevel_24h"]
            .rolling(ROLLING_SUM_WINDOW, min_periods=1).sum()
        )
    return df


def apply_consecutive_filter(results: pd.DataFrame, warn_t: float,
                              min_days: int) -> pd.DataFrame:
    if min_days <= 1:
        return results
    results = results.copy()
    probs  = results["flood_probability"].values
    tiers  = results["risk_tier"].tolist()
    streak = 0
    for i, p in enumerate(probs):
        if p >= warn_t:
            streak += 1
        else:
            streak = 0
        if streak < min_days and tiers[i] in ("WARNING", "DANGER"):
            tiers[i] = "WATCH"
    results["risk_tier"] = tiers
    return results


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------

def load_model() -> tuple:
    separator("Loading Model")
    if not os.path.exists(MODEL_FILE):
        sys.exit(
            f"\n  ERROR: Model file not found.\n"
            f"  Expected : {MODEL_FILE}\n"
        )
    artifact     = joblib.load(MODEL_FILE)
    model        = artifact["model"]
    feature_cols = artifact["feature_columns"]
    threshold    = artifact.get("threshold", DEFAULT_ALERT_THRESHOLD)
    watch_t      = artifact.get("watch_threshold",   threshold)
    warn_t       = artifact.get("warning_threshold", threshold + 0.10)

    global LAST_TRAINING_DATE
    LAST_TRAINING_DATE = artifact.get("last_training_date", LAST_TRAINING_DATE)

    print(f"  Model file        : {MODEL_FILE}")
    print(f"  Version           : {artifact.get('version', 'unknown')}")
    print(f"  Feature count     : {len(feature_cols)}")
    print(f"  WATCH threshold   : {watch_t:.2f}")
    print(f"  WARNING threshold : {warn_t:.2f}")
    print(f"  Last training date: {LAST_TRAINING_DATE}")
    print(f"  Live sensor input : showcase_merge.csv")
    print(f"  Predictions out   : showcase_predict.csv")

    return model, feature_cols, watch_t, warn_t


# ---------------------------------------------------------------------------
# 2a. Live mode
# ---------------------------------------------------------------------------

def load_live_features() -> pd.DataFrame:
    separator("Loading Showcase Merge Data")
    print(f"  Source : {LIVE_SENSOR_FILE}")
    if not os.path.exists(LIVE_SENSOR_FILE):
        sys.exit(
            f"\n  ERROR: Showcase merge file not found.\n"
            f"  Expected : {LIVE_SENSOR_FILE}\n"
            f"  Fix      : Run showcase_merge.py first.\n"
        )
    sensor_df, freq = load_sensor(sensor_path=LIVE_SENSOR_FILE)
    print(f"  Raw rows loaded  : {len(sensor_df):,}")
    print(f"  Raw date range   : {sensor_df.index.min().date()} → "
          f"{sensor_df.index.max().date()}")

    separator("Building Features")
    features = build_features(sensor_df, freq=freq, mode="sensor",
                              flood_log_path=FLOOD_LOG_PATH)
    features = append_rolling_features(features)

    core_cols = ["waterlevel", "soil_moisture", "humidity"]
    present   = [c for c in core_cols if c in features.columns]
    before    = len(features)
    features  = features.dropna(subset=present, how="any")
    dropped   = before - len(features)

    if dropped > 0:
        print(f"  ⚠️  Dropped {dropped} incomplete rows after feature engineering")
    else:
        print(f"  ✅  All {before} rows complete")

    print(f"  Total rows built : {len(features):,}")
    return features


def filter_new_rows(features: pd.DataFrame) -> pd.DataFrame:
    separator("Filtering to New Rows Only")

    already_predicted = set()

    if os.path.exists(PREDICTIONS_CSV):
        try:
            existing = pd.read_csv(PREDICTIONS_CSV, parse_dates=["timestamp"],
                                   index_col="timestamp")
            if not existing.empty:
                if existing.index.tzinfo is None:
                    existing.index = existing.index.tz_localize("UTC")
                already_predicted = set(existing.index)
                print(f"  showcase_predict.csv rows : {len(existing):,}")
            else:
                print("  showcase_predict.csv exists but is empty.")
        except Exception as e:
            print(f"  Could not read predictions CSV ({e}) — processing all rows")
    else:
        print("  showcase_predict.csv not found — processing all rows")

    new_df = features[~features.index.isin(already_predicted)]

    print(f"  Total rows       : {len(features):,}")
    print(f"  Already saved    : {len(already_predicted):,} rows (skipped)")
    print(f"  New rows         : {len(new_df):,}")

    if len(new_df) == 0:
        sys.exit(
            f"\n  No new rows found.\n"
            f"  Append new sensor rows to showcase_merge.csv and re-run.\n"
        )

    print(f"  Date range       : {new_df.index[0].date()} → {new_df.index[-1].date()}")
    return new_df


# ---------------------------------------------------------------------------
# 2b. Batch mode
# ---------------------------------------------------------------------------

def load_batch_features(data_path: str, feature_cols: list) -> tuple:
    separator("Loading Batch Data")
    if not os.path.exists(data_path):
        sys.exit(f"\n  ERROR: Data file not found.\n  Expected: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()
    print(f"  Rows loaded  : {len(df):,}")

    ground_truth  = df.get("flood_label")
    has_features  = all(c in df.columns for c in feature_cols)

    if has_features:
        features = df[feature_cols]
        features = append_rolling_features(features)
    else:
        raw_needed = ["waterlevel", "soil_moisture", "humidity"]
        missing    = [c for c in raw_needed if c not in df.columns]
        if missing:
            sys.exit(f"\n  ERROR: Missing raw columns: {missing}")
        from prepare_dataset import detect_sensor_frequency
        freq     = detect_sensor_frequency(df)
        features = build_features(df, freq=freq, mode="sensor",
                                  flood_label_series=ground_truth)
        features = append_rolling_features(features)

    return features, ground_truth


# ---------------------------------------------------------------------------
# 3. Run predictions
# ---------------------------------------------------------------------------

def run_predictions(model, feature_cols, features, watch_t, warn_t) -> tuple:
    separator("Running Predictions")
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        sys.exit(f"\n  ERROR: Missing feature columns:\n    {missing}")

    X        = features[feature_cols].values
    probs    = model.predict_proba(X)[:, 1]
    danger_t = warn_t + 0.10

    prob_std = probs.std()
    print(f"  Probability std  : {prob_std:.4f}")
    print(f"  Prob range       : {probs.min():.3f} – {probs.max():.3f}")

    def classify(prob):
        if   prob >= danger_t: return "DANGER"
        elif prob >= warn_t:   return "WARNING"
        elif prob >= watch_t:  return "WATCH"
        else:                  return "CLEAR"

    results = pd.DataFrame({
        "flood_probability": probs.round(4),
        "risk_tier":         [classify(p) for p in probs],
        "watch_threshold":   watch_t,
        "warning_threshold": warn_t,
        "danger_threshold":  danger_t,
    }, index=features.index)

    for col in feature_cols:
        results[col] = features[col].values

    if MIN_CONSECUTIVE_DAYS > 1:
        before  = (results["risk_tier"].isin(["WARNING", "DANGER"])).sum()
        results = apply_consecutive_filter(results, warn_t, MIN_CONSECUTIVE_DAYS)
        after   = (results["risk_tier"].isin(["WARNING", "DANGER"])).sum()
        if before - after > 0:
            print(f"  Consec. filter   : suppressed {before-after} single-day spikes → WATCH")

    print(f"  Predictions made : {len(results):,}")
    return results, danger_t


# ---------------------------------------------------------------------------
# 4. Print results
# ---------------------------------------------------------------------------

def print_results(results, ground_truth=None,
                  watch_t=None, warn_t=None, danger_t=None) -> None:
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


# ---------------------------------------------------------------------------
# 5. Save CSV
# ---------------------------------------------------------------------------

def save_csv(results, output_csv) -> None:
    separator("Saving showcase_predict.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv, parse_dates=["timestamp"],
                               index_col="timestamp")
        combined = pd.concat([existing, results])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = results
    combined.index.name = "timestamp"
    combined.to_csv(output_csv)
    print(f"  Saved → {output_csv}  ({len(combined):,} rows)")


# ---------------------------------------------------------------------------
# 6. Save plot
# ---------------------------------------------------------------------------

def save_plot(results, watch_t, warn_t, danger_t,
              plot_file, ground_truth=None) -> None:
    separator("Saving showcase_predict.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(results.index, results["flood_probability"],
            color="steelblue", linewidth=1.5, label="Flood Probability", zorder=3)
    ax.axhline(watch_t,  color="gold",   linestyle="--", linewidth=1,
               label=f"WATCH ({watch_t:.0%})")
    ax.axhline(warn_t,   color="orange", linestyle="--", linewidth=1,
               label=f"WARNING ({warn_t:.0%})")
    ax.axhline(danger_t, color="red",    linestyle="--", linewidth=1,
               label=f"DANGER ({danger_t:.0%})")
    ax.axhspan(watch_t, warn_t,   alpha=0.04, color="gold")
    ax.axhspan(warn_t,  danger_t, alpha=0.06, color="orange")
    ax.axhspan(danger_t, 1.0,     alpha=0.08, color="red")
    for tier, info in RISK_TIERS.items():
        mask = results["risk_tier"] == tier
        if mask.sum() > 0:
            ax.scatter(results.index[mask], results["flood_probability"][mask],
                       color=info["color"], s=50, zorder=4,
                       label=f"{info['emoji']} {tier} ({mask.sum()})")
    if ground_truth is not None:
        gt          = ground_truth.reindex(results.index).fillna(0)
        flood_dates = gt[gt == 1].index
        if len(flood_dates) > 0:
            ax.vlines(flood_dates, 0, 1, color="red", alpha=0.15,
                      linewidth=1.5, label=f"Actual Flood ({len(flood_dates)})")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Flood Probability")
    ax.set_xlabel("Date")
    ax.set_title(f"Flood Risk Prediction — {MODEL_LABEL}")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"  Saved → {plot_file}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_path=None, threshold_search=False) -> None:
    batch_mode = data_path is not None
    separator(f"Showcase Flood Prediction — XGBoost Inference")
    print(f"  Input  : {'BATCH — ' + data_path if batch_mode else 'LIVE — showcase_merge.csv'}")
    print(f"  Output : showcase_predict.csv  +  showcase_predict.png")

    model, feature_cols, watch_t, warn_t = load_model()

    if batch_mode:
        features, ground_truth = load_batch_features(data_path, feature_cols)
        stem        = os.path.splitext(os.path.basename(data_path))[0]
        output_csv  = os.path.join(OUTPUT_DIR, f"showcase_xgb_{stem}_predictions.csv")
        output_plot = os.path.join(OUTPUT_DIR, f"showcase_xgb_{stem}_predictions.png")
    else:
        all_features = load_live_features()
        features     = filter_new_rows(all_features)
        ground_truth = None
        output_csv   = PREDICTIONS_CSV
        output_plot  = PLOT_FILE

    results, danger_t = run_predictions(model, feature_cols, features, watch_t, warn_t)
    print_results(results, ground_truth, watch_t, warn_t, danger_t)
    save_csv(results, output_csv)
    save_plot(results, watch_t, warn_t, danger_t, output_plot, ground_truth)

    separator("DONE")
    print(f"  CSV  : {output_csv}")
    print(f"  Plot : {output_plot}")
    separator()


# ---------------------------------------------------------------------------
# Scheduled mode
# ---------------------------------------------------------------------------

def run_scheduled(interval_hours: int) -> None:
    print(f"\n  Scheduled mode — running every {interval_hours}h")
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
            time.sleep(interval_hours * 3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="showcase_predict — XGBoost inference on showcase_merge.csv"
    )
    parser.add_argument("--data",     type=str, default=None)
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--interval", type=int, default=12)
    args = parser.parse_args()

    if args.schedule:
        run_scheduled(args.interval)
    else:
        run_pipeline(data_path=args.data)