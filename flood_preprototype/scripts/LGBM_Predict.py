"""
predict_lgbm.py
=============
Flood Prediction — LightGBM Real-Time Inference

Runs the deployed LightGBM sensor model (flood_lgbm_sensor.pkl).
No satellite data required at inference time.

Design option  : LightGBM (Design Option 3 of 3)
Artifact       : flood_lgbm_sensor.pkl
Script         : predict_lgbm.py

Note           : If the artifact stores a CalibratedClassifierCV wrapper
                 (recommended after retraining), it is used transparently.
                 predict_proba() works identically — no code changes needed.
                 Raw LightGBM probabilities are compressed toward 0/1 extremes;
                 calibration corrects this for reliable threshold comparison.

Usage
-----
    python predict_lgbm.py
    python predict_lgbm.py --data ..\\data\\flood_dataset_test.csv
    python predict_lgbm.py --schedule --interval 12
    python predict_lgbm.py --data ..\\data\\flood_dataset_test.csv --threshold-search

BATCH MODE vs LIVE MODE
-----------------------
    Live mode  : reads the live sensor CSV, filters to rows after
                 LAST_TRAINING_DATE, builds features, runs model.

    Batch mode : reads any CSV directly. If it already has feature
                 columns uses them directly. If it only has raw sensor
                 columns, builds features. Ground truth (flood_label)
                 shown alongside predictions if present.

CHANGES FROM PREVIOUS VERSION
------------------------------
NEW 1  — FLOOD_LOG_PATH added to CONFIG.
         Operator-maintained CSV [timestamp, flood_label]. Append a row
         after each confirmed flood event. Used by
         compute_flood_history_features() for days_since_last_flood.
         Defaults to 999 if file does not exist — no crash.

NEW 2  — load_live_features() passes flood_log_path into build_features().

NEW 3  — load_batch_features() passes flood_label column into
         build_features() as flood_label_series so days_since_last_flood
         is computed correctly in batch / evaluation mode.

NEW 4  — append_rolling_features() retained as safety net.
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
from feature_engineering import build_features, SENSOR_FEATURE_COLUMNS


# ===========================================================================
# CONFIG
# ===========================================================================

MODEL_FILE       = r"..\model\flood_lgbm_sensor.pkl"
OUTPUT_DIR       = r"..\predictions"
LIVE_SENSOR_FILE = os.path.join(
    SCRIPT_DIR, r"..\data\sensor\obando_environmental_data.csv"
)
FLOOD_LOG_PATH = os.path.join(
    SCRIPT_DIR, r"..\data\flood_event_log.csv"
)

PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "flood_lgbm_sensor_predictions.csv")
PLOT_FILE       = os.path.join(OUTPUT_DIR, "flood_lgbm_sensor_predictions.png")

LAST_TRAINING_DATE      = "2024-12-31"
DEFAULT_ALERT_THRESHOLD = 0.50

SUSPECTED_MISLABELED_DATES = [
    "2025-07-09", "2025-07-10", "2025-07-11",
    "2025-11-09", "2025-11-10",
]

ROLLING_MEAN_WINDOW = 7
ROLLING_SUM_WINDOW  = 14

# ===========================================================================
# END CONFIG
# ===========================================================================

MODEL_LABEL = "LightGBM"

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


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------

def load_model() -> tuple:
    separator("Loading Model")
    if not os.path.exists(MODEL_FILE):
        sys.exit(
            f"\n  ERROR: Model file not found.\n"
            f"  Expected : {MODEL_FILE}\n"
            f"  Fix      : Run LGBM_train_flood_model.py first.\n"
        )
    artifact     = joblib.load(MODEL_FILE)
    model        = artifact["model"]
    feature_cols = artifact["feature_columns"]
    threshold    = artifact.get("threshold", DEFAULT_ALERT_THRESHOLD)
    watch_t      = artifact.get("watch_threshold",   threshold)
    warn_t       = artifact.get("warning_threshold", threshold + 0.10)

    global LAST_TRAINING_DATE
    LAST_TRAINING_DATE = artifact.get("last_training_date", LAST_TRAINING_DATE)

    model_type = type(model).__name__
    print(f"  Model loaded      : {MODEL_FILE}")
    print(f"  Model type        : {model_type}")
    if model_type == "CalibratedClassifierCV":
        print(f"  Calibration       : isotonic (probabilities are calibrated)")
        print(f"  Note              : Raw LightGBM probabilities are compressed")
        print(f"                      toward 0/1 extremes. Calibration corrects")
        print(f"                      this before threshold comparison.")
    print(f"  Feature count     : {len(feature_cols)}")
    print(f"  WATCH threshold   : {watch_t:.2f}  (recall-first, from artifact)")
    print(f"  WARNING threshold : {warn_t:.2f}  (precision-first, from artifact)")
    print(f"  Last training date: {LAST_TRAINING_DATE}  (from artifact)")
    return model, feature_cols, watch_t, warn_t


# ---------------------------------------------------------------------------
# 2a. Live mode
# ---------------------------------------------------------------------------

def load_live_features() -> pd.DataFrame:
    separator("Loading Live Sensor Data")
    sensor_df, freq = load_sensor(sensor_path=LIVE_SENSOR_FILE)
    separator("Building Sensor Features")
    features = build_features(
        sensor_df, freq=freq, mode="sensor",
        flood_log_path=FLOOD_LOG_PATH,
    )
    features = append_rolling_features(features)
    if os.path.exists(FLOOD_LOG_PATH):
        print(f"  Flood log        : {FLOOD_LOG_PATH}")
    else:
        print(f"  Flood log        : NOT FOUND — days_since_last_flood=999")
    print(f"  Total rows built : {len(features):,}")
    return features


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
            f"  Update LAST_TRAINING_DATE in predict_lgbm.py after retraining."
        )
    print(f"  Date range       : {new_df.index[0].date()} -> {new_df.index[-1].date()}")
    return new_df


# ---------------------------------------------------------------------------
# 2b. Batch mode
# ---------------------------------------------------------------------------

def load_batch_features(data_path: str, feature_cols: list) -> tuple:
    separator("Loading Batch Data")
    print(f"  File : {data_path}\n")
    if not os.path.exists(data_path):
        sys.exit(f"\n  ERROR: Data file not found.\n  Expected: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()

    print(f"  Rows loaded      : {len(df):,}")
    print(f"  Date range       : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Columns          : {list(df.columns)}")

    ground_truth       = df.get("flood_label")
    flood_label_series = ground_truth if ground_truth is not None else None

    has_features = all(c in df.columns for c in feature_cols)

    if has_features:
        print(f"\n  Feature columns found in CSV — using directly.")
        features = df[feature_cols]
        features = append_rolling_features(features)
    else:
        print(f"\n  Feature columns not found — building from raw sensor columns.")
        raw_needed = ["waterlevel", "soil_moisture", "humidity"]
        missing    = [c for c in raw_needed if c not in df.columns]
        if missing:
            sys.exit(
                f"\n  ERROR: CSV has neither pre-built features nor raw sensor columns.\n"
                f"  Missing: {missing}"
            )
        from prepare_dataset import detect_sensor_frequency
        freq     = detect_sensor_frequency(df)
        features = build_features(
            df, freq=freq, mode="sensor",
            flood_label_series=flood_label_series,
        )
        features = append_rolling_features(features)
        print(f"  Features built   : {len(features):,} rows")

    if ground_truth is not None:
        print(f"\n  Ground truth (flood_label) found:")
        print(f"    Flood=1 : {int(ground_truth.sum())}  ({100*ground_truth.mean():.1f}%)")
        print(f"    Flood=0 : {int((ground_truth==0).sum())}  "
              f"({100*(1-ground_truth.mean()):.1f}%)")
        idx_strs         = [str(ts.date()) for ts in df.index]
        found_mislabeled = [d for d in SUSPECTED_MISLABELED_DATES if d in idx_strs]
        if found_mislabeled:
            print(f"\n  ⚠️  LABEL WARNING: suspected mislabeled dates in this set:")
            for d in found_mislabeled:
                print(f"       {d}")

    return features, ground_truth


# ---------------------------------------------------------------------------
# 3. Run predictions
# ---------------------------------------------------------------------------

def run_predictions(model, feature_cols, features, watch_t, warn_t) -> tuple:
    separator("Running Predictions")
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        sys.exit(
            f"\n  ERROR: {len(missing)} feature column(s) missing:\n"
            f"    {missing}\n"
            f"  Retrain after updating feature_engineering.py.\n"
        )

    X        = features[feature_cols].values
    probs    = model.predict_proba(X)[:, 1]
    danger_t = warn_t + 0.10

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

    print(f"  Predictions made : {len(results):,}")
    return results, danger_t


# ---------------------------------------------------------------------------
# 3b. Threshold search
# ---------------------------------------------------------------------------

def run_threshold_search(model, feature_cols, features, ground_truth) -> None:
    from sklearn.metrics import precision_recall_curve
    separator("Threshold Search")

    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        print(f"  SKIP — missing feature columns: {missing}")
        return

    gt = ground_truth.reindex(features.index).dropna()
    if len(gt) == 0:
        print(f"  SKIP — no ground truth labels available.")
        return

    X     = features.loc[gt.index][feature_cols].values
    probs = model.predict_proba(X)[:, 1]
    prec, rec, thresholds = precision_recall_curve(gt.values, probs)

    watch_candidates = [(p, r, t) for p, r, t in zip(prec, rec, thresholds) if p >= 0.65]
    if watch_candidates:
        best = max(watch_candidates, key=lambda x: x[1])
        print(f"  Watch threshold  → {best[2]:.2f}  precision={best[0]:.3f}  recall={best[1]:.3f}")
    else:
        print(f"  Watch threshold  → no candidate with precision >= 0.65")

    warn_candidates = [(p, r, t) for p, r, t in zip(prec, rec, thresholds) if r >= 0.30]
    if warn_candidates:
        best = max(warn_candidates, key=lambda x: x[0])
        print(f"  Warning threshold→ {best[2]:.2f}  precision={best[0]:.3f}  recall={best[1]:.3f}")
    else:
        print(f"  Warning threshold→ no candidate with recall >= 0.30")

    print(f"\n  Apply with: python update_thresholds.py")
    print(f"    --model ..\\model\\flood_lgbm_sensor.pkl")
    print(f"    --watch <value> --warning <value>")


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

    if ground_truth is not None:
        gt = ground_truth.reindex(results.index).dropna()
        if len(gt) > 0:
            pred_watch = (results["flood_probability"] >= watch_t).astype(int)
            pred_warn  = (results["flood_probability"] >= warn_t).astype(int)
            gt_aligned = gt.reindex(pred_watch.index).fillna(0).astype(int)
            from sklearn.metrics import precision_score, recall_score
            print(f"\n  vs Ground Truth (flood_label):")
            print(f"  {'Threshold':<12} {'Precision':>10}  {'Recall':>8}  {'Alerts':>7}")
            print(f"  {'-'*12} {'-'*10}  {'-'*8}  {'-'*7}")
            for label, pred in [("WATCH", pred_watch), ("WARNING", pred_warn)]:
                prec = precision_score(gt_aligned, pred, zero_division=0)
                rec  = recall_score(gt_aligned, pred, zero_division=0)
                n    = pred.sum()
                print(f"  {label:<12} {prec:>10.3f}  {rec:>8.3f}  {n:>7}")

    print(f"\n  All predictions:")
    print(f"  {'Timestamp':<32} {'Probability':>12}  {'Risk':<10}")
    print(f"  {'-'*32} {'-'*12}  {'-'*10}")
    for ts, row in results.iterrows():
        t    = row["risk_tier"]
        gt_s = ""
        if ground_truth is not None and ts in ground_truth.index:
            gt_val = ground_truth[ts]
            gt_s   = f"  [actual={'FLOOD' if gt_val==1 else 'clear'}]"
        print(f"  {str(ts.date()):<32} {row['flood_probability']:>11.1%}  "
              f"{RISK_TIERS[t]['emoji']} {t}{gt_s}")

    alerts = results[results["risk_tier"].isin(["WARNING", "DANGER"])]
    if len(alerts) > 0:
        print(f"\n  ⚠️  WARNING/DANGER events: {len(alerts)}")
        for ts, row in alerts.iterrows():
            t = row["risk_tier"]
            print(f"    {RISK_TIERS[t]['emoji']}  {ts.date()}  "
                  f"{row['flood_probability']:.1%}  {t}")
    else:
        print(f"\n  ✅  No WARNING or DANGER events in this window.")


# ---------------------------------------------------------------------------
# 5. Save CSV
# ---------------------------------------------------------------------------

def save_csv(results, output_csv) -> None:
    separator("Saving CSV")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(output_csv):
        existing = pd.read_csv(
            output_csv, parse_dates=["timestamp"], index_col="timestamp")
        combined = pd.concat([existing, results])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = results
    combined.index.name = "timestamp"
    combined.to_csv(output_csv)
    print(f"  Saved → {output_csv}")
    print(f"  Total rows : {len(combined):,}")


# ---------------------------------------------------------------------------
# 6. Save plot
# ---------------------------------------------------------------------------

def save_plot(results, watch_t, warn_t, danger_t,
              plot_file, ground_truth=None) -> None:
    separator("Saving Plot")
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
    ax.set_title(f"Flood Risk Prediction — {MODEL_LABEL} Sensor Model")
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
    separator(f"Flood Prediction — {MODEL_LABEL} Sensor Inference Pipeline")
    print(f"  Model  : {MODEL_FILE}")
    print(f"  Mode   : {'BATCH — ' + data_path if batch_mode else 'LIVE — ' + LIVE_SENSOR_FILE}")

    model, feature_cols, watch_t, warn_t = load_model()

    if batch_mode:
        features, ground_truth = load_batch_features(data_path, feature_cols)
        stem        = os.path.splitext(os.path.basename(data_path))[0]
        output_csv  = os.path.join(OUTPUT_DIR, f"flood_lgbm_{stem}_predictions.csv")
        output_plot = os.path.join(OUTPUT_DIR, f"flood_lgbm_{stem}_predictions.png")
    else:
        all_features = load_live_features()
        features     = filter_new_rows(all_features)
        ground_truth = None
        output_csv   = PREDICTIONS_CSV
        output_plot  = PLOT_FILE

    results, danger_t = run_predictions(model, feature_cols, features, watch_t, warn_t)
    print_results(results, ground_truth, watch_t, warn_t, danger_t)
    if threshold_search and ground_truth is not None:
        run_threshold_search(model, feature_cols, features, ground_truth)
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
            time.sleep(interval_hours * 3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flood Prediction — LightGBM Sensor Model Inference"
    )
    parser.add_argument("--data",             type=str, default=None)
    parser.add_argument("--schedule",         action="store_true")
    parser.add_argument("--interval",         type=int, default=12)
    parser.add_argument("--threshold-search", action="store_true")
    args = parser.parse_args()
    if args.schedule:
        run_scheduled(args.interval)
    else:
        run_pipeline(data_path=args.data, threshold_search=args.threshold_search)