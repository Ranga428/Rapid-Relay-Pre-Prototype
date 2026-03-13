"""
prepare_dataset.py
==================
Merges sensor data and satellite/EO data into a single ML-ready dataset.

This script always produces a FULL dataset (sensor + satellite features
plus flood labels). It is used for training and revalidation only.

predict.py does NOT use this script — it loads sensor data directly.

Just run:
    python prepare_dataset.py

No arguments needed. Edit the CONFIG section below if paths change.

CHANGES (audit fixes)
---------------------
FIX 1 — load_sensor() now accepts an optional sensor_path parameter so
         predict.py can pass its own path instead of inheriting the
         hardcoded training path. The module-level constant is kept as
         the default so existing training usage is unchanged.

FIX 2 — load_sensor() now forward-fills humidity nulls (and warns about
         them) so training and inference handle nulls identically.
         Previously training dropped those rows via build_features()
         dropna() while predict.py filled them — a training-serving skew.

FIX 3 — detect_sensor_frequency() now recognises sub-daily non-hourly
         intervals (2h, 4h, 6h) instead of silently misclassifying them
         as daily.
"""

import sys
import os
import pandas as pd
import numpy as np
from feature_engineering import (
    build_features,
    align_satellite_labels,
    SENSOR_FEATURE_COLUMNS,
    FULL_FEATURE_COLUMNS,
)


# ===========================================================================
# CONFIG  — edit these if your paths or column names ever change
# ===========================================================================

SENSOR_FILE    = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\sensor\obando_environmental_data.csv"
SATELLITE_FILE = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\sentinel1\GEE-Processing\sentinel1_timeseries.csv"
OUTPUT_FILE    = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\flood_dataset.csv"

# Flood label column in satellite CSV.
# Set to 'flood_label' to use the pre-existing GEE-computed label directly.
# Set to None to recompute from flood_extent using FLOOD_THRESHOLD below.
USE_EXISTING_LABEL_COL = "flood_label"

# Only used if USE_EXISTING_LABEL_COL = None
FLOOD_THRESHOLD = 0.60

# How many hours after a sensor row to search for the next satellite pass
# for label alignment. 24h works well for daily sensor data.
LOOKBACK_HOURS = 24

# Water level floor — readings below this are treated as sensor offline/reset
# events, not real water level drops. These distort slope features significantly.
# All normal Obando readings are 2.0–2.6m; the ~0.6m cluster is anomalous.
WATERLEVEL_MIN = 1.0   # metres — rows below this are dropped

# ---------------------------------------------------------------------------
# SENSOR column name mapping
# Left  = internal name used by this script and feature_engineering.py
# Right = actual column name in your sensor CSV
# ---------------------------------------------------------------------------

SENSOR_COLUMN_MAP = {
    "timestamp":    "timestamp",
    "waterlevel":   "waterlevel",
    "soil_moisture":"soil_moisture",
    "humidity":     "humidity",
}

# ---------------------------------------------------------------------------
# SATELLITE column name mapping
# Left  = internal name used by this script and feature_engineering.py
# Right = actual column name in your satellite CSV
# ---------------------------------------------------------------------------

SATELLITE_COLUMN_MAP = {
    "timestamp":       "timestamp",
    "flood_extent":    "flood_extent",
    "soil_saturation": "soil_saturation",
    "wetness_trend":   "wetness_trend",
    "orbit_flag":      "orbit_flag",
    "flood_label":     "flood_label",
    "rainfall_1d":     "rainfall_1d",
    "rainfall_3d":     "rainfall_3d",
    "rainfall_7d":     "rainfall_7d",
    "era5_runoff_7d":  "era5_runoff_7d",
    "era5_soil_water": "era5_soil_water",
}

# ===========================================================================
# END CONFIG
# ===========================================================================


def separator(title=""):
    line = "=" * 55
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# FIX 3 — extended frequency detection to cover sub-daily non-hourly intervals.
# Previously anything > 70 minutes fell through to the daily branch, silently
# misclassifying 2h/4h/6h sensors and producing wrong rolling-window tick counts.
def detect_sensor_frequency(df: pd.DataFrame) -> str:
    """Detect whether sensor data is daily, hourly, or sub-daily."""
    if len(df) < 2:
        return "1D"
    diffs       = df.index.to_series().diff().dropna()
    median_diff = diffs.median()
    minutes     = median_diff.total_seconds() / 60

    if minutes <= 20:
        return "15min"
    elif minutes <= 70:
        return "1h"
    elif minutes <= 130:
        return "2h"
    elif minutes <= 250:
        return "4h"
    elif minutes <= 400:
        return "6h"
    else:
        return "1D"


def fix_gee_timestamps(series: pd.Series) -> pd.Series:
    """
    Fix the malformed GEE double-timezone suffix.
    e.g. 2017-05-26T10:05:58ZT00:00:00Z → 2017-05-26T10:05:58Z
    """
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        fixed   = series.str.replace(r"ZT.*$", "Z", regex=True)
        n_fixed = (fixed != series).sum()
        if n_fixed > 0:
            print(f"  Fixed {n_fixed} GEE double-timezone timestamps.")
        return fixed
    return series


def parse_timestamps_robust(series: pd.Series, source_name: str) -> pd.Series:
    """Try multiple timestamp formats before giving up."""
    series = fix_gee_timestamps(series)

    result   = pd.to_datetime(series, utc=True, errors="coerce")
    n_parsed = result.notna().sum()
    if n_parsed == len(series):
        print(f"  Timestamp format : auto-detected (all {n_parsed} parsed)")
        return result

    try:
        result2 = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        if result2.notna().sum() > n_parsed:
            result2  = result2.dt.tz_localize("UTC", ambiguous="infer",
                                               nonexistent="shift_forward")
            n_parsed = result2.notna().sum()
            print(f"  Timestamp format : inferred (parsed {n_parsed} / {len(series)})")
            return result2
    except Exception:
        pass

    if pd.api.types.is_numeric_dtype(series):
        try:
            result3 = pd.to_datetime(series, unit="D", origin="1899-12-30",
                                     utc=True, errors="coerce")
            n3 = result3.notna().sum()
            if n3 > n_parsed:
                print(f"  Timestamp format : Excel serial (parsed {n3} / {len(series)})")
                return result3
        except Exception:
            pass

    try:
        result4 = pd.to_datetime(series, format="mixed", dayfirst=False,
                                 errors="coerce", utc=True)
        n4 = result4.notna().sum()
        if n4 > n_parsed:
            print(f"  Timestamp format : mixed (parsed {n4} / {len(series)})")
            return result4
    except Exception:
        pass

    bad_vals = series[result.isna()].head(5).tolist()
    print(f"\n  Could not parse {result.isna().sum()} timestamps in {source_name}.")
    print(f"  Sample unparseable values: {bad_vals}")
    print(f"  Fix: convert timestamps to ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ")
    return result


# ---------------------------------------------------------------------------
# Step 1 — Load sensor CSV
#
# FIX 1 — accepts optional sensor_path so predict.py can supply its own
#          path rather than inheriting the hardcoded training constant.
#          Default keeps existing training usage unchanged.
#
# FIX 2 — forward-fills humidity nulls here so training and inference
#          both see filled data before rolling windows are computed.
#          Previously: training dropped null rows via build_features dropna();
#          inference filled them in predict.py. Now both paths are identical.
# ---------------------------------------------------------------------------

def load_sensor(sensor_path: str = None) -> tuple:
    separator("Step 1 — Loading Sensor Data")

    path = sensor_path if sensor_path is not None else SENSOR_FILE
    print(f"  Path : {path}\n")

    if not os.path.exists(path):
        sys.exit(
            f"\n  ERROR: Sensor file not found.\n"
            f"  Expected : {path}\n"
            f"  Fix      : Update SENSOR_FILE in the CONFIG section of prepare_dataset.py,\n"
            f"             or pass the correct path via --sensor on the command line."
        )

    df = pd.read_csv(path)
    print(f"  Raw shape    : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Raw columns  : {list(df.columns)}")

    reverse_map = {v: k for k, v in SENSOR_COLUMN_MAP.items()}
    df          = df.rename(columns=reverse_map)

    required = list(SENSOR_COLUMN_MAP.keys())
    missing  = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"\n  ERROR: Sensor CSV is missing expected columns: {missing}\n"
            f"  Found columns : {list(df.columns)}\n"
            f"  Fix           : Update SENSOR_COLUMN_MAP in the CONFIG section."
        )

    df["timestamp"] = parse_timestamps_robust(df["timestamp"], "sensor")
    bad = df["timestamp"].isna().sum()
    if bad:
        print(f"  Dropping {bad} rows with unparseable timestamps.")
        df = df.dropna(subset=["timestamp"])

    df   = df.set_index("timestamp").sort_index()
    freq = detect_sensor_frequency(df)
    print(f"  Detected frequency : {freq}")

    keep = [c for c in ["waterlevel", "soil_moisture", "humidity"] if c in df.columns]
    df   = df[keep]

    if "waterlevel" in df.columns:
        n_before  = len(df)
        df        = df[df["waterlevel"] >= WATERLEVEL_MIN]
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} rows with waterlevel < {WATERLEVEL_MIN}m "
                  f"(sensor offline/reset artifacts).")

    # FIX 2 — fill humidity nulls before returning so both training and
    # inference compute rolling windows on the same imputed values.
    # Previously predict.py filled here but training did not, causing the
    # 34 null rows to be dropped silently during build_features dropna().
    if "humidity" in df.columns:
        n_null = df["humidity"].isna().sum()
        if n_null > 0:
            df["humidity"] = df["humidity"].ffill()
            print(f"\n  WARNING — {n_null} null humidity value(s) found: forward-filled.")
        else:
            nulls = df.isnull().sum()
            if nulls.sum() > 0:
                print(f"\n  WARNING — Null values found:")
                print(nulls[nulls > 0].to_string())
    else:
        nulls = df.isnull().sum()
        if nulls.sum() > 0:
            print(f"\n  WARNING — Null values found:")
            print(nulls[nulls > 0].to_string())

    print(f"\n  Loaded rows  : {len(df):,}")
    print(f"  Date range   : {df.index[0]}  ->  {df.index[-1]}")
    print(f"  Columns kept : {list(df.columns)}")
    print(f"\n  Waterlevel stats:")
    print(df["waterlevel"].describe().round(3).to_string())
    print(f"\n  Sample data:")
    print(df.head(3).to_string())

    return df, freq


# ---------------------------------------------------------------------------
# Step 2 — Load satellite CSV
# ---------------------------------------------------------------------------

def load_satellite() -> pd.DataFrame:
    separator("Step 2 — Loading Satellite / EO Data")
    print(f"  Path : {SATELLITE_FILE}\n")

    if not os.path.exists(SATELLITE_FILE):
        sys.exit(
            f"\n  ERROR: Satellite file not found.\n"
            f"  Expected : {SATELLITE_FILE}\n"
            f"  Fix      : Update SATELLITE_FILE in the CONFIG section."
        )

    df = pd.read_csv(SATELLITE_FILE)
    df = df.dropna(how="all")

    print(f"  Raw shape    : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Raw columns  : {list(df.columns)}")

    ts_col = SATELLITE_COLUMN_MAP.get("timestamp", "timestamp")
    if ts_col in df.columns:
        print(f"\n  Raw timestamp samples : {df[ts_col].head(5).tolist()}")

    reverse_map = {v: k for k, v in SATELLITE_COLUMN_MAP.items() if v in df.columns}
    df          = df.rename(columns=reverse_map)

    required = ["timestamp", "flood_extent", "soil_saturation", "wetness_trend"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"\n  ERROR: Satellite CSV is missing expected columns: {missing}\n"
            f"  Found columns : {list(df.columns)}\n"
            f"  Fix           : Update SATELLITE_COLUMN_MAP in the CONFIG section."
        )

    df["timestamp"] = parse_timestamps_robust(df["timestamp"], "satellite")
    bad = df["timestamp"].isna().sum()
    if bad > 0:
        print(f"\n  WARNING: {bad} rows with unparseable timestamps — dropping.")
        df = df.dropna(subset=["timestamp"])

    if len(df) == 0:
        sys.exit(
            "\n  ERROR: All satellite timestamps failed to parse.\n"
            "  Check your sentinel1_timeseries.csv timestamp format."
        )

    df = df.set_index("timestamp").sort_index()

    if USE_EXISTING_LABEL_COL and USE_EXISTING_LABEL_COL in df.columns:
        df["flood_label"] = df[USE_EXISTING_LABEL_COL].astype(int)
        print(f"  Using pre-existing '{USE_EXISTING_LABEL_COL}' column from GEE.")
    else:
        df["flood_label"] = (df["flood_extent"] >= FLOOD_THRESHOLD).astype(int)
        print(f"  Computed flood_label from flood_extent >= {FLOOD_THRESHOLD}.")

    n_flood    = int(df["flood_label"].sum())
    n_no_flood = len(df) - n_flood

    print(f"\n  Loaded passes     : {len(df):,}")
    print(f"  Date range        : {df.index[0]}  ->  {df.index[-1]}")
    print(f"  Flood passes  (1) : {n_flood}  ({100*n_flood/len(df):.1f}%)")
    print(f"  Clear passes  (0) : {n_no_flood}  ({100*n_no_flood/len(df):.1f}%)")

    for col in ["rainfall_1d", "rainfall_3d", "rainfall_7d",
                "era5_runoff_7d", "era5_soil_water"]:
        if col in df.columns:
            print(f"  {col:<22}: {df[col].min():.3f}  ->  {df[col].max():.3f}")
        else:
            print(f"  {col:<22}: NOT FOUND in satellite CSV")

    show_cols = [c for c in
                 ["flood_extent", "soil_saturation", "wetness_trend",
                  "rainfall_7d", "era5_soil_water", "flood_label"]
                 if c in df.columns]
    print(f"\n  Sample data:")
    print(df[show_cols].head(5).to_string())

    return df


# ---------------------------------------------------------------------------
# Step 3 — Check temporal overlap
# ---------------------------------------------------------------------------

def check_overlap(sensor_df: pd.DataFrame, satellite_df: pd.DataFrame) -> None:
    separator("Step 3 — Checking Temporal Overlap")

    s_start, s_end = sensor_df.index[0],    sensor_df.index[-1]
    e_start, e_end = satellite_df.index[0], satellite_df.index[-1]

    overlap_start = max(s_start, e_start)
    overlap_end   = min(s_end,   e_end)

    print(f"  Sensor range      : {s_start.date()}  ->  {s_end.date()}")
    print(f"  Satellite range   : {e_start.date()}  ->  {e_end.date()}")

    if overlap_start >= overlap_end:
        sys.exit(
            f"\n  ERROR: Sensor and satellite data do NOT overlap in time.\n"
            f"  Sensor    : {s_start.date()} -> {s_end.date()}\n"
            f"  Satellite : {e_start.date()} -> {e_end.date()}"
        )

    overlap_days = (overlap_end - overlap_start).days
    print(f"  Overlap period    : {overlap_start.date()}  ->  {overlap_end.date()}")
    print(f"  Overlap length    : {overlap_days} days")

    if overlap_days < 30:
        print(f"\n  WARNING: Only {overlap_days} days of overlap — very few labeled rows.")
    else:
        print(f"  OK: Sufficient overlap found.")


# ---------------------------------------------------------------------------
# Step 4 — Forward-fill satellite columns into sensor timeline
# ---------------------------------------------------------------------------

def dedup_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n_dups = df.index.duplicated().sum()
    if n_dups > 0:
        print(f"  WARNING: {n_dups} duplicate timestamps in {label} — keeping last.")
        df = df[~df.index.duplicated(keep="last")]
    return df


def merge_satellite_columns(
    sensor_df: pd.DataFrame,
    satellite_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill satellite columns into the sensor timeline using
    merge_asof (backward direction).
    """
    separator("Step 4 — Merging Satellite Columns into Sensor Timeline")

    sensor_df    = sensor_df.copy()
    satellite_df = satellite_df.copy()

    sensor_df    = dedup_index(sensor_df,    "sensor")
    satellite_df = dedup_index(satellite_df, "satellite")

    sat_cols = [c for c in [
        "soil_saturation",
        "wetness_trend",
        "rainfall_1d",
        "rainfall_3d",
        "rainfall_7d",
        "era5_runoff_7d",
        "era5_soil_water",
    ] if c in satellite_df.columns]

    print(f"  Merging satellite columns : {sat_cols}")
    print(f"  Applying -1 day shift to SAR timestamps for same-day alignment.")

    sensor_reset    = sensor_df.reset_index()
    satellite_reset = satellite_df[sat_cols].reset_index().rename(
        columns={"timestamp": "timestamp"}
    )

    satellite_reset["timestamp"] = (
        satellite_reset["timestamp"] - pd.Timedelta(days=1)
    )

    merged    = pd.merge_asof(
        sensor_reset.sort_values("timestamp"),
        satellite_reset.sort_values("timestamp"),
        on        = "timestamp",
        direction = "backward",
    )
    sensor_df = merged.set_index("timestamp").sort_index()

    for col in sat_cols:
        filled   = sensor_df[col].notna().sum()
        unfilled = sensor_df[col].isna().sum()
        pct      = 100 * filled / len(sensor_df)
        if unfilled > 0:
            print(f"  WARNING: {unfilled:,} rows before first satellite pass "
                  f"have no {col} — will be dropped during feature building.")
        print(f"  {col:<22} : {filled:,} / {len(sensor_df):,} filled  ({pct:.1f}%)")

    return sensor_df


# ---------------------------------------------------------------------------
# Step 5 — Feature engineering + label alignment
# ---------------------------------------------------------------------------

def build_dataset(
    sensor_df: pd.DataFrame,
    satellite_df: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    separator("Step 5 — Feature Engineering + Label Alignment")

    print(f"  Sensor frequency : {freq}")
    print(f"  Building features in FULL mode (sensor + satellite)...")

    features = build_features(sensor_df, freq=freq, mode="full")

    use_existing = bool(
        USE_EXISTING_LABEL_COL and USE_EXISTING_LABEL_COL in satellite_df.columns
    )
    label_col = USE_EXISTING_LABEL_COL if use_existing else "flood_extent"

    if use_existing:
        print(f"\n  Using pre-existing label column '{label_col}' from satellite CSV.")
    else:
        print(f"\n  Deriving flood_label from '{label_col}' >= {FLOOD_THRESHOLD}.")

    print(f"  Aligning satellite labels (lookback = {LOOKBACK_HOURS}h)...")
    dataset = align_satellite_labels(
        features,
        satellite_df,
        label_col          = label_col,
        use_existing_label = use_existing,
        threshold          = FLOOD_THRESHOLD,
        lookback_hours     = LOOKBACK_HOURS,
    )

    return dataset


# ---------------------------------------------------------------------------
# Step 6 — Save + final report
# ---------------------------------------------------------------------------

def save_and_report(dataset: pd.DataFrame) -> None:
    separator("Saving Dataset")

    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Created output folder: {out_dir}")

    dataset.index.name = "timestamp"
    dataset.to_csv(OUTPUT_FILE)

    n_total = len(dataset)
    n_flood = int(dataset["flood_label"].sum())
    n_clear = n_total - n_flood
    ratio   = n_clear / n_flood if n_flood > 0 else float("inf")

    separator("DONE — Dataset Summary")
    print(f"  Output file    : {OUTPUT_FILE}")
    print(f"  Total rows     : {n_total:,}")
    print(f"  Flood     (1)  : {n_flood:,}   ({100*n_flood/n_total:.1f}%)")
    print(f"  No Flood  (0)  : {n_clear:,}   ({100*n_clear/n_total:.1f}%)")
    print(f"  Class ratio    : 1 : {ratio:.1f}   (no-flood per flood row)")
    print(f"\n  Feature columns in dataset:")
    feat_cols = [c for c in dataset.columns if c != "flood_label"]
    sensor_feats    = [c for c in feat_cols if c in SENSOR_FEATURE_COLUMNS]
    satellite_feats = [c for c in feat_cols if c not in SENSOR_FEATURE_COLUMNS]
    print(f"    Sensor features    ({len(sensor_feats)}) : {sensor_feats}")
    print(f"    Satellite features ({len(satellite_feats)}) : {satellite_feats}")
    print(f"\n  Sample output:")
    print(dataset.head(3).to_string())

    if n_total < 30:
        print(f"\n  WARNING: Only {n_total} labeled rows — very small for training.")
        print(f"           Try increasing LOOKBACK_HOURS in CONFIG.")

    print(f"\n  scale_pos_weight hint  : {ratio:.2f}")
    print(f"  (auto-computed by train_flood_model.py — no action needed)")
    print(f'\n  Next step: python train_flood_model.py --data "{OUTPUT_FILE}"')
    separator()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    separator("Flood Prediction — Dataset Preparation")
    print(f"  Sensor file    : {SENSOR_FILE}")
    print(f"  Satellite file : {SATELLITE_FILE}")
    print(f"  Output file    : {OUTPUT_FILE}")
    print(f"  Flood threshold: {FLOOD_THRESHOLD}")
    print(f"  Label lookback : {LOOKBACK_HOURS}h")
    print(f"  Waterlevel min : {WATERLEVEL_MIN}m  (below = sensor artifact, dropped)")
    separator()

    sensor_df,   freq = load_sensor()
    satellite_df      = load_satellite()
    check_overlap(sensor_df, satellite_df)
    sensor_df         = merge_satellite_columns(sensor_df, satellite_df)
    dataset           = build_dataset(sensor_df, satellite_df, freq)
    save_and_report(dataset)


if __name__ == "__main__":
    main()