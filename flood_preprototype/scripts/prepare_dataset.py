"""
prepare_dataset.py
==================
Merges sensor data and satellite/EO data into a single ML-ready dataset.

Just run:
    python prepare_dataset.py

No arguments needed. Edit the CONFIG section below if paths change.
"""

import sys
import os
import pandas as pd
import numpy as np
from feature_engineering import build_features, align_satellite_labels, FEATURE_COLUMNS


# ===========================================================================
# CONFIG  — edit these if your paths or column names ever change
# ===========================================================================

SENSOR_FILE    = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\sensor\obando_environmental_data.csv"
SATELLITE_FILE = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\sentinel1\GEE-Processing\sentinel1_timeseries.csv"
OUTPUT_FILE    = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\flood_dataset.csv"

# Flood label threshold: flood_extent >= this value -> flood_label = 1
FLOOD_THRESHOLD = 0.05

# How many hours after a sensor row to search for the next satellite pass
LOOKBACK_HOURS = 24

# ---------------------------------------------------------------------------
# Column name mapping
# Left  = what this script expects internally
# Right = actual column name in your CSV
# ---------------------------------------------------------------------------

SENSOR_COLUMN_MAP = {
    "timestamp":   "timestamp",
    "water_level": "water_level",
    "rainfall":    "rainfall",
    "humidity":    "humidity",
}

SATELLITE_COLUMN_MAP = {
    "timestamp":       "timestamp",
    "flood_extent":    "flood_extent",
    "soil_saturation": "soil_saturation",
    "wetness_trend":   "wetness_trend",
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


def detect_sensor_frequency(df: pd.DataFrame) -> str:
    """
    Detect whether sensor data is daily, hourly, or 15-minute.
    Returns one of: '15min', '1h', '1D'
    """
    if len(df) < 2:
        return "1D"
    diffs = df.index.to_series().diff().dropna()
    median_diff = diffs.median()
    minutes = median_diff.total_seconds() / 60

    if minutes <= 20:
        return "15min"
    elif minutes <= 70:
        return "1h"
    else:
        return "1D"


def fix_gee_timestamps(series: pd.Series) -> pd.Series:
    """
    Fix the malformed GEE double-timezone suffix.

    Google Earth Engine sometimes exports timestamps like:
        2017-05-26T10:05:58ZT00:00:00Z
    when it should be:
        2017-05-26T10:05:58Z

    This strips everything after the first Z.
    Safe to call on any series — non-matching values are left unchanged.
    """
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        fixed = series.str.replace(r"ZT.*$", "Z", regex=True)
        n_fixed = (fixed != series).sum()
        if n_fixed > 0:
            print(f"  Fixed {n_fixed} GEE double-timezone timestamps  "
                  f"(e.g. 2017-05-26T10:05:58ZT00:00:00Z -> 2017-05-26T10:05:58Z)")
        return fixed
    return series


def parse_timestamps_robust(series: pd.Series, source_name: str) -> pd.Series:
    """
    Try multiple timestamp formats before giving up.
    Handles: ISO8601, GEE double-suffix, Excel serial numbers, common date strings.
    """
    # Pre-process: fix GEE double-timezone suffix before any parsing attempt
    series = fix_gee_timestamps(series)

    # Try 1: standard pandas with utc
    result = pd.to_datetime(series, utc=True, errors="coerce")
    n_parsed = result.notna().sum()
    if n_parsed == len(series):
        print(f"  Timestamp format : auto-detected (all {n_parsed} parsed)")
        return result

    # Try 2: infer format explicitly (no utc first, then localize)
    try:
        result2 = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        if result2.notna().sum() > n_parsed:
            result2 = result2.dt.tz_localize("UTC", ambiguous="infer",
                                              nonexistent="shift_forward")
            n_parsed = result2.notna().sum()
            print(f"  Timestamp format : inferred (parsed {n_parsed} / {len(series)})")
            return result2
    except Exception:
        pass

    # Try 3: Excel serial date (numeric)
    if pd.api.types.is_numeric_dtype(series):
        try:
            result3 = pd.to_datetime(series, unit="D", origin="1899-12-30", utc=True,
                                     errors="coerce")
            n3 = result3.notna().sum()
            if n3 > n_parsed:
                print(f"  Timestamp format : Excel serial date (parsed {n3} / {len(series)})")
                return result3
        except Exception:
            pass

    # Try 4: mixed format string-by-string
    try:
        result4 = pd.to_datetime(series, format="mixed", dayfirst=False,
                                 errors="coerce", utc=True)
        n4 = result4.notna().sum()
        if n4 > n_parsed:
            print(f"  Timestamp format : mixed (parsed {n4} / {len(series)})")
            return result4
    except Exception:
        pass

    # If still broken, show sample values to help user diagnose
    bad_vals = series[result.isna()].head(5).tolist()
    print(f"\n  Could not parse {result.isna().sum()} timestamps in {source_name}.")
    print(f"  Sample unparseable values: {bad_vals}")
    print(f"  Fix: convert your timestamp column to ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ")

    return result


# ---------------------------------------------------------------------------
# Step 1 — Load sensor CSV
# ---------------------------------------------------------------------------

def load_sensor() -> pd.DataFrame:
    separator("Step 1 — Loading Sensor Data")
    print(f"  Path : {SENSOR_FILE}\n")

    if not os.path.exists(SENSOR_FILE):
        sys.exit(
            f"\n  ERROR: Sensor file not found.\n"
            f"  Expected : {SENSOR_FILE}\n"
            f"  Fix      : Update SENSOR_FILE in the CONFIG section at the top of this script."
        )

    df = pd.read_csv(SENSOR_FILE)
    print(f"  Raw shape    : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Raw columns  : {list(df.columns)}")

    # Rename columns to internal names
    reverse_map = {v: k for k, v in SENSOR_COLUMN_MAP.items()}
    df = df.rename(columns=reverse_map)

    # Check required columns
    required = list(SENSOR_COLUMN_MAP.keys())
    missing  = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"\n  ERROR: Sensor CSV is missing expected columns: {missing}\n"
            f"  Found columns : {list(df.columns)}\n"
            f"  Fix           : Update SENSOR_COLUMN_MAP in the CONFIG section."
        )

    # Parse timestamps robustly
    df["timestamp"] = parse_timestamps_robust(df["timestamp"], "sensor")
    bad = df["timestamp"].isna().sum()
    if bad:
        print(f"  Dropping {bad} rows with unparseable timestamps.")
        df = df.dropna(subset=["timestamp"])

    df = df.set_index("timestamp").sort_index()

    # Detect sampling frequency
    freq = detect_sensor_frequency(df)
    print(f"  Detected frequency : {freq}")

    # Keep only needed columns
    keep = [c for c in ["water_level", "rainfall", "humidity"] if c in df.columns]
    df   = df[keep]

    # Report nulls
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        print(f"\n  WARNING — Null values found:")
        print(nulls[nulls > 0].to_string())

    print(f"\n  Loaded rows  : {len(df):,}")
    print(f"  Date range   : {df.index[0]}  ->  {df.index[-1]}")
    print(f"  Columns kept : {list(df.columns)}")
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
            f"  Fix      : Update SATELLITE_FILE in the CONFIG section at the top of this script."
        )

    df = pd.read_csv(SATELLITE_FILE)
    print(f"  Raw shape    : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Raw columns  : {list(df.columns)}")

    # Show raw timestamp samples BEFORE parsing so user can see the format
    ts_col = SATELLITE_COLUMN_MAP.get("timestamp", "timestamp")
    if ts_col in df.columns:
        print(f"\n  Raw timestamp samples : {df[ts_col].head(5).tolist()}")

    # Rename columns
    reverse_map = {v: k for k, v in SATELLITE_COLUMN_MAP.items()}
    df = df.rename(columns=reverse_map)

    # Check required columns
    required = list(SATELLITE_COLUMN_MAP.keys())
    missing  = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"\n  ERROR: Satellite CSV is missing expected columns: {missing}\n"
            f"  Found columns : {list(df.columns)}\n"
            f"  Fix           : Update SATELLITE_COLUMN_MAP in the CONFIG section."
        )

    # Parse timestamps robustly
    df["timestamp"] = parse_timestamps_robust(df["timestamp"], "satellite")
    bad = df["timestamp"].isna().sum()
    if bad > 0:
        print(f"\n  WARNING: {bad} rows still have unparseable timestamps.")
        print(f"  Dropping those rows.")
        df = df.dropna(subset=["timestamp"])

    if len(df) == 0:
        sys.exit(
            f"\n  ERROR: All satellite timestamps failed to parse.\n"
            f"  Please paste a few lines of your sentinel1_timeseries.csv here\n"
            f"  so the timestamp format can be identified and fixed."
        )

    df = df.set_index("timestamp").sort_index()

    # Label preview
    df["flood_label"] = (df["flood_extent"] >= FLOOD_THRESHOLD).astype(int)
    n_flood    = int(df["flood_label"].sum())
    n_no_flood = len(df) - n_flood

    print(f"\n  Loaded passes     : {len(df):,}")
    print(f"  Date range        : {df.index[0]}  ->  {df.index[-1]}")
    print(f"  flood_extent range: {df['flood_extent'].min():.4f}  ->  {df['flood_extent'].max():.4f}")
    print(f"  Flood passes  (1) : {n_flood}   (flood_extent >= {FLOOD_THRESHOLD})")
    print(f"  Clear passes  (0) : {n_no_flood}")
    print(f"\n  Sample data:")
    print(df[["flood_extent", "soil_saturation", "wetness_trend", "flood_label"]].head(5).to_string())

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
        print(f"\n  WARNING: Only {overlap_days} days of overlap — you may get very few labeled rows.")
    else:
        print(f"  OK: Sufficient overlap found.")


# ---------------------------------------------------------------------------
# Step 4 — Forward-fill soil_saturation into sensor timeline
# ---------------------------------------------------------------------------

def dedup_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Remove duplicate index entries, keeping the last occurrence.
    Reports how many duplicates were found and dropped.
    """
    n_dups = df.index.duplicated().sum()
    if n_dups > 0:
        print(f"  WARNING: {n_dups} duplicate timestamps found in {label} — keeping last value.")
        dup_examples = df.index[df.index.duplicated(keep=False)].unique()[:3].tolist()
        print(f"  Example duplicates: {dup_examples}")
        df = df[~df.index.duplicated(keep="last")]
    return df


def merge_soil_saturation(sensor_df: pd.DataFrame, satellite_df: pd.DataFrame) -> pd.DataFrame:
    separator("Step 4 — Merging Soil Saturation into Sensor Timeline")

    sensor_df    = sensor_df.copy()
    satellite_df = satellite_df.copy()

    # Remove duplicate timestamps from both sides before reindexing
    sensor_df    = dedup_index(sensor_df,    "sensor")
    satellite_df = dedup_index(satellite_df, "satellite")

    # Forward-fill satellite soil_saturation into sensor timeline
    # merge_asof is more robust than reindex for mismatched datetime indices
    sensor_reset    = sensor_df.reset_index()
    satellite_reset = satellite_df[["soil_saturation"]].reset_index().rename(
        columns={"timestamp": "timestamp"}
    )

    merged = pd.merge_asof(
        sensor_reset.sort_values("timestamp"),
        satellite_reset.sort_values("timestamp"),
        on="timestamp",
        direction="backward",   # use the most recent satellite pass before each sensor row
    )
    merged = merged.set_index("timestamp").sort_index()
    sensor_df = merged

    soil_col = sensor_df["soil_saturation"]
    filled   = soil_col.notna().sum()
    unfilled = soil_col.isna().sum()

    if unfilled > 0:
        print(f"  WARNING: {unfilled:,} sensor rows fall before the first satellite pass")
        print(f"           and have no soil_saturation value — they will be dropped.")

    pct = 100 * filled / len(sensor_df)
    print(f"  Rows with soil_saturation : {filled:,} / {len(sensor_df):,}  ({pct:.1f}%)")
    print(f"  soil_saturation range     : {soil_col.min():.4f}  ->  {soil_col.max():.4f}")
    print(f"  Method                    : merge_asof (backward) from satellite passes")

    return sensor_df


# ---------------------------------------------------------------------------
# Step 5 — Feature engineering + label alignment (frequency-aware)
# ---------------------------------------------------------------------------

def build_dataset(
    sensor_df: pd.DataFrame,
    satellite_df: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    separator("Step 5 — Feature Engineering + Label Alignment")

    print(f"  Sensor frequency : {freq}")
    print("  Building rolling-window features...")

    features = build_features(sensor_df, freq=freq)

    print(f"  Feature rows  : {len(features):,}")
    print(f"  Feature cols  : {len(features.columns)}")
    print(f"  Features      : {list(features.columns)}")

    print(f"\n  Aligning satellite labels (lookback = {LOOKBACK_HOURS}h)...")
    dataset = align_satellite_labels(
        features,
        satellite_df,
        label_col      = "flood_extent",
        threshold      = FLOOD_THRESHOLD,
        lookback_hours = LOOKBACK_HOURS,
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
    print(f"  All columns    : {list(dataset.columns)}")
    print(f"\n  Sample output:")
    print(dataset.head(3).to_string())

    if n_total < 30:
        print(f"\n  WARNING: Only {n_total} labeled rows — very small for training.")
        print(f"           Try increasing LOOKBACK_HOURS in CONFIG.")

    print(f"\n  scale_pos_weight hint : {ratio:.2f}")
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
    separator()

    sensor_df,   freq = load_sensor()
    satellite_df      = load_satellite()
    check_overlap(sensor_df, satellite_df)
    sensor_df         = merge_soil_saturation(sensor_df, satellite_df)
    dataset           = build_dataset(sensor_df, satellite_df, freq)
    save_and_report(dataset)


if __name__ == "__main__":
    main()