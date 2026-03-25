"""
prepare_dataset.py
==================
Merges sensor data and satellite/EO data into a single ML-ready dataset.

This script always produces a FULL dataset (sensor + SAR satellite features
plus flood labels). Used for training and revalidation only.

predict_*.py does NOT use this script — it loads sensor data directly.

CHANGES FROM PREVIOUS VERSION
------------------------------
NEW 1  — build_dataset() now passes flood_label_series into build_features()
         so that days_since_last_flood is computed correctly during dataset
         preparation. The satellite flood_label column is used as the source.

NEW 2  — SAR_FEATURE_COLS and OUTPUT_FILE paths unchanged. No other
         structural changes — all new features are computed inside
         feature_engineering.py and picked up automatically via
         SENSOR_FEATURE_COLUMNS.

LABEL ALIGNMENT (unchanged)
----------------------------
Labels are assigned using direction='forward' with LOOKBACK_HOURS=288
(12 days). Each sensor day is labeled by the NEXT upcoming satellite pass
within the next 12 days.
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
# CONFIG
# ===========================================================================

# AFTER
_PREP_DIR      = os.path.dirname(os.path.abspath(__file__))
SENSOR_FILE    = os.path.join(_PREP_DIR, "..", "data", "sensor", "obando_environmental_data.csv")
SATELLITE_FILE = os.path.join(_PREP_DIR, "..", "data", "sentinel1", "GEE-Processing", "sentinel1_timeseries.csv")
OUTPUT_FILE    = os.path.join(_PREP_DIR, "..", "data", "flood_dataset.csv")

USE_EXISTING_LABEL_COL = "flood_label"
FLOOD_THRESHOLD        = 0.60
LOOKBACK_HOURS         = 288

WATERLEVEL_MIN  = None
WATERLEVEL_ZMAX = 6.0

SENSOR_COLUMN_MAP = {
    "timestamp":     "timestamp",
    "waterlevel":    "waterlevel",
    "soil_moisture": "soil_moisture",
    "humidity":      "humidity",
}

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
    "era5_runoff_1d":  "era5_runoff_1d",
    "era5_runoff_7d":  "era5_runoff_7d",
    "era5_soil_water": "era5_soil_water",
}

# SAR-only columns merged into sensor timeline (ERA5/GPM excluded — label leak)
SAR_FEATURE_COLS = [
    "soil_saturation",
    "wetness_trend",
    "orbit_flag",
]

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
    if len(df) < 2:
        return "1D"
    diffs       = df.index.to_series().diff().dropna()
    median_diff = diffs.median()
    minutes     = median_diff.total_seconds() / 60
    if minutes <= 20:    return "15min"
    elif minutes <= 70:  return "1h"
    elif minutes <= 130: return "2h"
    elif minutes <= 250: return "4h"
    elif minutes <= 400: return "6h"
    else:                return "1D"


def fix_gee_timestamps(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        fixed   = series.str.replace(r"ZT.*$", "Z", regex=True)
        n_fixed = (fixed != series).sum()
        if n_fixed > 0:
            print(f"  Fixed {n_fixed} GEE double-timezone timestamps.")
        return fixed
    return series


def parse_timestamps_robust(series: pd.Series, source_name: str) -> pd.Series:
    series   = fix_gee_timestamps(series)
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
    return result


# ---------------------------------------------------------------------------
# Step 1 — Load sensor CSV
# ---------------------------------------------------------------------------

def load_sensor(sensor_path: str = None) -> tuple:
    separator("Step 1 — Loading Sensor Data")
    path = sensor_path if sensor_path is not None else SENSOR_FILE
    print(f"  Path : {path}\n")

    if not os.path.exists(path):
        sys.exit(f"\n  ERROR: Sensor file not found.\n  Expected : {path}")

    df = pd.read_csv(path)
    print(f"  Raw shape    : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Raw columns  : {list(df.columns)}")

    reverse_map = {v: k for k, v in SENSOR_COLUMN_MAP.items()}
    df          = df.rename(columns=reverse_map)

    required = list(SENSOR_COLUMN_MAP.keys())
    missing  = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"\n  ERROR: Sensor CSV missing columns: {missing}\n"
            f"  Found: {list(df.columns)}\n"
            f"  Fix: Update SENSOR_COLUMN_MAP in CONFIG."
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
        df        = df.dropna(subset=["waterlevel"])
        n_nan     = n_before - len(df)
        if n_nan > 0:
            print(f"  Dropped {n_nan} rows with NaN waterlevel.")

        n_before2 = len(df)
        df        = df[df["waterlevel"].abs() <= WATERLEVEL_ZMAX]
        n_extreme = n_before2 - len(df)
        if n_extreme > 0:
            print(f"  Dropped {n_extreme} rows with |waterlevel z-score| > "
                  f"{WATERLEVEL_ZMAX} (sensor fault artifacts).")

        print(f"  Kept {len(df):,} / {n_before:,} sensor rows "
              f"({100*len(df)/n_before:.1f}%) after dropout filter.")

    if "humidity" in df.columns:
        n_null = df["humidity"].isna().sum()
        if n_null > 0:
            df["humidity"] = df["humidity"].ffill()
            print(f"\n  WARNING — {n_null} null humidity value(s): forward-filled.")
        else:
            nulls = df.isnull().sum()
            if nulls.sum() > 0:
                print(f"\n  WARNING — Null values found:")
                print(nulls[nulls > 0].to_string())

    print(f"\n  Loaded rows  : {len(df):,}")
    print(f"  Date range   : {df.index[0]}  ->  {df.index[-1]}")
    print(f"  Columns kept : {list(df.columns)}")
    print(f"\n  Waterlevel stats (z-scored):")
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
        sys.exit(f"\n  ERROR: Satellite file not found.\n  Expected : {SATELLITE_FILE}")

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
            f"\n  ERROR: Satellite CSV missing columns: {missing}\n"
            f"  Found: {list(df.columns)}"
        )

    df["timestamp"] = parse_timestamps_robust(df["timestamp"], "satellite")
    bad = df["timestamp"].isna().sum()
    if bad > 0:
        print(f"\n  WARNING: {bad} unparseable timestamps — dropping.")
        df = df.dropna(subset=["timestamp"])

    if len(df) == 0:
        sys.exit("\n  ERROR: All satellite timestamps failed to parse.")

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

    print(f"\n  ERA5/GPM diagnostics (reference only — NOT used as features):")
    for col in ["rainfall_1d", "rainfall_3d", "rainfall_7d",
                "era5_runoff_1d", "era5_runoff_7d", "era5_soil_water"]:
        if col in df.columns:
            print(f"  {col:<22}: {df[col].min():.3f}  ->  {df[col].max():.3f}")
        else:
            print(f"  {col:<22}: NOT FOUND in satellite CSV")

    print(f"\n  SAR features (used as model features):")
    for col in ["soil_saturation", "wetness_trend", "orbit_flag"]:
        if col in df.columns:
            print(f"  {col:<22}: {df[col].min():.3f}  ->  {df[col].max():.3f}")

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
            f"\n  ERROR: No temporal overlap.\n"
            f"  Sensor    : {s_start.date()} -> {s_end.date()}\n"
            f"  Satellite : {e_start.date()} -> {e_end.date()}"
        )

    overlap_days = (overlap_end - overlap_start).days
    print(f"  Overlap period    : {overlap_start.date()}  ->  {overlap_end.date()}")
    print(f"  Overlap length    : {overlap_days} days")

    if overlap_days < 30:
        print(f"\n  WARNING: Only {overlap_days} days of overlap.")
    else:
        print(f"  OK: Sufficient overlap found.")


# ---------------------------------------------------------------------------
# Step 4 — Forward-fill SAR satellite columns into sensor timeline
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
    separator("Step 4 — Merging SAR Satellite Columns into Sensor Timeline")

    sensor_df    = sensor_df.copy()
    satellite_df = satellite_df.copy()
    sensor_df    = dedup_index(sensor_df,    "sensor")
    satellite_df = dedup_index(satellite_df, "satellite")

    sat_cols = [c for c in SAR_FEATURE_COLS if c in satellite_df.columns]

    print(f"  Merging SAR columns     : {sat_cols}")
    print(f"  Excluded (label source) : rainfall_1d/3d/7d, era5_runoff_1d/7d, era5_soil_water")

    sensor_reset    = sensor_df.reset_index()
    satellite_reset = (
        satellite_df[sat_cols]
        .reset_index()
        .rename(columns={"timestamp": "timestamp"})
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
            print(f"  WARNING: {unfilled:,} rows before first pass "
                  f"have no {col} — will be dropped during feature building.")
        print(f"  {col:<22} : {filled:,} / {len(sensor_df):,} filled  ({pct:.1f}%)")

    return sensor_df


# ---------------------------------------------------------------------------
# Step 5 — Feature engineering + label alignment
# NEW: passes satellite flood_label series into build_features() so that
#      days_since_last_flood is computed from confirmed satellite flood events.
# ---------------------------------------------------------------------------

def build_dataset(
    sensor_df: pd.DataFrame,
    satellite_df: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    separator("Step 5 — Feature Engineering + Label Alignment")

    print(f"  Sensor frequency  : {freq}")
    print(f"  Label strategy    : forward, {LOOKBACK_HOURS}h window")
    print(f"  Building features in FULL mode (sensor + SAR satellite)...")

    # --- NEW: extract satellite flood labels for days_since_last_flood ---
    # Forward-fill the satellite flood_label onto the sensor daily index
    # so compute_flood_history_features() has a label at each sensor row.
    sat_labels = None
    if "flood_label" in satellite_df.columns:
        sat_labels = (
            satellite_df["flood_label"]
            .reindex(sensor_df.index, method="ffill")
            .fillna(0)
            .astype(int)
        )
        n_flood_days = int(sat_labels.sum())
        print(f"  Flood label series : {n_flood_days} flood days passed to "
              f"compute_flood_history_features()")

    features = build_features(
        sensor_df,
        freq=freq,
        mode="full",
        flood_label_series=sat_labels,   # NEW: passed in
    )

    use_existing = bool(
        USE_EXISTING_LABEL_COL and USE_EXISTING_LABEL_COL in satellite_df.columns
    )
    label_col = USE_EXISTING_LABEL_COL if use_existing else "flood_extent"

    print(f"\n  Aligning satellite labels (lookback = {LOOKBACK_HOURS}h, direction = forward)...")
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
    print(f"  Class ratio    : 1 : {ratio:.1f}")

    print(f"\n  Feature columns in dataset:")
    feat_cols       = [c for c in dataset.columns if c != "flood_label"]
    sensor_feats    = [c for c in feat_cols if c in SENSOR_FEATURE_COLUMNS]
    satellite_feats = [c for c in feat_cols if c not in SENSOR_FEATURE_COLUMNS]
    print(f"    Sensor features    ({len(sensor_feats)}) : {sensor_feats}")
    print(f"    SAR features       ({len(satellite_feats)}) : {satellite_feats}")

    print(f"\n  Sample output:")
    print(dataset.head(3).to_string())

    if n_total < 100:
        print(f"\n  WARNING: Only {n_total} labeled rows.")
        print(f"           Consider increasing LOOKBACK_HOURS (now {LOOKBACK_HOURS}h).")

    print(f"\n  scale_pos_weight hint  : {ratio:.2f}")
    print(f'\n  Next step: python RF_train_flood_model.py')
    print(f'             python XGB_train_flood_model.py')
    print(f'             python LGBM_train_flood_model.py')
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
    print(f"  Label lookahead: {LOOKBACK_HOURS}h ({LOOKBACK_HOURS//24} days, forward)")
    print(f"  SAR features   : {SAR_FEATURE_COLS}")
    print(f"  Excluded cols  : rainfall_1d/3d/7d, era5_runoff_1d/7d, era5_soil_water")
    separator()

    sensor_df,   freq = load_sensor()
    satellite_df      = load_satellite()
    check_overlap(sensor_df, satellite_df)
    sensor_df         = merge_satellite_columns(sensor_df, satellite_df)
    dataset           = build_dataset(sensor_df, satellite_df, freq)
    save_and_report(dataset)


if __name__ == "__main__":
    main()