"""
feature_engineering.py
=======================
Feature engineering for XGBoost Flood Prediction.

TWO FEATURE SETS
----------------
This module defines two explicit feature sets to support the
two-model training architecture:

    SENSOR_FEATURE_COLUMNS  — sensor-only features, safe for inference.
                              No satellite data required at prediction time.
                              Used by the deployed model (flood_xgb_sensor.pkl)
                              and by predict.py.

    FULL_FEATURE_COLUMNS    — sensor + satellite features, used during
                              training and revalidation only.
                              Used by the full model (flood_xgb_full.pkl).
                              Never used at inference time.

WHY TWO SETS?
-------------
Satellite passes occur every ~12 days. At inference time (real-time alerting),
no fresh satellite data is available. Satellite features like soil_saturation
and wetness_trend would be 0-12 days stale — acceptable for slow-changing
variables during revalidation, but architecturally cleaner to exclude entirely
from the deployed model so there is zero training-serving skew.

The satellite data's role at inference time is zero. Its role during training
is to provide accurate flood_labels and richer context for the full model.

DATA SOURCES
------------
Sensor CSV  : waterlevel, soil_moisture, humidity  (daily, real-time)
Satellite   : soil_saturation, wetness_trend, rainfall_1d/3d/7d,
              era5_runoff_7d, era5_soil_water, flood_label
              (every ~12 days, forward-filled during dataset preparation)

SAMPLING FREQUENCIES SUPPORTED
-------------------------------
    '15min' : 15-minute intervals (96 ticks/day)
    '1h'    : hourly intervals    (24 ticks/day)
    '2h'    : 2-hour intervals    (12 ticks/day)
    '4h'    : 4-hour intervals    ( 6 ticks/day)
    '6h'    : 6-hour intervals    ( 4 ticks/day)
    '1D'    : daily intervals     ( 1 tick/day)

CHANGES (audit fixes)
---------------------
FIX 1 — build_features() now raises a hard error (instead of printing a
         note and silently continuing) when expected feature columns are
         missing. Previously a sensor going offline during training would
         produce a model artifact with fewer features than expected, causing
         predict.py's missing-feature guard to fire at inference time with
         no indication that the model itself was trained incorrectly.

FIX 2 — align_satellite_labels() now uses merge_asof instead of a Python
         loop over each feature row. On daily data this is a minor speed
         improvement; on 15-minute data with years of history it is
         substantially faster.

FIX 3 — TICKS table extended to cover 2h, 4h, and 6h frequencies to match
         the extended detect_sensor_frequency() in prepare_dataset.py.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Ticks-per-window lookup
# Maps human-readable window sizes to row counts at each sampling frequency.
# FIX 3 — added 2h, 4h, 6h entries to match extended frequency detection.
# ---------------------------------------------------------------------------

TICKS = {
    "15min": {"1h": 4,   "3h": 12,  "6h": 24,  "24h": 96,  "48h": 192},
    "1h":    {"1h": 1,   "3h": 3,   "6h": 6,   "24h": 24,  "48h": 48},
    "2h":    {"1h": 1,   "3h": 2,   "6h": 3,   "24h": 12,  "48h": 24},
    "4h":    {"1h": 1,   "3h": 1,   "6h": 2,   "24h": 6,   "48h": 12},
    "6h":    {"1h": 1,   "3h": 1,   "6h": 1,   "24h": 4,   "48h": 8},
    "1D":    {"1h": 1,   "3h": 1,   "6h": 1,   "24h": 7,   "48h": 14},
}


# ---------------------------------------------------------------------------
# SENSOR-ONLY feature columns
# These are the ONLY features used by predict.py and flood_xgb_sensor.pkl.
# Every column here must be derivable from sensor CSV data alone.
# No satellite CSV access required at inference time.
# ---------------------------------------------------------------------------

SENSOR_FEATURE_COLUMNS = [
    # Water level dynamics
    "max_waterlevel_6h",
    "max_waterlevel_24h",
    "waterlevel_slope_3h",
    "waterlevel_slope_6h",
    "waterlevel_std_24h",
    # Sensor soil moisture dynamics
    "sensor_soilmoisture_mean_6h",
    "sensor_soilmoisture_mean_24h",
    "sensor_soilmoisture_trend_6h",
    # Atmospheric conditions
    "humidity_mean_24h",
    "humidity_trend_6h",
]


# ---------------------------------------------------------------------------
# FULL feature columns (sensor + satellite)
# Used only during training and revalidation.
# flood_xgb_full.pkl is trained on these.
# Never used by predict.py.
# ---------------------------------------------------------------------------

FULL_FEATURE_COLUMNS = SENSOR_FEATURE_COLUMNS + [
    "soil_saturation",
    "wetness_trend",
    "rainfall_1d",
    "rainfall_3d",
    "rainfall_7d",
    "era5_runoff_7d",
    "era5_soil_water",
]


# ---------------------------------------------------------------------------
# 1. Water Level Features
# ---------------------------------------------------------------------------

def compute_waterlevel_features(
    df: pd.DataFrame,
    col: str = "waterlevel",
    freq: str = "1D",
) -> pd.DataFrame:
    t  = TICKS[freq]
    df = df.copy()

    df["max_waterlevel_6h"]    = df[col].rolling(t["6h"],  min_periods=1).max()
    df["max_waterlevel_24h"]   = df[col].rolling(t["24h"], min_periods=1).max()
    df["waterlevel_std_24h"]   = df[col].rolling(t["24h"], min_periods=2).std().fillna(0)
    df["waterlevel_slope_3h"]  = (df[col] - df[col].shift(t["3h"])) / max(t["3h"], 1)
    df["waterlevel_slope_6h"]  = (df[col] - df[col].shift(t["6h"])) / max(t["6h"], 1)

    return df


# ---------------------------------------------------------------------------
# 2. Sensor Soil Moisture Features
# ---------------------------------------------------------------------------

def compute_sensor_soilmoisture_features(
    df: pd.DataFrame,
    col: str = "soil_moisture",
    freq: str = "1D",
) -> pd.DataFrame:
    t  = TICKS[freq]
    df = df.copy()

    df["sensor_soilmoisture_mean_6h"]  = df[col].rolling(t["6h"],  min_periods=1).mean()
    df["sensor_soilmoisture_mean_24h"] = df[col].rolling(t["24h"], min_periods=1).mean()
    df["sensor_soilmoisture_trend_6h"] = df[col] - df[col].shift(t["6h"])

    return df


# ---------------------------------------------------------------------------
# 3. Humidity Features
# ---------------------------------------------------------------------------

def compute_humidity_features(
    df: pd.DataFrame,
    col: str = "humidity",
    freq: str = "1D",
) -> pd.DataFrame:
    t  = TICKS[freq]
    df = df.copy()

    df["humidity_mean_24h"] = df[col].rolling(t["24h"], min_periods=1).mean()
    df["humidity_trend_6h"] = df[col] - df[col].shift(t["6h"])

    return df


# ---------------------------------------------------------------------------
# 4. Master Feature Pipeline
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    waterlevel_col:    str = "waterlevel",
    soilmoisture_col:  str = "soil_moisture",
    humidity_col:      str = "humidity",
    freq:              str = "1D",
    mode:              str = "sensor",
) -> pd.DataFrame:
    """
    Build all features from a sensor DataFrame.

    Args:
        df               : Sensor DataFrame with datetime index.
                           For mode='full', must also contain satellite
                           columns forward-filled by prepare_dataset.py.
        waterlevel_col   : Column name for water level readings.
        soilmoisture_col : Column name for sensor soil moisture readings.
        humidity_col     : Column name for humidity readings.
        freq             : Sampling frequency — '15min', '1h', '2h', '4h',
                           '6h', or '1D'. Auto-detected by prepare_dataset.py.
        mode             : 'sensor' — build SENSOR_FEATURE_COLUMNS only.
                           'full'   — build FULL_FEATURE_COLUMNS.

    Returns:
        DataFrame with selected feature columns only, NaN rows dropped.
    """
    if mode not in ("sensor", "full"):
        raise ValueError(
            f"mode='{mode}' is invalid. Choose 'sensor' or 'full'.\n"
            f"  'sensor' : inference-safe, sensor columns only\n"
            f"  'full'   : training/revalidation, sensor + satellite columns"
        )

    if freq not in TICKS:
        raise ValueError(
            f"freq='{freq}' is not supported. Choose from: {list(TICKS.keys())}"
        )

    # Validate required sensor columns
    required_sensor = [waterlevel_col, soilmoisture_col, humidity_col]
    missing_sensor  = [c for c in required_sensor if c not in df.columns]
    if missing_sensor:
        raise ValueError(
            f"Missing sensor columns in DataFrame: {missing_sensor}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Check SENSOR_COLUMN_MAP in prepare_dataset.py."
        )

    # Validate satellite columns when running in full mode
    if mode == "full":
        required_satellite = [
            "soil_saturation", "wetness_trend",
            "rainfall_1d", "rainfall_3d", "rainfall_7d",
            "era5_runoff_7d", "era5_soil_water",
        ]
        missing_satellite = [c for c in required_satellite if c not in df.columns]
        if missing_satellite:
            raise ValueError(
                f"mode='full' requires satellite columns but these are missing: {missing_satellite}\n"
                f"Available columns: {list(df.columns)}\n"
                f"These should have been forward-filled by prepare_dataset.py.\n"
                f"Run prepare_dataset.py before train_flood_model.py."
            )

    # Build all sensor features
    df = compute_waterlevel_features(df,           col=waterlevel_col,   freq=freq)
    df = compute_sensor_soilmoisture_features(df,  col=soilmoisture_col, freq=freq)
    df = compute_humidity_features(df,             col=humidity_col,     freq=freq)

    target_cols = SENSOR_FEATURE_COLUMNS if mode == "sensor" else FULL_FEATURE_COLUMNS

    # FIX 1 — hard stop on missing feature columns instead of silently
    # skipping them. Previously a missing column would print a note and
    # continue, producing a model artifact with fewer features than expected.
    # This caused predict.py's missing-feature guard to fire at inference
    # time even though the model itself was already corrupt.
    missing_features = [c for c in target_cols if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Expected feature columns are missing after build: {missing_features}\n"
            f"This means one or more sensor columns failed to produce their\n"
            f"derived features. Check the input DataFrame for gaps or wrong\n"
            f"column names. Cannot continue — a model trained on incomplete\n"
            f"features would be unreliable at inference time."
        )

    result = df[target_cols].dropna()

    print(f"  Feature mode   : {mode.upper()}")
    print(f"  Feature count  : {len(target_cols)}")
    print(f"  Feature rows   : {len(result):,}  (after dropping NaN rows)")

    return result


# ---------------------------------------------------------------------------
# 5. Label Alignment
#
# FIX 2 — replaced the Python loop over feature rows with merge_asof.
# The original loop ran satellite_df.loc[] once per feature row, which is
# O(n * m) and slow on high-frequency or multi-year data. merge_asof is
# a single vectorized operation equivalent to a sorted nearest-join.
#
# Behaviour is identical: each feature row gets the label from the next
# satellite pass within lookback_hours, or is dropped if none exists.
# ---------------------------------------------------------------------------

def align_satellite_labels(
    features:           pd.DataFrame,
    satellite_df:       pd.DataFrame,
    label_col:          str   = "flood_label",
    use_existing_label: bool  = True,
    threshold:          float = 0.05,
    lookback_hours:     int   = 24,
) -> pd.DataFrame:
    """
    Assign a binary flood label to each feature row by finding the next
    satellite pass within lookback_hours after that row's timestamp.

    Labels always come from satellite data regardless of model mode.
    The sensor-only model and the full model share identical labels —
    only their feature sets differ.

    Args:
        features           : Feature DataFrame with datetime index.
        satellite_df       : Satellite DataFrame with datetime index.
        label_col          : Column to use for labelling.
        use_existing_label : If True, use label_col directly as binary label.
                             If False, threshold flood_extent >= threshold.
        threshold          : Only used when use_existing_label=False.
        lookback_hours     : Hours after each sensor row to search for a pass.
    """
    satellite_df = satellite_df.copy()

    if use_existing_label and label_col in satellite_df.columns:
        satellite_df["flood_label"] = satellite_df[label_col].astype(int)
    else:
        flood_extent_col = label_col if label_col != "flood_label" else "flood_extent"
        if flood_extent_col not in satellite_df.columns:
            raise ValueError(
                f"Cannot derive flood_label: '{flood_extent_col}' not found.\n"
                f"Available columns: {list(satellite_df.columns)}"
            )
        satellite_df["flood_label"] = (
            satellite_df[flood_extent_col] >= threshold
        ).astype(int)

    # FIX 2 — vectorized label alignment with merge_asof.
    #
    # For each feature row find the next satellite pass (direction='forward').
    # We carry sat_timestamp alongside flood_label so the lookback window
    # filter can be applied as a single vectorized comparison after the join.

    window = pd.Timedelta(hours=lookback_hours)

    # Ensure index has a name before reset_index so the column is predictable.
    sat_work = satellite_df[["flood_label"]].copy()
    sat_work.index.name = "timestamp"
    feat_work = features.copy()
    feat_work.index.name = "timestamp"

    sat_for_join = (
        sat_work
        .reset_index()
        .rename(columns={"timestamp": "sat_timestamp"})
        .sort_values("sat_timestamp")
    )

    feat_for_join = (
        feat_work
        .reset_index()
        .sort_values("timestamp")
    )

    joined = pd.merge_asof(
        feat_for_join,
        sat_for_join,
        left_on   = "timestamp",
        right_on  = "sat_timestamp",
        direction = "forward",
    )

    # Keep only rows where the satellite pass is within lookback_hours.
    in_window = (
        joined["sat_timestamp"].notna() &
        ((joined["sat_timestamp"] - joined["timestamp"]) <= window)
    )
    result = (
        joined[in_window]
        .drop(columns=["sat_timestamp"])
        .set_index("timestamp")
        .sort_index()
    )

    if len(result) == 0:
        raise ValueError(
            "No satellite observations matched any feature window.\n"
            "Check that sensor and satellite timestamps overlap and that\n"
            f"LOOKBACK_HOURS ({lookback_hours}h) is wide enough to catch passes.\n"
            "Try increasing LOOKBACK_HOURS in prepare_dataset.py CONFIG."
        )

    total   = len(result)
    flooded = int(result["flood_label"].sum())

    print(f"  Labeled rows   : {total:,}")
    print(f"  Flood     (1)  : {flooded:,}  ({100*flooded/total:.1f}%)")
    print(f"  No Flood  (0)  : {total - flooded:,}  ({100*(total-flooded)/total:.1f}%)")

    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Smoke test — daily frequency\n")

    idx = pd.date_range("2017-01-01", periods=730, freq="1D", tz="UTC")
    rng = np.random.default_rng(42)

    sensor = pd.DataFrame({
        "waterlevel":    rng.uniform(1.5, 3.0, len(idx)),
        "soil_moisture": rng.uniform(0.1, 0.5, len(idx)),
        "humidity":      rng.uniform(1, 5, len(idx)),
    }, index=idx)

    sat_idx   = pd.date_range("2017-01-01", periods=20, freq="12D", tz="UTC")
    satellite = pd.DataFrame({
        "flood_extent":    rng.uniform(0, 0.2, 20),
        "soil_saturation": rng.uniform(0.2, 0.8, 20),
        "wetness_trend":   rng.integers(0, 2, 20),
        "rainfall_1d":     rng.uniform(0, 50, 20),
        "rainfall_3d":     rng.uniform(0, 120, 20),
        "rainfall_7d":     rng.uniform(0, 250, 20),
        "era5_runoff_7d":  rng.uniform(0, 200, 20),
        "era5_soil_water": rng.uniform(0.2, 0.49, 20),
        "flood_label":     rng.integers(0, 2, 20),
    }, index=sat_idx)

    sat_cols = ["soil_saturation", "wetness_trend", "rainfall_1d",
                "rainfall_3d", "rainfall_7d", "era5_runoff_7d", "era5_soil_water"]
    for col in sat_cols:
        sensor[col] = satellite[col].reindex(sensor.index, method="ffill")

    print("--- SENSOR MODE ---")
    sensor_features = build_features(sensor, freq="1D", mode="sensor")
    print(f"Columns: {list(sensor_features.columns)}\n")

    print("--- FULL MODE ---")
    full_features = build_features(sensor, freq="1D", mode="full")
    print(f"Columns: {list(full_features.columns)}\n")

    dataset = align_satellite_labels(full_features, satellite,
                                     label_col="flood_label",
                                     use_existing_label=True)
    print(f"\nFinal dataset shape: {dataset.shape}")
    print(dataset.head(3).to_string())