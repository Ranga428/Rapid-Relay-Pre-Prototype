"""
feature_engineering.py
=======================
Feature engineering functions for XGBoost Flood Prediction Model.

Supports multiple sensor sampling frequencies:
    - '15min' : 15-minute intervals (96 ticks per day)
    - '1h'    : hourly intervals    (24 ticks per day)
    - '1D'    : daily intervals     (1 tick per day)

Data Sources
------------
Sensor:
    water_level, rainfall, humidity

Satellite / EO (per pass):
    flood_extent, soil_saturation, wetness_trend, flood_label
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Ticks-per-window lookup table
# Tells each feature function how many rows = N hours given the frequency
# ---------------------------------------------------------------------------

TICKS = {
    "15min": {"1h": 4,  "3h": 12,  "6h": 24,  "24h": 96,  "48h": 192},
    "1h":    {"1h": 1,  "3h": 3,   "6h": 6,   "24h": 24,  "48h": 48},
    "1D":    {"1h": 1,  "3h": 1,   "6h": 1,   "24h": 7,   "48h": 14},
    # Daily: use 7-day and 14-day windows to capture weekly patterns
}


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
    "humidity_mean_24h",
    "humidity_trend_6h",
    "soil_saturation",
    "soil_saturation_mean_24h",
]


# ---------------------------------------------------------------------------
# 1. Water Level Features
# ---------------------------------------------------------------------------

def compute_water_level_features(
    df: pd.DataFrame,
    col: str = "water_level",
    freq: str = "15min",
) -> pd.DataFrame:
    t = TICKS[freq]
    df = df.copy()

    df["max_water_level_6h"]   = df[col].rolling(t["6h"],  min_periods=1).max()
    df["max_water_level_24h"]  = df[col].rolling(t["24h"], min_periods=1).max()
    df["water_level_std_24h"]  = df[col].rolling(t["24h"], min_periods=2).std().fillna(0)
    df["water_level_slope_3h"] = (df[col] - df[col].shift(t["3h"])) / max(t["3h"], 1)
    df["water_level_slope_6h"] = (df[col] - df[col].shift(t["6h"])) / max(t["6h"], 1)

    return df


# ---------------------------------------------------------------------------
# 2. Rainfall Features
# ---------------------------------------------------------------------------

def compute_rainfall_features(
    df: pd.DataFrame,
    col: str = "rainfall",
    freq: str = "15min",
) -> pd.DataFrame:
    t = TICKS[freq]
    df = df.copy()

    df["rainfall_sum_1h"]        = df[col].rolling(t["1h"],  min_periods=1).sum()
    df["rainfall_sum_6h"]        = df[col].rolling(t["6h"],  min_periods=1).sum()
    df["rainfall_sum_24h"]       = df[col].rolling(t["24h"], min_periods=1).sum()
    df["rainfall_max_intensity"] = df[col].rolling(t["1h"],  min_periods=1).max()

    return df


# ---------------------------------------------------------------------------
# 3. Humidity Features
# ---------------------------------------------------------------------------

def compute_humidity_features(
    df: pd.DataFrame,
    col: str = "humidity",
    freq: str = "15min",
) -> pd.DataFrame:
    t = TICKS[freq]
    df = df.copy()

    df["humidity_mean_24h"] = df[col].rolling(t["24h"], min_periods=1).mean()
    df["humidity_trend_6h"] = df[col] - df[col].shift(t["6h"])

    return df


# ---------------------------------------------------------------------------
# 4. Soil Saturation Features (from satellite, forward-filled)
# ---------------------------------------------------------------------------

def compute_soil_saturation_features(
    df: pd.DataFrame,
    col: str = "soil_saturation",
    freq: str = "15min",
) -> pd.DataFrame:
    t = TICKS[freq]
    df = df.copy()

    df["soil_saturation_mean_24h"] = df[col].rolling(t["24h"], min_periods=1).mean()

    return df


# ---------------------------------------------------------------------------
# 5. Master Feature Pipeline
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    water_col:    str = "water_level",
    rainfall_col: str = "rainfall",
    humidity_col: str = "humidity",
    soil_col:     str = "soil_saturation",
    freq:         str = "15min",
) -> pd.DataFrame:
    """
    Build all features from a sensor DataFrame.

    Args:
        df           : Sensor DataFrame with datetime index.
                       Must already contain soil_saturation (forward-filled
                       from satellite — handled in prepare_dataset.py).
        water_col    : Column name for water level.
        rainfall_col : Column name for rainfall.
        humidity_col : Column name for humidity.
        soil_col     : Column name for forward-filled soil saturation.
        freq         : Sampling frequency — '15min', '1h', or '1D'.
                       Detected automatically by prepare_dataset.py.

    Returns:
        DataFrame with FEATURE_COLUMNS only, NaN rows dropped.
    """
    required = [water_col, rainfall_col, humidity_col, soil_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in input DataFrame: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Tip: soil_saturation must be forward-filled from satellite data "
            f"before calling build_features(). See prepare_dataset.py."
        )

    if freq not in TICKS:
        raise ValueError(
            f"Unsupported freq='{freq}'. Choose from: {list(TICKS.keys())}"
        )

    df = compute_water_level_features(df,     col=water_col,    freq=freq)
    df = compute_rainfall_features(df,        col=rainfall_col, freq=freq)
    df = compute_humidity_features(df,        col=humidity_col, freq=freq)
    df = compute_soil_saturation_features(df, col=soil_col,     freq=freq)

    result = df[FEATURE_COLUMNS].dropna()
    return result


# ---------------------------------------------------------------------------
# 6. Label Alignment
# ---------------------------------------------------------------------------

def align_satellite_labels(
    features:       pd.DataFrame,
    satellite_df:   pd.DataFrame,
    label_col:      str   = "flood_extent",
    threshold:      float = 0.05,
    lookback_hours: int   = 24,
) -> pd.DataFrame:
    """
    Assign a binary flood label to each feature row by finding the next
    satellite pass within `lookback_hours` after that row's timestamp.
    """
    satellite_df = satellite_df.copy()
    satellite_df["flood_label"] = (
        satellite_df[label_col] >= threshold
    ).astype(int)

    labels = []
    for ts in features.index:
        window_end = ts + pd.Timedelta(hours=lookback_hours)
        match = satellite_df.loc[
            (satellite_df.index >= ts) & (satellite_df.index <= window_end),
            "flood_label",
        ]
        if len(match) > 0:
            labels.append((ts, match.iloc[0]))

    if not labels:
        raise ValueError(
            "No satellite observations matched any feature window.\n"
            "Check that sensor and satellite timestamps overlap and that\n"
            "LOOKBACK_HOURS is wide enough to catch satellite passes.\n"
            "Current setting: LOOKBACK_HOURS = " + str(lookback_hours)
        )

    label_series = (
        pd.DataFrame(labels, columns=["timestamp", "flood_label"])
        .set_index("timestamp")
    )
    result = features.join(label_series, how="inner")

    total   = len(result)
    flooded = result["flood_label"].sum()
    print(f"  Labeled rows   : {total}")
    print(f"  Flood (1)      : {flooded}  ({100*flooded/total:.1f}%)")
    print(f"  No Flood (0)   : {total - flooded}  ({100*(total-flooded)/total:.1f}%)")

    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Smoke test — daily frequency (matching your current sensor data)\n")

    # Daily sensor data for 2 years
    idx = pd.date_range("2017-01-01", periods=730, freq="1D", tz="UTC")
    rng = np.random.default_rng(42)

    sensor = pd.DataFrame({
        "water_level": rng.uniform(-3, 5, len(idx)),
        "rainfall":    rng.uniform(0, 1, len(idx)),
        "humidity":    rng.uniform(1, 5, len(idx)),
    }, index=idx)

    # Satellite passes every 12 days
    sat_idx = pd.date_range("2017-01-01", periods=20, freq="12D", tz="UTC")
    satellite = pd.DataFrame({
        "flood_extent":    rng.uniform(0, 0.2, 20),
        "soil_saturation": rng.uniform(0.2, 0.8, 20),
        "wetness_trend":   rng.integers(0, 2, 20),
    }, index=sat_idx)

    # Forward-fill soil saturation
    sensor["soil_saturation"] = (
        satellite["soil_saturation"].reindex(sensor.index, method="ffill")
    )

    features = build_features(sensor, freq="1D")
    print(f"Features shape : {features.shape}")
    print(f"Columns        : {list(features.columns)}\n")

    dataset = align_satellite_labels(features, satellite)
    print(f"\nFinal dataset  : {dataset.shape}")
    print(dataset.head(3).to_string())