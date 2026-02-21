"""
feature_engineering.py
=======================
Feature engineering functions for XGBoost Flood Prediction Model.
Input:  Raw 15-minute sensor DataFrame
Output: Aggregated rolling-window feature DataFrame (one row per window)
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Hydrological Features
# ---------------------------------------------------------------------------

def compute_water_level_features(df: pd.DataFrame, col: str = "water_level") -> pd.DataFrame:
    """
    Compute rolling water level statistics.

    Args:
        df  : DataFrame with datetime index and a water level column.
        col : Name of the water level column.

    Returns:
        DataFrame with new water level feature columns appended.
    """
    # Number of 15-min ticks in each window
    ticks_3h  = 3  * 4   # 12
    ticks_6h  = 6  * 4   # 24
    ticks_24h = 24 * 4   # 96

    df = df.copy()
    df["max_water_level_6h"]    = df[col].rolling(ticks_6h,  min_periods=1).max()
    df["max_water_level_24h"]   = df[col].rolling(ticks_24h, min_periods=1).max()
    df["water_level_std_24h"]   = df[col].rolling(ticks_24h, min_periods=1).std()

    # Slope = difference over window (units per hour)
    df["water_level_slope_3h"]  = (
        df[col] - df[col].shift(ticks_3h)
    ) / 3.0
    df["water_level_slope_6h"]  = (
        df[col] - df[col].shift(ticks_6h)
    ) / 6.0

    return df


# ---------------------------------------------------------------------------
# 2. Rainfall Features
# ---------------------------------------------------------------------------

def compute_rainfall_features(df: pd.DataFrame, col: str = "rainfall_mm") -> pd.DataFrame:
    """
    Compute rolling rainfall accumulation and peak intensity.

    Args:
        df  : DataFrame with datetime index and a rainfall column (mm / 15 min).
        col : Name of the rainfall column.

    Returns:
        DataFrame with new rainfall feature columns appended.
    """
    ticks_1h  = 4
    ticks_6h  = 24
    ticks_24h = 96

    df = df.copy()
    df["rainfall_sum_1h"]       = df[col].rolling(ticks_1h,  min_periods=1).sum()
    df["rainfall_sum_6h"]       = df[col].rolling(ticks_6h,  min_periods=1).sum()
    df["rainfall_sum_24h"]      = df[col].rolling(ticks_24h, min_periods=1).sum()
    df["rainfall_max_intensity"] = df[col].rolling(ticks_1h, min_periods=1).max()

    return df


# ---------------------------------------------------------------------------
# 3. Soil Moisture Features
# ---------------------------------------------------------------------------

def compute_soil_features(df: pd.DataFrame, col: str = "soil_moisture") -> pd.DataFrame:
    """
    Compute rolling soil moisture mean and 48-hour trend.

    Args:
        df  : DataFrame with datetime index and a soil moisture column.
        col : Name of the soil moisture column.

    Returns:
        DataFrame with new soil feature columns appended.
    """
    ticks_24h = 96
    ticks_48h = 192

    df = df.copy()
    df["soil_moisture_mean_24h"]  = df[col].rolling(ticks_24h, min_periods=1).mean()
    df["soil_moisture_trend_48h"] = (
        df[col] - df[col].shift(ticks_48h)
    )

    return df


# ---------------------------------------------------------------------------
# 4. Tidal Features  (optional – skip if no tidal sensor)
# ---------------------------------------------------------------------------

def compute_tidal_features(df: pd.DataFrame, col: str = "tidal_height") -> pd.DataFrame:
    """
    Compute tidal height statistics and phase encoding.

    Args:
        df  : DataFrame with datetime index and a tidal height column.
        col : Name of the tidal height column.

    Returns:
        DataFrame with new tidal feature columns appended.
    """
    ticks_24h = 96

    df = df.copy()
    df["tidal_height_current"]  = df[col]
    df["tidal_height_max_24h"]  = df[col].rolling(ticks_24h, min_periods=1).max()

    # Approximate tidal phase: hours since last local trough
    # (replace with actual harmonic analysis if available)
    df["tidal_phase"] = (df.index.hour + df.index.minute / 60.0) % 12.42

    return df


# ---------------------------------------------------------------------------
# 5. Master Pipeline
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Hydrological
    "max_water_level_6h",
    "max_water_level_24h",
    "water_level_slope_3h",
    "water_level_slope_6h",
    "water_level_std_24h",
    # Rainfall
    "rainfall_sum_1h",
    "rainfall_sum_6h",
    "rainfall_sum_24h",
    "rainfall_max_intensity",
    # Soil
    "soil_moisture_mean_24h",
    "soil_moisture_trend_48h",
    # Tidal
    "tidal_height_current",
    "tidal_height_max_24h",
    "tidal_phase",
]


def build_features(
    df: pd.DataFrame,
    water_col:   str = "water_level",
    rainfall_col: str = "rainfall_mm",
    soil_col:    str = "soil_moisture",
    tidal_col:   str = "tidal_height",
    include_tidal: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df            : Raw 15-minute sensor DataFrame with datetime index.
        water_col     : Column name for water level readings.
        rainfall_col  : Column name for rainfall readings.
        soil_col      : Column name for soil moisture readings.
        tidal_col     : Column name for tidal height readings.
        include_tidal : Set False if no tidal sensor is available.

    Returns:
        DataFrame containing only the engineered feature columns (NaN rows dropped).
    """
    df = compute_water_level_features(df, col=water_col)
    df = compute_rainfall_features(df,   col=rainfall_col)
    df = compute_soil_features(df,       col=soil_col)

    if include_tidal:
        df = compute_tidal_features(df, col=tidal_col)
        cols = FEATURE_COLUMNS
    else:
        cols = [c for c in FEATURE_COLUMNS
                if c not in ("tidal_height_current", "tidal_height_max_24h", "tidal_phase")]

    result = df[cols].dropna()
    return result


# ---------------------------------------------------------------------------
# 6. Label Alignment Helper
# ---------------------------------------------------------------------------

def align_satellite_labels(
    features: pd.DataFrame,
    satellite_df: pd.DataFrame,
    label_col: str = "flood_extent",
    threshold: float = 0.05,
    lookback_hours: int = 24,
) -> pd.DataFrame:
    """
    Assign a binary flood label to each feature row by looking up
    the satellite pass that falls within `lookback_hours` after that row.

    Args:
        features      : Feature DataFrame with datetime index.
        satellite_df  : DataFrame of satellite passes with datetime index
                        and a flood_extent column (0–1 fraction).
        label_col     : Column in satellite_df containing flood extent.
        threshold     : Flood extent fraction above which label = 1.
        lookback_hours: Maximum hours after a feature row to search for a
                        satellite observation.

    Returns:
        Feature DataFrame with a 'flood_label' column; rows without a
        matching satellite pass are dropped.
    """
    satellite_df = satellite_df.copy()
    satellite_df["flood_label"] = (satellite_df[label_col] >= threshold).astype(int)

    labels = []
    for ts in features.index:
        window_end = ts + pd.Timedelta(hours=lookback_hours)
        match = satellite_df.loc[
            (satellite_df.index >= ts) & (satellite_df.index <= window_end),
            "flood_label",
        ]
        if len(match) > 0:
            labels.append((ts, match.iloc[0]))
        # else: no satellite pass in this window → skip row

    if not labels:
        raise ValueError("No satellite observations matched any feature window.")

    label_series = pd.DataFrame(labels, columns=["timestamp", "flood_label"]).set_index("timestamp")
    result = features.join(label_series, how="inner")
    return result


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate synthetic 15-min data for 30 days
    idx = pd.date_range("2023-01-01", periods=30 * 24 * 4, freq="15min")
    rng = np.random.default_rng(42)

    raw = pd.DataFrame({
        "water_level":  rng.uniform(0.5, 3.0, len(idx)),
        "rainfall_mm":  rng.uniform(0.0, 5.0, len(idx)),
        "soil_moisture": rng.uniform(0.2, 0.9, len(idx)),
        "tidal_height": np.sin(np.linspace(0, 60 * np.pi, len(idx))) * 1.5 + 1.5,
    }, index=idx)

    features = build_features(raw)
    print("Feature shape:", features.shape)
    print(features.head())
