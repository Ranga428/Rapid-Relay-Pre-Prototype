"""
feature_engineering.py
=======================
Feature engineering for Flood Prediction (XGBoost / RF / LightGBM).

CHANGES FROM PREVIOUS VERSION
------------------------------
REMOVED — days_since_last_flood dropped from SENSOR_FEATURE_COLUMNS.
    The feature defaulted to 999 for almost all rows because no
    flood_event_log.csv exists yet. A near-constant feature adds no
    signal and was diluting recall-sensitive splits. The function
    compute_flood_history_features() is retained so the feature can be
    re-added once a populated flood event log is available.

    To re-enable: uncomment "days_since_last_flood" in
    SENSOR_FEATURE_COLUMNS and ensure flood_event_log.csv exists with
    confirmed flood event dates before retraining.

SENSOR_FEATURE_COLUMNS now has 25 features (down from 26 — one removed).
FULL_FEATURE_COLUMNS retains the 3 SAR-derived satellite columns on top.

TWO FEATURE SETS
----------------
    SENSOR_FEATURE_COLUMNS  — sensor-only, safe for real-time inference.
                              Used by the deployed sensor models.

    FULL_FEATURE_COLUMNS    — sensor + SAR-derived satellite features.
                              Used during training / revalidation only.

LABEL ALIGNMENT STRATEGY
--------------------------
align_satellite_labels() uses direction='forward' with a 288-hour
(12-day) lookahead. Each sensor day is labeled by the NEXT upcoming
satellite pass — the correct target for early warning.
"""

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Ticks-per-window lookup
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
# SENSOR-ONLY feature columns (25 total)
# Safe for real-time inference — no satellite data required.
#
# days_since_last_flood is intentionally excluded until a populated
# flood_event_log.csv exists. See module docstring.
# ---------------------------------------------------------------------------

SENSOR_FEATURE_COLUMNS = [
    # --- Water level (point / window stats) ---
    "max_waterlevel_6h",
    "max_waterlevel_24h",
    "waterlevel_slope_3h",
    "waterlevel_slope_6h",
    "waterlevel_std_24h",
    "waterlevel_rise_rate_48h",

    # --- Water level (lag / memory) ---
    "waterlevel_lag_1d",
    "waterlevel_lag_2d",
    "waterlevel_lag_3d",

    # --- Water level (context / proximity) ---
    "waterlevel_days_above_threshold",
    "waterlevel_pct_rank_30d",
    "waterlevel_distance_to_max",

    # --- Water level (slow-build rolling) ---
    "waterlevel_mean_7d",
    "waterlevel_cumrise_14d",

    # --- Soil moisture ---
    "sensor_soilmoisture_mean_6h",
    "sensor_soilmoisture_mean_24h",
    "sensor_soilmoisture_trend_6h",

    # --- Soil moisture (lag) ---
    "soilmoisture_lag_1d",
    "soilmoisture_lag_2d",

    # --- Humidity ---
    "humidity_mean_24h",
    "humidity_trend_6h",

    # --- Cross-sensor interactions ---
    "waterlevel_x_soilmoisture",
    "humidity_x_waterlevel_slope",

    # --- Season (month encoding) ---
    "season_sin",
    "season_cos",

    # --- Season (week-of-year encoding) ---
    "week_sin",
    "week_cos",

    # days_since_last_flood — REMOVED until flood_event_log.csv is populated
    # "days_since_last_flood",
]


# ---------------------------------------------------------------------------
# FULL feature columns (sensor + SAR-derived satellite)
# Used only during training and revalidation.
# ERA5/GPM columns excluded — label leakage (see original notes).
# ---------------------------------------------------------------------------

FULL_FEATURE_COLUMNS = SENSOR_FEATURE_COLUMNS + [
    "soil_saturation",   # SAR VV/VH normalized ratio
    "wetness_trend",     # 30-day VV backscatter trend
    "orbit_flag",        # ASCENDING/DESCENDING geometry
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

    # Existing point / window features
    df["max_waterlevel_6h"]        = df[col].rolling(t["6h"],  min_periods=1).max()
    df["max_waterlevel_24h"]       = df[col].rolling(t["24h"], min_periods=1).max()
    df["waterlevel_std_24h"]       = df[col].rolling(t["24h"], min_periods=2).std().fillna(0)
    df["waterlevel_slope_3h"]      = (df[col] - df[col].shift(t["3h"])) / max(t["3h"], 1)
    df["waterlevel_slope_6h"]      = (df[col] - df[col].shift(t["6h"])) / max(t["6h"], 1)
    df["waterlevel_rise_rate_48h"] = (df[col] - df[col].shift(t["48h"])) / max(t["48h"], 1)

    # Consecutive ticks above 1.5 sigma — sustained high water state
    above     = (df[col] > 1.5).astype(int)
    group_key = (above != above.shift()).cumsum()
    df["waterlevel_days_above_threshold"] = (
        above.groupby(group_key).cumcount().add(1).mul(above)
    )

    # Percentile rank vs rolling 30-day window
    window_30d = t["24h"] * 30
    df["waterlevel_pct_rank_30d"] = (
        df[col]
        .rolling(window_30d, min_periods=7)
        .apply(lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]), raw=False)
    )

    # Distance from expanding historical maximum — headroom to danger mark
    historical_max = df[col].expanding(min_periods=1).max()
    df["waterlevel_distance_to_max"] = historical_max - df[col]

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
# 4. Season Features
# ---------------------------------------------------------------------------

def compute_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sin/cos encoding for month and week-of-year.

    Month encoding: broad wet/dry season cycle.
    Week encoding : finer resolution — peak Obando flood risk is
                    weeks 35–38 (late Aug to mid-Sep), which month
                    encoding cannot distinguish within August/September.
    """
    df    = df.copy()
    month = df.index.month.astype(float)

    try:
        week = df.index.isocalendar().week.astype(float).values
    except AttributeError:
        week = pd.Series(df.index).dt.isocalendar().week.astype(float).values

    df["season_sin"] = np.sin(2 * np.pi * month / 12)
    df["season_cos"] = np.cos(2 * np.pi * month / 12)
    df["week_sin"]   = np.sin(2 * np.pi * week / 52)
    df["week_cos"]   = np.cos(2 * np.pi * week / 52)
    return df


# ---------------------------------------------------------------------------
# 5. Lag Features
# ---------------------------------------------------------------------------

def compute_lag_features(
    df: pd.DataFrame,
    waterlevel_col:   str = "waterlevel",
    soilmoisture_col: str = "soil_moisture",
    freq: str = "1D",
) -> pd.DataFrame:
    """
    Explicit prior-day values as model inputs.

    Gives the model direct memory of what the river level actually was
    yesterday vs. just a slope estimate — captures ascending multi-day
    trends that rolling stats compress into a single derivative.
    """
    t   = TICKS[freq]
    df  = df.copy()
    tpd = t["24h"]

    df["waterlevel_lag_1d"]   = df[waterlevel_col].shift(tpd * 1)
    df["waterlevel_lag_2d"]   = df[waterlevel_col].shift(tpd * 2)
    df["waterlevel_lag_3d"]   = df[waterlevel_col].shift(tpd * 3)
    df["soilmoisture_lag_1d"] = df[soilmoisture_col].shift(tpd * 1)
    df["soilmoisture_lag_2d"] = df[soilmoisture_col].shift(tpd * 2)
    return df


# ---------------------------------------------------------------------------
# 6. Interaction Features
# ---------------------------------------------------------------------------

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit cross-sensor products for compound risk states.

    waterlevel_x_soilmoisture:
        High water + saturated soil — every raindrop becomes runoff,
        compounding flood risk beyond what either signal alone captures.

    humidity_x_waterlevel_slope:
        Active rainfall loading (high humidity) during a rising river.
        Only the positive (rising) slope component is used.
    """
    df = df.copy()

    if "max_waterlevel_24h" in df.columns and "sensor_soilmoisture_mean_24h" in df.columns:
        df["waterlevel_x_soilmoisture"] = (
            df["max_waterlevel_24h"] * df["sensor_soilmoisture_mean_24h"]
        )
    else:
        df["waterlevel_x_soilmoisture"] = 0.0

    if "humidity_mean_24h" in df.columns and "waterlevel_slope_6h" in df.columns:
        df["humidity_x_waterlevel_slope"] = (
            df["humidity_mean_24h"] * df["waterlevel_slope_6h"].clip(lower=0)
        )
    else:
        df["humidity_x_waterlevel_slope"] = 0.0

    return df


# ---------------------------------------------------------------------------
# 7. Rolling Waterlevel Features
# ---------------------------------------------------------------------------

def append_rolling_waterlevel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slow-build flood features for Aug–Oct patterns where water rises
    gradually over many days rather than surging in a single event.
    Idempotent — safe to call even if columns already exist.
    """
    if "max_waterlevel_24h" not in df.columns:
        return df
    df = df.copy()
    if "waterlevel_mean_7d" not in df.columns:
        df["waterlevel_mean_7d"] = (
            df["max_waterlevel_24h"].rolling(7, min_periods=1).mean()
        )
    if "waterlevel_cumrise_14d" not in df.columns:
        df["waterlevel_cumrise_14d"] = (
            df["max_waterlevel_24h"].rolling(14, min_periods=1).sum()
        )
    return df


# ---------------------------------------------------------------------------
# 8. Flood History Features (retained but NOT in SENSOR_FEATURE_COLUMNS)
# ---------------------------------------------------------------------------

def compute_flood_history_features(
    df: pd.DataFrame,
    flood_label_series: pd.Series = None,
    flood_log_path: str = None,
) -> pd.DataFrame:
    """
    Computes days_since_last_flood — watershed recovery proxy.

    NOT currently in SENSOR_FEATURE_COLUMNS. Re-enable once a populated
    flood_event_log.csv exists with confirmed event dates.

    Batch mode : pass flood_label_series (pd.Series, same index as df).
    Live mode  : pass flood_log_path to operator-maintained CSV
                 with columns [timestamp, flood_label].
    Neither    : feature set to 999 (unknown).
    """
    df     = df.copy()
    labels = None

    if flood_label_series is not None:
        labels = flood_label_series.reindex(df.index).fillna(0)
    elif flood_log_path is not None and os.path.exists(flood_log_path):
        try:
            log    = pd.read_csv(
                flood_log_path,
                parse_dates=["timestamp"],
                index_col="timestamp",
            )
            labels = log["flood_label"].reindex(df.index).fillna(0)
        except Exception as e:
            print(f"  WARNING: Could not load flood log ({e}). "
                  f"days_since_last_flood set to 999.")

    if labels is None:
        df["days_since_last_flood"] = 999
        return df

    flood_dates = labels[labels == 1].index.sort_values()
    days_since  = []
    for ts in df.index:
        past = flood_dates[flood_dates < ts]
        days_since.append(999 if len(past) == 0 else int((ts - past[-1]).days))

    df["days_since_last_flood"] = days_since
    return df


# ---------------------------------------------------------------------------
# 9. Master Feature Pipeline
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    waterlevel_col:     str = "waterlevel",
    soilmoisture_col:   str = "soil_moisture",
    humidity_col:       str = "humidity",
    freq:               str = "1D",
    mode:               str = "sensor",
    flood_label_series: pd.Series = None,
    flood_log_path:     str = None,
) -> pd.DataFrame:
    """
    Build all features from a sensor DataFrame.

    Args:
        df                 : Sensor DataFrame with DatetimeIndex.
        waterlevel_col     : Column name for water level readings.
        soilmoisture_col   : Column name for soil moisture readings.
        humidity_col       : Column name for humidity readings.
        freq               : Sampling frequency string.
        mode               : 'sensor' or 'full'.
        flood_label_series : Optional flood labels for days_since_last_flood
                             (batch / training mode). Currently unused since
                             that feature is excluded from SENSOR_FEATURE_COLUMNS.
        flood_log_path     : Optional path to flood event log CSV
                             (live mode). Currently unused for same reason.

    Returns:
        DataFrame with selected feature columns only, NaN rows dropped.
    """
    if mode not in ("sensor", "full"):
        raise ValueError(f"mode='{mode}' is invalid. Choose 'sensor' or 'full'.")
    if freq not in TICKS:
        raise ValueError(
            f"freq='{freq}' is not supported. Choose from: {list(TICKS.keys())}"
        )

    required_sensor = [waterlevel_col, soilmoisture_col, humidity_col]
    missing_sensor  = [c for c in required_sensor if c not in df.columns]
    if missing_sensor:
        raise ValueError(
            f"Missing sensor columns in DataFrame: {missing_sensor}\n"
            f"Available columns: {list(df.columns)}"
        )

    if mode == "full":
        required_satellite = ["soil_saturation", "wetness_trend", "orbit_flag"]
        missing_satellite  = [c for c in required_satellite if c not in df.columns]
        if missing_satellite:
            raise ValueError(
                f"mode='full' requires SAR satellite columns but these are missing: "
                f"{missing_satellite}\n"
                f"Run prepare_dataset.py before train scripts."
            )

    df = compute_waterlevel_features(df,          col=waterlevel_col,   freq=freq)
    df = compute_sensor_soilmoisture_features(df, col=soilmoisture_col, freq=freq)
    df = compute_humidity_features(df,            col=humidity_col,     freq=freq)
    df = compute_season_features(df)
    df = compute_lag_features(df,
                               waterlevel_col=waterlevel_col,
                               soilmoisture_col=soilmoisture_col,
                               freq=freq)
    df = compute_interaction_features(df)
    df = append_rolling_waterlevel(df)

    # days_since_last_flood excluded from target_cols — computed here
    # only if explicitly requested (flood_label_series or flood_log_path passed)
    if flood_label_series is not None or flood_log_path is not None:
        df = compute_flood_history_features(
            df,
            flood_label_series=flood_label_series,
            flood_log_path=flood_log_path,
        )

    target_cols      = SENSOR_FEATURE_COLUMNS if mode == "sensor" else FULL_FEATURE_COLUMNS
    missing_features = [c for c in target_cols if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Expected feature columns are missing after build: {missing_features}\n"
            f"Check the input DataFrame for gaps or wrong column names."
        )

    result = df[target_cols].dropna()

    print(f"  Feature mode   : {mode.upper()}")
    print(f"  Feature count  : {len(target_cols)}")
    print(f"  Feature rows   : {len(result):,}  (after dropping NaN rows)")

    return result


# ---------------------------------------------------------------------------
# 10. Label Alignment (unchanged)
# ---------------------------------------------------------------------------

def align_satellite_labels(
    features:           pd.DataFrame,
    satellite_df:       pd.DataFrame,
    label_col:          str   = "flood_label",
    use_existing_label: bool  = True,
    threshold:          float = 0.05,
    lookback_hours:     int   = 288,
) -> pd.DataFrame:
    """
    Assign a binary flood label to each feature row using direction='forward'.
    Each sensor row is matched to the NEXT upcoming satellite pass within
    lookback_hours. Physical meaning: "Will the next satellite pass observe
    a flood?" — the correct target for an early warning system.
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

    window               = pd.Timedelta(hours=lookback_hours)
    sat_work             = satellite_df[["flood_label"]].copy()
    sat_work.index.name  = "timestamp"
    feat_work            = features.copy()
    feat_work.index.name = "timestamp"

    sat_for_join = (
        sat_work.reset_index()
        .rename(columns={"timestamp": "sat_timestamp"})
        .sort_values("sat_timestamp")
    )
    feat_for_join = feat_work.reset_index().sort_values("timestamp")

    joined = pd.merge_asof(
        feat_for_join,
        sat_for_join,
        left_on   = "timestamp",
        right_on  = "sat_timestamp",
        direction = "forward",
    )

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
            f"Check timestamp overlap and that lookback_hours ({lookback_hours}h) "
            "is wide enough."
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
        "waterlevel":    rng.uniform(-2.0, 3.0, len(idx)),
        "soil_moisture": rng.uniform(0.1, 0.5, len(idx)),
        "humidity":      rng.uniform(1, 5, len(idx)),
    }, index=idx)

    for col in ["soil_saturation", "wetness_trend", "orbit_flag"]:
        sensor[col] = rng.uniform(0, 1, len(idx))

    print("--- SENSOR MODE ---")
    sf = build_features(sensor, freq="1D", mode="sensor")
    print(f"Columns ({len(sf.columns)}): {list(sf.columns)}")
    print(f"Expected 27 sensor features\n")

    print("--- FULL MODE ---")
    ff = build_features(sensor, freq="1D", mode="full")
    print(f"Columns ({len(ff.columns)}): {list(ff.columns)}")
    print(f"Expected 30 full features\n")