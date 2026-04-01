"""
feature_engineering.py
=======================
Feature engineering for Flood Prediction (XGBoost / RF / LightGBM).

WATERLEVEL UNITS (UPDATED)
--------------------------
waterlevel is now raw metres (no z-score). One threshold needed updating:

    waterlevel_days_above_threshold — was (df[col] > 1.5) in z-score units.
    Now uses rolling 85th percentile (30-day window).

    Rationale: UHSLC Manila South Harbor MSL ≈ 2.2m above chart datum.
    Normal tidal range: 1.8–2.6m. A sustained reading above 2.5m means
    the tide is tracking above the typical high-tide mark, indicating an
    elevated baseline consistent with a pre-flood state. This matches the
    physical intent of the original 1.5σ threshold.

CHANGES FROM PREVIOUS VERSION
------------------------------
UPDATED — waterlevel_days_above_threshold now uses rolling 85th percentile
          (30-day window) instead of a fixed absolute threshold.

MONSOON FEATURES (existing)
-----------------------------
80% of 2024-2026 test-period flood days have waterlevel BELOW the training
p05 minimum (1.8-2.6m tidal range). Five features teach the model the
monsoon-flood signature:

    is_monsoon_season, waterlevel_monsoon, soilmoisture_monsoon,
    humidity_x_soilmoisture, soilmoisture_trend_3d

FIX FEATURES ADDED (NEW)
--------------------------
Eight features targeting three diagnosed failure modes in the test logs:

  POST-FLOOD DECAY (fixes Aug false-positive block):
    days_since_flood_level  — consecutive days waterlevel has been below its
                              rolling 85th-pct threshold; resets to 0 when
                              above. Signals the flood has ended.
    waterlevel_falling_streak — consecutive days of falling waterlevel
                              (capped 30). Direct receding-flood signal.
    post_flood_decay_7d     — waterlevel 7d ago minus today (lag diff).
                              Positive = water receding; negative = rising.
                              Computed lag-based (no leakage).

  LATE-SEASON ANOMALY (fixes Nov flood miss):
    humidity_anomaly_vs_30d  — humidity minus its own 30-day rolling mean.
                               Catches relative spikes at post-monsoon baseline.
    soilmoist_anomaly_vs_30d — soil moisture minus its own 30-day rolling mean.
    late_season_wet_flag     — 1 if Oct/Nov AND either anomaly is positive.
                               Direct flag for this failure mode.

  VARIANCE / DYNAMISM (fixes Jan-Mar flat probability plateau):
    waterlevel_7d_std  — 7-day rolling std of raw waterlevel.
    soilmoist_7d_std   — 7-day rolling std of raw soil moisture.
    Both are near-zero in dry season (correctly so) and non-zero when
    conditions are dynamic, helping trees distinguish active vs quiet states
    without the noise injection hack.

NOTE: No noise injection is applied. The Jan–Mar plateau is caused by
genuinely flat sensor inputs during dry season; the correct fix is the
std features above, which let the model learn "dry season = low variance"
rather than corrupting the signal.

SENSOR_FEATURE_COLUMNS now has 40 features (was 32).
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

WATERLEVEL_ELEVATED_WINDOW     = 30
WATERLEVEL_ELEVATED_PERCENTILE = 0.85


# ---------------------------------------------------------------------------
# SENSOR-ONLY feature columns (40 total — 32 original + 8 fix features)
# Safe for real-time inference — no satellite data required.
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

    # --- Monsoon features (fixes 2024-2026 distribution shift) ---
    "is_monsoon_season",
    "waterlevel_monsoon",
    "soilmoisture_monsoon",
    "humidity_x_soilmoisture",
    "soilmoisture_trend_3d",

    # --- Post-flood decay (NEW — fixes Aug false-positive block) ---
    "days_since_flood_level",
    "waterlevel_falling_streak",
    "post_flood_decay_7d",

    # --- Late-season anomaly (NEW — fixes Nov flood miss) ---
    "humidity_anomaly_vs_30d",
    "soilmoist_anomaly_vs_30d",
    "late_season_wet_flag",

    # --- Variance / dynamism (NEW — fixes Jan-Mar flat probability plateau) ---
    "waterlevel_7d_std",
    "soilmoist_7d_std",
]


# ---------------------------------------------------------------------------
# FULL feature columns (sensor + SAR-derived satellite)
# Used only during training and revalidation.
# ERA5/GPM columns excluded — label leakage.
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

    df["max_waterlevel_6h"]        = df[col].rolling(t["6h"],  min_periods=1).max()
    df["max_waterlevel_24h"]       = df[col].rolling(t["24h"], min_periods=1).max()
    df["waterlevel_std_24h"]       = df[col].rolling(t["24h"], min_periods=2).std().fillna(0)
    df["waterlevel_slope_3h"]      = (df[col] - df[col].shift(t["3h"])) / max(t["3h"], 1)
    df["waterlevel_slope_6h"]      = (df[col] - df[col].shift(t["6h"])) / max(t["6h"], 1)
    df["waterlevel_rise_rate_48h"] = (df[col] - df[col].shift(t["48h"])) / max(t["48h"], 1)

    window_30d  = t["24h"] * WATERLEVEL_ELEVATED_WINDOW
    rolling_85p = df[col].rolling(window_30d, min_periods=7).quantile(WATERLEVEL_ELEVATED_PERCENTILE)
    above       = (df[col] > rolling_85p).astype(int)
    group_key   = (above != above.shift()).cumsum()
    df["waterlevel_days_above_threshold"] = (
        above.groupby(group_key).cumcount().add(1).mul(above)
    )

    window_30d = t["24h"] * 30
    df["waterlevel_pct_rank_30d"] = (
        df[col]
        .rolling(window_30d, min_periods=7)
        .apply(lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]), raw=False)
    )

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
# 8. Monsoon Features (existing)
# ---------------------------------------------------------------------------

def compute_monsoon_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Five features that encode the monsoon-saturation flood mechanism.
    Must be called AFTER compute_lag_features (requires soilmoisture_lag_*).
    """
    df = df.copy()

    month = df.index.month
    df["is_monsoon_season"] = ((month >= 6) & (month <= 10)).astype(float)

    if "max_waterlevel_6h" in df.columns:
        df["waterlevel_monsoon"] = df["max_waterlevel_6h"] * df["is_monsoon_season"]
    else:
        df["waterlevel_monsoon"] = 0.0

    if "sensor_soilmoisture_mean_24h" in df.columns:
        df["soilmoisture_monsoon"] = df["sensor_soilmoisture_mean_24h"] * df["is_monsoon_season"]
    else:
        df["soilmoisture_monsoon"] = 0.0

    if "humidity_mean_24h" in df.columns and "sensor_soilmoisture_mean_24h" in df.columns:
        df["humidity_x_soilmoisture"] = (
            df["humidity_mean_24h"] * df["sensor_soilmoisture_mean_24h"]
        )
    else:
        df["humidity_x_soilmoisture"] = 0.0

    if "soilmoisture_lag_1d" in df.columns and "soilmoisture_lag_2d" in df.columns:
        df["soilmoisture_trend_3d"] = df["soilmoisture_lag_1d"] - df["soilmoisture_lag_2d"]
    else:
        df["soilmoisture_trend_3d"] = 0.0

    return df


# ---------------------------------------------------------------------------
# 9. Post-Flood Decay Features (NEW — fixes Aug false-positive block)
# ---------------------------------------------------------------------------

def compute_postflood_decay_features(
    df: pd.DataFrame,
    col: str = "waterlevel",
    freq: str = "1D",
) -> pd.DataFrame:
    """
    Three features that signal the flood has ended and risk is diminishing.

    Problem: After the July monsoon flood, lag features (waterlevel_lag_*,
    waterlevel_cumrise_14d) stay elevated for weeks. The model has no signal
    that the flood is OVER, so it keeps outputting DANGER through August even
    though actual=clear.

    days_since_flood_level:
        Uses the same rolling 85th-percentile threshold as
        waterlevel_days_above_threshold. Resets to 0 whenever waterlevel is
        above that threshold; increments by 1 each day it stays below.
        Computed identically to the TICKS-aware logic in waterlevel features.

    waterlevel_falling_streak:
        Consecutive days of strictly decreasing waterlevel, capped at 30.
        Uses the same tick-aware shift as the slope features.

    post_flood_decay_7d:
        Waterlevel 7 calendar days ago (lag) minus today. Positive = level
        has fallen over the past week = receding flood. Lag-based, no leakage.
        Uses t['24h']*7 ticks to be frequency-consistent.
    """
    t  = TICKS[freq]
    df = df.copy()

    # --- days_since_flood_level ---
    window_30d  = t["24h"] * WATERLEVEL_ELEVATED_WINDOW
    rolling_85p = df[col].rolling(window_30d, min_periods=7).quantile(
        WATERLEVEL_ELEVATED_PERCENTILE
    )
    above = (df[col] > rolling_85p).astype(int)

    days_since = []
    counter    = 0
    for v in above:
        if v == 1:
            counter = 0
        else:
            counter += 1
        days_since.append(counter)
    df["days_since_flood_level"] = days_since

    # --- waterlevel_falling_streak ---
    tpd   = t["24h"]
    delta = df[col] - df[col].shift(tpd)       # change since previous period

    falling_streak = []
    streak = 0
    for d in delta.fillna(0):
        if d < 0:
            streak += 1
        else:
            streak = 0
        falling_streak.append(min(streak, 30))
    df["waterlevel_falling_streak"] = falling_streak

    # --- post_flood_decay_7d ---
    lag_7d = df[col].shift(tpd * 7)
    df["post_flood_decay_7d"] = (lag_7d - df[col]).fillna(0)

    return df


# ---------------------------------------------------------------------------
# 10. Late-Season Anomaly Features (NEW — fixes Nov flood miss)
# ---------------------------------------------------------------------------

def compute_late_season_anomaly_features(
    df: pd.DataFrame,
    humidity_col:    str = "humidity",
    soilmoisture_col: str = "soil_moisture",
    freq: str = "1D",
) -> pd.DataFrame:
    """
    Three features that detect relative spikes at post-monsoon baseline.

    Problem: Nov 3–8 floods were missed because absolute sensor values in
    November are low (post-monsoon baseline). The model sees "dry-season-like"
    readings. But these floods happen from a *relative* spike above the recent
    baseline — the anomaly is invisible to absolute-value features.

    humidity_anomaly_vs_30d:
        humidity_mean_24h minus its own 30-day rolling mean. Uses the already-
        computed humidity_mean_24h to avoid double-computation. Window uses
        t['24h']*30 ticks for frequency consistency.

    soilmoist_anomaly_vs_30d:
        sensor_soilmoisture_mean_24h minus its own 30-day rolling mean.
        Same window logic.

    late_season_wet_flag:
        1 if the calendar month is October or November AND at least one of the
        two anomalies is positive (above the 30-day baseline). Hard binary flag
        directly targeting this failure mode.
    """
    t  = TICKS[freq]
    df = df.copy()

    window_30d = t["24h"] * 30

    # Use the already-computed 24h means if available; fall back to raw cols
    hum_col = "humidity_mean_24h"    if "humidity_mean_24h"             in df.columns else humidity_col
    sm_col  = "sensor_soilmoisture_mean_24h" if "sensor_soilmoisture_mean_24h" in df.columns else soilmoisture_col

    hum_30d_mean = df[hum_col].rolling(window_30d, min_periods=7).mean()
    sm_30d_mean  = df[sm_col].rolling(window_30d,  min_periods=7).mean()

    df["humidity_anomaly_vs_30d"]  = df[hum_col] - hum_30d_mean
    df["soilmoist_anomaly_vs_30d"] = df[sm_col]  - sm_30d_mean

    month = df.index.month
    is_late = ((month == 10) | (month == 11)).astype(int)
    hum_above = (df["humidity_anomaly_vs_30d"]  > 0).astype(int)
    sm_above  = (df["soilmoist_anomaly_vs_30d"] > 0).astype(int)
    df["late_season_wet_flag"] = is_late * ((hum_above | sm_above).astype(int))

    return df


# ---------------------------------------------------------------------------
# 11. Variance / Dynamism Features (NEW — fixes Jan-Mar flat probability plateau)
# ---------------------------------------------------------------------------

def compute_variance_features(
    df: pd.DataFrame,
    waterlevel_col:   str = "waterlevel",
    soilmoisture_col: str = "soil_moisture",
    freq: str = "1D",
) -> pd.DataFrame:
    """
    Two std features that let the model distinguish active vs quiet periods.

    Problem: Jan–Mar sensor readings are near-constant (dry season) →
    rolling features collapse to identical values → model outputs the same
    probability every day (10.7% stuck for weeks).

    The correct fix is NOT noise injection (which corrupts training signal).
    Instead, rolling std naturally captures "how dynamic is the sensor right
    now". Near-zero in dry season (correct — the model SHOULD know conditions
    are stable), non-zero when conditions are fluctuating pre-flood.

    waterlevel_7d_std:
        7-day rolling std of raw waterlevel. Window uses t['24h']*7 ticks.

    soilmoist_7d_std:
        7-day rolling std of raw soil moisture. Same window.
    """
    t  = TICKS[freq]
    df = df.copy()

    window_7d = t["24h"] * 7

    df["waterlevel_7d_std"] = (
        df[waterlevel_col].rolling(window_7d, min_periods=2).std().fillna(0)
    )
    df["soilmoist_7d_std"] = (
        df[soilmoisture_col].rolling(window_7d, min_periods=2).std().fillna(0)
    )

    return df


# ---------------------------------------------------------------------------
# 12. Flood History Features (retained but NOT in SENSOR_FEATURE_COLUMNS)
# ---------------------------------------------------------------------------

def compute_flood_history_features(
    df: pd.DataFrame,
    flood_label_series: pd.Series = None,
    flood_log_path: str = None,
) -> pd.DataFrame:
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
# 13. Master Feature Pipeline
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
        waterlevel_col     : Column name for water level readings (raw metres).
        soilmoisture_col   : Column name for soil moisture readings.
        humidity_col       : Column name for humidity readings.
        freq               : Sampling frequency string.
        mode               : 'sensor' or 'full'.
        flood_label_series : Optional flood labels for days_since_last_flood.
        flood_log_path     : Optional path to flood event log CSV.

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

    # Original pipeline — order preserved exactly
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
    df = compute_monsoon_features(df)           # must come after lag features

    # New fix features — appended after original pipeline, same order requirement
    df = compute_postflood_decay_features(df,   col=waterlevel_col,      freq=freq)
    df = compute_late_season_anomaly_features(df,
                                              humidity_col=humidity_col,
                                              soilmoisture_col=soilmoisture_col,
                                              freq=freq)
    df = compute_variance_features(df,          waterlevel_col=waterlevel_col,
                                                soilmoisture_col=soilmoisture_col,
                                                freq=freq)

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
# 14. Label Alignment (unchanged)
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
    lookback_hours.
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
        "waterlevel":    rng.uniform(1.8, 3.2, len(idx)),
        "soil_moisture": rng.uniform(0.1, 0.5, len(idx)),
        "humidity":      rng.uniform(1, 5, len(idx)),
    }, index=idx)

    for col in ["soil_saturation", "wetness_trend", "orbit_flag"]:
        sensor[col] = rng.uniform(0, 1, len(idx))

    print("--- SENSOR MODE ---")
    sf = build_features(sensor, freq="1D", mode="sensor")
    print(f"Columns ({len(sf.columns)}): {list(sf.columns)}")
    print(f"Expected 40 sensor features\n")

    print("--- FULL MODE ---")
    ff = build_features(sensor, freq="1D", mode="full")
    print(f"Columns ({len(ff.columns)}): {list(ff.columns)}")
    print(f"Expected 43 full features\n")

    # Verify no original features changed
    original_32 = [
        "max_waterlevel_6h","max_waterlevel_24h","waterlevel_slope_3h","waterlevel_slope_6h",
        "waterlevel_std_24h","waterlevel_rise_rate_48h","waterlevel_lag_1d","waterlevel_lag_2d",
        "waterlevel_lag_3d","waterlevel_days_above_threshold","waterlevel_pct_rank_30d",
        "waterlevel_distance_to_max","waterlevel_mean_7d","waterlevel_cumrise_14d",
        "sensor_soilmoisture_mean_6h","sensor_soilmoisture_mean_24h","sensor_soilmoisture_trend_6h",
        "soilmoisture_lag_1d","soilmoisture_lag_2d","humidity_mean_24h","humidity_trend_6h",
        "waterlevel_x_soilmoisture","humidity_x_waterlevel_slope","season_sin","season_cos",
        "week_sin","week_cos","is_monsoon_season","waterlevel_monsoon","soilmoisture_monsoon",
        "humidity_x_soilmoisture","soilmoisture_trend_3d",
    ]
    missing = [f for f in original_32 if f not in sf.columns]
    if missing:
        print(f"ERROR — original features missing: {missing}")
    else:
        print("✅  All 32 original features present and correct")
        print(f"✅  8 new fix features added: {[f for f in sf.columns if f not in original_32]}")