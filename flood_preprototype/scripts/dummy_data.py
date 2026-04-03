"""
generate_dummy_sensor_data.py
=============================
Generates synthetic hardware sensor data in the format of
obando_sensor_data.csv (Date, Time, Soil (%), Humidity (%), Distance (m))

HOW IT WORKS
------------
Two data sources inform the generation:

1. obando_environmental_data.csv  (CSV1) — 9 years of daily proxy data
   Used to derive monthly seasonal patterns: what the mean and variance
   of each variable looks like across the year (Jan–Dec). These monthly
   statistics are converted back to hardware units so the synthetic
   readings follow real seasonal behavior (drier soil in Feb, wetter
   in Jul–Aug; higher water levels Sep–Oct, etc.)

2. obando_sensor_data.csv  (CSV2) — 53 real hardware readings
   Used to derive:
     - Per-minute noise level (std of minute-to-minute diff)
     - Autocorrelation (humidity is persistent across minutes; others
       are noisier, refreshing each minute)
     - Valid ranges for each sensor

GENERATION STRATEGY
-------------------
For each day requested:
  1. Look up that month's seasonal mean + std from CSV1 (in proxy units)
  2. Convert those means/stds to hardware units (inverse of calibration)
  3. Sample a "daily baseline" for each sensor from that monthly distribution
  4. Walk through each minute of the day applying:
       - A slow diurnal drift (water peaks at high tide mid-morning/evening,
         humidity peaks at night, soil moisture stable intraday)
       - Per-minute Gaussian noise matching CSV2 observed noise
       - AR(1) smoothing where the real data showed high autocorrelation

RAINY SEASON ADJUSTMENTS (Jun–Oct)
-----------------------------------
  - Soil moisture means raised (+0.02–0.03 m³/m³) for wetter ground
  - Humidity means raised (+0.4–1.4 cm CWV) for more atmospheric moisture
  - Water level means raised (+0.2–0.3 m) in Jul–Oct for higher river level
  - DIST_WET anchor lowered (0.08→0.04 m) — sensor nearly touching water
    at monsoon peak
  - sm_t clamp extended (1.0→1.8) — allows very wet days to push distance
    below DIST_WET
  - Tidal amplitude doubled in rainy months (0.05→0.10 m, 0.02→0.04 m)
  - Distance noise scale raised (0.05→0.08) to reflect turbulent water

OUTPUT
------
Columns: Date, Time, Soil (%), Humidity (%), Distance (m)
Format matches obando_sensor_data.csv exactly.
Rows: one per minute, 1440 rows/day × N days requested.

USAGE
-----
    # Generate 1 year of 1-minute data (default output)
    python generate_dummy_sensor_data.py

    # Custom date range
    python generate_dummy_sensor_data.py --start 2025-01-01 --end 2025-12-31

    # Control output path and minute interval
    python generate_dummy_sensor_data.py --start 2026-01-01 --end 2026-06-30 \
        --interval 5 --out my_dummy_data.csv

    # Dry run: just show stats, don't write
    python generate_dummy_sensor_data.py --dry-run
"""

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("dummy_gen")

# ---------------------------------------------------------------------------
# CALIBRATION CONSTANTS  (must match sensor_ingest.py exactly)
# These are used in REVERSE — proxy units → hardware units
# ---------------------------------------------------------------------------

SENSOR_HEIGHT_ABOVE_DATUM_M = 0.958

SOIL_HW_DRY    = 71.8
SOIL_HW_WET    = 85.0
SOIL_PROXY_DRY = 0.301
SOIL_PROXY_WET = 0.463

HUMIDITY_HW_MIN    = 75.12
HUMIDITY_HW_MAX    = 88.78
HUMIDITY_PROXY_MIN = 0.15
HUMIDITY_PROXY_MAX = 6.87

# ---------------------------------------------------------------------------
# NOISE PARAMETERS  (derived from CSV2 real hardware data)
# ---------------------------------------------------------------------------

# Per-minute diff std from CSV2 (with dropout rows removed)
SOIL_NOISE_STD     = 0.11   # % per minute
HUMIDITY_NOISE_STD = 1.58   # %RH per minute
DISTANCE_NOISE_STD = 0.66   # m per minute (filtered valid only)

# AR(1) coefficients from CSV2 lag-1 autocorrelation
# Humidity is strongly persistent (0.936); others are near-random
HUMIDITY_AR1 = 0.936
SOIL_AR1     = 0.0    # essentially white noise between minutes
DISTANCE_AR1 = 0.0

# Sensor valid range (from sensor_ingest.py calibration guards)
DISTANCE_MIN_VALID = 0.05
DISTANCE_MAX_VALID = 4.039

# Hardware observed ranges (from CSV2 describe())
SOIL_HW_OBSERVED_MIN     = 71.62
SOIL_HW_OBSERVED_MAX     = 71.97
HUMIDITY_HW_OBSERVED_MIN = 75.12
HUMIDITY_HW_OBSERVED_MAX = 88.78

# Dropout rate: fraction of distance readings that are sensor dropouts (0.00 m)
# CSV2 had 7 dropouts out of 53 rows = ~13%
DROPOUT_RATE = 0.13

# ---------------------------------------------------------------------------
# MONTHLY SEASONAL STATISTICS  (derived from CSV1 proxy data)
# Stored in PROXY units, converted to HW units during generation.
#
# CHANGED: Jun–Oct rows have been raised to produce stronger rainy-season
# signals across all three sensors:
#   - wl_mean  raised +0.2–0.3 m in Jul–Oct  (higher river / flood level)
#   - sm_mean  raised +0.02–0.03 m³/m³       (wetter soil)
#   - hum_mean raised +0.4–1.4 cm CWV        (more atmospheric moisture)
# ---------------------------------------------------------------------------

MONTHLY_PROXY = {
    #month: (wl_mean, wl_std, sm_mean, sm_std, hum_mean, hum_std)
    1:  (1.855, 0.510, 0.356, 0.070, 2.065, 1.093),
    2:  (1.879, 0.501, 0.301, 0.066, 2.233, 1.094),
    3:  (1.970, 0.438, 0.254, 0.038, 2.677, 1.003),
    4:  (2.156, 0.170, 0.260, 0.046, 2.842, 1.136),
    5:  (2.171, 0.217, 0.346, 0.088, 3.080, 1.515),
    # --- rainy season (Jun–Oct): all three sensors raised ---
    6:  (2.275, 0.469, 0.470, 0.026, 3.400, 1.719),   # CHANGED: sm +0.019, hum +0.46
    7:  (2.338, 0.481, 0.490, 0.014, 3.518, 1.673),   # CHANGED: wl +0.20, sm +0.024, hum +1.30
    8:  (2.497, 0.425, 0.495, 0.017, 3.681, 1.817),   # CHANGED: wl +0.30, sm +0.032, hum +1.40
    9:  (2.575, 0.179, 0.490, 0.014, 3.570, 1.708),   # CHANGED: wl +0.30, sm +0.026, hum +1.20
    10: (2.446, 0.132, 0.471, 0.045, 3.204, 1.524),   # CHANGED: wl +0.20, sm +0.030, hum +1.00
    # --- dry season resumes ---
    11: (2.132, 0.291, 0.397, 0.059, 2.224, 1.280),
    12: (1.960, 0.449, 0.380, 0.067, 2.045, 1.293),
}

# CHANGED: set of rainy-season months used to gate amplitude and noise tweaks
RAINY_MONTHS = {6, 7, 8, 9, 10}

# ---------------------------------------------------------------------------
# UNIT CONVERSION HELPERS  (inverse of calibration)
# ---------------------------------------------------------------------------

def proxy_wl_to_hw_distance(wl_m: float) -> float:
    """Inverse of calibration: waterlevel (m above datum) → distance (m from sensor)"""
    return SENSOR_HEIGHT_ABOVE_DATUM_M - wl_m


def proxy_sm_to_hw_pct(sm_proxy: float) -> float:
    """Inverse soil calibration: m³/m³ → hardware %"""
    t = (sm_proxy - SOIL_PROXY_DRY) / (SOIL_PROXY_WET - SOIL_PROXY_DRY)
    t = max(0.0, min(1.0, t))
    return SOIL_HW_DRY + t * (SOIL_HW_WET - SOIL_HW_DRY)


def proxy_hum_to_hw_rh(hum_proxy: float) -> float:
    """Inverse humidity calibration: cm CWV → hardware %RH"""
    t = (hum_proxy - HUMIDITY_PROXY_MIN) / (HUMIDITY_PROXY_MAX - HUMIDITY_PROXY_MIN)
    t = max(0.0, min(1.0, t))
    return HUMIDITY_HW_MIN + t * (HUMIDITY_HW_MAX - HUMIDITY_HW_MIN)


# ---------------------------------------------------------------------------
# DIURNAL DRIFT SHAPES
# ---------------------------------------------------------------------------

def diurnal_waterlevel_offset(minute_of_day: int, rainy: bool = False) -> float:
    """
    Simulate semi-diurnal tidal influence on water level.
    Obando is tidal — two high-water periods per day roughly 12h apart.
    Returns a signed offset in proxy metres.

    CHANGED: amplitude is doubled in rainy-season months to reflect a more
    dynamic water surface when the river is running high.
      Dry:   ±0.05 m primary, ±0.02 m second harmonic
      Rainy: ±0.10 m primary, ±0.04 m second harmonic
    """
    t = minute_of_day / 1440.0 * 2 * np.pi
    amp  = 0.10 if rainy else 0.05   # CHANGED
    amp2 = 0.04 if rainy else 0.02   # CHANGED
    return amp * np.sin(2 * t) + amp2 * np.sin(4 * t)


def diurnal_humidity_offset(minute_of_day: int) -> float:
    """
    RH is highest at night/early morning, lowest mid-afternoon.
    Returns offset in hardware %RH (±5%).
    """
    t = minute_of_day / 1440.0 * 2 * np.pi
    # Peak at midnight (minute 0), trough at noon (minute 720)
    return -5.0 * np.cos(t)


def diurnal_soil_offset(minute_of_day: int) -> float:
    """
    Soil moisture changes very little intraday — tiny drift only.
    Returns offset in hardware % (±0.05%)
    """
    t = minute_of_day / 1440.0 * 2 * np.pi
    return 0.05 * np.sin(t)


# ---------------------------------------------------------------------------
# CORE GENERATOR
# ---------------------------------------------------------------------------

def generate_day(
    target_date: date,
    rng: np.random.Generator,
    interval_minutes: int = 1,
) -> pd.DataFrame:
    """
    Generate one day of synthetic hardware sensor readings.

    Parameters
    ----------
    target_date      : calendar date to generate
    rng              : numpy random generator (for reproducibility)
    interval_minutes : how many minutes between readings (default 1)

    Returns
    -------
    DataFrame with columns: Date, Time, Soil (%), Humidity (%), Distance (m)
    """
    month = target_date.month
    rainy = month in RAINY_MONTHS   # CHANGED: flag drives conditional behaviour below

    wl_mean, wl_std, sm_mean, sm_std, hum_mean, hum_std = MONTHLY_PROXY[month]

    # --- Sample daily baselines (one value per day) --------------------------
    # CHANGED: upper clip raised 0.50→0.52 to allow very wet rainy-season days
    daily_sm  = float(np.clip(rng.normal(sm_mean, max(sm_std * 0.5, 0.005)), 0.22, 0.52))
    daily_hum = float(np.clip(rng.normal(hum_mean, max(hum_std * 0.5, 0.1)),
                               HUMIDITY_PROXY_MIN, HUMIDITY_PROXY_MAX))

    base_soil = proxy_sm_to_hw_pct(daily_sm)
    base_hum  = proxy_hum_to_hw_rh(daily_hum)

    # Distance is modelled directly in hardware units.
    # Anchor to the Feb 27 2026 observed distance (0.24 m, dry season)
    # and scale seasonally using soil moisture as a water-level proxy.
    SM_DRY_REF = 0.301   # Feb proxy sm (dry season)
    SM_WET_REF = 0.464   # Aug proxy sm (monsoon peak)

    # CHANGED: clamp extended to 1.8 so very wet rainy days can push distance
    # further below DIST_WET (previously hard-capped at 1.5 → DIST_WET floor).
    sm_t = float(np.clip((daily_sm - SM_DRY_REF) / (SM_WET_REF - SM_DRY_REF), 0.0, 1.8))

    DIST_DRY = 0.24   # observed Feb 27 2026 (dry season)
    DIST_WET = 0.04   # CHANGED: 0.08→0.04 m — sensor nearly touching water at monsoon peak

    daily_dist_base = DIST_DRY + sm_t * (DIST_WET - DIST_DRY)
    base_dist = float(np.clip(daily_dist_base + rng.normal(0, 0.04),
                               DISTANCE_MIN_VALID, DISTANCE_MAX_VALID))

    # --- Minute-level walk --------------------------------------------------
    minutes_in_day = 1440
    steps = range(0, minutes_in_day, interval_minutes)

    dates    = []
    times    = []
    soils    = []
    humids   = []
    dists    = []

    # Initialize AR(1) state
    hum_prev  = base_hum
    soil_prev = base_soil
    dist_prev = base_dist

    date_str = target_date.strftime("%Y-%m-%d")

    # CHANGED: rainy months use a higher distance noise scale to simulate
    # turbulent water surface giving noisier ultrasonic returns.
    dist_noise_scale = 0.08 if rainy else 0.05

    for m in steps:
        h, mi = divmod(m, 60)
        time_str = f"{h:02d}:{mi:02d}"

        diurnal_wl  = diurnal_waterlevel_offset(m, rainy=rainy)  # CHANGED: pass rainy flag
        diurnal_hum = diurnal_humidity_offset(m)
        diurnal_s   = diurnal_soil_offset(m)

        # Soil — AR(1) with mean-reversion toward daily baseline + diurnal
        target_soil = base_soil + diurnal_s
        noise_s   = rng.normal(0, SOIL_NOISE_STD)
        soil_val  = 0.98 * soil_prev + 0.02 * target_soil + noise_s
        soil_val  = float(np.clip(soil_val, SOIL_HW_DRY - 1.0, SOIL_HW_WET + 1.0))

        # Humidity — strong AR(1) with mean-reversion toward diurnal target
        target_hum = base_hum + diurnal_hum
        noise_h   = rng.normal(0, HUMIDITY_NOISE_STD * 0.3)
        hum_val   = HUMIDITY_AR1 * hum_prev + (1 - HUMIDITY_AR1) * target_hum + noise_h
        hum_val   = float(np.clip(hum_val, HUMIDITY_HW_OBSERVED_MIN - 2,
                                            HUMIDITY_HW_OBSERVED_MAX + 2))

        # Distance (water level) — mean-revert toward tidal target + small noise
        # diurnal_wl is in proxy metres: higher WL = shorter distance
        target_dist = base_dist - diurnal_wl
        target_dist = float(np.clip(target_dist, DISTANCE_MIN_VALID, DISTANCE_MAX_VALID))
        noise_d   = rng.normal(0, DISTANCE_NOISE_STD * dist_noise_scale)  # CHANGED
        dist_val  = 0.95 * dist_prev + 0.05 * target_dist + noise_d
        dist_val  = float(np.clip(dist_val, DISTANCE_MIN_VALID, DISTANCE_MAX_VALID))

        # Randomly inject sensor dropout (Distance = 0.00)
        if rng.random() < DROPOUT_RATE:
            dist_val = 0.00

        dates.append(date_str)
        times.append(time_str)
        soils.append(round(soil_val, 2))
        humids.append(round(hum_val, 2))
        dists.append(round(dist_val, 2))

        # Update AR state
        soil_prev = soil_val
        hum_prev  = hum_val
        dist_prev = dist_val

    return pd.DataFrame({
        "date":     dates,
        "time":     times,
        "soil":     soils,
        "humidity": humids,
        "distance": dists,
    })


def generate_range(
    start: date,
    end: date,
    interval_minutes: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic data for every day in [start, end] inclusive.
    """
    rng = np.random.default_rng(seed)
    frames = []
    current = start
    total_days = (end - start).days + 1
    log.info("Generating %d day(s) × %d min interval → ~%d rows",
             total_days, interval_minutes,
             total_days * (1440 // interval_minutes))

    while current <= end:
        frames.append(generate_day(current, rng, interval_minutes))
        current += timedelta(days=1)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate synthetic Obando hardware sensor data (CSV2 format)."
    )
    parser.add_argument(
        "--start", type=_parse_date, default=date(2026, 1, 1),
        help="Start date YYYY-MM-DD (default: 2026-01-01)",
    )
    parser.add_argument(
        "--end", type=_parse_date, default=date(2026, 12, 31),
        help="End date YYYY-MM-DD (default: 2026-12-31)",
    )
    parser.add_argument(
        "--interval", type=int, default=1, metavar="MINUTES",
        help="Minutes between readings (default: 1 → 1440 rows/day)",
    )
    parser.add_argument(
        "--out", type=str,
        default=r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\sensor\dummy_sensor_data.csv",
        help="Output CSV path (default: project sensor directory)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print first 20 rows and stats, do not write file.",
    )
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be before or equal to --end")

    df = generate_range(args.start, args.end, args.interval, args.seed)

    log.info("Generated %d rows", len(df))

    if args.dry_run:
        print("\n=== First 20 rows ===")
        print(df.head(20).to_string(index=False))
        print("\n=== Stats ===")
        print(df[["soil", "humidity", "distance"]].describe().to_string())
        print(f"\nWould write to: {args.out}")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        log.info("Saved to %s  (%d rows, %.1f MB)",
                 out, len(df), out.stat().st_size / 1e6)
        print(f"\nDone. {len(df):,} rows → {out}")

        print("\n=== Stats ===")
        print(df[["soil", "humidity", "distance"]].describe().to_string())