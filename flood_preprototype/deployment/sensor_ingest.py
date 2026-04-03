"""
sensor_ingest.py
================
Hardware Sensor Ingestion + Calibration → ML Pipeline CSV
Flood Early Warning System — Obando, Bulacan

WHAT THIS FILE DOES
-------------------
Reads raw hardware sensor readings from Supabase
(`obando_environmental_data` table), converts each reading into the
same physical units the RF model was trained on, then writes them
incrementally into the local ML-pipeline CSV.

Two CSVs are produced on every ingest run:
    obando_sensor_data.csv      ← calibrated values (used by RF model)
    obando_sensor_data_raw.csv  ← raw hardware values (no conversion)

This file replaces both the old sensor_ingest.py AND sensor_normalize.py.
The calibration is simple enough (3 conversions) that a separate file
adds complexity with no benefit.

─────────────────────────────────────────────────────────────────────
WHY CALIBRATION IS NEEDED
─────────────────────────────────────────────────────────────────────
The RF model was trained on proxy satellite data with specific units:
    waterlevel    → metres above UHSLC chart datum (Manila South Harbor)
    soil_moisture → ERA5-Land volumetric water content  (m³/m³, 0.22–0.50)
    humidity      → MODIS Column Water Vapor            (cm,    0.15–6.87)

Your hardware sensors output completely different units:
    Distance (m)  → metres from sensor face DOWN to water surface
    Soil (%)      → capacitive sensor dielectric output  (~71–72% in dry season)
    Humidity (%)  → relative humidity                    (~75–89% RH)

Feeding raw hardware values to the model would produce wrong predictions
because every decision-tree split threshold was learned in proxy units.
The three conversions below fix this.

─────────────────────────────────────────────────────────────────────
CONVERSION 1 — WATER LEVEL  (two-step: actual level + linear stretch)
─────────────────────────────────────────────────────────────────────
Physical setup:
    The ultrasonic sensor sits at the top of the dike.
    Dike height = 13 ft 3 in = 4.039 m above the river bed.
    The sensor reads the air gap DOWN to the water surface.

WHY A SIMPLE DATUM SHIFT IS NOT ENOUGH:
    A single fixed offset aligned one anchor point but could not capture
    the full proxy range. The proxy waterlevel swings ~0.718 m (dry) →
    ~2.197 m (monsoon peak), a range of ~1.479 m. The hardware water level
    only swings ~3.799 m → ~3.999 m (a range of ~0.200 m) because the
    sensor is mounted high on the dike and the distance reading changes
    very little day-to-day. A linear stretch is required to map the narrow
    hardware range onto the wide proxy range.

Step 1 — Actual water level on the dike wall (metres above river bed):
    hw_wl_m = DIKE_HEIGHT_M - distance_m
    e.g. 4.039 - 0.240 = 3.799 m
    (the water surface is sitting 3.799 m up the 4.039 m dike wall)

Step 2 — Linear stretch onto proxy-equivalent range:
    t          = (hw_wl_m - HW_WL_DRY) / (HW_WL_WET - HW_WL_DRY)
    waterlevel = PROXY_WL_DRY + t × (PROXY_WL_WET - PROXY_WL_DRY)

Anchor points:
    Dry anchor (Feb 27 2026 observed):
        distance = 0.240 m  →  hw_wl = 3.799 m  →  proxy = 0.718 m  ✓
    Wet anchor (monsoon estimate, distance ≈ 0.040 m):
        distance = 0.040 m  →  hw_wl = 3.999 m  →  proxy = 2.197 m  ✓

⚠ HW_WL_WET and PROXY_WL_WET are estimates until a paired monsoon
  observation is collected (July–August 2026). That single field
  measurement is the most important calibration action remaining.

Bad readings to discard:
    Distance = 0.00 m  → sensor dropout (no return signal)
    Distance > 4.039 m → physically impossible (exceeds dike height)
    Distance < 0.05 m  → sensor face reflection / debris

─────────────────────────────────────────────────────────────────────
CONVERSION 2 — SOIL MOISTURE  (linear stretch, interim)
─────────────────────────────────────────────────────────────────────
The problem:
    ERA5 measures volumetric water content of the top 7 cm of soil
    across a 9 km grid cell — a physical quantity in m³/m³.
    Your capacitive sensor measures the dielectric permittivity of
    soil at one point and reports it as a percentage. These are
    related but NOT the same thing, and have no fixed formula.

    Proxy Feb (dry season):  ~0.242 m³/m³
    Proxy Aug (monsoon):     ~0.463 m³/m³   seasonal swing = 0.221

    HW Feb 27 (dry season):  ~71.8%
    HW monsoon reading:      UNKNOWN — collect in July/August 2026

Current approach — min-max linear stretch:
    We know one anchor point (dry season: HW=71.8 → proxy=0.242).
    We assume a wet-season ceiling of HW=85.0% until measured.
    We stretch the HW range linearly onto the proxy range.

    t             = (hw_pct - HW_DRY) / (HW_WET - HW_DRY)
    soil_moisture = PROXY_DRY + t × (PROXY_WET - PROXY_DRY)

Examples with current constants:
    HW = 71.8%  →  t=0.00  →  0.242 m³/m³  (dry season floor ✓)
    HW = 78.4%  →  t=0.50  →  0.353 m³/m³  (midpoint)
    HW = 85.0%  →  t=1.00  →  0.463 m³/m³  (monsoon ceiling ✓)

⚠ IMPORTANT: HW_WET = 85.0 is an estimate.
  Replace it with the actual sensor reading once you collect
  a monsoon-season observation (July–August 2026).

─────────────────────────────────────────────────────────────────────
CONVERSION 3 — HUMIDITY  (linear rescale, low-risk approximation)
─────────────────────────────────────────────────────────────────────
The problem:
    MODIS Column Water Vapor (proxy) measures the total water vapour
    in a vertical column of the atmosphere, in centimetres.
    Your sensor measures Relative Humidity — how saturated the air is
    at that point, in percent.
    These are physically different quantities with no exact formula.

Why it is acceptable to approximate:
    Humidity features do NOT appear in the model's top 8 by importance.
    Water level and soil moisture features dominate (top ~60% importance).
    A linear rescale that gets humidity into the right numerical range
    is sufficient — small errors here have minimal prediction impact.

The fix — stretch observed HW range onto observed proxy range:
    t        = (hw_rh - HW_MIN) / (HW_MAX - HW_MIN)
    humidity = PROXY_MIN + t × (PROXY_MAX - PROXY_MIN)

Examples:
    HW = 78.5% RH  →  t≈0.00  →  0.15 cm  (proxy min ✓)
    HW = 83.6% RH  →  t≈0.50  →  3.51 cm
    HW = 88.78% RH →  t≈1.00  →  6.87 cm  (proxy max ✓)

─────────────────────────────────────────────────────────────────────
RAW OUTPUT CSV
─────────────────────────────────────────────────────────────────────
In addition to the calibrated CSV, every ingest run also writes a
second CSV (obando_sensor_data_raw.csv) containing the original
hardware readings with no conversion applied. Columns:
    timestamp     — ISO datetime string (UTC), midnight = daily aggregate
    waterlevel    — daily mean of DIKE_HEIGHT_M - distance_m
                    (physical water height on dike wall, metres above river bed)
    soil_moisture — daily mean raw capacitive soil sensor reading (%)
    humidity      — daily mean raw relative humidity reading (%)

This raw file is useful for:
    - Auditing sensor hardware health over time
    - Collecting paired readings for future calibration updates
    - Debugging unexpected model predictions

─────────────────────────────────────────────────────────────────────
INTEGRATION WITH Start.py
─────────────────────────────────────────────────────────────────────
    from sensor_ingest import ingest_latest, log_prediction

    ingest_latest()                   # before RF_Predict.run_pipeline()
    log_prediction(tier, prob, ts)    # after predictions are made

─────────────────────────────────────────────────────────────────────
SUPABASE TABLE EXPECTED
─────────────────────────────────────────────────────────────────────
obando_environmental_data
    id          BIGINT PRIMARY KEY
    date        DATE NOT NULL          ← calendar date
    time        TIME NOT NULL          ← time of reading
    soil        NUMERIC NOT NULL       ← raw Soil (%) from capacitive sensor
    humidity    NUMERIC NOT NULL       ← raw Humidity (%) from RH sensor
    distance    NUMERIC NOT NULL       ← raw Distance (m) from ultrasonic

predictions
    id            BIGINT PRIMARY KEY
    timestamp     TEXT NOT NULL
    probability   REAL NOT NULL
    risk_tier     TEXT NOT NULL
    model         TEXT NOT NULL
    created_at    TIMESTAMPTZ DEFAULT now()

REQUIRED ENVIRONMENT VARIABLES (.env at project root)
    SUPABASE_URL         = https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY = <service_role key>

USAGE
-----
    python sensor_ingest.py                     # ingest all new rows
    python sensor_ingest.py --date 2025-04-01   # backfill a specific date
    python sensor_ingest.py --show 7            # print last 7 days from Supabase
    python sensor_ingest.py --dry-run           # preview without writing
    python sensor_ingest.py --check-calibration # print a calibration summary table
"""

import os
import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH  = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".env"))

SENSOR_CSV = Path(os.getenv(
    "SENSOR_CSV_PATH",
    os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "sensor", "obando_sensor_data.csv"))
))

SENSOR_CSV_RAW = Path(os.getenv(
    "SENSOR_CSV_RAW_PATH",
    os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "sensor", "obando_sensor_data_raw.csv"))
))

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
_loaded = load_dotenv(_ENV_PATH, override=False)

from supabase import create_client, Client

# ===========================================================================
# ███████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗
# ██╔════╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗
# ███████╗█████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝
# ╚════██║██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗
# ███████║███████╗██║ ╚████║███████║╚██████╔╝██║  ██║
# ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
#
# HARDWARE CALIBRATION CONSTANTS
# Update these as new field measurements are collected.
# ===========================================================================

# ── WATER LEVEL ─────────────────────────────────────────────────────────────
# Two-step conversion: distance → actual water level → proxy-equivalent.
#
# Step 1 constant — physical dike height (sensor mount above river bed):
#   DIKE_HEIGHT_M = 13 ft 3 in = 4.039 m
#   hw_wl_m = DIKE_HEIGHT_M - distance_m
#
# Step 2 constants — linear stretch anchors.
#   Maps hardware water level (m above river bed) onto the same scale as the
#   proxy waterlevel (m above UHSLC datum) that the RF model was trained on.
#
#   Dry anchor (Feb 27 2026 observed paired reading):
#     distance = 0.240 m  →  hw_wl = 4.039 - 0.240 = 3.799 m  →  proxy = 0.718 m
#
#   Wet anchor (monsoon peak estimate, distance ≈ 0.040 m):
#     distance = 0.040 m  →  hw_wl = 4.039 - 0.040 = 3.999 m  →  proxy = 2.197 m
#
# ⚠ HW_WL_WET = 3.999 and PROXY_WL_WET = 2.197 are ESTIMATES.
#   Replace both with actual paired readings from July–August 2026 monsoon.
#   That paired observation is the most important calibration action remaining.
#
DIKE_HEIGHT_M = 4.039    # metres — 13 ft 3 in, sensor mount above river bed

HW_WL_DRY    = 3.819    # hw water level (m above river bed), dry season   (Feb 27 2026 daily mean, ≈distance 0.220 m)
HW_WL_WET    = 3.999    # hw water level (m above river bed), monsoon peak (⚠ ESTIMATE)
PROXY_WL_DRY = 0.718    # proxy waterlevel (m above UHSLC datum), dry season   (Feb 27 2026 observed)
PROXY_WL_WET = 2.197    # proxy waterlevel (m above UHSLC datum), monsoon peak (⚠ ESTIMATE)

# Ultrasonic dropout / error thresholds
DISTANCE_MIN_VALID_M = 0.05    # below this = sensor face reflection or debris
DISTANCE_MAX_VALID_M = 4.039   # above this = physically impossible (exceeds dike)

# ── SOIL MOISTURE ────────────────────────────────────────────────────────────
# Linear stretch: maps hardware % range onto proxy m³/m³ range.
# Dry anchor  → observed Feb 27 2026
# Wet ceiling → ⚠ ESTIMATED at 85.0% — REPLACE with actual Jul/Aug 2026 reading
#
SOIL_HW_DRY    = 71.8    # hardware % reading in dry season  (Feb observed)
SOIL_HW_WET    = 85.0    # hardware % reading in monsoon     (⚠ ESTIMATE — update Jul/Aug 2026)
SOIL_PROXY_DRY = 0.242   # proxy m³/m³ in dry season        (ERA5 Feb 27 2026 actual)
SOIL_PROXY_WET = 0.463   # proxy m³/m³ in monsoon           (ERA5 Aug mean)

# ── HUMIDITY ─────────────────────────────────────────────────────────────────
# Linear rescale: maps observed RH% range onto observed proxy CWV range.
# Low model importance — linear approximation is acceptable.
#
HUMIDITY_HW_MIN    = 78.5    # raised to compress t → lowers output toward proxy values
HUMIDITY_HW_MAX    = 88.78   # max observed hardware %RH
HUMIDITY_PROXY_MIN = 0.15    # min proxy column water vapour  (cm)
HUMIDITY_PROXY_MAX = 6.87    # max proxy column water vapour  (cm)

# ===========================================================================
# CONFIG
# ===========================================================================

USE_HARDWARE = True

TABLE_ENV_DATA    = "obando_environmental_data"
TABLE_PREDICTIONS = "predictions"

# Columns to SELECT — matches actual Postgres column names (all lowercase)
_SELECT = "date, time, soil, humidity, distance"

# ===========================================================================
# END CONFIG
# ===========================================================================

log = logging.getLogger("sensor_ingest")

# ---------------------------------------------------------------------------
# Supabase client (lazy singleton)
# ---------------------------------------------------------------------------

_supabase_client: Client | None = None


def get_client() -> Client:
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise RuntimeError(
            f"Supabase credentials not found.\n"
            f"  .env path checked    : {_ENV_PATH}\n"
            f"  .env file found      : {_loaded}\n"
            f"  SUPABASE_URL         : {'SET' if url else 'MISSING'}\n"
            f"  SUPABASE_SERVICE_KEY : {'SET' if key else 'MISSING'}\n"
        )

    _supabase_client = create_client(url, key)
    log.debug("Supabase client initialised — %s", url)
    return _supabase_client


# ---------------------------------------------------------------------------
# ██████╗ █████╗ ██╗     ██╗██████╗ ██████╗  █████╗ ████████╗██╗ ██████╗ ███╗  ██╗
# ██╔════╝██╔══██╗██║     ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗ ██║
# ██║     ███████║██║     ██║██████╔╝██████╔╝███████║   ██║   ██║██║   ██║██╔██╗██║
# ██║     ██╔══██║██║     ██║██╔══██╗██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚████║
# ╚██████╗██║  ██║███████╗██║██████╔╝██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚███║
#  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═╝
# ---------------------------------------------------------------------------


def _calibrate_waterlevel(distance_m: float) -> float | None:
    """
    Two-step conversion: raw sensor distance → proxy-equivalent waterlevel.

    Step 1 — Actual water level on the dike wall (metres above river bed):
        hw_wl_m = DIKE_HEIGHT_M - distance_m
        e.g. 4.039 - 0.240 = 3.799 m
        (the water surface is sitting 3.799 m up the 4.039 m dike wall)

    Step 2 — Linear stretch onto proxy-equivalent range (m above UHSLC datum):
        t          = (hw_wl_m - HW_WL_DRY) / (HW_WL_WET - HW_WL_DRY)
        waterlevel = PROXY_WL_DRY + t × (PROXY_WL_WET - PROXY_WL_DRY)

    Verification — dry anchor (Feb 27 2026):
        hw_wl = 3.799 m  →  t = 0.0  →  proxy = 0.718 m  ✓
    Verification — wet anchor (monsoon estimate):
        hw_wl = 3.999 m  →  t = 1.0  →  proxy = 2.197 m  ✓

    WHY LINEAR STRETCH:
        Hardware water level has a narrow range (~0.200 m swing) because the
        sensor sits high on the dike. The proxy has a wide range (~1.479 m
        swing). A fixed offset cannot bridge this gap.

    Returns None if the reading is a known sensor error:
        distance <= 0.05 m  → sensor face reflection or debris
        distance >= 4.039 m → physically impossible (exceeds dike height)
    """
    if distance_m <= DISTANCE_MIN_VALID_M or distance_m >= DISTANCE_MAX_VALID_M:
        return None

    # Step 1 — actual water height on the dike wall
    hw_wl_m = DIKE_HEIGHT_M - distance_m

    # Step 2 — linear stretch onto proxy range
    t = (hw_wl_m - HW_WL_DRY) / (HW_WL_WET - HW_WL_DRY)
    t = max(0.0, min(1.0, t))   # clamp — never exceed proxy anchor range
    waterlevel = PROXY_WL_DRY + t * (PROXY_WL_WET - PROXY_WL_DRY)

    return round(waterlevel, 6)


def _calibrate_soil_moisture(hw_pct: float) -> float:
    """
    Convert capacitive sensor % reading to ERA5-equivalent volumetric
    water content (m³/m³).

    WHAT IT IS:
        ERA5 volumetric water content is a physical quantity — litres of
        water per litre of soil. Your sensor outputs a dielectric reading
        as a percentage. They correlate but have no fixed formula.

    HOW IT WORKS:
        We use a linear stretch between two anchor points:
            Dry anchor:  HW=71.8%  →  proxy=0.242 m³/m³  (Feb 27 2026 observed)
            Wet ceiling: HW=85.0%  →  proxy=0.463 m³/m³  (⚠ estimated)

        t             = (hw_pct - DRY) / (WET - DRY)   # 0.0 to 1.0
        soil_moisture = PROXY_DRY + t × (PROXY_WET - PROXY_DRY)

    ⚠ HW_WET=85.0 is an estimate. Update SOIL_HW_WET with the actual
      sensor reading during July–August 2026 monsoon peak.
    """
    t = (hw_pct - SOIL_HW_DRY) / (SOIL_HW_WET - SOIL_HW_DRY)
    t = max(0.0, min(1.0, t))
    return round(SOIL_PROXY_DRY + t * (SOIL_PROXY_WET - SOIL_PROXY_DRY), 6)


def _calibrate_humidity(hw_rh: float) -> float:
    """
    Convert relative humidity % to proxy-equivalent column water vapour (cm).

    WHAT IT IS:
        MODIS Column Water Vapour measures the total atmospheric water
        in a vertical column, in centimetres of precipitable water.
        Your sensor measures how saturated the local air is, in percent.
        These are different quantities — no exact formula exists.

    HOW IT WORKS:
        Because humidity features are low-importance in this model
        (not in the top 8), a linear rescale is acceptable:

        t        = (hw_rh - HW_MIN) / (HW_MAX - HW_MIN)
        humidity = PROXY_MIN + t × (PROXY_MAX - PROXY_MIN)

        Maps HW range [73.0, 88.78] %RH
        onto proxy range [0.15, 6.87] cm CWV.
    """
    t = (hw_rh - HUMIDITY_HW_MIN) / (HUMIDITY_HW_MAX - HUMIDITY_HW_MIN)
    t = max(0.0, min(1.0, t))
    return round(HUMIDITY_PROXY_MIN + t * (HUMIDITY_PROXY_MAX - HUMIDITY_PROXY_MIN), 6)


def calibrate_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all three calibrations to a raw hardware DataFrame, then
    aggregate to one row per day by taking the daily mean.

    Expects columns: timestamp, waterlevel (raw distance m),
                     soil_moisture (raw %), humidity (raw %RH)
    Returns columns: timestamp, waterlevel (m above datum),
                     soil_moisture (m³/m³), humidity (cm CWV)

    Invalid distance readings are dropped before averaging.
    The daily mean matches how proxy training data is aggregated
    (ERA5, MODIS, UHSLC are all daily mean values).
    """
    out = df.copy()

    # Water level — two-step calibration, drop invalid rows
    out["waterlevel"] = out["waterlevel"].apply(_calibrate_waterlevel)
    invalid = out["waterlevel"].isna().sum()
    out = out.dropna(subset=["waterlevel"])

    # Soil moisture
    out["soil_moisture"] = out["soil_moisture"].apply(_calibrate_soil_moisture)

    # Humidity
    out["humidity"] = out["humidity"].apply(_calibrate_humidity)

    # Extract date from timestamp for grouping
    out["date"] = out["timestamp"].str[:10]

    # Aggregate to daily mean — matches proxy training data granularity
    daily = (
        out.groupby("date", sort=True)[["waterlevel", "soil_moisture", "humidity"]]
        .mean()
        .round(6)
        .reset_index()
    )

    # Reconstruct timestamp as midnight UTC to represent the daily aggregate
    daily["timestamp"] = daily["date"] + "T00:00:00"
    daily = daily[["timestamp", "waterlevel", "soil_moisture", "humidity"]]

    log.info(
        "Aggregated %d calibrated reading(s) → %d daily mean row(s) (dropped %d invalid).",
        len(out), len(daily), invalid,
    )

    return daily


def print_calibration_summary() -> None:
    """
    Print a human-readable table showing what the calibration does
    at key reference points. Useful for field verification.
    """
    print()
    print("=" * 78)
    print("  SENSOR CALIBRATION SUMMARY — Obando Flood Early Warning")
    print("=" * 78)

    print()
    print("  WATER LEVEL  (two-step: actual level + linear stretch)")
    print(f"  Dike height  : {DIKE_HEIGHT_M} m  (13 ft 3 in, sensor mount above river bed)")
    print(f"  Step 1       : hw_wl_m    = {DIKE_HEIGHT_M} − distance_m")
    print(f"  Step 2       : waterlevel = PROXY_WL_DRY + t × (PROXY_WL_WET − PROXY_WL_DRY)")
    print(f"  Dry anchor   : hw_wl={HW_WL_DRY} m  →  proxy={PROXY_WL_DRY} m  (Feb 27 2026 observed)")
    print(f"  Wet anchor   : hw_wl={HW_WL_WET} m  →  proxy={PROXY_WL_WET} m  (⚠ ESTIMATE — update Jul/Aug 2026)")
    print()
    print(f"  {'Distance (m)':<16} {'hw_wl (m)':<14} {'t':<12} {'Proxy WL (m)':<16} Notes")
    print(f"  {'-'*78}")
    for d, note in [
        (0.04,  "⚠ INVALID — too close"),
        (0.24,  "Feb 27 2026 dry anchor"),
        (0.50,  "example — early wet season"),
        (1.00,  "example — rising water"),
        (2.00,  "example — flood level"),
        (3.50,  "example — near-full dike"),
        (4.039, "⚠ INVALID — exceeds dike"),
    ]:
        if DISTANCE_MIN_VALID_M < d < DISTANCE_MAX_VALID_M:
            hw_wl = round(DIKE_HEIGHT_M - d, 4)
            t_raw = (hw_wl - HW_WL_DRY) / (HW_WL_WET - HW_WL_DRY)
            t_clamped = max(0.0, min(1.0, t_raw))
            proxy = round(PROXY_WL_DRY + t_clamped * (PROXY_WL_WET - PROXY_WL_DRY), 4)
            t_str = f"{t_raw:.4f}" + (" *clamped" if t_raw != t_clamped else "")
            print(f"  {d:<16.3f} {hw_wl:<14.4f} {t_str:<12} {proxy:<16.4f} {note}")
        else:
            print(f"  {d:<16.3f} {'—':<14} {'—':<12} {'DISCARDED':<16} {note}")

    print()
    print("  SOIL MOISTURE  (linear stretch, interim)")
    print(f"  Dry anchor:   HW={SOIL_HW_DRY}%  →  {SOIL_PROXY_DRY} m³/m³  (Feb 27 2026 observed)")
    print(f"  Wet ceiling:  HW={SOIL_HW_WET}%  →  {SOIL_PROXY_WET} m³/m³  (⚠ ESTIMATED — update Jul/Aug 2026)")
    print()
    print(f"  {'HW Soil (%)':<16} {'→  Soil moisture (m³/m³)':<28} Notes")
    print(f"  {'-'*65}")
    for s, note in [
        (71.8, "dry season floor (Feb 27 2026 observed)"),
        (75.0, "example"),
        (78.4, "midpoint"),
        (82.0, "example"),
        (85.0, "wet ceiling (⚠ estimated)"),
    ]:
        sm = _calibrate_soil_moisture(s)
        print(f"  {s:<16.1f} →  {sm:<28.4f} {note}")

    print()
    print("  HUMIDITY  (linear rescale — low model importance)")
    print(f"  HW range:    [{HUMIDITY_HW_MIN}, {HUMIDITY_HW_MAX}] %RH")
    print(f"  Proxy range: [{HUMIDITY_PROXY_MIN}, {HUMIDITY_PROXY_MAX}] cm column water vapour")
    print()
    print(f"  {'HW Humidity (%RH)':<20} {'→  Humidity (cm CWV)':<26} Notes")
    print(f"  {'-'*60}")
    for h, note in [
        (73.0,  "adjusted min"),
        (81.3,  "observed mean"),
        (85.0,  "example"),
        (88.78, "observed max"),
    ]:
        hm = _calibrate_humidity(h)
        print(f"  {h:<20.2f} →  {hm:<26.4f} {note}")

    print()
    print("  ⚠ ACTIONS REQUIRED")
    print("  1. Collect a paired distance + proxy waterlevel observation during")
    print("     July–August 2026 monsoon peak. Update HW_WL_WET and PROXY_WL_WET.")
    print("  2. Collect a soil sensor reading at monsoon peak. Update SOIL_HW_WET.")
    print("     These two monsoon observations are the most important remaining")
    print("     calibration actions for this system.")
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Supabase fetch helpers
# ---------------------------------------------------------------------------

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "waterlevel", "soil_moisture", "humidity"])


def _rows_to_df(data: list[dict]) -> pd.DataFrame:
    """
    Convert raw Supabase rows (split date + time columns) into the internal
    DataFrame format expected by calibrate_df():
        timestamp     — ISO string  (date + time combined, UTC)
        waterlevel    — raw Distance (m)
        soil_moisture — raw Soil (%)
        humidity      — raw Humidity (%)

    Postgres returns:
        date     → "YYYY-MM-DD"
        time     → "HH:MM:SS"
        soil     → float  (raw %)
        humidity → float  (raw %RH)
        distance → float  (raw metres)
    """
    if not data:
        return _empty_df()

    df = pd.DataFrame(data)

    # Combine date + time into a single UTC timestamp string
    df["timestamp"] = (
        pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            utc=True,
        )
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
    )

    # Rename to internal names expected by calibrate_df()
    df = df.rename(columns={
        "distance": "waterlevel",    # raw distance → converted to water level
        "soil":     "soil_moisture", # raw soil %   → converted to m³/m³
        # "humidity" column name is already correct
    })

    return df[["timestamp", "waterlevel", "soil_moisture", "humidity"]]


def fetch_rows_since(after_timestamp: str | None) -> pd.DataFrame:
    """
    Fetch all rows strictly newer than after_timestamp. None = full table.

    Paginates automatically to bypass the default 1000-row Supabase limit.
    Filters on the date column (coarse) then drops any rows whose
    reconstructed timestamp is not strictly newer (fine).
    """
    PAGE_SIZE = 1000
    all_rows: list[dict] = []
    offset = 0

    cutoff_date = str(after_timestamp)[:10] if after_timestamp else None

    while True:
        try:
            query = (
                get_client()
                .table(TABLE_ENV_DATA)
                .select(_SELECT)
                .order("date", desc=False)
                .order("time", desc=False)
                .range(offset, offset + PAGE_SIZE - 1)
            )
            if cutoff_date:
                query = query.gte("date", cutoff_date)

            response = query.execute()
        except Exception as exc:
            log.error("Supabase fetch failed at offset %d: %s", offset, exc)
            break

        batch = response.data or []
        all_rows.extend(batch)
        log.debug("Fetched page at offset %d: %d row(s)", offset, len(batch))

        if len(batch) < PAGE_SIZE:
            break  # last page

        offset += PAGE_SIZE

    log.info("Total raw rows fetched: %d", len(all_rows))

    df = _rows_to_df(all_rows)

    # Fine filter — drop rows not strictly newer than the last CSV timestamp
    if after_timestamp and not df.empty:
        df = df[df["timestamp"] > str(after_timestamp)]

    return df


def fetch_rows_for_date(target_date: str) -> pd.DataFrame:
    """Fetch all rows whose date column equals target_date (YYYY-MM-DD). Paginated."""
    PAGE_SIZE = 1000
    all_rows: list[dict] = []
    offset = 0

    while True:
        try:
            response = (
                get_client()
                .table(TABLE_ENV_DATA)
                .select(_SELECT)
                .eq("date", target_date)
                .order("time", desc=False)
                .range(offset, offset + PAGE_SIZE - 1)
                .execute()
            )
        except Exception as exc:
            log.error("Supabase fetch failed at offset %d: %s", offset, exc)
            break

        batch = response.data or []
        all_rows.extend(batch)

        if len(batch) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return _rows_to_df(all_rows)


def fetch_rows_for_range(days: int) -> pd.DataFrame:
    """Fetch all rows from the last N calendar days. Paginated."""
    PAGE_SIZE = 1000
    all_rows: list[dict] = []
    offset = 0
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    while True:
        try:
            response = (
                get_client()
                .table(TABLE_ENV_DATA)
                .select(_SELECT)
                .gte("date", cutoff)
                .order("date", desc=False)
                .order("time", desc=False)
                .range(offset, offset + PAGE_SIZE - 1)
                .execute()
            )
        except Exception as exc:
            log.error("Supabase fetch failed at offset %d: %s", offset, exc)
            break

        batch = response.data or []
        all_rows.extend(batch)

        if len(batch) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return _rows_to_df(all_rows)


def insert_prediction_row(record: dict) -> bool:
    """Insert one prediction record into the predictions table."""
    try:
        get_client().table(TABLE_PREDICTIONS).insert(record).execute()
        return True
    except Exception as exc:
        log.error("Failed to insert prediction: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def get_latest_csv_timestamp() -> str | None:
    if not SENSOR_CSV.exists():
        return None
    try:
        df = pd.read_csv(SENSOR_CSV)
        if df.empty or "timestamp" not in df.columns:
            return None
        return df["timestamp"].astype(str).max()
    except Exception:
        return None


def date_already_in_csv(target_date: str) -> bool:
    if not SENSOR_CSV.exists():
        return False
    try:
        df = pd.read_csv(SENSOR_CSV)
        return df["timestamp"].astype(str).str.startswith(target_date).any()
    except Exception:
        return False


def append_rows_to_csv(rows: pd.DataFrame, dry_run: bool = False) -> None:
    """Write calibrated rows to the ML pipeline CSV, creating the file if needed."""
    if rows.empty:
        log.warning("Nothing to write — DataFrame is empty.")
        return

    rows = rows[["timestamp", "waterlevel", "soil_moisture", "humidity"]].copy()
    SENSOR_CSV.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.info("[DRY RUN] Would append %d calibrated row(s):", len(rows))
        print(rows.to_string(index=False))
        return

    if not SENSOR_CSV.exists():
        rows.to_csv(SENSOR_CSV, index=False)
        log.info("Created %s with %d row(s)", SENSOR_CSV, len(rows))
    else:
        rows.to_csv(SENSOR_CSV, mode="a", header=False, index=False)
        log.info("Appended %d new row(s) to %s", len(rows), SENSOR_CSV)


def append_raw_rows_to_csv(rows: pd.DataFrame, dry_run: bool = False) -> None:
    """
    Write raw (pre-calibration) hardware readings to a separate CSV.

    Columns written:
        timestamp     — ISO datetime string (UTC), midnight = daily aggregate
        waterlevel    — daily mean of DIKE_HEIGHT_M - distance_m
                        (physical water height on dike wall, metres above river bed)
        soil_moisture — daily mean raw capacitive soil sensor reading (%)
        humidity      — daily mean raw relative humidity reading (%)

    Aggregated to daily means (same granularity as the calibrated CSV)
    for consistent side-by-side comparison. Each row represents the
    mean of all readings taken on that calendar day.
    """
    if rows.empty:
        log.warning("Nothing to write — raw DataFrame is empty.")
        return

    raw = rows[["timestamp", "waterlevel", "soil_moisture", "humidity"]].copy()

    # Step 1 dike calculation — DIKE_HEIGHT_M - distance_m = physical water height
    # This replaces the raw distance column entirely in the raw output
    raw["waterlevel"] = (DIKE_HEIGHT_M - raw["waterlevel"]).round(6)

    # Aggregate to daily mean — matches calibrated CSV granularity
    raw["date"] = raw["timestamp"].str[:10]
    daily_raw = (
        raw.groupby("date", sort=True)[["waterlevel", "soil_moisture", "humidity"]]
        .mean()
        .round(6)
        .reset_index()
    )
    daily_raw["timestamp"] = daily_raw["date"] + "T00:00:00"
    raw = daily_raw[["timestamp", "waterlevel", "soil_moisture", "humidity"]]

    log.info("Aggregated raw readings -> %d daily mean row(s).", len(raw))

    SENSOR_CSV_RAW.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.info("[DRY RUN] Would append %d raw row(s):", len(raw))
        print(raw.to_string(index=False))
        return

    if not SENSOR_CSV_RAW.exists():
        raw.to_csv(SENSOR_CSV_RAW, index=False)
        log.info("Created %s with %d raw row(s)", SENSOR_CSV_RAW, len(raw))
    else:
        raw.to_csv(SENSOR_CSV_RAW, mode="a", header=False, index=False)
        log.info("Appended %d raw row(s) to %s", len(raw), SENSOR_CSV_RAW)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_latest(dry_run: bool = False) -> bool:
    """
    Default mode — incremental sync with calibration applied.

    Fetches only rows newer than the latest CSV timestamp from Supabase,
    applies all three sensor calibrations, and appends to both CSVs:
        obando_sensor_data.csv      ← calibrated (used by RF model)
        obando_sensor_data_raw.csv  ← raw hardware values (no conversion)

    On first run (no CSV) pulls and calibrates the full table.
    Returns True if at least one new row was written.
    """
    if not USE_HARDWARE:
        log.info("USE_HARDWARE=False — ingestion skipped.")
        return False

    latest_ts = get_latest_csv_timestamp()

    if latest_ts:
        log.info("CSV latest timestamp: %s — fetching newer rows only.", latest_ts)
    else:
        log.info("CSV missing or empty — bootstrapping full history from Supabase.")

    raw = fetch_rows_since(after_timestamp=latest_ts)

    if raw.empty:
        log.info("No new rows since %s — CSV is already up to date.", latest_ts)
        return False

    log.info("Fetched %d raw row(s) — saving raw output and applying calibration.", len(raw))

    # Save raw hardware values before any conversion
    append_raw_rows_to_csv(raw, dry_run=dry_run)

    # Apply calibration and save to ML pipeline CSV
    calibrated = calibrate_df(raw)

    if calibrated.empty:
        log.warning("All fetched rows were discarded as invalid sensor readings.")
        return False

    append_rows_to_csv(calibrated, dry_run=dry_run)
    return True


def ingest_date(target_date: str, dry_run: bool = False) -> bool:
    """
    Backfill a specific date with calibration applied.
    Writes both the calibrated CSV and the raw CSV.
    Skips if that date already exists in the calibrated CSV.
    """
    if not USE_HARDWARE:
        log.info("USE_HARDWARE=False — ingestion skipped.")
        return False

    if date_already_in_csv(target_date):
        log.info("%s already in CSV — skipping.", target_date)
        return False

    raw = fetch_rows_for_date(target_date)

    if raw.empty:
        log.warning("No Supabase rows found for %s.", target_date)
        return False

    # Save raw hardware values before any conversion
    append_raw_rows_to_csv(raw, dry_run=dry_run)

    # Apply calibration and save to ML pipeline CSV
    calibrated = calibrate_df(raw)
    log.info("Ingesting %d calibrated row(s) for %s.", len(calibrated), target_date)
    append_rows_to_csv(calibrated, dry_run=dry_run)
    return True


def log_prediction(risk_tier: str, probability: float, timestamp: str) -> None:
    """
    Called by Start.py after each prediction run.
    Stores the result in Supabase for remote monitoring.
    """
    record = {
        "timestamp":   timestamp,
        "probability": round(probability, 4),
        "risk_tier":   risk_tier,
        "model":       "RF",
    }
    ok = insert_prediction_row(record)
    if ok:
        log.info("Prediction logged: %s  tier=%s  prob=%.1f%%",
                 timestamp, risk_tier, probability * 100)
    else:
        log.warning("Prediction NOT logged (insert failed): %s  tier=%s",
                    timestamp, risk_tier)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
    )

    log.info(".env path  : %s", _ENV_PATH)
    log.info(".env found : %s", _loaded)

    parser = argparse.ArgumentParser(
        description="Ingest + calibrate hardware sensor data into the ML pipeline CSV."
    )
    parser.add_argument("--date",    type=str, default=None,
                        help="Backfill a specific date (YYYY-MM-DD).")
    parser.add_argument("--show",    type=int, metavar="DAYS", default=None,
                        help="Print last N days of raw rows from Supabase (no calibration applied).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview calibrated and raw output without writing to disk.")
    parser.add_argument("--check-calibration", action="store_true",
                        help="Print calibration summary table and exit.")
    args = parser.parse_args()

    if args.check_calibration:
        print_calibration_summary()

    elif args.show:
        df = fetch_rows_for_range(days=args.show)
        print(f"\nLast {args.show} day(s) — RAW from Supabase (pre-calibration):\n")
        print(df.to_string(index=False) if not df.empty else "No data found.")

    elif args.date:
        ingest_date(args.date, dry_run=args.dry_run)

    else:
        ingest_latest(dry_run=args.dry_run)