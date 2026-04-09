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
    obando_sensor_data.csv      ← calibrated values (used by XGB model)
    obando_sensor_data_raw.csv  ← raw hardware values (no conversion)

This file replaces both the old sensor_ingest.py AND sensor_normalize.py.
The calibration is simple enough (3 conversions) that a separate file
adds complexity with no benefit.

─────────────────────────────────────────────────────────────────────
WHY CALIBRATION IS NEEDED
─────────────────────────────────────────────────────────────────────
The XGB model was trained on proxy satellite data with specific units:
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
DUPLICATE PREVENTION
─────────────────────────────────────────────────────────────────────
Duplicates were previously caused by two separate bugs:

BUG 1 — fetch_rows_since used date >= cutoff_date (inclusive) at the
  Supabase query level. Since the calibrated CSV stores midnight
  timestamps (e.g. 2026-04-03T00:00:00), the cutoff date was always
  re-fetched in the next run.
  FIX: query uses date > cutoff_date (strictly greater), so the day
  that is already in the CSV is never re-fetched from Supabase.

BUG 2 — append_rows_to_csv and append_raw_rows_to_csv blindly appended
  new rows to the CSV with no existence check.
  FIX: both functions now load the existing CSV first, drop any rows
  whose timestamp already exists, then write only genuinely new rows.
  This acts as a safety net even if the Supabase filter ever returns
  an overlapping row.

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

    ingest_latest()                   # before XGB_Predict.run_pipeline()
    log_prediction(tier, prob, ts)    # after predictions are made

─────────────────────────────────────────────────────────────────────
SUPABASE TABLE EXPECTED
─────────────────────────────────────────────────────────────────────
obando_environmental_data
    id              BIGINT PRIMARY KEY
    "Date"          DATE NOT NULL          ← calendar date
    "Time"          TIME NOT NULL          ← time of reading
    "Soil Moisture" REAL NOT NULL          ← raw Soil (%) from capacitive sensor
    "Humidity"      REAL NOT NULL          ← raw Humidity (%) from RH sensor
    "Final Distance" REAL NULL             ← raw Distance (m) from ultrasonic

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
DIKE_HEIGHT_M = 4.039    # metres — 13 ft 3 in, sensor mount above river bed

HW_WL_DRY    = 3.819    # hw water level (m above river bed), dry season   (Feb 27 2026 daily mean)
HW_WL_WET    = 3.999    # hw water level (m above river bed), monsoon peak (⚠ ESTIMATE)
PROXY_WL_DRY = 0.718    # proxy waterlevel (m above UHSLC datum), dry season
PROXY_WL_WET = 2.197    # proxy waterlevel (m above UHSLC datum), monsoon peak (⚠ ESTIMATE)

DISTANCE_MIN_VALID_M = 0.05    # below this = sensor face reflection or debris
DISTANCE_MAX_VALID_M = 4.039   # above this = physically impossible (exceeds dike)

# ── SOIL MOISTURE ────────────────────────────────────────────────────────────
SOIL_HW_DRY    = 71.8    # hardware % reading in dry season  (Feb observed)
SOIL_HW_WET    = 85.0    # hardware % reading in monsoon     (⚠ ESTIMATE — update Jul/Aug 2026)
SOIL_PROXY_DRY = 0.242   # proxy m³/m³ in dry season
SOIL_PROXY_WET = 0.463   # proxy m³/m³ in monsoon

# ── HUMIDITY ─────────────────────────────────────────────────────────────────
HUMIDITY_HW_MIN    = 78.5    # min observed hardware %RH
HUMIDITY_HW_MAX    = 88.78   # max observed hardware %RH
HUMIDITY_PROXY_MIN = 0.15    # min proxy column water vapour  (cm)
HUMIDITY_PROXY_MAX = 6.87    # max proxy column water vapour  (cm)

# ===========================================================================
# CONFIG
# ===========================================================================

USE_HARDWARE = True

TABLE_ENV_DATA    = "obando_environmental_data"
TABLE_PREDICTIONS = "predictions"

# ── Column names matching the actual Supabase table schema ──────────────────
COL_DATE     = "Date"
COL_TIME     = "Time"
COL_SOIL     = "Soil Moisture"
COL_HUMIDITY = "Humidity"
COL_DISTANCE = "Final Distance"

_SELECT = f'"Date", "Time", "Soil Moisture", "Humidity", "Final Distance"'

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
# Calibration functions
# ---------------------------------------------------------------------------

def _calibrate_waterlevel(distance_m: float) -> float | None:
    """
    Two-step conversion: raw sensor distance → proxy-equivalent waterlevel.

    Step 1 — Actual water level on the dike wall (metres above river bed):
        hw_wl_m = DIKE_HEIGHT_M - distance_m

    Step 2 — Linear stretch onto proxy-equivalent range (m above UHSLC datum):
        t          = (hw_wl_m - HW_WL_DRY) / (HW_WL_WET - HW_WL_DRY)
        waterlevel = PROXY_WL_DRY + t × (PROXY_WL_WET - PROXY_WL_DRY)

    Returns None for known sensor errors:
        distance <= DISTANCE_MIN_VALID_M  → sensor face reflection or debris
        distance >= DISTANCE_MAX_VALID_M  → physically impossible
    """
    if distance_m <= DISTANCE_MIN_VALID_M or distance_m >= DISTANCE_MAX_VALID_M:
        return None

    hw_wl_m = DIKE_HEIGHT_M - distance_m
    t = (hw_wl_m - HW_WL_DRY) / (HW_WL_WET - HW_WL_DRY)
    t = max(0.0, min(1.0, t))
    waterlevel = PROXY_WL_DRY + t * (PROXY_WL_WET - PROXY_WL_DRY)

    return round(waterlevel, 6)


def _calibrate_soil_moisture(hw_pct: float) -> float:
    """Linear stretch: hardware % → ERA5-equivalent volumetric water content (m³/m³)."""
    t = (hw_pct - SOIL_HW_DRY) / (SOIL_HW_WET - SOIL_HW_DRY)
    t = max(0.0, min(1.0, t))
    return round(SOIL_PROXY_DRY + t * (SOIL_PROXY_WET - SOIL_PROXY_DRY), 6)


def _calibrate_humidity(hw_rh: float) -> float:
    """Linear rescale: relative humidity % → proxy-equivalent column water vapour (cm)."""
    t = (hw_rh - HUMIDITY_HW_MIN) / (HUMIDITY_HW_MAX - HUMIDITY_HW_MIN)
    t = max(0.0, min(1.0, t))
    return round(HUMIDITY_PROXY_MIN + t * (HUMIDITY_PROXY_MAX - HUMIDITY_PROXY_MIN), 6)


def calibrate_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all three calibrations to a raw hardware DataFrame, then
    aggregate to one row per day (daily mean).

    Input columns : timestamp, waterlevel (raw distance m),
                    soil_moisture (raw %), humidity (raw %RH)
    Output columns: timestamp, waterlevel (m above datum),
                    soil_moisture (m³/m³), humidity (cm CWV)

    Invalid distance readings are dropped before averaging.
    """
    out = df.copy()

    out["waterlevel"] = out["waterlevel"].apply(_calibrate_waterlevel)
    invalid = out["waterlevel"].isna().sum()
    out = out.dropna(subset=["waterlevel"])

    out["soil_moisture"] = out["soil_moisture"].apply(_calibrate_soil_moisture)
    out["humidity"]      = out["humidity"].apply(_calibrate_humidity)

    out["date"] = out["timestamp"].str[:10]

    daily = (
        out.groupby("date", sort=True)[["waterlevel", "soil_moisture", "humidity"]]
        .mean()
        .round(6)
        .reset_index()
    )

    daily["timestamp"] = daily["date"] + "T00:00:00"
    daily = daily[["timestamp", "waterlevel", "soil_moisture", "humidity"]]

    log.info(
        "Aggregated %d calibrated reading(s) → %d daily mean row(s) (dropped %d invalid).",
        len(out), len(daily), invalid,
    )

    return daily


def print_calibration_summary() -> None:
    """Print a human-readable calibration verification table."""
    print()
    print("=" * 78)
    print("  SENSOR CALIBRATION SUMMARY — Obando Flood Early Warning")
    print("=" * 78)

    print()
    print("  WATER LEVEL  (two-step: actual level + linear stretch)")
    print(f"  Dike height  : {DIKE_HEIGHT_M} m  (13 ft 3 in, sensor mount above river bed)")
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
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Supabase fetch helpers
# ---------------------------------------------------------------------------

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "waterlevel", "soil_moisture", "humidity"])


def _rows_to_df(data: list[dict]) -> pd.DataFrame:
    """
    Convert raw Supabase rows into the internal DataFrame format
    expected by calibrate_df().

    Maps actual table columns → internal pipeline names:
        "Date"          → date component of timestamp
        "Time"          → time component of timestamp
        "Soil Moisture" → soil_moisture
        "Humidity"      → humidity
        "Final Distance"→ waterlevel (raw distance, calibrated later)
    """
    if not data:
        return _empty_df()

    df = pd.DataFrame(data)

    df["timestamp"] = (
        pd.to_datetime(
            df[COL_DATE].astype(str) + " " + df[COL_TIME].astype(str),
            utc=True,
        )
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
    )

    df = df.rename(columns={
        COL_DISTANCE: "waterlevel",    # "Final Distance" → raw distance m (calibrated later)
        COL_SOIL:     "soil_moisture", # "Soil Moisture"  → raw %
        COL_HUMIDITY: "humidity",      # "Humidity"       → raw %RH
    })

    return df[["timestamp", "waterlevel", "soil_moisture", "humidity"]]


def fetch_rows_since(after_timestamp: str | None) -> pd.DataFrame:
    """
    Fetch all rows strictly newer than after_timestamp. None = full table.

    Uses COL_DATE ("Date") to match the actual Supabase column name.
    Uses gt (>) instead of gte (>=) to exclude the cutoff day entirely,
    preventing duplicate rows on incremental runs.

    Paginates automatically to bypass the default 1000-row Supabase limit.
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
                .order(COL_DATE, desc=False)
                .order(COL_TIME, desc=False)
                .range(offset, offset + PAGE_SIZE - 1)
            )
            if cutoff_date:
                query = query.gt(COL_DATE, cutoff_date)

            response = query.execute()
        except Exception as exc:
            log.error("Supabase fetch failed at offset %d: %s", offset, exc)
            break

        batch = response.data or []
        all_rows.extend(batch)
        log.debug("Fetched page at offset %d: %d row(s)", offset, len(batch))

        if len(batch) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    log.info("Total raw rows fetched from Supabase: %d", len(all_rows))

    return _rows_to_df(all_rows)


def fetch_rows_for_date(target_date: str) -> pd.DataFrame:
    """Fetch all rows whose Date column equals target_date (YYYY-MM-DD). Paginated."""
    PAGE_SIZE = 1000
    all_rows: list[dict] = []
    offset = 0

    while True:
        try:
            response = (
                get_client()
                .table(TABLE_ENV_DATA)
                .select(_SELECT)
                .eq(COL_DATE, target_date)
                .order(COL_TIME, desc=False)
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
                .gte(COL_DATE, cutoff)
                .order(COL_DATE, desc=False)
                .order(COL_TIME, desc=False)
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
    """Return the latest timestamp string already in SENSOR_CSV, or None."""
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


def _existing_timestamps(csv_path: Path) -> set[str]:
    """Return the set of timestamp strings already present in a CSV file."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["timestamp"])
        return set(df["timestamp"].astype(str).tolist())
    except Exception:
        return set()


def append_rows_to_csv(rows: pd.DataFrame, dry_run: bool = False) -> int:
    """
    Write calibrated rows to the ML pipeline CSV.
    Drops any incoming row whose timestamp is already present (dedup safety net).
    Returns the number of rows actually written.
    """
    if rows.empty:
        log.warning("Nothing to write — DataFrame is empty.")
        return 0

    rows = rows[["timestamp", "waterlevel", "soil_moisture", "humidity"]].copy()

    existing = _existing_timestamps(SENSOR_CSV)
    before   = len(rows)
    rows     = rows[~rows["timestamp"].astype(str).isin(existing)]
    skipped  = before - len(rows)
    if skipped:
        log.warning("Skipped %d row(s) already present in %s.", skipped, SENSOR_CSV)

    if rows.empty:
        log.info("No new calibrated rows to write after dedup check.")
        return 0

    SENSOR_CSV.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.info("[DRY RUN] Would append %d calibrated row(s):", len(rows))
        print(rows.to_string(index=False))
        return len(rows)

    if not SENSOR_CSV.exists():
        rows.to_csv(SENSOR_CSV, index=False)
        log.info("Created %s with %d row(s)", SENSOR_CSV, len(rows))
    else:
        rows.to_csv(SENSOR_CSV, mode="a", header=False, index=False)
        log.info("Appended %d new row(s) to %s", len(rows), SENSOR_CSV)

    return len(rows)


def append_raw_rows_to_csv(rows: pd.DataFrame, dry_run: bool = False) -> int:
    """
    Write raw (pre-calibration) hardware readings to the raw CSV.
    Drops rows whose timestamp is already in obando_sensor_data_raw.csv.
    Returns the number of rows actually written.
    """
    if rows.empty:
        log.warning("Nothing to write — raw DataFrame is empty.")
        return 0

    raw = rows[["timestamp", "waterlevel", "soil_moisture", "humidity"]].copy()

    # Step 1 dike calculation — physical water height on dike wall
    raw["waterlevel"] = (DIKE_HEIGHT_M - raw["waterlevel"]).round(6)

    # Aggregate to daily mean
    raw["date"] = raw["timestamp"].str[:10]
    daily_raw = (
        raw.groupby("date", sort=True)[["waterlevel", "soil_moisture", "humidity"]]
        .mean()
        .round(6)
        .reset_index()
    )
    daily_raw["timestamp"] = daily_raw["date"] + "T00:00:00"
    raw = daily_raw[["timestamp", "waterlevel", "soil_moisture", "humidity"]]

    log.info("Aggregated raw readings → %d daily mean row(s).", len(raw))

    existing = _existing_timestamps(SENSOR_CSV_RAW)
    before   = len(raw)
    raw      = raw[~raw["timestamp"].astype(str).isin(existing)]
    skipped  = before - len(raw)
    if skipped:
        log.warning("Skipped %d raw row(s) already present in %s.", skipped, SENSOR_CSV_RAW)

    if raw.empty:
        log.info("No new raw rows to write after dedup check.")
        return 0

    SENSOR_CSV_RAW.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.info("[DRY RUN] Would append %d raw row(s):", len(raw))
        print(raw.to_string(index=False))
        return len(raw)

    if not SENSOR_CSV_RAW.exists():
        raw.to_csv(SENSOR_CSV_RAW, index=False)
        log.info("Created %s with %d raw row(s)", SENSOR_CSV_RAW, len(raw))
    else:
        raw.to_csv(SENSOR_CSV_RAW, mode="a", header=False, index=False)
        log.info("Appended %d raw row(s) to %s", len(raw), SENSOR_CSV_RAW)

    return len(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_latest(dry_run: bool = False) -> bool:
    """
    Default mode — incremental sync with calibration applied.

    Fetches only rows from dates strictly newer than the latest CSV timestamp,
    applies all three sensor calibrations, and appends to both CSVs:
        obando_sensor_data.csv      ← calibrated (used by XGB model)
        obando_sensor_data_raw.csv  ← raw hardware values (no conversion)

    On first run (no CSV) pulls and calibrates the full table.
    Returns True if at least one new row was written.
    """
    if not USE_HARDWARE:
        log.info("USE_HARDWARE=False — ingestion skipped.")
        return False

    latest_ts = get_latest_csv_timestamp()

    if latest_ts:
        log.info("CSV latest timestamp: %s — fetching rows from dates after %s.",
                 latest_ts, latest_ts[:10])
    else:
        log.info("CSV missing or empty — bootstrapping full history from Supabase.")

    raw = fetch_rows_since(after_timestamp=latest_ts)

    if raw.empty:
        log.info("No new rows after %s — CSV is already up to date.", latest_ts)
        return False

    log.info("Fetched %d raw row(s) — saving raw output and applying calibration.", len(raw))

    append_raw_rows_to_csv(raw, dry_run=dry_run)

    calibrated = calibrate_df(raw)

    if calibrated.empty:
        log.warning("All fetched rows were discarded as invalid sensor readings.")
        return False

    written = append_rows_to_csv(calibrated, dry_run=dry_run)
    return written > 0


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

    append_raw_rows_to_csv(raw, dry_run=dry_run)

    calibrated = calibrate_df(raw)
    log.info("Ingesting %d calibrated row(s) for %s.", len(calibrated), target_date)
    written = append_rows_to_csv(calibrated, dry_run=dry_run)
    return written > 0


def log_prediction(risk_tier: str, probability: float, timestamp: str) -> None:
    """
    Called by Start.py after each prediction run.
    Stores the result in Supabase for remote monitoring.
    """
    record = {
        "timestamp":   timestamp,
        "probability": round(probability, 4),
        "risk_tier":   risk_tier,
        "model":       "XGB",
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