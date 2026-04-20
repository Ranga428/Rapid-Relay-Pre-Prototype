"""
stress_test_gen.py
==================
Rapid Relay EWS — Stress Test CSV Generator
Obando, Bulacan

Generates synthetic input CSVs that simulate each flood alert tier.
Water level, soil moisture, and humidity are now driven by a pyswmm
SWMM simulation rather than pure NumPy ranges:

  waterlevel    ← node depth  at J_SENSOR   (m)
  soil_moisture ← subcatchment soil moisture at SC_OBANDO (normalized 0–1)
  humidity      ← subcatchment runoff        at SC_OBANDO (scaled to CWV proxy)

Rainfall forcing per tier is injected programmatically into the SWMM
time-series before each simulation run, so every scenario produces
hydraulically-consistent sensor values.

Usage examples:
  # Append one row per tier to speedtest_merge.csv:
  python stress_test_gen.py --scenario clear   --append
  python stress_test_gen.py --scenario watch   --append
  python stress_test_gen.py --scenario warning --append
  python stress_test_gen.py --scenario danger  --append

  # Generate full stress-test CSVs (30-day sims) for XGB_Predict --data:
  python stress_test_gen.py --scenario clear   --days 30 --csv
  python stress_test_gen.py --scenario all     --days 30 --csv

  # Simulate an escalating event (one appended row per tier):
  python stress_test_gen.py --scenario escalate --append

  # Insert one raw hardware row per tier into Supabase:
  python stress_test_gen.py --scenario clear   --supabase
  python stress_test_gen.py --scenario all     --supabase
  python stress_test_gen.py --scenario escalate --supabase

  # Combine with --append or --csv as needed:
  python stress_test_gen.py --scenario warning --append --supabase

SWMM model:
  Default path: <project_root>/data/swmm/obando_bulacan.inp
  Override:     --inp /path/to/your_model.inp
  If the file does not exist, a minimal placeholder is auto-written.

Value ranges from HEC-HMS calibration data (2017–2026):
  waterlevel   : 0.40 – 3.00 m  (expanded for extreme stress test coverage)
  soil_moisture: 0.15 – 0.55 (normalized)
  humidity     : 0.05 – 8.50 (CWV scaled)

─────────────────────────────────────────────────────────────────────
--supabase FLAG  (NEW)
─────────────────────────────────────────────────────────────────────
Inserts one synthetic row per tier into the live Supabase table
`obando_environmental_data` using raw hardware units — the same units
the real sensors produce.

The calibrated (proxy) values from the SWMM sim are inverted back to
hardware units before insert:

  FINAL DISTANCE (m)  ← reverse water-level calibration
      proxy_wl → hw_wl_m  →  distance = DIKE_HEIGHT_M - hw_wl_m
      (reverse of sensor_ingest.py CONVERSION 1)

  SOIL MOISTURE (%)   ← reverse soil calibration
      proxy_sm → t  →  hw_pct = SOIL_HW_DRY + t*(SOIL_HW_WET-SOIL_HW_DRY)
      (reverse of sensor_ingest.py CONVERSION 2)

  HUMIDITY (%RH)      ← reverse humidity calibration
      proxy_hum → t  →  hw_rh = HUMIDITY_HW_MIN + t*(HUMIDITY_HW_MAX-HUMIDITY_HW_MIN)
      (reverse of sensor_ingest.py CONVERSION 3)

Each inserted row uses today's date + a time of 12:00:00 (or offset
minutes apart for escalate/all to avoid PK collisions).

Requires the same .env file as sensor_ingest.py:
  SUPABASE_URL         = https://<project-ref>.supabase.co
  SUPABASE_SERVICE_KEY = <service_role key>
"""

from __future__ import annotations

import os
import sys
import shutil
import argparse
import tempfile
import warnings
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# pyswmm + swmm.toolkit types
try:
    from pyswmm import Simulation, Nodes, Subcatchments, RainGages
    from pyswmm.swmm5 import PySWMM
    PYSWMM_AVAILABLE = True
except ImportError:
    PYSWMM_AVAILABLE = False
    warnings.warn(
        "pyswmm is not installed — falling back to NumPy synthetic generation.\n"
        "Install with:  pip install pyswmm",
        stacklevel=2,
    )

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

SCRIPT_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = SCRIPT_DIR.parent

MERGE_FILE    = _PROJECT_ROOT / "data" / "sensor" / "speedtest_merge.csv"
STRESS_CSV_DIR = _PROJECT_ROOT / "data" / "stress_test"
DEFAULT_INP   = _PROJECT_ROOT / "data" / "swmm" / "obando_bulacan.inp"

# .env location (mirrors sensor_ingest.py)
_ENV_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".env"))

# SWMM element names that must exist in the .inp model
SWMM_SENSOR_NODE   = "J_SENSOR"     # junction → waterlevel
SWMM_SUBCATCHMENT  = "SC_OBANDO"    # subcatchment → soil_moisture, humidity proxy
SWMM_RAINGAUGE     = "RG_OBANDO"    # rain gauge whose timeseries we override

# ---------------------------------------------------------------------------
# TIER RAINFALL FORCING  (mm/hr)
# ---------------------------------------------------------------------------

TIER_RAINFALL_MM_HR = {
    "clear":   0.5,    # near-zero rainfall — deep dry season baseline
    "watch":   10.0,
    "warning": 35.0,   # slightly more aggressive than before
    "danger":  90.0,   # typhoon-level forcing — extreme event
}

# ---------------------------------------------------------------------------
# TIER PROFILES
# ---------------------------------------------------------------------------

TIER_PROFILES = {
    "clear": {
        "label":         "CLEAR",
        "description":   "Deep dry season — bone-dry soil, near-zero humidity, minimal water.",
        "waterlevel":    (0.40, 0.65),    # was (0.60, 0.75) — pushed lower
        "soil_moisture": (0.15, 0.24),    # was (0.22, 0.30) — truly dry soil
        "humidity":      (0.05, 0.80),    # was (0.15, 1.50) — near-zero atmospheric moisture
    },
    "watch": {
        "label":         "WATCH",
        "description":   "Early wet season — water rising, soil moistening.",
        "waterlevel":    (0.75, 1.20),
        "soil_moisture": (0.30, 0.38),
        "humidity":      (1.50, 3.00),
    },
    "warning": {
        "label":         "WARNING",
        "description":   "Active flood conditions — high water, saturating soil.",
        "waterlevel":    (1.35, 2.10),    # was (1.20, 2.00) — shifted higher
        "soil_moisture": (0.40, 0.47),    # was (0.38, 0.45) — approaching saturation
        "humidity":      (3.20, 5.00),    # was (3.00, 4.50) — shifted higher
    },
    "danger": {
        "label":         "DANGER",
        "description":   "Catastrophic flood — overtopping risk, soil fully saturated.",
        "waterlevel":    (2.20, 3.00),    # was (2.00, 2.58) — exceeds historical peak
        "soil_moisture": (0.47, 0.55),    # was (0.45, 0.50) — beyond field saturation
        "humidity":      (5.50, 8.50),    # was (4.50, 6.87) — well above calibrated max
    },
}

ESCALATION_ORDER = ["clear", "watch", "warning", "danger"]

# ---------------------------------------------------------------------------
# SENSOR SCALING CONSTANTS
# ---------------------------------------------------------------------------

SM_OUT_LO, SM_OUT_HI = 0.15, 0.55   # updated to match new CLEAR low / DANGER high
SM_IN_LO,  SM_IN_HI  = 0.00, 1.00

HUMIDITY_RUNOFF_PEAK = 50.0
HUMIDITY_CWV_MIN     = 0.05          # updated to match new CLEAR low
HUMIDITY_CWV_MAX     = 8.50          # updated to match new DANGER high

# ---------------------------------------------------------------------------
# CALIBRATION CONSTANTS  (must mirror sensor_ingest.py exactly)
# Used for reverse-calibration when inserting into Supabase.
# ---------------------------------------------------------------------------

DIKE_HEIGHT_M = 4.039

HW_WL_DRY    = 3.819
HW_WL_WET    = 3.999
PROXY_WL_DRY = 0.718
PROXY_WL_WET = 2.197

SOIL_HW_DRY    = 71.8
SOIL_HW_WET    = 85.0
SOIL_PROXY_DRY = 0.242
SOIL_PROXY_WET = 0.463

HUMIDITY_HW_MIN    = 78.5
HUMIDITY_HW_MAX    = 88.78
HUMIDITY_PROXY_MIN = 0.05            # updated to match new CLEAR low
HUMIDITY_PROXY_MAX = 8.50            # updated to match new DANGER high

# Supabase table + column names (must mirror sensor_ingest.py)
TABLE_ENV_DATA = "obando_environmental_data"
COL_DATE       = "Date"
COL_TIME       = "Time"
COL_SOIL       = "Soil Moisture"
COL_HUMIDITY   = "Humidity"
COL_DISTANCE   = "Final Distance"

# ---------------------------------------------------------------------------
# MINIMAL .INP TEMPLATE
# ---------------------------------------------------------------------------

MINIMAL_INP_TEMPLATE = """\
[TITLE]
Rapid Relay EWS - Obando Bulacan placeholder

[OPTIONS]
FLOW_UNITS           CMS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS         DEPTH
MIN_SLOPE            0
ALLOW_PONDING        NO
SKIP_STEADY_STATE    NO
START_DATE           {start_date}
START_TIME           00:00:00
END_DATE             {end_date}
END_TIME             23:00:00
REPORT_START_DATE    {start_date}
REPORT_START_TIME    00:00:00
DRY_DAYS             0
REPORT_STEP          01:00:00
WET_STEP             00:05:00
DRY_STEP             01:00:00
ROUTING_STEP         0:00:30

[EVAPORATION]
CONSTANT         0.0
DRY_ONLY         NO

[RAINGAGES]
;;Name   Format    Interval SCF Source
{raingauge}   INTENSITY 1:00   1.0  TIMESERIES TS_RAIN

[SUBCATCHMENTS]
;;Name  Rain_Gage  Outlet  Area  %Imperv  Width  %Slope  CurbLen
{subcatchment}  {raingauge}  {sensor_node}  250  35  500  1.5  0

[SUBAREAS]
;;Subcatch  N-Imperv  N-Perv  S-Imperv  S-Perv  PctZero  RouteTo
{subcatchment}  0.015  0.24  1.8  3.5  25  OUTLET

[INFILTRATION]
;;Subcatch  MaxRate  MinRate  Decay  DryTime  MaxInfil
{subcatchment}  76.2  3.81  4.14  7  0

[JUNCTIONS]
;;Name  Elev  MaxDepth  InitDepth  SurDepth  Aponded
{sensor_node}  1.2  3.0  0.4  0  500

[OUTFALLS]
;;Name  Elev  Type  Gated
OUT_MANILA_BAY  0.0  FREE  NO

[CONDUITS]
;;Name  FromNode  ToNode  Length  Roughness  InOffset  OutOffset
C_MAIN  {sensor_node}  OUT_MANILA_BAY  8000  0.018  0  0

[XSECTIONS]
;;Link  Shape  Geom1  Geom2  Geom3  Geom4  Barrels
C_MAIN  TRAPEZOIDAL  3.0  20  2  2  1

[TIMESERIES]
;;Name  Date  Time  Value
{rain_timeseries}

[REPORT]
SUBCATCHMENTS ALL
NODES ALL
LINKS ALL

[COORDINATES]
;;Node  X-Coord  Y-Coord
{sensor_node}  120.9520  14.8320
OUT_MANILA_BAY  120.9600  14.8150

[Polygons]
;;Subcatchment  X-Coord  Y-Coord
{subcatchment}  120.9300  14.8400
{subcatchment}  120.9700  14.8400
{subcatchment}  120.9700  14.8200
{subcatchment}  120.9300  14.8200

[SYMBOLS]
;;Gage  X-Coord  Y-Coord
{raingauge}  120.9400  14.8500
"""


def _write_minimal_inp(inp_path: Path, rain_mm_hr: float, days: int) -> None:
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.utcnow()
    end   = today + pd.Timedelta(days=max(days - 1, 1))

    ts_rows: list[str] = []
    current = today
    while current <= end + pd.Timedelta(days=1):
        for hour in [0, 6, 12, 18]:
            ts_rows.append(
                f"TS_RAIN  {current.strftime('%m/%d/%Y')}  "
                f"{hour:02d}:00  {rain_mm_hr}"
            )
        current += pd.Timedelta(days=1)

    content = MINIMAL_INP_TEMPLATE.format(
        start_date      = today.strftime("%m/%d/%Y"),
        end_date        = end.strftime("%m/%d/%Y"),
        rain_timeseries = "\n".join(ts_rows),
        raingauge       = SWMM_RAINGAUGE,
        subcatchment    = SWMM_SUBCATCHMENT,
        sensor_node     = SWMM_SENSOR_NODE,
    )
    inp_path.write_text(content)
    print(f"  [INP] Wrote placeholder model → {inp_path}")


def _patch_inp_dates_and_rain(
    template_inp: Path,
    rain_mm_hr: float,
    days: int,
) -> Path:
    today = pd.Timestamp.utcnow()
    end   = today + pd.Timedelta(days=max(days - 1, 1))
    start_str = today.strftime("%m/%d/%Y")
    end_str   = end.strftime("%m/%d/%Y")

    new_ts_rows: list[str] = []
    current = today
    while current <= end + pd.Timedelta(days=1):
        for hour in [0, 6, 12, 18]:
            new_ts_rows.append(
                f"TS_RAIN  {current.strftime('%m/%d/%Y')}  "
                f"{hour:02d}:00  {rain_mm_hr}"
            )
        current += pd.Timedelta(days=1)

    src = template_inp.read_text()
    lines_out: list[str] = []
    in_timeseries = False
    ts_written    = False

    for line in src.splitlines():
        stripped = line.strip()

        if stripped.startswith("["):
            in_timeseries = stripped.upper() == "[TIMESERIES]"
            if in_timeseries and not ts_written:
                lines_out.append(line)
                lines_out.extend(new_ts_rows)
                ts_written = True
                continue

        if in_timeseries and stripped.startswith("TS_RAIN") and not stripped.startswith(";;"):
            continue

        if stripped.upper().startswith("START_DATE") and not stripped.startswith("REPORT"):
            line = f"START_DATE           {start_str}"
        elif stripped.upper().startswith("END_DATE"):
            line = f"END_DATE             {end_str}"
        elif stripped.upper().startswith("REPORT_START_DATE"):
            line = f"REPORT_START_DATE    {start_str}"

        lines_out.append(line)

    tmp_dir = Path(tempfile.mkdtemp(prefix="swmm_stress_"))
    tmp_inp = tmp_dir / template_inp.name
    tmp_inp.write_text("\n".join(lines_out))
    return tmp_inp


# ---------------------------------------------------------------------------
# SWMM SIMULATION
# ---------------------------------------------------------------------------

def _scale(value: float, in_lo: float, in_hi: float,
           out_lo: float, out_hi: float) -> float:
    if in_hi == in_lo:
        return (out_lo + out_hi) / 2
    ratio = (value - in_lo) / (in_hi - in_lo)
    return float(np.clip(out_lo + ratio * (out_hi - out_lo), out_lo, out_hi))


def run_swmm_simulation(
    inp_path: Path,
    tier_key: str,
    days: int,
    seed: int = 42,
) -> pd.DataFrame:
    if not PYSWMM_AVAILABLE:
        print("  [SWMM] pyswmm unavailable — using NumPy fallback.")
        return _numpy_fallback(tier_key, days, seed)

    rain_mm_hr = TIER_RAINFALL_MM_HR[tier_key]
    label      = TIER_PROFILES[tier_key]["label"]

    tmp_inp = _patch_inp_dates_and_rain(inp_path, rain_mm_hr, days)
    tmp_dir = tmp_inp.parent

    records: list[dict] = []
    try:
        print(f"  [SWMM] Running {label} scenario  "
              f"(rain={rain_mm_hr} mm/hr, days={days}) …")

        with Simulation(str(tmp_inp)) as sim:
            node_collection = Nodes(sim)
            sub_collection  = Subcatchments(sim)

            sensor_node  = node_collection[SWMM_SENSOR_NODE]
            subcatchment = sub_collection[SWMM_SUBCATCHMENT]

            sim.step_advance(3600)

            for step in sim:
                ts = pd.Timestamp(sim.current_time)

                wl = float(sensor_node.depth)
                wl = round(max(0.0, wl), 6)

                infil   = max(0.0, float(getattr(subcatchment, "infiltration_loss", 0.0)))
                runoff_ = max(0.0, float(getattr(subcatchment, "runoff", 0.0)))
                denom   = runoff_ + infil + 1e-9
                wetness = runoff_ / denom
                sm = _scale(wetness, SM_IN_LO, SM_IN_HI, SM_OUT_LO, SM_OUT_HI)
                sm = round(sm, 6)

                runoff = float(getattr(subcatchment, "runoff", 0.0))
                hum = _scale(
                    runoff, 0.0, HUMIDITY_RUNOFF_PEAK,
                    HUMIDITY_CWV_MIN, HUMIDITY_CWV_MAX
                )
                hum = round(hum, 6)

                records.append({
                    "timestamp":    ts.strftime("%Y-%m-%dT%H:%M:%S"),
                    "waterlevel":   wl,
                    "soil_moisture": sm,
                    "humidity":     hum,
                    "_tier_label":  label,
                })

                if len(records) >= days * 24:
                    break

        print(f"  [SWMM] Simulation complete — {len(records)} hourly steps collected.")

    except Exception as exc:
        print(f"  [SWMM] Simulation error: {exc}")
        print("  [SWMM] Falling back to NumPy synthetic generation.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _numpy_fallback(tier_key, days, seed)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not records:
        print("  [SWMM] No records produced — falling back to NumPy.")
        return _numpy_fallback(tier_key, days, seed)

    df = pd.DataFrame(records)

    df["_dt"] = pd.to_datetime(df["timestamp"])
    df = (
        df.set_index("_dt")
          .resample("D")
          .agg({
              "waterlevel":    "max",
              "soil_moisture": "max",
              "humidity":      "max",
              "_tier_label":   "first",
          })
          .reset_index()
    )
    df["timestamp"] = df["_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df = df.drop(columns=["_dt"]).head(days)
    df = df[["timestamp", "waterlevel", "soil_moisture", "humidity", "_tier_label"]]

    _range_check(df, tier_key)
    return df


def _range_check(df: pd.DataFrame, tier_key: str) -> None:
    p = TIER_PROFILES[tier_key]
    for col, (lo, hi) in [
        ("waterlevel",    p["waterlevel"]),
        ("soil_moisture", p["soil_moisture"]),
        ("humidity",      p["humidity"]),
    ]:
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if not out_of_range.empty:
            pct = 100 * len(out_of_range) / len(df)
            warnings.warn(
                f"  [RANGE] {col}: {len(out_of_range)} rows ({pct:.1f}%) "
                f"outside calibrated {tier_key.upper()} range [{lo}, {hi}]. "
                f"Consider re-calibrating the SWMM model or rainfall forcing.",
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# NUMPY FALLBACK
# ---------------------------------------------------------------------------

def _ramp(lo: float, hi: float, n: int,
          noise: float = 0.02, seed: int = 0) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    base = np.linspace(lo, hi, n)
    base += rng.normal(0, (hi - lo) * noise, n)
    return np.clip(base, lo, hi).round(6)


def _numpy_fallback(tier_key: str, days: int, seed: int = 42) -> pd.DataFrame:
    p     = TIER_PROFILES[tier_key]
    label = p["label"]
    start = pd.Timestamp.utcnow().floor("D")
    dates = pd.date_range(start, periods=days, freq="D", tz="UTC")

    df = pd.DataFrame({
        "timestamp":    [d.strftime("%Y-%m-%dT%H:%M:%S") for d in dates],
        "waterlevel":   _ramp(*p["waterlevel"],   days, seed=seed),
        "soil_moisture": _ramp(*p["soil_moisture"], days, seed=seed + 1),
        "humidity":     _ramp(*p["humidity"],     days, seed=seed + 2),
        "_tier_label":  label,
    })
    return df


# ---------------------------------------------------------------------------
# SINGLE ROW (for --append and --supabase)
# ---------------------------------------------------------------------------

def single_row_from_swmm(
    tier_key: str,
    inp_path: Path,
    offset_days: int = 0,
    position: str = "mid",
) -> dict:
    df = run_swmm_simulation(inp_path, tier_key, days=2, seed=0)
    if df.empty:
        return _single_row_fallback(tier_key, offset_days, position)

    if position == "lo":
        row = df.loc[df["waterlevel"].idxmin()].to_dict()
    elif position == "hi":
        row = df.loc[df["waterlevel"].idxmax()].to_dict()
    else:
        mid_idx = len(df) // 2
        row = df.iloc[mid_idx].to_dict()

    ts = pd.Timestamp.utcnow().floor("D") + pd.Timedelta(days=offset_days)
    row["timestamp"] = ts.strftime("%Y-%m-%dT%H:%M:%S")
    return row


def _single_row_fallback(
    tier_key: str,
    offset_days: int = 0,
    position: str = "mid",
) -> dict:
    p = TIER_PROFILES[tier_key]
    pick = {"lo": 0, "mid": 1, "hi": 2}[position]
    vals = [
        (p["waterlevel"][0],    p["soil_moisture"][0],    p["humidity"][0]),
        (
            round(sum(p["waterlevel"])   / 2, 6),
            round(sum(p["soil_moisture"]) / 2, 6),
            round(sum(p["humidity"])     / 2, 6),
        ),
        (p["waterlevel"][1],    p["soil_moisture"][1],    p["humidity"][1]),
    ]
    wl, sm, hum = vals[pick]
    ts = pd.Timestamp.utcnow().floor("D") + pd.Timedelta(days=offset_days)
    return {
        "timestamp":    ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "waterlevel":   wl,
        "soil_moisture": sm,
        "humidity":     hum,
        "_tier_label":  p["label"],
    }


# ===========================================================================
# REVERSE CALIBRATION  (proxy units → raw hardware units)
#
# Inverts the three conversions in sensor_ingest.py so that synthetic
# calibrated values can be stored in obando_environmental_data in the
# same raw units the physical sensors produce.
# ===========================================================================

def _reverse_waterlevel(proxy_wl: float) -> float:
    """
    Proxy waterlevel (m above UHSLC datum) → Final Distance (m).

    Inverse of sensor_ingest._calibrate_waterlevel():
      1. Reverse linear stretch  →  hw_wl_m
      2. hw_wl_m → distance_m = DIKE_HEIGHT_M - hw_wl_m

    Result is clamped to the valid sensor range [0.05, 4.039].
    """
    t = (proxy_wl - PROXY_WL_DRY) / (PROXY_WL_WET - PROXY_WL_DRY)
    t = max(0.0, min(1.0, t))
    hw_wl_m = HW_WL_DRY + t * (HW_WL_WET - HW_WL_DRY)
    distance_m = DIKE_HEIGHT_M - hw_wl_m
    # Clamp to the valid sensor window used by sensor_ingest.py
    distance_m = max(0.05, min(DIKE_HEIGHT_M, distance_m))
    return round(distance_m, 4)


def _reverse_soil_moisture(proxy_sm: float) -> float:
    """
    Proxy soil moisture (m³/m³) → raw capacitive sensor reading (%).

    Inverse of sensor_ingest._calibrate_soil_moisture().
    """
    t = (proxy_sm - SOIL_PROXY_DRY) / (SOIL_PROXY_WET - SOIL_PROXY_DRY)
    t = max(0.0, min(1.0, t))
    hw_pct = SOIL_HW_DRY + t * (SOIL_HW_WET - SOIL_HW_DRY)
    return round(hw_pct, 4)


def _reverse_humidity(proxy_hum: float) -> float:
    """
    Proxy humidity (cm CWV) → raw relative humidity reading (%RH).

    Inverse of sensor_ingest._calibrate_humidity().
    """
    t = (proxy_hum - HUMIDITY_PROXY_MIN) / (HUMIDITY_PROXY_MAX - HUMIDITY_PROXY_MIN)
    t = max(0.0, min(1.0, t))
    hw_rh = HUMIDITY_HW_MIN + t * (HUMIDITY_HW_MAX - HUMIDITY_HW_MIN)
    return round(hw_rh, 4)


def _row_to_hardware(row: dict, time_str: str) -> dict:
    """
    Convert one calibrated synthetic row → raw hardware record ready
    for insertion into obando_environmental_data.

    Parameters
    ----------
    row      : dict with keys timestamp, waterlevel, soil_moisture, humidity
    time_str : HH:MM:SS string to use for the Time column

    Returns
    -------
    dict matching the Supabase column names exactly.
    """
    ts = pd.Timestamp(row["timestamp"])
    return {
        COL_DATE:     ts.strftime("%Y-%m-%d"),   # "Date"
        COL_TIME:     time_str,                   # "Time"
        COL_DISTANCE: _reverse_waterlevel(row["waterlevel"]),
        COL_SOIL:     _reverse_soil_moisture(row["soil_moisture"]),
        COL_HUMIDITY: _reverse_humidity(row["humidity"]),
        "Temperature":   0,
        "Pressure":      0,
    }


# ---------------------------------------------------------------------------
# SUPABASE CLIENT  (lazy, mirrors sensor_ingest.py pattern)
# ---------------------------------------------------------------------------

_supabase_client = None


def _get_supabase_client():
    """
    Lazy singleton Supabase client.
    Loads credentials from the same .env as sensor_ingest.py.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH, override=False)
    except ImportError:
        pass  # python-dotenv optional; credentials may already be in env

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise RuntimeError(
            f"Supabase credentials not found.\n"
            f"  .env path checked    : {_ENV_PATH}\n"
            f"  SUPABASE_URL         : {'SET' if url else 'MISSING'}\n"
            f"  SUPABASE_SERVICE_KEY : {'SET' if key else 'MISSING'}\n"
            "  Make sure your .env file exists at the project root."
        )

    try:
        from supabase import create_client
    except ImportError:
        raise RuntimeError(
            "supabase-py is not installed.\n"
            "Install with:  pip install supabase"
        )

    _supabase_client = create_client(url, key)
    return _supabase_client


def _supabase_insert_row(record: dict, dry_run: bool = False) -> bool:
    """
    Insert one hardware-unit record into obando_environmental_data.
    Returns True on success, False on failure.
    """
    if dry_run:
        print(f"  [DRY RUN] Would insert → {record}")
        return True

    try:
        client = _get_supabase_client()
        client.table(TABLE_ENV_DATA).insert(record).execute()
        return True
    except Exception as exc:
        print(f"  [SUPABASE] Insert failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# PUBLIC: insert one stress-test row per tier into Supabase
# ---------------------------------------------------------------------------

def insert_stress_rows_to_supabase(
    rows_df: pd.DataFrame,
    dry_run: bool = False,
    base_time_offset_minutes: int = 0,
    count: int = 1,
) -> int:
    if rows_df.empty:
        print("  [SUPABASE] Nothing to insert — DataFrame is empty.")
        return 0

    inserted = 0
    # Get the current system time to use as the base for all inserts
    now = datetime.now()

    for repeat in range(count):
        for i, (_, row) in enumerate(rows_df.iterrows()):
            # Calculate offset: starting from 'now', increment by 1 minute per row
            total_offset = base_time_offset_minutes + (repeat * len(rows_df)) + i
            current_ts = now + timedelta(minutes=total_offset)
            
            time_str = current_ts.strftime("%H:%M:%S")
            date_str = current_ts.strftime("%Y-%m-%d")

            # Update the row's timestamp so reverse calibration uses the correct date/time context
            row_with_offset = row.copy()
            row_with_offset["timestamp"] = current_ts.strftime("%Y-%m-%dT%H:%M:%S")

            # Convert to hardware units using the dynamic time strings
            record = _row_to_hardware(row_with_offset, time_str)
            record[COL_DATE] = date_str # Ensure Date column matches our calculated TS
            
            tier = row.get("_tier_label", "?")

            print(f"  [SUPABASE] Inserting {tier} row {repeat+1}/{count} → "
                  f"Date={record[COL_DATE]}  Time={record[COL_TIME]}  "
                  f"Dist={record[COL_DISTANCE]:.4f}m  "
                  f"Soil={record[COL_SOIL]:.2f}%  "
                  f"RH={record[COL_HUMIDITY]:.2f}%")

            ok = _supabase_insert_row(record, dry_run=dry_run)
            if ok:
                inserted += 1
            else:
                print(f"  [SUPABASE] ✗  Failed to insert {tier} row.")

    label = "would insert" if dry_run else "inserted"
    print(f"  [SUPABASE] {label} {inserted}/{count * len(rows_df)} row(s) into {TABLE_ENV_DATA}.")
    return inserted


# ---------------------------------------------------------------------------
# I/O HELPERS
# ---------------------------------------------------------------------------

def append_to_merge(rows_df: pd.DataFrame, merge_path: Path) -> None:
    merge_path.parent.mkdir(parents=True, exist_ok=True)
    out = rows_df.drop(columns=["_tier_label"], errors="ignore")

    if merge_path.exists():
        existing = pd.read_csv(merge_path)
        combined = pd.concat([existing, out], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        print(f"  speedtest_merge.csv not found — creating: {merge_path}")
        combined = out

    combined.to_csv(merge_path, index=False)
    print(f"  Appended {len(out)} row(s) → {merge_path}  (total: {len(combined)} rows)")


def save_stress_csv(df: pd.DataFrame, tier_key: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"stress_{tier_key}.csv"
    path  = out_dir / fname
    df.drop(columns=["_tier_label"], errors="ignore").to_csv(path, index=False)
    print(f"  Saved stress CSV → {path}  ({len(df)} rows)")
    print(f"  Run: python XGB_Predict.py --data \"{path}\"")
    plot_stress_csv(df, tier_key, out_dir)
    return path


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

TIER_COLORS = {
    "clear":              {"bg": "#e8f5e9", "line": "#2e7d32", "label": "CLEAR"},
    "watch":              {"bg": "#fffde7", "line": "#f9a825", "label": "WATCH"},
    "warning":            {"bg": "#fff3e0", "line": "#e65100", "label": "WARNING"},
    "danger":             {"bg": "#ffebee", "line": "#b71c1c", "label": "DANGER"},
    "escalate":           {"bg": "#f3e5f5", "line": "#6a1b9a", "label": "ESCALATE"},
    "all_tiers_sequence": {"bg": "#e3f2fd", "line": "#0d47a1", "label": "ALL TIERS"},
}

# Updated thresholds to match new tier profile boundaries
WL_THRESHOLDS = {
    "WATCH entry":   (0.75,  "gold"),
    "WARNING entry": (1.35,  "orange"),   # was 1.20
    "DANGER entry":  (2.20,  "red"),      # was 2.00
}

_COL_THRESHOLDS = [
    [(0.75, "gold",   "WATCH"), (1.35, "orange", "WARNING"), (2.20, "red",    "DANGER")],
    [(0.30, "gold",   "WATCH"), (0.40, "orange", "WARNING"), (0.47, "red",    "DANGER")],
    [(1.50, "gold",   "WATCH"), (3.20, "orange", "WARNING"), (5.50, "red",    "DANGER")],
]
_COL_SPANS = [
    [(0, 0.75, "green"), (0.75, 1.35, "gold"), (1.35, 2.20, "orange"), (2.20, 3.20, "red")],
    [(0, 0.30, "green"), (0.30, 0.40, "gold"), (0.40, 0.47, "orange"), (0.47, 0.60, "red")],
    [(0, 1.50, "green"), (1.50, 3.20, "gold"), (3.20, 5.50, "orange"), (5.50, 9.00, "red")],
]
_TIER_ZONE_LEGEND = [
    mpatches.Patch(color="green",      alpha=0.5, label="CLEAR zone"),
    mpatches.Patch(color="gold",       alpha=0.5, label="WATCH zone"),
    mpatches.Patch(color="darkorange", alpha=0.5, label="WARNING zone"),
    mpatches.Patch(color="red",        alpha=0.5, label="DANGER zone"),
]


def _add_threshold_bands(ax, col_i: int) -> None:
    for (ylo, yhi, bc) in _COL_SPANS[col_i]:
        ax.axhspan(ylo, yhi, alpha=0.06, color=bc, zorder=0)
    for (tval, tcolor, _) in _COL_THRESHOLDS[col_i]:
        ax.axhline(tval, color=tcolor, linestyle="--", linewidth=1.0, alpha=0.8)


def _fmt_xaxis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)


def _plot_all_tiers_sequence(df: pd.DataFrame, out_dir: Path) -> Path:
    tiers  = ESCALATION_ORDER
    n      = len(df) // 4
    chunks = [df.iloc[i * n: (i + 1) * n].copy() for i in range(4)]

    fig, axes = plt.subplots(
        4, 3, figsize=(18, 14), sharex=False,
        gridspec_kw={"hspace": 0.55, "wspace": 0.3}
    )
    fig.suptitle(
        "Rapid Relay EWS — Stress Test Input Data  (SWMM-driven)\n"
        "All Tier Scenarios in Sequence  |  Obando, Bulacan",
        fontsize=13, fontweight="bold",
    )

    col_titles = ["Water Level (m)", "Soil Moisture", "Humidity (CWV)"]
    cols_data  = ["waterlevel", "soil_moisture", "humidity"]
    col_colors = ["steelblue", "sienna", "teal"]

    for row_i, (tier_key, chunk) in enumerate(zip(tiers, chunks)):
        tc    = TIER_COLORS[tier_key]
        dates = pd.to_datetime(chunk["timestamp"])

        for col_i, (col, color) in enumerate(zip(cols_data, col_colors)):
            ax = axes[row_i][col_i]
            ax.set_facecolor(tc["bg"])
            ax.fill_between(dates, chunk[col], alpha=0.25, color=color)
            ax.plot(dates, chunk[col], color=color, linewidth=1.8)
            _add_threshold_bands(ax, col_i)
            ax.grid(axis="y", alpha=0.25)
            _fmt_xaxis(ax)

            if col_i == 0:
                ax.set_ylabel(
                    f"{tc['label']}\n{col_titles[0]}",
                    fontsize=8, fontweight="bold", color=tc["line"],
                )
            else:
                ax.set_ylabel(col_titles[col_i], fontsize=8)

            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=10, fontweight="bold", pad=4)

    fig.legend(
        handles=_TIER_ZONE_LEGEND, loc="lower center", ncol=4, fontsize=9,
        bbox_to_anchor=(0.5, 0.005), framealpha=0.85,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plot_path = out_dir / "stress_all_tiers_sequence.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot   → {plot_path}")
    return plot_path


def plot_stress_csv(df: pd.DataFrame, tier_key: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    if tier_key == "all_tiers_sequence":
        return _plot_all_tiers_sequence(df, out_dir)

    tc    = TIER_COLORS.get(tier_key, {"bg": "#f5f5f5", "line": "steelblue", "label": tier_key.upper()})
    dates = pd.to_datetime(df["timestamp"])
    source_note = "SWMM-driven" if PYSWMM_AVAILABLE else "NumPy fallback"

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.patch.set_facecolor(tc["bg"])
    fig.suptitle(
        f"Rapid Relay EWS — Stress Test Input Data  ({source_note})\n"
        f"Scenario: {tc['label']}  |  {len(df)} days  |  Obando, Bulacan",
        fontsize=13, fontweight="bold", y=0.98,
    )

    ax1 = axes[0]
    ax1.set_facecolor(tc["bg"])
    ax1.fill_between(dates, df["waterlevel"], alpha=0.25, color=tc["line"])
    ax1.plot(dates, df["waterlevel"], color=tc["line"], linewidth=2,
             label="Water Level (m)  ← SWMM J_SENSOR depth", zorder=3)
    for tlabel, (tval, tcolor) in WL_THRESHOLDS.items():
        ax1.axhline(tval, color=tcolor, linestyle="--", linewidth=1.1,
                    alpha=0.8, label=f"{tlabel} ({tval}m)")
    ax1.axhspan(0,    0.75, alpha=0.06, color="green",  zorder=0)
    ax1.axhspan(0.75, 1.35, alpha=0.07, color="gold",   zorder=0)
    ax1.axhspan(1.35, 2.20, alpha=0.07, color="orange", zorder=0)
    ax1.axhspan(2.20, 3.20, alpha=0.08, color="red",    zorder=0)
    for ylo, yhi, tlabel, tcolor in [
        (0,    0.75, "CLEAR",   "green"),
        (0.75, 1.35, "WATCH",   "goldenrod"),
        (1.35, 2.20, "WARNING", "darkorange"),
        (2.20, 3.20, "DANGER",  "red"),
    ]:
        ax1.annotate(
            tlabel, xy=(1.002, (ylo + yhi) / 2),
            xycoords=("axes fraction", "data"),
            fontsize=7.5, color=tcolor, fontweight="bold", va="center",
        )
    ax1.set_ylabel("Water Level (m)", fontsize=10)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(labelbottom=False)

    ax2 = axes[1]
    ax2.set_facecolor(tc["bg"])
    ax2.fill_between(dates, df["soil_moisture"], alpha=0.25, color="sienna")
    ax2.plot(dates, df["soil_moisture"], color="sienna", linewidth=2,
             label="Soil Moisture  ← SWMM SC_OBANDO moisture_deficit (inverted)", zorder=3)
    ax2.axhline(0.30, color="gold",   linestyle="--", linewidth=1.1, alpha=0.8, label="WATCH entry (0.30)")
    ax2.axhline(0.40, color="orange", linestyle="--", linewidth=1.1, alpha=0.8, label="WARNING entry (0.40)")
    ax2.axhline(0.47, color="red",    linestyle="--", linewidth=1.1, alpha=0.8, label="DANGER entry (0.47)")
    ax2.axhspan(0,    0.30, alpha=0.06, color="green",  zorder=0)
    ax2.axhspan(0.30, 0.40, alpha=0.07, color="gold",   zorder=0)
    ax2.axhspan(0.40, 0.47, alpha=0.07, color="orange", zorder=0)
    ax2.axhspan(0.47, 0.60, alpha=0.08, color="red",    zorder=0)
    ax2.set_ylabel("Soil Moisture", fontsize=10)
    ax2.set_ylim(0, 0.60)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(labelbottom=False)

    ax3 = axes[2]
    ax3.set_facecolor(tc["bg"])
    ax3.fill_between(dates, df["humidity"], alpha=0.25, color="teal")
    ax3.plot(dates, df["humidity"], color="teal", linewidth=2,
             label="Humidity (CWV proxy)  ← SWMM SC_OBANDO runoff scaled", zorder=3)
    ax3.axhline(1.50, color="gold",   linestyle="--", linewidth=1.1, alpha=0.8, label="WATCH entry (1.50)")
    ax3.axhline(3.20, color="orange", linestyle="--", linewidth=1.1, alpha=0.8, label="WARNING entry (3.20)")
    ax3.axhline(5.50, color="red",    linestyle="--", linewidth=1.1, alpha=0.8, label="DANGER entry (5.50)")
    ax3.axhspan(0,    1.50, alpha=0.06, color="green",  zorder=0)
    ax3.axhspan(1.50, 3.20, alpha=0.07, color="gold",   zorder=0)
    ax3.axhspan(3.20, 5.50, alpha=0.07, color="orange", zorder=0)
    ax3.axhspan(5.50, 9.00, alpha=0.08, color="red",    zorder=0)
    ax3.set_ylabel("Humidity (CWV)", fontsize=10)
    ax3.set_ylim(bottom=0)
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax3.grid(axis="y", alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=9)
    ax3.set_xlabel("Date", fontsize=10)

    fig.legend(
        handles=_TIER_ZONE_LEGEND, loc="lower center", ncol=4, fontsize=9,
        bbox_to_anchor=(0.5, 0.01), framealpha=0.8,
    )
    plt.tight_layout(rect=[0, 0.04, 0.98, 0.96])
    plot_path = out_dir / f"stress_{tier_key}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot   → {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# CONSOLE PREVIEW
# ---------------------------------------------------------------------------

def preview(rows_df: pd.DataFrame, scenario_label: str) -> None:
    source = "SWMM" if PYSWMM_AVAILABLE else "NumPy fallback"
    print(f"\n  {'─'*60}")
    print(f"  Scenario : {scenario_label}  [{source}]")
    print(f"  Rows     : {len(rows_df)}")
    print(f"  {'─'*60}")
    print(f"  {'Timestamp':<22} {'WaterLvl':>9} {'SoilMoist':>10} {'Humidity':>9}  Tier")
    print(f"  {'─'*22} {'─'*9} {'─'*10} {'─'*9}  {'─'*8}")
    for _, row in rows_df.iterrows():
        tier = row.get("_tier_label", "")
        print(
            f"  {row['timestamp']:<22} {row['waterlevel']:>9.4f} "
            f"{row['soil_moisture']:>10.4f} {row['humidity']:>9.4f}  {tier}"
        )
    print(f"  {'─'*60}\n")


def preview_hardware(rows_df: pd.DataFrame) -> None:
    """
    Print the reverse-calibrated hardware values that will actually be
    inserted into Supabase, so you can verify them before committing.
    """
    print(f"\n  {'─'*72}")
    print(f"  Supabase insert preview  (raw hardware units — what will be stored)")
    print(f"  {'─'*72}")
    print(f"  {'Date':<12} {'Time':<10} {'Dist (m)':>10} {'Soil (%)':>10} {'RH (%)':>10}  Tier")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10}  {'─'*8}")
    for i, (_, row) in enumerate(rows_df.iterrows()):
        offset_min = i
        hour       = 12 + offset_min // 60
        minute     = offset_min % 60
        time_str   = f"{hour:02d}:{minute:02d}:00"
        hw = _row_to_hardware(row, time_str)
        tier = row.get("_tier_label", "")
        print(
            f"  {hw[COL_DATE]:<12} {hw[COL_TIME]:<10} "
            f"{hw[COL_DISTANCE]:>10.4f} {hw[COL_SOIL]:>10.4f} "
            f"{hw[COL_HUMIDITY]:>10.4f}  {tier}"
        )
    print(f"  {'─'*72}\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rapid Relay EWS — Stress Test CSV Generator (pyswmm edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  clear    — deep dry-season baseline  (0.5 mm/hr rainfall forcing)
  watch    — early wet season          (10 mm/hr)
  warning  — active flooding           (35 mm/hr)
  danger   — catastrophic flood peak   (90 mm/hr)
  all      — all four tiers in sequence
  escalate — one appended row per tier (escalation order)

Output modes (can be combined freely):
  --append     append row(s) to speedtest_merge.csv
  --csv        save full CSV for XGB_Predict.py --data
  --supabase   insert one raw hardware row per tier into Supabase
               (reverse-calibrates proxy values back to sensor units)
  --dry-run    preview all outputs without writing or inserting anything

SWMM model:
  --inp    path to .inp model (auto-created if absent)

Examples:
  python stress_test_gen.py --scenario clear   --append
  python stress_test_gen.py --scenario all     --days 30 --csv
  python stress_test_gen.py --scenario warning --days 14 --csv --append
  python stress_test_gen.py --scenario escalate --append

  # Supabase insert examples:
  python stress_test_gen.py --scenario clear    --supabase
  python stress_test_gen.py --scenario warning  --supabase --dry-run
  python stress_test_gen.py --scenario all      --supabase
  python stress_test_gen.py --scenario escalate --supabase
  python stress_test_gen.py --scenario danger   --append --supabase
        """,
    )

    parser.add_argument("--scenario", "-s", required=True,
                        choices=["clear", "watch", "warning", "danger", "all", "escalate"],
                        help="Alert tier scenario to generate.")
    parser.add_argument("--append", "-a", action="store_true",
                        help="Append generated row(s) to speedtest_merge.csv.")
    parser.add_argument("--csv", "-c", action="store_true",
                        help="Save full stress-test CSV for batch prediction input.")
    parser.add_argument("--supabase", action="store_true",
                        help=(
                            "Insert one raw hardware row per tier into Supabase "
                            "obando_environmental_data. Values are reverse-calibrated "
                            "from proxy units back to sensor units before insert."
                        ))
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview all outputs without writing or inserting anything.")
    parser.add_argument("--days", "-d", type=int, default=30,
                        help="Simulation length in days for --csv mode (default: 30).")
    parser.add_argument("--position", "-p", default="mid",
                        choices=["lo", "mid", "hi"],
                        help="Where in the tier range to sample for single-row modes.")
    parser.add_argument("--inp", type=str, default=str(DEFAULT_INP),
                        help=f"Path to SWMM .inp model (default: {DEFAULT_INP}).")
    parser.add_argument("--merge-file", type=str, default=str(MERGE_FILE),
                        help="Path to speedtest_merge.csv.")
    parser.add_argument("--out-dir", type=str, default=str(STRESS_CSV_DIR),
                        help="Output directory for stress CSVs.")
    parser.add_argument("--count", type=int, default=1,
                    help="Number of rows to insert per tier into Supabase (default: 1).")

    args = parser.parse_args()

    if not args.append and not args.csv and not args.supabase:
        print("\n  ⚠️  No output mode selected. Use --append, --csv, and/or --supabase.\n")
        parser.print_help()
        sys.exit(1)

    inp_path   = Path(args.inp)
    merge_path = Path(args.merge_file)
    out_dir    = Path(args.out_dir)

    # Auto-create .inp if missing
    if not inp_path.exists():
        default_rain = TIER_RAINFALL_MM_HR.get(
            args.scenario if args.scenario not in ("all", "escalate") else "clear",
            0.5,
        )
        _write_minimal_inp(inp_path, default_rain, args.days)

    dry_tag = "  [DRY RUN]" if args.dry_run else ""
    print(f"\n{'='*60}")
    print(f"  Rapid Relay — Stress Test CSV Generator  (pyswmm){dry_tag}")
    print(f"{'='*60}")
    print(f"  Scenario   : {args.scenario}")
    modes = " + ".join(
        m for m, flag in [("--append", args.append), ("--csv", args.csv), ("--supabase", args.supabase)]
        if flag
    )
    print(f"  Mode       : {modes}")
    print(f"  SWMM .inp  : {inp_path}")
    print(f"  pyswmm     : {'available ✓' if PYSWMM_AVAILABLE else 'NOT installed — NumPy fallback'}")
    if args.csv:
        print(f"  Days       : {args.days}")
    print(f"  Position   : {args.position}  (for single-row modes)")
    if args.dry_run:
        print(f"  ⚠  DRY RUN — nothing will be written or inserted.")

    # ── ESCALATE ──────────────────────────────────────────────────────
    if args.scenario == "escalate":
        rows = []
        for i, tier_key in enumerate(ESCALATION_ORDER):
            row = single_row_from_swmm(tier_key, inp_path, offset_days=i,
                                       position=args.position)
            rows.append(row)
        df = pd.DataFrame(rows)
        preview(df, "Escalation: CLEAR → WATCH → WARNING → DANGER")

        if args.append:
            if not args.dry_run:
                append_to_merge(df, merge_path)
            else:
                print(f"  [DRY RUN] Would append {len(df)} row(s) to {merge_path}")

        if args.supabase:
            preview_hardware(df)
            insert_stress_rows_to_supabase(df, dry_run=args.dry_run, count=args.count)

        if args.csv:
            if not args.dry_run:
                save_stress_csv(df, "escalate", out_dir)
            else:
                print(f"  [DRY RUN] Would save stress CSV for escalate scenario.")

    # ── ALL ───────────────────────────────────────────────────────────
    elif args.scenario == "all":
        all_dfs: list[pd.DataFrame] = []

        # Collect one representative row per tier for --append and --supabase
        single_rows = []
        for i, tier_key in enumerate(ESCALATION_ORDER):
            if args.csv:
                df = run_swmm_simulation(inp_path, tier_key, args.days, seed=i * 10)
                preview(df, f"{TIER_PROFILES[tier_key]['label']} — {args.days} days")
                if not args.dry_run:
                    save_stress_csv(df, tier_key, out_dir)
                else:
                    print(f"  [DRY RUN] Would save stress CSV for {tier_key}.")
                all_dfs.append(df)

            if args.append or args.supabase:
                row = single_row_from_swmm(tier_key, inp_path, offset_days=i,
                                            position=args.position)
                single_rows.append(row)

        if single_rows:
            single_df = pd.DataFrame(single_rows)

            if args.append:
                preview(single_df, "All tiers — single rows")
                if not args.dry_run:
                    append_to_merge(single_df, merge_path)
                else:
                    print(f"  [DRY RUN] Would append {len(single_df)} row(s) to {merge_path}")

            if args.supabase:
                preview_hardware(single_df)
                insert_stress_rows_to_supabase(single_df, dry_run=args.dry_run, count=args.count)

        if args.csv and all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            if not args.dry_run:
                save_stress_csv(combined, "all_tiers_sequence", out_dir)
            else:
                print(f"  [DRY RUN] Would save combined stress CSV.")

    # ── SINGLE TIER ───────────────────────────────────────────────────
    else:
        tier_key = args.scenario
        label    = TIER_PROFILES[tier_key]["label"]

        if args.append or args.supabase:
            row    = single_row_from_swmm(tier_key, inp_path, offset_days=0,
                                          position=args.position)
            row_df = pd.DataFrame([row])

            if args.append:
                preview(row_df, f"{label} — single row ({args.position})")
                if not args.dry_run:
                    append_to_merge(row_df, merge_path)
                else:
                    print(f"  [DRY RUN] Would append 1 row to {merge_path}")

            if args.supabase:
                preview_hardware(row_df)
                insert_stress_rows_to_supabase(row_df, dry_run=args.dry_run, count=args.count)

        if args.csv:
            df = run_swmm_simulation(inp_path, tier_key, args.days, seed=42)
            preview(df, f"{label} — {args.days} days")
            if not args.dry_run:
                save_stress_csv(df, tier_key, out_dir)
            else:
                print(f"  [DRY RUN] Would save stress CSV for {tier_key}.")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()