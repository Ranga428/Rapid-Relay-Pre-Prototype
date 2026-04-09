"""
speed_test_runner_v2_REALTIME_DAILY.py
========================================
HYBRID PIPELINE — Real-time Ingestion + Daily Prediction
Flood EWS for Obando, Bulacan

CORE LOGIC
----------
INGESTION (Real-time, every tick):
  - Fetch ALL rows from Supabase
  - For each new row (id > last_processed_id):
    → Calibrate → Append to sensor CSV → Track in ingestion audit
  - No prediction yet, just accumulate sensor data

PREDICTION (Once daily at midnight):
  - Check if it's past midnight (00:00–00:30 UTC or configurable window)
  - If yes: Run single prediction on FULL accumulated sensor history
  - Post alert for daily prediction result
  - Log to prediction audit

BENEFITS
--------
✅ Real-time sensor data accumulation (no lag)
✅ Handles all incoming rows efficiently
✅ Daily prediction on complete historical context
✅ Separates ingestion concerns from model concerns
✅ Predictable alert schedule (once per day max)
✅ Lower compute cost (predict 1x/day, not N times/day)

CSV SCHEMAS
-----------
INGESTION AUDIT (speedtest_ingestion_audit.csv):
  run_number, run_timestamp, rows_fetched, new_rows_found, new_rows_ingested,
  last_ingested_id, last_ingested_timestamp, ingestion_errors

SENSOR DATA (speedtest_sensor_data.csv):
  timestamp, waterlevel, soil_moisture, humidity, row_created_at
  (accumulates across all runs)

PREDICTION AUDIT (speedtest_prediction_audit.csv):
  prediction_number, prediction_timestamp, rows_in_history, flood_probability,
  risk_tier, watch_threshold, warning_threshold, danger_threshold,
  fb_posted, prediction_created_at

USAGE
-----
  python speed_test_runner_v2_REALTIME_DAILY.py                   # 30s interval
  python speed_test_runner_v2_REALTIME_DAILY.py --interval 10     # 10s interval
  python speed_test_runner_v2_REALTIME_DAILY.py --once            # Single ingest + check predict
  python speed_test_runner_v2_REALTIME_DAILY.py --force-predict   # Force prediction now
  python speed_test_runner_v2_REALTIME_DAILY.py --dry-run         # Preview only
"""

# ===========================================================================
# IMPORTS
# ===========================================================================

import os
import sys
import time
import json
import uuid
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import joblib
import pytz
import matplotlib
matplotlib.use("Agg")

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

# ===========================================================================
# PATHS
# ===========================================================================

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

_ENV_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".env"))
load_dotenv(dotenv_path=_ENV_PATH)

MODEL_FILE           = os.path.join(_PROJECT_ROOT, "model", "flood_xgb_sensor.pkl")
SPEEDTEST_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "predictions")
INGESTION_AUDIT_CSV  = os.path.join(SPEEDTEST_OUTPUT_DIR, "speedtest_ingestion_audit.csv")
PREDICTION_AUDIT_CSV = os.path.join(SPEEDTEST_OUTPUT_DIR, "speedtest_prediction_audit.csv")
SPEEDTEST_PRED_CSV   = os.path.join(SPEEDTEST_OUTPUT_DIR, "speedtest_predictions.csv")
AVAILABILITY_CSV     = os.path.join(SPEEDTEST_OUTPUT_DIR, "speedtest_availability.csv")

SENSOR_CSV = Path(os.path.normpath(
    os.path.join(_PROJECT_ROOT, "data", "sensor", "speedtest_sensor_data.csv")
))

FLOOD_LOG_PATH = os.path.join(_PROJECT_ROOT, "data", "flood_event_log.csv")

# State tracking
STATE_FILE = os.path.join(SPEEDTEST_OUTPUT_DIR, "speedtest_state.json")

_ML_PIPELINE = os.path.join(_PROJECT_ROOT, "ml_pipeline")
sys.path.insert(0, _ML_PIPELINE)

# ===========================================================================
# LOGGING
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("speed_test_runner_realtime_daily")

# ===========================================================================
# CONFIG
# ===========================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

TABLE_ENV_DATA      = "obando_environmental_data"
SUPABASE_PRED_TABLE = "flood_predictions_speedtest"
SUPABASE_BATCH_SIZE = 500

COL_ID       = "id"
COL_SOIL     = "Soil Moisture"
COL_HUMIDITY = "Humidity"
COL_DISTANCE = "Final Distance"
COL_DATE     = "Date"
COL_TIME     = "Time"

_SELECT = '"id", "Date", "Time", "Soil Moisture", "Humidity", "Final Distance"'

_supabase_client = None

# Calibration constants
DIKE_HEIGHT_M        = 4.039
HW_WL_DRY            = 3.819
HW_WL_WET            = 3.999
PROXY_WL_DRY         = 0.718
PROXY_WL_WET         = 2.197
DISTANCE_MIN_VALID_M = 0.05
DISTANCE_MAX_VALID_M = 4.039

SOIL_HW_DRY    = 71.8
SOIL_HW_WET    = 85.0
SOIL_PROXY_DRY = 0.242
SOIL_PROXY_WET = 0.463

HUMIDITY_HW_MIN    = 78.5
HUMIDITY_HW_MAX    = 88.78
HUMIDITY_PROXY_MIN = 0.15
HUMIDITY_PROXY_MAX = 6.87

# Prediction config
DEFAULT_ALERT_THRESHOLD = 0.50
MIN_CONSECUTIVE_DAYS    = 2
ROLLING_MEAN_WINDOW     = 7
ROLLING_SUM_WINDOW      = 14
LAST_TRAINING_DATE      = "2024-12-31"

# Daily prediction timing
PREDICTION_HOUR_UTC = 0  # Predict at 00:00 UTC (midnight)
PREDICTION_WINDOW_MINUTES = 30  # Allow prediction window: 00:00–00:30 UTC
SENSOR_TIMEZONE = os.getenv("SENSOR_TIMEZONE", "UTC")

# Facebook
FB_PAGE_ID     = os.getenv("FB_PAGE_ID")
FB_PAGE_TOKEN  = os.getenv("FB_PAGE_TOKEN")
FB_API_VERSION = "v23.0"
FB_STATE_PATH  = os.path.join(SCRIPT_DIR, "speedtest_last_fb_prediction_ts.json")

# Allow forcing FB posts via environment (useful for CI/testing or always-on posting)
# Default to true so the runner posts predictions by default (set ALWAYS_POST_FB=false to disable)
ALWAYS_POST_FB = os.getenv("ALWAYS_POST_FB", "true").lower() in ("1", "true", "yes")

FB_TIER_MESSAGE = {
    "CLEAR": (
        "🟢 DAILY CLEAR — Obando, Bulacan\n\n"
        "No flood risk detected in today's data by Rapid Relay EWS.\n"
        "Conditions normal. No action required.\n\n"
        "Flood Probability : {prob_pct}\n"
        "Prediction Time   : {timestamp}\n\n"
        "Stay informed. Follow official DRRMO advisories."
    ),
    "WATCH": (
        "🟡 DAILY WATCH — Obando, Bulacan\n\n"
        "Elevated flood risk detected in today's data by Rapid Relay EWS.\n"
        "Monitor water levels. Be prepared to act.\n\n"
        "Flood Probability : {prob_pct}\n"
        "Prediction Time   : {timestamp}\n\n"
        "Stay safe. Follow official DRRMO advisories."
    ),
    "WARNING": (
        "🟠 DAILY WARNING — Obando, Bulacan\n\n"
        "High flood risk detected in today's data by Rapid Relay EWS.\n"
        "Prepare for possible flooding. Move valuables to higher ground.\n\n"
        "Flood Probability : {prob_pct}\n"
        "Prediction Time   : {timestamp}\n\n"
        "Stay safe. Follow official DRRMO advisories."
    ),
    "DANGER": (
        "🔴 DAILY DANGER — Obando, Bulacan\n\n"
        "IMMINENT/ONGOING flood detected in today's data by Rapid Relay EWS.\n"
        "Take immediate action. Evacuate if in flood-prone areas.\n\n"
        "Flood Probability : {prob_pct}\n"
        "Prediction Time   : {timestamp}\n\n"
        "🚨 Follow DRRMO evacuation orders immediately."
    ),
}

RISK_TIERS = {
    "CLEAR":   {"emoji": "🟢", "color": "green"},
    "WATCH":   {"emoji": "🟡", "color": "gold"},
    "WARNING": {"emoji": "🟠", "color": "orange"},
    "DANGER":  {"emoji": "🔴", "color": "red"},
}

# Session
_SESSION_ID = str(uuid.uuid4())[:8]

# ===========================================================================
# HELPERS
# ===========================================================================

def sep(title=""):
    line = "=" * 70
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def countdown(seconds: int, label: str = "Next run in") -> None:
    try:
        for remaining in range(seconds, 0, -1):
            mins, secs = divmod(remaining, 60)
            time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
            print(f"\r  ⏳  {label}: {time_str}   ", end="", flush=True)
            time.sleep(1)
        print(f"\r  ✅  Starting next run...          ", flush=True)
        time.sleep(0.3)
        print(f"\r{' ' * 50}\r", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\r{' ' * 50}\r", end="", flush=True)
        raise


def is_prediction_time() -> bool:
    """Check if current time is within the daily prediction window (UTC)."""
    now_utc = datetime.now(pytz.UTC)
    hour = now_utc.hour
    minute = now_utc.minute
    
    # Prediction window: PREDICTION_HOUR_UTC to PREDICTION_HOUR_UTC + PREDICTION_WINDOW_MINUTES
    if hour == PREDICTION_HOUR_UTC and minute < PREDICTION_WINDOW_MINUTES:
        return True
    return False


def get_last_prediction_timestamp() -> str | None:
    """Return timestamp of last successful prediction."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            return state.get("last_prediction_timestamp")
    except Exception:
        return None


def save_state(last_ingest_id: int | None = None, last_predict_ts: str | None = None) -> None:
    """Persist ingestion and prediction state."""
    os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
    
    # Read existing state
    state = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
        except Exception:
            pass
    
    # Update
    if last_ingest_id is not None:
        state["last_ingested_id"] = last_ingest_id
    if last_predict_ts is not None:
        state["last_prediction_timestamp"] = last_predict_ts
    
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log.warning("Could not save state: %s", e)


def get_last_ingested_id() -> int | None:
    """Return highest row id ingested so far."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            return state.get("last_ingested_id")
    except Exception:
        return None


def get_supabase():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            f"Supabase credentials missing.\n"
            f"  .env checked         : {_ENV_PATH}\n"
            f"  SUPABASE_URL         : {'SET' if SUPABASE_URL else 'MISSING'}\n"
            f"  SUPABASE_SERVICE_KEY : {'SET' if SUPABASE_KEY else 'MISSING'}\n"
        )
    from supabase import create_client
    _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# ===========================================================================
# SUPABASE FETCH
# ===========================================================================

def fetch_all_rows() -> list | None:
    """Fetch ALL rows from Supabase ordered by id DESC."""
    try:
        supabase = get_supabase()
        response = (
            supabase.table(TABLE_ENV_DATA)
            .select(_SELECT)
            .order("id", desc=True)
            .execute()
        )
    except Exception as exc:
        log.error("Supabase fetch failed: %s", exc)
        return None

    rows = response.data or []
    log.info("Fetched %d total row(s) from Supabase.", len(rows))
    return rows


def fetch_latest_row() -> list | None:
    """Fetch only the latest row from Supabase (ordered by id DESC, limit=1)."""
    try:
        supabase = get_supabase()
        response = (
            supabase.table(TABLE_ENV_DATA)
            .select(_SELECT)
            .order("id", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        log.error("Supabase fetch latest failed: %s", exc)
        return None

    rows = response.data or []
    log.info("Fetched %d latest row(s) from Supabase.", len(rows))
    return rows


# ===========================================================================
# CALIBRATION
# ===========================================================================

def _calibrate_waterlevel(distance_m):
    try:
        d = float(distance_m)
    except (TypeError, ValueError):
        return None
    if np.isnan(d) or d <= DISTANCE_MIN_VALID_M or d >= DISTANCE_MAX_VALID_M:
        return None
    hw_wl_m = DIKE_HEIGHT_M - d
    t = max(0.0, min(1.0, (hw_wl_m - HW_WL_DRY) / (HW_WL_WET - HW_WL_DRY)))
    return round(PROXY_WL_DRY + t * (PROXY_WL_WET - PROXY_WL_DRY), 6)


def _calibrate_soil(hw_pct) -> float:
    t = max(0.0, min(1.0, (float(hw_pct) - SOIL_HW_DRY) / (SOIL_HW_WET - SOIL_HW_DRY)))
    return round(SOIL_PROXY_DRY + t * (SOIL_PROXY_WET - SOIL_PROXY_DRY), 6)


def _calibrate_humidity(hw_rh) -> float:
    t = max(0.0, min(1.0, (float(hw_rh) - HUMIDITY_HW_MIN) / (HUMIDITY_HW_MAX - HUMIDITY_HW_MIN)))
    return round(HUMIDITY_PROXY_MIN + t * (HUMIDITY_PROXY_MAX - HUMIDITY_PROXY_MIN), 6)


def calibrate_row(row: dict) -> dict | None:
    """Calibrate a single row. Returns dict or None if invalid."""
    cal_wl = _calibrate_waterlevel(row.get(COL_DISTANCE))
    if cal_wl is None:
        log.debug(
            "Row id=%s has invalid Final Distance (%s) — skipping.",
            row.get(COL_ID), row.get(COL_DISTANCE)
        )
        return None

    return {
        "cal_waterlevel":    cal_wl,
        "cal_soil_moisture": _calibrate_soil(row.get(COL_SOIL, 0)),
        "cal_humidity":      _calibrate_humidity(row.get(COL_HUMIDITY, 0)),
    }


def _row_timestamp_to_utc_iso(row: dict) -> str | None:
    """Combine Date+Time from a Supabase row and return an ISO UTC timestamp.

    Returns string like '2026-04-09T08:04:07Z' or None if parsing fails.
    """
    date = row.get(COL_DATE)
    time = row.get(COL_TIME)
    if not date or not time:
        return None
    raw = f"{date}T{time}"
    try:
        ts = pd.to_datetime(raw, utc=False, errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(raw, utc=True, errors="coerce")
        if pd.isna(ts):
            return None

        # If naive, localize to configured sensor timezone then convert to UTC
        if ts.tzinfo is None:
            try:
                tz = pytz.timezone(SENSOR_TIMEZONE)
                ts = tz.localize(ts)
            except Exception:
                # Fallback: assume UTC
                ts = ts.replace(tzinfo=pytz.UTC)

        ts_utc = ts.astimezone(pytz.UTC)
        # Use Z format for readability and robust parsing later
        return ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def get_availability_state() -> tuple[int, int]:
    """Return cumulative (uptime_ticks, total_ticks) from availability CSV."""
    if not os.path.exists(AVAILABILITY_CSV):
        return 0, 0
    try:
        df = pd.read_csv(AVAILABILITY_CSV)
        if df.empty:
            return 0, 0
        last = df.iloc[-1]
        return int(last.get("uptime_ticks", 0)), int(last.get("total_ticks", 0))
    except Exception:
        return 0, 0


def write_availability_row(status: str, supabase_row_id: int | None = None, dry_run: bool = False) -> None:
    """Append one availability row to the availability CSV."""
    uptime_prev, total_prev = get_availability_state()
    total_ticks = total_prev + 1
    uptime_ticks = uptime_prev + (1 if status == "ACTIVE" else 0)
    avail_pct = round((uptime_ticks / total_ticks) * 100, 4) if total_ticks > 0 else 0.0

    # session-local tick number
    tick_number = 1
    if os.path.exists(AVAILABILITY_CSV):
        try:
            df = pd.read_csv(AVAILABILITY_CSV)
            tick_number = int(df[df["session_id"] == _SESSION_ID].shape[0]) + 1
        except Exception:
            tick_number = 1

    row = {
        "session_id":      _SESSION_ID,
        "tick_number":     tick_number,
        "tick_timestamp":  now_iso(),
        "status":          status,
        "supabase_row_id": supabase_row_id,
        "uptime_ticks":    uptime_ticks,
        "total_ticks":     total_ticks,
        "availability_pct": avail_pct,
    }

    df_row = pd.DataFrame([row])
    os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(AVAILABILITY_CSV):
        df_row.to_csv(AVAILABILITY_CSV, index=False)
    else:
        df_row.to_csv(AVAILABILITY_CSV, mode="a", header=False, index=False)

    log.info(
        "Availability → tick=%d  status=%-7s  Aₒ=%.2f%%  (%d/%d ticks)",
        tick_number, status, avail_pct, uptime_ticks, total_ticks
    )


def save_speedtest_prediction_row(pred: dict, sensor_timestamp: str | None = None, transmission_speed_s: float | None = None, dry_run: bool = False) -> None:
    """Append a single prediction summary row to SPEEDTEST_PRED_CSV."""
    row = {
        "timestamp":            pred.get("timestamp"),
        "flood_probability":    pred.get("probability"),
        "risk_tier":            pred.get("tier"),
        "watch_threshold":      pred.get("watch_threshold"),
        "warning_threshold":    pred.get("warning_threshold"),
        "danger_threshold":     pred.get("danger_threshold"),
        "transmission_speed_s": transmission_speed_s,
        "sensor_timestamp":     sensor_timestamp,
        "prediction_created_at": pred.get("prediction_created_at"),
        "row_created_at":       now_iso(),
    }

    df_row = pd.DataFrame([row])
    os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
    if dry_run:
        log.info("[DRY RUN] Would append to %s:\n%s", SPEEDTEST_PRED_CSV, df_row.to_string(index=False))
        return

    if not os.path.exists(SPEEDTEST_PRED_CSV):
        df_row.to_csv(SPEEDTEST_PRED_CSV, index=False)
    else:
        df_row.to_csv(SPEEDTEST_PRED_CSV, mode="a", header=False, index=False)


def sync_prediction_to_supabase(pred: dict, dry_run: bool = False) -> bool:
    """Sync a single prediction to Supabase table `flood_predictions_speedtest`.

    Returns True on success (or dry-run), False on failure.
    """
    supabase = None
    try:
        supabase = get_supabase()
    except Exception as e:
        log.error("Supabase client unavailable: %s", e)
        return False

    # Use prediction_created_at (UTC ISO) if available
    ts = pred.get("prediction_created_at") or pred.get("timestamp")
    if not ts:
        # fallback to now UTC
        ts = datetime.now(pytz.UTC).isoformat()

    record = {
        "timestamp": ts,
        "flood_probability": float(pred.get("probability", 0.0)),
        "risk_tier": str(pred.get("tier", "")),
    }

    if dry_run:
        log.info("[DRY RUN] Would upsert prediction to Supabase: %s", record)
        return True

    try:
        # Use upsert on conflict timestamp to avoid unique constraint errors
        res = supabase.table(SUPABASE_PRED_TABLE).upsert([record], on_conflict="timestamp").execute()
        # supabase client returns .data on success
        if getattr(res, "error", None):
            log.error("Supabase upsert error: %s", res.error)
            return False
        log.info("Synced prediction to Supabase (timestamp=%s)", ts)
        return True
    except Exception as e:
        log.error("Supabase upsert failed: %s", e)
        return False


def _append_ingestion_audit_single(run_number: int, rows_fetched: int, new_rows_found: int, new_rows_ingested: int, last_ingested_id: int | None, last_ingested_timestamp: str | None, dry_run: bool = False) -> None:
    """Append a single ingestion audit row (used by realtime handler)."""
    audit = {
        "run_number": run_number,
        "run_timestamp": now_iso(),
        "rows_fetched": rows_fetched,
        "new_rows_found": new_rows_found,
        "new_rows_ingested": new_rows_ingested,
        "last_ingested_id": last_ingested_id,
        "last_ingested_timestamp": last_ingested_timestamp,
        "ingestion_errors": new_rows_found - new_rows_ingested,
    }
    df = pd.DataFrame([audit])
    os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
    if dry_run:
        log.info("[DRY RUN] Would append ingestion audit:\n%s", df.to_string(index=False))
        return
    if not os.path.exists(INGESTION_AUDIT_CSV):
        df.to_csv(INGESTION_AUDIT_CSV, index=False)
    else:
        df.to_csv(INGESTION_AUDIT_CSV, mode="a", header=False, index=False)


def append_to_sensor_csv(row: dict, cal: dict, dry_run: bool = False) -> bool:
    """
    Append calibrated row to sensor CSV.
    Returns True if appended, False if skipped.
    """
    # Normalize Date+Time to UTC ISO (Z) for robust parsing later
    ts_iso = _row_timestamp_to_utc_iso(row)
    if ts_iso is None:
        log.debug("Could not parse timestamp for row id=%s — skipping.", row.get(COL_ID))
        return False

    new_row = pd.DataFrame([{
        "timestamp":      ts_iso,
        "waterlevel":     cal["cal_waterlevel"],
        "soil_moisture":  cal["cal_soil_moisture"],
        "humidity":       cal["cal_humidity"],
        "row_created_at": now_iso(),
    }])

    if dry_run:
        log.info("[DRY RUN] Would append to sensor CSV:\n%s", new_row.to_string(index=False))
        return True

    SENSOR_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not SENSOR_CSV.exists():
        new_row.to_csv(SENSOR_CSV, index=False)
        log.debug("Created sensor CSV: %s", SENSOR_CSV)
        return True

    # Read existing timestamps and normalize them to UTC ISO for accurate dedupe
    try:
        existing = pd.read_csv(SENSOR_CSV, usecols=["timestamp"])['timestamp'].astype(str)
        existing_norm = pd.to_datetime(existing, utc=True, errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        if ts_iso in existing_norm.values:
            log.debug("Timestamp %s already in sensor CSV — skipping.", ts_iso)
            return False
    except Exception as e:
        log.debug("Could not read existing sensor CSV timestamps (%s) — proceeding to append.", e)

    new_row.to_csv(SENSOR_CSV, mode="a", header=False, index=False)
    log.debug("Appended to sensor CSV: %s", ts_iso)
    return True


# ===========================================================================
# PREDICTION
# ===========================================================================

_model_cache = None


def load_model():
    global _model_cache, LAST_TRAINING_DATE
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

    artifact     = joblib.load(MODEL_FILE)
    model        = artifact["model"]
    feature_cols = artifact["feature_columns"]
    threshold    = artifact.get("threshold", DEFAULT_ALERT_THRESHOLD)
    watch_t      = artifact.get("watch_threshold", threshold)
    warn_t       = artifact.get("warning_threshold", threshold + 0.10)
    LAST_TRAINING_DATE = artifact.get("last_training_date", LAST_TRAINING_DATE)

    log.info("Model loaded: %s", os.path.basename(MODEL_FILE))
    log.info("  Features: %d  WATCH: %.2f  WARNING: %.2f", len(feature_cols), watch_t, warn_t)

    _model_cache = (model, feature_cols, watch_t, warn_t)
    return _model_cache


def append_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    if "max_waterlevel_24h" not in df.columns:
        return df
    df = df.copy()
    if "waterlevel_mean_7d" not in df.columns:
        df["waterlevel_mean_7d"] = (
            df["max_waterlevel_24h"].rolling(ROLLING_MEAN_WINDOW, min_periods=1).mean()
        )
    if "waterlevel_cumrise_14d" not in df.columns:
        df["waterlevel_cumrise_14d"] = (
            df["max_waterlevel_24h"].rolling(ROLLING_SUM_WINDOW, min_periods=1).sum()
        )
    return df


def classify_prob(prob: float, watch_t: float, warn_t: float, danger_t: float) -> str:
    if   prob >= danger_t: return "DANGER"
    elif prob >= warn_t:   return "WARNING"
    elif prob >= watch_t:  return "WATCH"
    else:                  return "CLEAR"


def apply_consecutive_filter(results: pd.DataFrame, warn_t: float) -> pd.DataFrame:
    if MIN_CONSECUTIVE_DAYS <= 1:
        return results
    results = results.copy()
    probs   = results["flood_probability"].values
    tiers   = results["risk_tier"].tolist()
    streak  = 0
    for i, p in enumerate(probs):
        streak = streak + 1 if p >= warn_t else 0
        if streak < MIN_CONSECUTIVE_DAYS and tiers[i] in ("WARNING", "DANGER"):
            tiers[i] = "WATCH"
    results["risk_tier"] = tiers
    return results


def run_prediction() -> dict | None:
    """Run prediction on full sensor history. Returns dict or None."""
    from prepare_dataset import load_sensor
    from feature_engineering import build_features

    if not SENSOR_CSV.exists():
        log.error("Sensor CSV not found — cannot predict.")
        return None

    try:
        model, feature_cols, watch_t, warn_t = load_model()
    except FileNotFoundError as e:
        log.error("Cannot load model: %s", e)
        return None

    try:
        sensor_df, freq = load_sensor(sensor_path=str(SENSOR_CSV))
        features = build_features(
            sensor_df, freq=freq, mode="sensor",
            flood_log_path=FLOOD_LOG_PATH if os.path.exists(FLOOD_LOG_PATH) else None
        )
        features = append_rolling_features(features)
    except Exception as e:
        log.error("Feature engineering failed: %s", e)
        return None

    core_cols = ["waterlevel", "soil_moisture", "humidity"]
    present   = [c for c in core_cols if c in features.columns]
    features  = features.dropna(subset=present, how="any")

    training_cutoff = pd.Timestamp(LAST_TRAINING_DATE, tz="UTC")
    features = features[features.index > training_cutoff]

    if features.empty:
        log.warning("No feature rows after training cutoff — cannot predict.")
        return None

    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        log.error("Missing feature columns: %s", missing)
        return None

    danger_t = warn_t + 0.10
    X        = features[feature_cols].values
    probs    = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "flood_probability": probs.round(4),
        "risk_tier": [classify_prob(p, watch_t, warn_t, danger_t) for p in probs],
        "watch_threshold":   watch_t,
        "warning_threshold": warn_t,
        "danger_threshold":  danger_t,
    }, index=features.index)

    results = apply_consecutive_filter(results, warn_t)

    latest      = results.iloc[-1]
    latest_tier = latest["risk_tier"]
    latest_prob = float(latest["flood_probability"])
    pred_ts     = datetime.now(pytz.UTC).isoformat()
    pred_epoch  = time.time()

    log.info(
        "Prediction → %s %s  (%.1f%%)  on %d rows",
        RISK_TIERS[latest_tier]["emoji"], latest_tier, latest_prob * 100, len(features)
    )

    return {
        "tier":                  latest_tier,
        "probability":           latest_prob,
        "timestamp":             pred_ts,
        "prediction_created_at": pred_ts,
        "pred_epoch":            pred_epoch,
        "watch_threshold":       watch_t,
        "warning_threshold":     warn_t,
        "danger_threshold":      danger_t,
        "rows_in_history":       len(features),
    }


# ===========================================================================
# FACEBOOK ALERT
# ===========================================================================

def _load_fb_last_posted():
    try:
        if not os.path.exists(FB_STATE_PATH):
            return None
        with open(FB_STATE_PATH, "r") as f:
            return json.load(f).get("last_posted_ts")
    except Exception:
        return None


def _save_fb_last_posted(timestamp: str) -> None:
    try:
        with open(FB_STATE_PATH, "w") as f:
            json.dump({"last_posted_ts": timestamp}, f)
    except Exception as e:
        log.warning("Could not save FB state: %s", e)


def _mask_token(tok: str | None) -> str:
    """Return a masked version of a token for safe logging."""
    if not tok:
        return "<missing>"
    s = str(tok)
    if len(s) <= 10:
        return s[:4] + "..." + s[-2:]
    return s[:6] + "..." + s[-4:]


def fb_token_info(dry_run: bool = False) -> dict | None:
    """Perform a lightweight token/info check against Graph API `/me`.

    Returns parsed JSON on success, or a dict with `_raw` on non-JSON response.
    Returns None if token missing or request fails.
    """
    if not FB_PAGE_TOKEN:
        log.error("FB token missing — cannot run token diagnostics.")
        return None

    url = f"https://graph.facebook.com/{FB_API_VERSION}/me"
    params = {"access_token": FB_PAGE_TOKEN, "fields": "id,name"}
    if dry_run:
        log.info("[DRY RUN] FB token diagnostics: GET %s params=%s", url, {"access_token": _mask_token(FB_PAGE_TOKEN), "fields": "id,name"})
        return None

    try:
        res = requests.get(url, params=params, timeout=10)
        try:
            data = res.json()
        except ValueError:
            data = {"_raw": res.text, "status_code": res.status_code}

        # Log token info at info/debug level for diagnostics
        try:
            if isinstance(data, dict) and data.get("error"):
                log.error("FB token diagnostics failed: status=%s body=%s", res.status_code, data)
            else:
                log.info("FB token valid: status=%s body=%s", res.status_code, data)
        except Exception:
            log.debug("FB token diagnostics raw: %s", data)

        return data
    except requests.Timeout:
        log.error("FB token diagnostics timed out")
        return None
    except Exception as e:
        log.exception("FB token diagnostics failed: %s", e)
        return None


def send_fb_alert(
    tier: str,
    probability: float,
    timestamp: str,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """Send daily alert to Facebook. Only post once per day per tier."""
    if not FB_PAGE_ID or not FB_PAGE_TOKEN:
        log.error(
            "FB credentials missing — skipping. FB_PAGE_ID=%s FB_PAGE_TOKEN=%s",
            'SET' if FB_PAGE_ID else 'MISSING', 'SET' if FB_PAGE_TOKEN else 'MISSING'
        )
        return False

    last = _load_fb_last_posted()
    today = datetime.now().strftime("%Y-%m-%d")
    # Only post once per day unless forced
    if not force and last and last.startswith(today):
        log.info("FB: already posted today — skipping.")
        return True
    if force:
        log.info("FB: force post enabled — bypassing once-per-day guard")

    template = FB_TIER_MESSAGE.get(tier, FB_TIER_MESSAGE["WARNING"])
    message  = template.format(prob_pct=f"{probability:.1%}", timestamp=timestamp)
    url      = f"https://graph.facebook.com/{FB_API_VERSION}/{FB_PAGE_ID}/feed"

    log.info("Posting daily %s alert to Facebook...", tier)

    if dry_run:
        log.info("[DRY RUN] Would post to Facebook:\n%s", message)
        _save_fb_last_posted(now_iso())
        return True

    try:
        # Diagnostic: log masked token and a short preview of the message
        log.debug("FB POST -> url=%s token=%s message_len=%d", url, _mask_token(FB_PAGE_TOKEN), len(message))

        # Run token diagnostics once before posting (helps reveal common issues)
        try:
            token_info = fb_token_info(dry_run=dry_run)
            if token_info is None and not dry_run:
                log.debug("FB token diagnostics returned no data — continuing to post attempt")
        except Exception:
            log.debug("FB token diagnostics raised an exception; continuing to post")

        payload = {"message": message, "access_token": FB_PAGE_TOKEN}
        res = requests.post(url, data=payload, timeout=15)

        # Response diagnostics: headers, status, raw body
        try:
            data = res.json()
        except ValueError:
            data = {"_raw": res.text, "status_code": res.status_code}

        log.debug("FB response status=%s headers=%s", res.status_code, dict(res.headers))
        log.debug("FB response body=%s", data)

        # Graph API error handling: include trace id and full error object in logs
        if res.status_code >= 400 or (isinstance(data, dict) and data.get("error")):
            err = data.get("error") if isinstance(data, dict) else data
            err_type = err.get("type") if isinstance(err, dict) else None
            err_code = err.get("code") if isinstance(err, dict) else None
            err_msg = err.get("message") if isinstance(err, dict) else str(err)
            trace = res.headers.get("x-fb-trace-id") or res.headers.get("x-fb-debug") or "-"
            log.error("FB API error status=%s type=%s code=%s trace=%s message=%s full=%s", res.status_code, err_type, err_code, trace, err_msg, err)
            return False

        post_id = data.get("id") if isinstance(data, dict) else None
        trace = res.headers.get("x-fb-trace-id") or "-"
        log.info("FB post successful — post id: %s trace=%s", post_id or "<unknown>", trace)
        _save_fb_last_posted(now_iso())
        return True
    except requests.Timeout:
        log.error("FB request timed out.")
        return False
    except Exception as e:
        log.exception("FB unexpected error: %s", e)
        return False


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_once(
    run_number: int,
    ingest_number: int,
    force_predict: bool = False,
    dry_run: bool = False,
    force_fb: bool = False,
) -> tuple[int, bool]:
    """
    Run one tick: ingest all new rows, optionally predict.
    
    Returns:
        (new_rows_ingested, prediction_ran)
    """
    tick_start = time.time()
    run_ts     = now_iso()

    sep(f"TICK #{run_number}  |  {run_ts}")

    # ------------------------------------------------------------------
    # PHASE 1: REAL-TIME INGESTION
    # ------------------------------------------------------------------
    sep("PHASE 1 — Real-time Ingestion")

    all_rows = fetch_latest_row()
    if all_rows is None:
        log.error("Fetch failed")
        return 0, False

    # Provide a richer scan summary: fetched count, id range, last ingested id,
    # number of new rows, id range for new rows, missing timestamps in new rows,
    # and a sample newest/oldest timestamp (UTC) to help debugging.
    total_fetched = len(all_rows)
    if total_fetched == 0:
        log.info("Scanned Supabase: 0 rows returned.")
        last_ingest_id = get_last_ingested_id()
        new_rows = []
    else:
        try:
            fetched_high = all_rows[0].get(COL_ID) or "<none>"
            fetched_low = all_rows[-1].get(COL_ID) or "<none>"
        except Exception:
            fetched_high = fetched_low = "<err>"

        last_ingest_id = get_last_ingested_id()
        if last_ingest_id is None:
            new_rows = all_rows
        else:
            new_rows = [r for r in all_rows if (r.get(COL_ID) or 0) > (last_ingest_id or 0)]

        new_count = len(new_rows)
        if new_count:
            try:
                new_high = new_rows[0].get(COL_ID) or "<none>"
                new_low = new_rows[-1].get(COL_ID) or "<none>"
            except Exception:
                new_high = new_low = "<err>"
        else:
            new_high = new_low = "-"

        def _sample_ts(r):
            try:
                return _row_timestamp_to_utc_iso(r) or "<no-ts>"
            except Exception:
                return "<err>"

        sample_newest_ts = _sample_ts(all_rows[0])
        sample_oldest_ts = _sample_ts(all_rows[-1])
        missing_ts_in_new = sum(1 for r in new_rows if not r.get(COL_DATE) or not r.get(COL_TIME))

        log.info(
            "Scanned Supabase: fetched=%d ids=%s..%s last_ingest_id=%s new=%d ids=%s..%s missing_ts_in_new=%d newest_ts=%s oldest_ts=%s",
            total_fetched, fetched_low, fetched_high, last_ingest_id, new_count, new_low, new_high, missing_ts_in_new, sample_newest_ts, sample_oldest_ts
        )

    # Ingest new rows
    new_ingested = 0
    last_row_data = None

    for row in new_rows:
        row_id = row.get(COL_ID)
        
        # Calibrate
        cal = calibrate_row(row)
        if cal is None:
            log.debug("Row id=%s calibration failed", row_id)
            continue
        
        # Append to sensor CSV
        if append_to_sensor_csv(row, cal, dry_run=dry_run):
            new_ingested += 1
            last_row_data = {
                "row_id": row_id,
                "date": row.get(COL_DATE),
                "time": row.get(COL_TIME),
                "cal": cal,
            }

    if new_ingested > 0:
        log.info("✅ Ingested %d new row(s)", new_ingested)
        if not dry_run:
            save_state(last_ingest_id=last_row_data["row_id"])

    # Write ingestion audit
    if not dry_run:
        ingestion_audit = {
            "run_number": run_number,
            "run_timestamp": run_ts,
            "rows_fetched": len(all_rows),
            "new_rows_found": len(new_rows),
            "new_rows_ingested": new_ingested,
            "last_ingested_id": last_row_data["row_id"] if last_row_data else None,
            "last_ingested_timestamp": (
                f"{last_row_data['date']} {last_row_data['time']}"
                if last_row_data else None
            ),
            "ingestion_errors": len(new_rows) - new_ingested,
        }
        os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
        audit_df = pd.DataFrame([ingestion_audit])
        if not os.path.exists(INGESTION_AUDIT_CSV):
            audit_df.to_csv(INGESTION_AUDIT_CSV, index=False)
        else:
            audit_df.to_csv(INGESTION_AUDIT_CSV, mode="a", header=False, index=False)
        log.info("Ingestion audit recorded")

    # ------------------------------------------------------------------
    # PHASE 2: DAILY PREDICTION (once per day, within window)
    # ------------------------------------------------------------------
    prediction_ran = False

    # Decide whether to run daily prediction.
    # Run if forced, in the UTC prediction window, or if we've just ingested rows
    # that belong to a new UTC date compared to the last prediction.
    sep_flag = False
    last_pred_ts_raw = get_last_prediction_timestamp()
    last_pred_date = None
    if last_pred_ts_raw:
        try:
            parsed = pd.to_datetime(last_pred_ts_raw, utc=True, errors="coerce")
            if not pd.isna(parsed):
                last_pred_date = parsed.strftime("%Y-%m-%d")
            else:
                # Fallback: assume stored value may already be a date string
                last_pred_date = str(last_pred_ts_raw)[:10]
        except Exception:
            last_pred_date = str(last_pred_ts_raw)[:10]

    should_predict = False
    if force_predict:
        should_predict = True
    elif is_prediction_time():
        should_predict = True
    else:
        # If we ingested new rows, check whether the latest ingested row has a
        # different UTC date than the last prediction — if so, run prediction now.
        if new_ingested > 0 and last_row_data:
            tmp_row = {COL_DATE: last_row_data.get("date"), COL_TIME: last_row_data.get("time")}
            last_row_iso = _row_timestamp_to_utc_iso(tmp_row)
            last_row_date_utc = None
            if last_row_iso:
                try:
                    last_row_date_utc = pd.to_datetime(last_row_iso, utc=True).strftime("%Y-%m-%d")
                except Exception:
                    last_row_date_utc = str(last_row_iso)[:10]
            # Run prediction if last prediction date is missing or different
            if last_row_date_utc and last_row_date_utc != last_pred_date:
                should_predict = True

    if should_predict:
        sep("PHASE 2 — Daily Prediction")
        sep_flag = True
        # Double-check: avoid duplicate prediction for the same UTC date
        if last_pred_date and new_ingested > 0 and last_row_data:
            try:
                tmp_row = {COL_DATE: last_row_data.get("date"), COL_TIME: last_row_data.get("time")}
                last_row_iso = _row_timestamp_to_utc_iso(tmp_row)
                last_row_date_utc = pd.to_datetime(last_row_iso, utc=True).strftime("%Y-%m-%d") if last_row_iso else None
                if last_row_date_utc and last_row_date_utc == last_pred_date and not force_predict:
                    log.info("Already predicted for UTC date %s — skipping.", last_pred_date)
                    should_predict = False
            except Exception:
                pass

    if should_predict:
        log.info("Running daily prediction on accumulated history...")
        prediction = run_prediction()

        if prediction is None:
            log.error("Prediction failed")
        else:
            prediction_ran = True
            tier = prediction["tier"]
            prob = prediction["probability"]

            # Compute sensor timestamp and transmission speed (if possible)
            sensor_iso = None
            transmission_speed_s = None
            if last_row_data:
                try:
                    sensor_iso = _row_timestamp_to_utc_iso({COL_DATE: last_row_data.get("date"), COL_TIME: last_row_data.get("time")})
                    if sensor_iso:
                        sensor_dt = pd.to_datetime(sensor_iso, utc=True)
                        pred_dt = pd.to_datetime(prediction.get("prediction_created_at"), utc=True)
                        transmission_speed_s = (pred_dt.timestamp() - sensor_dt.timestamp())
                except Exception:
                    sensor_iso = None
                    transmission_speed_s = None

            # Save summary prediction to speedtest_predictions.csv
            try:
                save_speedtest_prediction_row(prediction, sensor_timestamp=sensor_iso, transmission_speed_s=transmission_speed_s, dry_run=dry_run)
                # Also sync to Supabase
                try:
                    sync_prediction_to_supabase(prediction, dry_run=dry_run)
                except Exception as e:
                    log.debug("Could not sync prediction to Supabase: %s", e)
            except Exception as e:
                log.debug("Could not save speedtest prediction row: %s", e)

            # Send alert
            fb_posted = send_fb_alert(
                tier=tier,
                probability=prob,
                timestamp=prediction["timestamp"],
                dry_run=dry_run,
                force=force_fb,
            )

            # Write prediction audit
            if not dry_run:
                pred_audit = {
                    "prediction_number": ingest_number,
                    "prediction_timestamp": prediction["timestamp"],
                    "rows_in_history": prediction["rows_in_history"],
                    "flood_probability": prob,
                    "risk_tier": tier,
                    "watch_threshold": prediction.get("watch_threshold"),
                    "warning_threshold": prediction.get("warning_threshold"),
                    "danger_threshold": prediction.get("danger_threshold"),
                    "fb_posted": fb_posted,
                    "prediction_created_at": prediction["prediction_created_at"],
                }
                os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
                pred_df = pd.DataFrame([pred_audit])
                if not os.path.exists(PREDICTION_AUDIT_CSV):
                    pred_df.to_csv(PREDICTION_AUDIT_CSV, index=False)
                else:
                    pred_df.to_csv(PREDICTION_AUDIT_CSV, mode="a", header=False, index=False)
                # Update state with UTC date only
                try:
                    save_state(last_predict_ts=datetime.now(pytz.UTC).strftime("%Y-%m-%d"))
                except Exception:
                    save_state(last_predict_ts=prediction.get("prediction_created_at", ""))

            # Summary
            sep("PREDICTION RESULT")
            print(f"  {RISK_TIERS[tier]['emoji']}  {tier}")
            print(f"  Probability: {prob:.1%}")
            print(f"  History rows: {prediction['rows_in_history']}")
            print(f"  FB posted: {fb_posted}")

    elapsed = time.time() - tick_start

    # Determine availability status for this tick
    try:
        if all_rows is None:
            status = "FAILED"
        elif new_ingested > 0:
            # If prediction was requested this tick, require successful prediction
            if 'should_predict' in locals() and should_predict:
                status = "ACTIVE" if prediction_ran else "FAILED"
            else:
                status = "ACTIVE"
        else:
            status = "SKIPPED"
    except Exception:
        status = "FAILED"

    supabase_row_id = last_row_data["row_id"] if last_row_data else None
    try:
        write_availability_row(status=status, supabase_row_id=supabase_row_id, dry_run=dry_run)
    except Exception as e:
        log.debug("Could not write availability row: %s", e)

    sep("TICK COMPLETE")
    print(f"  New rows ingested: {new_ingested}")
    print(f"  Prediction ran: {prediction_ran}")
    print(f"  Elapsed: {elapsed:.2f}s")
    sep()

    return new_ingested, prediction_ran


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time Ingestion + Daily Prediction Pipeline"
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Loop interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once and exit"
    )
    parser.add_argument(
        "--force-predict", action="store_true",
        help="Force daily prediction now (ignore time window)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview only, no writes"
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="Run a realtime listener that ingests on new INSERTs"
    )
    parser.add_argument(
        "--predict-on-insert", action="store_true",
        help="If set with --realtime, run prediction on every new INSERT (expensive)"
    )
    parser.add_argument(
        "--realtime-interval", type=int, default=5,
        help="Polling interval in seconds when realtime subscription unavailable (default: 5)"
    )
    parser.add_argument(
        "--test-fb", action="store_true",
        help="Test Facebook posting once and exit (respect --dry-run)"
    )
    parser.add_argument(
        "--fb-debug", action="store_true",
        help="Run FB token diagnostics and a verbose test post (respect --dry-run)"
    )
    parser.add_argument(
        "--force-fb", action="store_true",
        help="Ignore once-per-day FB guard and post every prediction"
    )
    args = parser.parse_args()

    print(f"\n  🚀  Real-time Ingestion + Daily Prediction Pipeline")
    print(f"  📂  Sensor CSV   : {SENSOR_CSV}")
    print(f"  📋  Ingestion    : {INGESTION_AUDIT_CSV}")
    print(f"  📊  Predictions  : {PREDICTION_AUDIT_CSV}")
    print(f"  ⏰  Predict time : {PREDICTION_HOUR_UTC:02d}:00–{PREDICTION_HOUR_UTC:02d}:{PREDICTION_WINDOW_MINUTES:02d} UTC")
    print(f"  🔑  Session ID   : {_SESSION_ID}")
    print(f"  Press Ctrl+C to stop.\n")

    # If the user requested a Facebook test, run it now and exit.
    if args.test_fb:
        print(f"\n  🔎  Testing Facebook posting (dry_run={args.dry_run})...")
        ok = send_fb_alert(tier="WARNING", probability=0.5, timestamp=now_iso(), dry_run=args.dry_run, force=(args.force_fb or ALWAYS_POST_FB))
        print(f"\n  Facebook test result: {'OK' if ok else 'FAILED'}\n")
        sys.exit(0)

    if args.fb_debug:
        print(f"\n  🐞  Running Facebook diagnostics (dry_run={args.dry_run})...")
        # Run token diagnostics first
        info = fb_token_info(dry_run=args.dry_run)
        print("\n  FB token diagnostics result (check logs for full details):")
        print(f"    token_present = {'yes' if FB_PAGE_TOKEN else 'no'}")
        if info is not None:
            print(f"    token_info = {info}")
        else:
            print("    token_info = <none or dry-run>")

        # Attempt a verbose test post (respects --dry-run)
        ok = send_fb_alert(tier="WARNING", probability=0.5, timestamp=now_iso(), dry_run=args.dry_run, force=(args.force_fb or ALWAYS_POST_FB))
        print(f"\n  Facebook post attempt result: {'OK' if ok else 'FAILED'}\n")
        sys.exit(0)

    if args.once:
        run_number = 1
        ingest_number = 1
        new_rows, pred_ran = run_once(
            run_number=run_number,
            ingest_number=ingest_number,
            force_predict=args.force_predict,
            dry_run=args.dry_run,
            force_fb=(args.force_fb or ALWAYS_POST_FB),
        )
        print(f"\n  Result: {new_rows} rows ingested, prediction={'RAN' if pred_ran else 'SKIPPED'}\n")
        sys.exit(0)

    # Continuous loop
    run_number = 1
    ingest_number = 1
    # If realtime mode requested, start realtime listener instead of polling loop
    if args.realtime:
        def start_realtime_listener(poll_interval: int = 5, dry_run: bool = False, predict_on_insert: bool = False):
            supabase = get_supabase()

            def handle_payload(payload):
                # normalize different payload shapes from client
                new = None
                if isinstance(payload, dict):
                    new = payload.get("new") or payload.get("record")
                    if new is None:
                        payload_inner = payload.get("payload")
                        if isinstance(payload_inner, dict):
                            new = payload_inner.get("new") or payload_inner.get("record")

                if not new:
                    log.debug("Realtime payload without record: %s", payload)
                    return

                row_id = new.get(COL_ID)
                log.info("Realtime INSERT id=%s", row_id)

                # Calibrate
                cal = calibrate_row(new)
                if cal is None:
                    log.debug("Realtime row id=%s calibration failed", row_id)
                    write_availability_row("FAILED", supabase_row_id=row_id, dry_run=dry_run)
                    _append_ingestion_audit_single(run_number=1, rows_fetched=1, new_rows_found=1, new_rows_ingested=0, last_ingested_id=None, last_ingested_timestamp=None, dry_run=dry_run)
                    return

                appended = append_to_sensor_csv(new, cal, dry_run=dry_run)
                if appended and not dry_run:
                    try:
                        save_state(last_ingest_id=row_id)
                    except Exception:
                        pass

                # Optionally run prediction on every insert
                pred_ok = False
                if predict_on_insert:
                    pred = run_prediction()
                    if pred is not None:
                        pred_ok = True
                        # compute sensor ts and speed
                        sensor_iso = _row_timestamp_to_utc_iso(new)
                        transmission_speed_s = None
                        try:
                            if sensor_iso:
                                sensor_dt = pd.to_datetime(sensor_iso, utc=True)
                                pred_dt = pd.to_datetime(pred.get("prediction_created_at"), utc=True)
                                transmission_speed_s = pred_dt.timestamp() - sensor_dt.timestamp()
                        except Exception:
                            transmission_speed_s = None
                        try:
                            save_speedtest_prediction_row(pred, sensor_timestamp=sensor_iso, transmission_speed_s=transmission_speed_s, dry_run=dry_run)
                        except Exception as e:
                            log.debug("Could not save realtime prediction row: %s", e)
                        try:
                            sync_prediction_to_supabase(pred, dry_run=dry_run)
                        except Exception as e:
                            log.debug("Could not sync realtime prediction to Supabase: %s", e)
                        # Attempt to post realtime prediction to Facebook (if configured)
                        try:
                            fb_ok = send_fb_alert(
                                tier=pred.get("tier"),
                                probability=pred.get("probability"),
                                timestamp=pred.get("prediction_created_at") or pred.get("timestamp"),
                                dry_run=dry_run,
                                force=(predict_on_insert and (args.force_fb or ALWAYS_POST_FB)),
                            )
                            log.info("Realtime FB post attempted: %s", "posted" if fb_ok else "not-posted")
                            if not fb_ok:
                                log.debug("Realtime FB post returned False — check FB credentials / token or use --fb-debug for diagnostics")
                        except Exception as e:
                            log.exception("Realtime FB post failed: %s", e)

                # Ingestion audit for this single insert
                try:
                    _append_ingestion_audit_single(run_number=1, rows_fetched=1, new_rows_found=1, new_rows_ingested=1 if appended else 0, last_ingested_id=row_id if appended else None, last_ingested_timestamp=(f"{new.get(COL_DATE)} {new.get(COL_TIME)}" if appended else None), dry_run=dry_run)
                except Exception:
                    pass

                # Availability
                status = "ACTIVE" if appended and (not predict_on_insert or pred_ok) else ("SKIPPED" if not appended else "FAILED")
                try:
                    write_availability_row(status=status, supabase_row_id=row_id, dry_run=dry_run)
                except Exception as e:
                    log.debug("Could not write realtime availability row: %s", e)

            # Try realtime subscribe first
            try:
                log.info("Starting Supabase realtime listener on table '%s'...", TABLE_ENV_DATA)
                supabase.from_(TABLE_ENV_DATA).on("INSERT", lambda payload: handle_payload(payload)).subscribe()
                # keep process alive
                while True:
                    time.sleep(poll_interval)
            except Exception as e:
                log.warning("Realtime subscribe failed (%s). Falling back to polling every %ds", e, poll_interval)
                # Fallback polling loop
                last_seen = get_last_ingested_id() or 0
                try:
                    while True:
                        rows = fetch_latest_row() or []
                        total_fetched = len(rows)
                        if total_fetched == 0:
                            log.debug("Realtime poll: 0 rows returned from Supabase.")
                            new_rows = []
                        else:
                            try:
                                fetched_high = rows[0].get(COL_ID) or "<none>"
                                fetched_low = rows[-1].get(COL_ID) or "<none>"
                            except Exception:
                                fetched_high = fetched_low = "<err>"
                            new_rows = [r for r in rows if (r.get(COL_ID) or 0) > (last_seen or 0)]
                            new_count = len(new_rows)
                            if new_count:
                                try:
                                    new_high = new_rows[0].get(COL_ID) or "<none>"
                                    new_low = new_rows[-1].get(COL_ID) or "<none>"
                                except Exception:
                                    new_high = new_low = "<err>"
                            else:
                                new_high = new_low = "-"
                            log.info(
                                "Realtime poll: fetched=%d ids=%s..%s last_seen=%s new=%d ids=%s..%s",
                                total_fetched, fetched_low, fetched_high, last_seen, new_count, new_low, new_high
                            )

                        if new_rows:
                            # process oldest-first
                            for r in reversed(new_rows):
                                handle_payload({"new": r})
                                last_seen = max(last_seen, r.get(COL_ID) or last_seen)
                        time.sleep(poll_interval)
                except KeyboardInterrupt:
                    log.info("Realtime polling stopped by user.")

        # Start realtime listener (blocking)
        start_realtime_listener(poll_interval=args.realtime_interval, dry_run=args.dry_run, predict_on_insert=args.predict_on_insert)
        sys.exit(0)

    while True:
        print(f"\n{'─'*70}")
        print(f"  RUN #{run_number}  |  {now_iso()}")
        print(f"{'─'*70}")

        try:
            new_rows, pred_ran = run_once(
                run_number=run_number,
                ingest_number=ingest_number,
                force_predict=args.force_predict,
                dry_run=args.dry_run,
                force_fb=(args.force_fb or ALWAYS_POST_FB),
            )
            if new_rows > 0 or pred_ran:
                ingest_number += 1
            run_number += 1
        except KeyboardInterrupt:
            print("\n  Stopped by user.")
            sys.exit(0)
        except Exception as exc:
            log.error("Unhandled error: %s", exc, exc_info=True)

        try:
            countdown(args.interval, label="Next run in")
        except KeyboardInterrupt:
            print("\n  Stopped by user.")
            sys.exit(0)