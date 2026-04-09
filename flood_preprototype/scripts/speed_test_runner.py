"""
speed_test_runner_v2.py
=======================
Combined Speed Test — Flood EWS Full Pipeline
Obando, Bulacan

PIPELINE BEHAVIOR
-----------------
- Fetches ONLY the single latest row (by Date+Time) from Supabase.
- If that row was already processed in a previous run → skip entirely.
- If it is new → calibrate → predict → sync → post FB alert.
- Writes one audit row per successful run to speedtest_audit.csv.
- Tracks operational availability (Aₒ) per tick to speedtest_availability.csv.

SPEEDTEST PREDICTIONS CSV COLUMNS
----------------------------------
    timestamp               str    — sensor date (midnight UTC, model granularity)
    flood_probability       float  — XGBoost predicted probability (0–1)
    risk_tier               str    — CLEAR / WATCH / WARNING / DANGER
    watch_threshold         float  — model watch threshold
    warning_threshold       float  — model warning threshold
    danger_threshold        float  — model danger threshold
    transmission_speed_s    float  — latency from sensor timestamp → prediction_created_at (seconds)
    sensor_timestamp        str    — raw sensor Date+Time from Supabase (ISO)
    prediction_created_at   str    — when the prediction was produced (local ISO)
    row_created_at          str    — when this CSV row was written (local ISO)

AUDIT CSV COLUMNS
-----------------
    run_number              int    — pipeline run counter
    run_timestamp           str    — when this tick started (local ISO)
    supabase_row_id         int    — id of the source row in Supabase
    supabase_saved_at       str    — Date + Time from Supabase (sensor timestamp)
    cal_waterlevel          float  — calibrated water level (m above datum)
    cal_soil_moisture       float  — calibrated soil moisture (m³/m³)
    cal_humidity            float  — calibrated humidity (cm CWV)
    flood_probability       float  — XGBoost predicted probability (0–1)
    risk_tier               str    — CLEAR / WATCH / WARNING / DANGER
    prediction_created_at   str    — when the prediction row was written (local ISO)
    fb_posted               bool   — whether FB alert was sent this tick

AVAILABILITY CSV COLUMNS
------------------------
    session_id              str    — UUID generated once per script launch
    tick_number             int    — increments every loop regardless of outcome
    tick_timestamp          str    — when the tick ran (local ISO)
    status                  str    — ACTIVE | SKIPPED | FAILED
    supabase_row_id         int    — id of latest row seen (even if skipped)
    uptime_ticks            int    — running count of ACTIVE ticks
    total_ticks             int    — running count of all ticks
    availability_pct        float  — uptime_ticks / total_ticks * 100 (running)

AVAILABILITY STATUS DEFINITIONS
--------------------------------
    ACTIVE   — new row detected, calibrated, and prediction produced end-to-end (true uptime)
    SKIPPED  — latest Supabase row already processed in a prior tick (sensor gap, not system failure)
    FAILED   — fetch error, NULL data, calibration out of range, or model failure (true downtime)

    Strict Aₒ     = ACTIVE / total_ticks
    Soft health   = (ACTIVE + SKIPPED) / total_ticks  (separates sensor gaps from system failures)

SUPABASE SOURCE TABLE SCHEMA (obando_environmental_data)
---------------------------------------------------------
    id               BIGINT GENERATED ALWAYS AS IDENTITY
    "Soil Moisture"  REAL NOT NULL
    "Temperature"    REAL NOT NULL
    "Humidity"       REAL NOT NULL
    "Pressure"       REAL NOT NULL
    "Final Distance" REAL NULL
    "Date"           DATE NULL
    "Time"           TIME NULL
    "Device"         TEXT NULL

USAGE
-----
    python speed_test_runner_v2.py                  # run every 30 seconds (default)
    python speed_test_runner_v2.py --interval 10    # run every 10 seconds
    python speed_test_runner_v2.py --once           # run exactly once and exit
    python speed_test_runner_v2.py --dry-run        # preview, no writes

ENVIRONMENT VARIABLES (.env two levels above this file)
    SUPABASE_URL          = https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY  = <service_role key>
    FB_PAGE_ID            = <numeric page id>
    FB_PAGE_TOKEN         = <long-lived page access token>
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
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import joblib
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
SPEEDTEST_CSV        = os.path.join(SPEEDTEST_OUTPUT_DIR, "speedtest_predictions.csv")
AUDIT_CSV            = os.path.join(SPEEDTEST_OUTPUT_DIR, "speedtest_audit.csv")
AVAILABILITY_CSV     = os.path.join(SPEEDTEST_OUTPUT_DIR, "speedtest_availability.csv")

SENSOR_CSV = Path(os.path.normpath(
    os.path.join(_PROJECT_ROOT, "data", "sensor", "speedtest_sensor_data.csv")
))

FLOOD_LOG_PATH = os.path.join(_PROJECT_ROOT, "data", "flood_event_log.csv")

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
log = logging.getLogger("speed_test_runner_v2")

# ===========================================================================
# SUPABASE CONFIG
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
# FACEBOOK CONFIG
# ===========================================================================

FB_PAGE_ID     = os.getenv("FB_PAGE_ID")
FB_PAGE_TOKEN  = os.getenv("FB_PAGE_TOKEN")
FB_API_VERSION = "v23.0"
FB_STATE_PATH  = os.path.join(SCRIPT_DIR, "speedtest_last_posted.json")

FB_TIER_MESSAGE = {
    "CLEAR": (
        "🟢 ALL CLEAR — Obando, Bulacan\n\n"
        "No flood risk detected by the Rapid Relay Early Warning System.\n"
        "Conditions are normal. No action required.\n\n"
        "Flood Probability : {prob_pct}\n"
        "Date              : {timestamp}\n\n"
        "Stay informed. Follow official advisories from local DRRMO."
    ),
    "WATCH": (
        "🟡 FLOOD WATCH — Obando, Bulacan\n\n"
        "Elevated flood risk detected by the Rapid Relay Early Warning System.\n"
        "Monitor water levels closely. Be prepared to act.\n\n"
        "Flood Probability : {prob_pct}\n"
        "Date              : {timestamp}\n\n"
        "Stay safe. Follow official advisories from local DRRMO."
    ),
    "WARNING": (
        "🟠 FLOOD WARNING — Obando, Bulacan\n\n"
        "High flood risk detected by the Rapid Relay Early Warning System.\n"
        "Prepare for possible flooding. Move valuables to higher ground.\n\n"
        "Flood Probability : {prob_pct}\n"
        "Date              : {timestamp}\n\n"
        "Stay safe. Follow official advisories from local DRRMO."
    ),
    "DANGER": (
        "🔴 FLOOD DANGER — Obando, Bulacan\n\n"
        "IMMINENT or ONGOING flood detected by the Rapid Relay Early Warning System.\n"
        "Take immediate action. Evacuate if in flood-prone areas.\n\n"
        "Flood Probability : {prob_pct}\n"
        "Date              : {timestamp}\n\n"
        "🚨 Follow evacuation orders from local DRRMO immediately."
    ),
}

# ===========================================================================
# CALIBRATION CONSTANTS
# ===========================================================================

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

# ===========================================================================
# PREDICTION CONFIG
# ===========================================================================

DEFAULT_ALERT_THRESHOLD = 0.50
MIN_CONSECUTIVE_DAYS    = 2
ROLLING_MEAN_WINDOW     = 7
ROLLING_SUM_WINDOW      = 14
LAST_TRAINING_DATE      = "2024-12-31"

RISK_TIERS = {
    "CLEAR":   {"emoji": "🟢", "color": "green"},
    "WATCH":   {"emoji": "🟡", "color": "gold"},
    "WARNING": {"emoji": "🟠", "color": "orange"},
    "DANGER":  {"emoji": "🔴", "color": "red"},
}

AUDIT_COLUMNS = [
    "run_number",
    "run_timestamp",
    "supabase_row_id",
    "supabase_saved_at",
    "cal_waterlevel",
    "cal_soil_moisture",
    "cal_humidity",
    "flood_probability",
    "risk_tier",
    "prediction_created_at",
    "fb_posted",
]

AVAIL_COLUMNS = [
    "session_id",
    "tick_number",
    "tick_timestamp",
    "status",
    "supabase_row_id",
    "uptime_ticks",
    "total_ticks",
    "availability_pct",
]

# Generated once per script launch — groups all ticks in this session
_SESSION_ID = str(uuid.uuid4())[:8]

# ===========================================================================
# HELPERS
# ===========================================================================

def sep(title=""):
    line = "=" * 60
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


def get_run_number() -> int:
    """Return next run number based on existing audit CSV row count."""
    if not os.path.exists(AUDIT_CSV):
        return 1
    try:
        df = pd.read_csv(AUDIT_CSV, usecols=["run_number"])
        return int(df["run_number"].max()) + 1 if not df.empty else 1
    except Exception:
        return 1


def get_last_processed_id() -> int | None:
    """Return the Supabase row id processed in the most recent audit entry."""
    if not os.path.exists(AUDIT_CSV):
        return None
    try:
        df = pd.read_csv(AUDIT_CSV, usecols=["run_number", "supabase_row_id"])
        if df.empty:
            return None
        return int(df.loc[df["run_number"].idxmax(), "supabase_row_id"])
    except Exception:
        return None


def write_audit_row(row: dict, dry_run: bool = False) -> None:
    """Append one row to the audit CSV, creating it with headers if needed."""
    os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame([{col: row.get(col, None) for col in AUDIT_COLUMNS}])

    if dry_run:
        log.info("[DRY RUN] Audit row:\n%s", df.to_string(index=False))
        return

    if not os.path.exists(AUDIT_CSV):
        df.to_csv(AUDIT_CSV, index=False)
        log.info("Created audit CSV: %s", AUDIT_CSV)
    else:
        df.to_csv(AUDIT_CSV, mode="a", header=False, index=False)
        log.info("Appended audit row → run #%s  id=%s  tier=%s",
                 row.get("run_number"), row.get("supabase_row_id"), row.get("risk_tier"))


# ===========================================================================
# AVAILABILITY TRACKING
# ===========================================================================

def get_availability_state() -> tuple[int, int]:
    """
    Read existing availability CSV and return
    (cumulative uptime_ticks, cumulative total_ticks) across all sessions.
    Returns (0, 0) if no CSV exists yet.
    """
    if not os.path.exists(AVAILABILITY_CSV):
        return 0, 0
    try:
        df = pd.read_csv(AVAILABILITY_CSV, usecols=["uptime_ticks", "total_ticks"])
        if df.empty:
            return 0, 0
        last = df.iloc[-1]
        return int(last["uptime_ticks"]), int(last["total_ticks"])
    except Exception:
        return 0, 0


def write_availability_row(
    tick_number: int,
    status: str,          # "ACTIVE" | "SKIPPED" | "FAILED"
    supabase_row_id,
    uptime_ticks: int,
    total_ticks: int,
    dry_run: bool = False,
) -> None:
    """Append one availability row to the CSV."""
    avail_pct = round((uptime_ticks / total_ticks) * 100, 4) if total_ticks > 0 else 0.0

    row = {
        "session_id":       _SESSION_ID,
        "tick_number":      tick_number,
        "tick_timestamp":   now_iso(),
        "status":           status,
        "supabase_row_id":  supabase_row_id,
        "uptime_ticks":     uptime_ticks,
        "total_ticks":      total_ticks,
        "availability_pct": avail_pct,
    }

    df = pd.DataFrame([{col: row.get(col) for col in AVAIL_COLUMNS}])

    if dry_run:
        log.info(
            "[DRY RUN] Availability → tick=%d  status=%s  Aₒ=%.2f%%",
            tick_number, status, avail_pct
        )
        return

    os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(AVAILABILITY_CSV):
        df.to_csv(AVAILABILITY_CSV, index=False)
        log.info("Created availability CSV: %s", AVAILABILITY_CSV)
    else:
        df.to_csv(AVAILABILITY_CSV, mode="a", header=False, index=False)

    log.info(
        "Availability → tick=%d  status=%-7s  Aₒ=%.2f%%  (%d/%d ticks)",
        tick_number, status, avail_pct, uptime_ticks, total_ticks
    )


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


# ===========================================================================
# STEP 1 — FETCH LATEST ROW ONLY
# ===========================================================================

def fetch_latest_row() -> dict | None:
    """
    Fetch the single most recent row from Supabase ordered by Date DESC,
    Time DESC. Scans up to 50 rows to find the first one where Date,
    Time, and Final Distance are all non-NULL.
    Returns a raw dict or None.
    """
    try:
        response = (
            get_supabase()
            .table(TABLE_ENV_DATA)
            .select(_SELECT)
            .order(COL_DATE, desc=True)
            .order(COL_TIME, desc=True)
            .limit(50)
            .execute()
        )
    except Exception as exc:
        log.error("Supabase fetch failed: %s", exc)
        return None

    rows = response.data or []
    log.info("Fetched %d candidate row(s) from Supabase.", len(rows))

    for row in rows:
        if (
            row.get(COL_DATE)     is not None
            and row.get(COL_TIME)     is not None
            and row.get(COL_DISTANCE) is not None
        ):
            log.info(
                "Latest valid row → id=%s  Date=%s  Time=%s  Distance=%s",
                row.get(COL_ID), row.get(COL_DATE),
                row.get(COL_TIME), row.get(COL_DISTANCE),
            )
            return row

    log.warning("No valid row found (all candidates have NULL Date/Time/Distance).")
    return None


def calibrate_row(row: dict) -> dict | None:
    """
    Calibrate a single raw Supabase row.
    Returns dict with cal_ fields, or None if waterlevel is invalid.
    """
    cal_wl = _calibrate_waterlevel(row.get(COL_DISTANCE))
    if cal_wl is None:
        log.warning(
            "Row id=%s has invalid Final Distance (%s) — skipping.",
            row.get(COL_ID), row.get(COL_DISTANCE)
        )
        return None

    return {
        "cal_waterlevel":    cal_wl,
        "cal_soil_moisture": _calibrate_soil(row.get(COL_SOIL, 0)),
        "cal_humidity":      _calibrate_humidity(row.get(COL_HUMIDITY, 0)),
    }


def append_to_sensor_csv(row: dict, cal: dict, dry_run: bool = False) -> None:
    """
    Append the calibrated row to the running sensor CSV so that
    feature engineering always has the full history available.
    Uses midnight timestamp (daily granularity) to match model expectations.
    """
    ts_str = f"{row[COL_DATE]}T00:00:00"

    new_row = pd.DataFrame([{
        "timestamp":      ts_str,
        "waterlevel":     cal["cal_waterlevel"],
        "soil_moisture":  cal["cal_soil_moisture"],
        "humidity":       cal["cal_humidity"],
        "row_created_at": now_iso(),
    }])

    if dry_run:
        log.info("[DRY RUN] Would append sensor row:\n%s", new_row.to_string(index=False))
        return

    SENSOR_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not SENSOR_CSV.exists():
        new_row.to_csv(SENSOR_CSV, index=False)
        log.info("Created sensor CSV: %s", SENSOR_CSV)
        return

    existing = pd.read_csv(SENSOR_CSV, usecols=["timestamp"])
    if ts_str in existing["timestamp"].astype(str).values:
        log.info("Sensor CSV already has row for %s — skipping append.", ts_str)
        return

    new_row.to_csv(SENSOR_CSV, mode="a", header=False, index=False)
    log.info("Sensor CSV updated → %s", SENSOR_CSV)


# ===========================================================================
# STEP 2 — PREDICTION
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

    log.info("Model loaded  : %s", os.path.basename(MODEL_FILE))
    log.info("  Version     : %s", artifact.get("version", "unknown"))
    log.info("  Features    : %d", len(feature_cols))
    log.info("  WATCH thr   : %.2f   WARNING thr : %.2f", watch_t, warn_t)

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
    """
    Run XGBoost inference using the full sensor CSV history.
    Returns prediction dict or None on failure.
    """
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
    pred_ts     = now_iso()
    pred_epoch  = time.time()   # wall-clock at prediction completion

    log.info(
        "Prediction → %s %s  (%.1f%%)",
        RISK_TIERS[latest_tier]["emoji"], latest_tier, latest_prob * 100
    )

    return {
        "tier":                  latest_tier,
        "probability":           latest_prob,
        "timestamp":             results.index[-1].strftime("%Y-%m-%d"),
        "prediction_created_at": pred_ts,
        "pred_epoch":            pred_epoch,
        "results_df":            results,
    }


def save_predictions_csv(
    results_df: pd.DataFrame,
    transmission_speed_s: float | None = None,
    sensor_timestamp: str | None = None,
    prediction_created_at: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    Persist prediction results to speedtest_predictions.csv.

    The latest row gets transmission_speed_s, sensor_timestamp, and
    prediction_created_at stamped in; historical rows keep whatever
    values they already have (NaN if written by an older version).
    """
    if dry_run:
        log.info(
            "[DRY RUN] Would save predictions CSV  "
            "(transmission_speed_s=%.3fs  sensor_ts=%s)",
            transmission_speed_s or 0.0, sensor_timestamp
        )
        return

    os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
    out = results_df.copy()
    now = now_iso()
    out["row_created_at"] = now

    # Stamp speed columns only on the latest row (last index position)
    out["transmission_speed_s"]  = None
    out["sensor_timestamp"]       = None
    out["prediction_created_at"]  = None
    if len(out) > 0:
        out.iloc[-1, out.columns.get_loc("transmission_speed_s")]  = (
            round(transmission_speed_s, 4) if transmission_speed_s is not None else None
        )
        out.iloc[-1, out.columns.get_loc("sensor_timestamp")]       = sensor_timestamp
        out.iloc[-1, out.columns.get_loc("prediction_created_at")]  = prediction_created_at

    if os.path.exists(SPEEDTEST_CSV):
        existing = pd.read_csv(
            SPEEDTEST_CSV, parse_dates=["timestamp"], index_col="timestamp"
        )
        if existing.index.tzinfo is None:
            existing.index = existing.index.tz_localize("UTC")
        combined = pd.concat([existing, out])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = out

    combined.index.name = "timestamp"
    combined.to_csv(SPEEDTEST_CSV)
    log.info(
        "Predictions CSV saved → %s  (%d total rows)  speed=%.3fs",
        SPEEDTEST_CSV, len(combined), transmission_speed_s or 0.0
    )


# ===========================================================================
# STEP 3 — SUPABASE SYNC
# ===========================================================================

def sync_to_supabase(dry_run: bool = False) -> None:
    sep(f"STEP 3 — Supabase Sync → {SUPABASE_PRED_TABLE}")

    if not os.path.exists(SPEEDTEST_CSV):
        log.warning("No predictions CSV — skipping Supabase sync.")
        return

    try:
        df = pd.read_csv(SPEEDTEST_CSV)
    except Exception as e:
        log.error("Could not read predictions CSV: %s", e)
        return

    if df.empty:
        log.warning("Predictions CSV is empty — nothing to sync.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    records = [
        {
            "timestamp":         row["timestamp"].isoformat(),
            "flood_probability": float(row["flood_probability"]),
            "risk_tier":         str(row["risk_tier"]),
            "row_created_at":    str(row.get("row_created_at", now_iso())),
        }
        for _, row in df.iterrows()
    ]

    log.info("Syncing %d record(s) to %s ...", len(records), SUPABASE_PRED_TABLE)

    if dry_run:
        log.info("[DRY RUN] Would upsert %d record(s).", len(records))
        return

    try:
        sb    = get_supabase()
        total = 0
        for i in range(0, len(records), SUPABASE_BATCH_SIZE):
            batch = records[i:i + SUPABASE_BATCH_SIZE]
            sb.table(SUPABASE_PRED_TABLE).upsert(
                batch, on_conflict="timestamp"
            ).execute()
            total += len(batch)
            log.info("  Upserted batch %d / %d", total, len(records))
        log.info("Supabase sync complete: %d row(s).", total)
    except Exception as e:
        log.error("Supabase sync failed: %s", e)


# ===========================================================================
# STEP 4 — FACEBOOK ALERT
# ===========================================================================

def _load_fb_last_posted():
    try:
        if not os.path.exists(FB_STATE_PATH):
            return None
        with open(FB_STATE_PATH, "r") as f:
            return json.load(f).get("last_posted_timestamp")
    except Exception:
        return None


def _save_fb_last_posted(timestamp: str) -> None:
    try:
        with open(FB_STATE_PATH, "w") as f:
            json.dump({"last_posted_timestamp": timestamp}, f, indent=2)
        log.info("FB state saved — last posted: %s", timestamp)
    except Exception as e:
        log.warning("Could not save FB state: %s", e)


def send_fb_alert(
    tier: str,
    probability: float,
    timestamp: str,
    dry_run: bool = False,
) -> bool:
    sep("STEP 4 — Facebook Alert")

    if not FB_PAGE_ID or not FB_PAGE_TOKEN:
        log.warning(
            "FB credentials missing — skipping.\n"
            "  FB_PAGE_ID    : %s\n"
            "  FB_PAGE_TOKEN : %s",
            "SET" if FB_PAGE_ID else "MISSING",
            "SET" if FB_PAGE_TOKEN else "MISSING",
        )
        return False

    last = _load_fb_last_posted()
    if last and last == timestamp:
        log.info("FB: already posted for %s — skipping.", timestamp)
        return True

    template = FB_TIER_MESSAGE.get(tier, FB_TIER_MESSAGE["WARNING"])
    message  = template.format(prob_pct=f"{probability:.1%}", timestamp=timestamp)
    url      = f"https://graph.facebook.com/{FB_API_VERSION}/{FB_PAGE_ID}/feed"

    log.info("Posting %s alert (prob=%.1f%%, ts=%s) ...", tier, probability * 100, timestamp)

    if dry_run:
        log.info("[DRY RUN] Would post to Facebook:\n%s", message)
        _save_fb_last_posted(timestamp)
        return True

    try:
        res  = requests.post(
            url,
            data={"message": message, "access_token": FB_PAGE_TOKEN},
            timeout=15,
        )
        data = res.json()
        if "id" in data:
            log.info("FB post successful — post id: %s", data["id"])
            _save_fb_last_posted(timestamp)
            return True
        else:
            err = data.get("error", {})
            log.error("FB API error: %s", err.get("message", data))
            return False
    except requests.Timeout:
        log.error("FB request timed out.")
        return False
    except Exception as e:
        log.error("FB unexpected error: %s", e)
        return False


# ===========================================================================
# FULL PIPELINE — one tick
# ===========================================================================

def run_once(
    run_number: int,
    tick_number: int,
    uptime_ticks: int,
    total_ticks: int,
    dry_run: bool = False,
) -> tuple[str, int | None]:
    """
    Execute one full pipeline tick.

    Returns
    -------
    (status, supabase_row_id) where status is 'ACTIVE', 'SKIPPED', or 'FAILED'.
    """
    tick_start = time.time()
    run_ts     = now_iso()

    sep(f"SPEED TEST PIPELINE  [Run #{run_number}  |  {run_ts}]")
    print(f"  Sensor CSV   : {SENSOR_CSV}")
    print(f"  Predictions  : {SPEEDTEST_CSV}")
    print(f"  Audit CSV    : {AUDIT_CSV}")
    print(f"  Dry run      : {dry_run}")

    # ------------------------------------------------------------------
    # STEP 1 — Fetch latest row from Supabase
    # ------------------------------------------------------------------
    sep("STEP 1 — Fetch Latest Sensor Row")

    latest_row = fetch_latest_row()

    if latest_row is None:
        log.warning("No valid sensor row found — skipping tick entirely.")
        return "FAILED", None

    row_id = latest_row.get(COL_ID)

    # Skip if already processed
    last_processed_id = get_last_processed_id()
    if last_processed_id is not None and row_id == last_processed_id:
        log.info(
            "Row id=%s already processed in a previous run — skipping tick entirely.",
            row_id
        )
        return "SKIPPED", row_id

    supabase_saved_at = f"{latest_row.get(COL_DATE)} {latest_row.get(COL_TIME)}"
    log.info("New row detected → id=%s  saved_at=%s", row_id, supabase_saved_at)

    # Wall-clock at the moment the sensor timestamp was confirmed as new.
    # Used as the "transmission start" reference for latency calculation.
    sensor_epoch = time.time()

    # ------------------------------------------------------------------
    # STEP 1b — Calibrate
    # ------------------------------------------------------------------
    cal = calibrate_row(latest_row)
    if cal is None:
        log.warning("Calibration failed — skipping tick entirely.")
        return "FAILED", row_id

    log.info(
        "Calibrated → waterlevel=%.6f  soil=%.6f  humidity=%.6f",
        cal["cal_waterlevel"], cal["cal_soil_moisture"], cal["cal_humidity"]
    )

    # Append calibrated row to sensor CSV for feature engineering
    append_to_sensor_csv(latest_row, cal, dry_run=dry_run)

    # ------------------------------------------------------------------
    # STEP 2 — Predict
    # ------------------------------------------------------------------
    sep("STEP 2 — XGBoost Prediction")

    prediction = run_prediction()

    if prediction is None:
        log.warning("Prediction failed — skipping tick entirely.")
        return "FAILED", row_id

    # Latency: sensor timestamp confirmed → prediction produced
    transmission_speed_s = round(prediction["pred_epoch"] - sensor_epoch, 4)
    log.info("Transmission speed (sensor→prediction) : %.4fs", transmission_speed_s)

    save_predictions_csv(
        prediction["results_df"],
        transmission_speed_s=transmission_speed_s,
        sensor_timestamp=supabase_saved_at,
        prediction_created_at=prediction["prediction_created_at"],
        dry_run=dry_run,
    )

    # ------------------------------------------------------------------
    # STEP 3 — Supabase sync
    # ------------------------------------------------------------------
    sync_to_supabase(dry_run=dry_run)

    # ------------------------------------------------------------------
    # STEP 4 — Facebook alert
    # ------------------------------------------------------------------
    fb_posted = send_fb_alert(
        tier=prediction["tier"],
        probability=prediction["probability"],
        timestamp=prediction["timestamp"],
        dry_run=dry_run,
    )

    # ------------------------------------------------------------------
    # STEP 5 — Write audit row
    # ------------------------------------------------------------------
    sep("STEP 5 — Audit CSV")

    audit_row = {
        "run_number":            run_number,
        "run_timestamp":         run_ts,
        "supabase_row_id":       row_id,
        "supabase_saved_at":     supabase_saved_at,
        "cal_waterlevel":        cal["cal_waterlevel"],
        "cal_soil_moisture":     cal["cal_soil_moisture"],
        "cal_humidity":          cal["cal_humidity"],
        "flood_probability":     prediction["probability"],
        "risk_tier":             prediction["tier"],
        "prediction_created_at": prediction["prediction_created_at"],
        "fb_posted":             fb_posted,
    }

    write_audit_row(audit_row, dry_run=dry_run)

    # ------------------------------------------------------------------
    # Tick summary
    # ------------------------------------------------------------------
    elapsed = time.time() - tick_start
    tier    = prediction["tier"]
    sep("TICK COMPLETE")
    print(f"  {RISK_TIERS[tier]['emoji']}  {tier:<8}  prob={prediction['probability']:.1%}  ts={prediction['timestamp']}")
    print(f"  Supabase row id      : {row_id}")
    print(f"  Supabase saved at    : {supabase_saved_at}")
    print(f"  cal_waterlevel       : {cal['cal_waterlevel']}")
    print(f"  cal_soil_moisture    : {cal['cal_soil_moisture']}")
    print(f"  cal_humidity         : {cal['cal_humidity']}")
    print(f"  Prediction created   : {prediction['prediction_created_at']}")
    print(f"  Transmission speed   : {transmission_speed_s:.4f}s  (sensor→prediction)")
    print(f"  FB posted            : {fb_posted}")
    print(f"  Elapsed              : {elapsed:.2f}s")
    print(f"  Audit CSV            : {AUDIT_CSV}")
    sep()

    return "ACTIVE", row_id


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flood EWS Speed Test v2 — Full Pipeline Runner with Availability Tracking"
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Loop interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run pipeline exactly once and exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview all steps without writing to disk, Supabase, or Facebook"
    )
    args = parser.parse_args()

    print(f"\n  🚀  Speed test runner v2 starting — interval={args.interval}s")
    print(f"  📂  Model        : {MODEL_FILE}")
    print(f"  📊  Sensor CSV   : {SENSOR_CSV}")
    print(f"  💾  Output CSV   : {SPEEDTEST_CSV}")
    print(f"  📋  Audit CSV    : {AUDIT_CSV}")
    print(f"  📈  Avail CSV    : {AVAILABILITY_CSV}")
    print(f"  🗄️   Supabase src : {TABLE_ENV_DATA}")
    print(f"  🗄️   Supabase dst : {SUPABASE_PRED_TABLE}")
    print(f"  📘  FB Page ID   : {FB_PAGE_ID or 'MISSING'}")
    print(f"  🔑  Session ID   : {_SESSION_ID}")
    print(f"  Press Ctrl+C to stop.\n")

    # ------------------------------------------------------------------
    # --once mode
    # ------------------------------------------------------------------
    if args.once:
        run_number = get_run_number()
        uptime_ticks, total_ticks = get_availability_state()
        total_ticks += 1

        status, row_id = run_once(
            run_number=run_number,
            tick_number=total_ticks,
            uptime_ticks=uptime_ticks,
            total_ticks=total_ticks,
            dry_run=args.dry_run,
        )

        if status == "ACTIVE":
            uptime_ticks += 1

        write_availability_row(
            tick_number=total_ticks,
            status=status,
            supabase_row_id=row_id,
            uptime_ticks=uptime_ticks,
            total_ticks=total_ticks,
            dry_run=args.dry_run,
        )

        live_ao = round(uptime_ticks / total_ticks * 100, 2) if total_ticks > 0 else 0.0
        print(f"\n  📊  Final Availability (Aₒ) : {live_ao}%  ({uptime_ticks} active / {total_ticks} total ticks)")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Continuous loop mode
    # ------------------------------------------------------------------
    run_number = get_run_number()
    uptime_ticks, total_ticks = get_availability_state()

    while True:
        total_ticks += 1
        live_ao_display = (
            f"{round(uptime_ticks / (total_ticks - 1) * 100, 2)}%"
            if total_ticks > 1 else "—"
        )

        print(f"\n{'─'*60}")
        print(f"  RUN #{run_number}  |  {now_iso()}  |  Aₒ so far: {live_ao_display}")
        print(f"{'─'*60}")

        status = "FAILED"
        row_id = None

        try:
            status, row_id = run_once(
                run_number=run_number,
                tick_number=total_ticks,
                uptime_ticks=uptime_ticks,
                total_ticks=total_ticks,
                dry_run=args.dry_run,
            )
        except KeyboardInterrupt:
            print("\n  Stopped by user.")
            sys.exit(0)
        except Exception as exc:
            log.error("Unhandled error in pipeline: %s", exc, exc_info=True)

        if status == "ACTIVE":
            uptime_ticks += 1
            run_number   += 1

        write_availability_row(
            tick_number=total_ticks,
            status=status,
            supabase_row_id=row_id,
            uptime_ticks=uptime_ticks,
            total_ticks=total_ticks,
            dry_run=args.dry_run,
        )

        live_ao = round(uptime_ticks / total_ticks * 100, 2)
        print(f"\n  📊  Live Availability (Aₒ) : {live_ao}%  ({uptime_ticks} active / {total_ticks} total ticks)")

        try:
            countdown(args.interval, label="Next run in")
        except KeyboardInterrupt:
            print("\n  Stopped by user.")
            sys.exit(0)