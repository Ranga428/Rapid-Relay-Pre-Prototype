"""
Rapid_Relay_Showcase.py
========================================
FULL HYBRID PIPELINE — Real-time Ingestion + GEE Proxy + Merge + Daily Prediction
Flood EWS for Obando, Bulacan

CORE LOGIC
----------
INGESTION (Real-time, every tick):
  - Fetch latest row from Supabase hardware sensor table
  - For each new row (id > last_processed_id):
    → Calibrate → Append to sensor CSV → Track in ingestion audit

SATELLITE PROXY (Once daily or on new sensor date):
  - Fetch updated proxy data via Sat_SensorData_proxy.py (GEE)
  - Incremental: only fetches rows newer than latest in obando_environmental_data.csv

MERGE (After proxy fetch):
  - Run merge_sensor.py to align sensor + proxy into combined_sensor_context.csv
  - Incremental: only merges rows newer than latest in combined CSV

PREDICTION (Once daily, after merge):
  - Run prediction on combined_sensor_context.csv (merged sensor + proxy)
  - Post alert for daily prediction result
  - Log to prediction audit with PST-normalized timestamps

OUTPUTS (all CSVs include PST-normalized timestamps)
------------------------------------------------------
INGESTION AUDIT  → speedtest_ingestion_audit.csv
SENSOR DATA      → speedtest_sensor_data.csv
PREDICTION AUDIT → speedtest_prediction_audit.csv
PREDICTIONS      → speedtest_predictions.csv        (with PST timestamps)
AVAILABILITY     → speedtest_availability.csv

TIMESTAMP NORMALIZATION
-----------------------
All timestamps stored in PST (Asia/Manila, UTC+8).
Naive timestamps from Supabase assumed UTC, then converted.
time_diff_seconds = row_created_at - sensor_timestamp (pipeline latency).

USAGE
-----
  python Rapid_Relay_Showcase.py                      # 30s interval
  python Rapid_Relay_Showcase.py --interval 10        # 10s interval
  python Rapid_Relay_Showcase.py --once               # Single tick
  python Rapid_Relay_Showcase.py --force-predict      # Force prediction now
  python Rapid_Relay_Showcase.py --dry-run            # Preview only
  python Rapid_Relay_Showcase.py --realtime           # Supabase realtime listener
  python Rapid_Relay_Showcase.py --realtime --predict-on-insert
  python Rapid_Relay_Showcase.py --test-fb            # Test FB post and exit
  python Rapid_Relay_Showcase.py --fb-debug           # FB token diagnostics
  python Rapid_Relay_Showcase.py --skip-proxy         # Skip GEE proxy fetch
  python Rapid_Relay_Showcase.py --skip-merge         # Skip sensor-proxy merge
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
import traceback
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

# Sensor-only CSV (hardware readings, calibrated)
SENSOR_CSV = Path(os.path.normpath(
    os.path.join(_PROJECT_ROOT, "data", "sensor", "speedtest_sensor_data.csv")
))

# Merged sensor + proxy CSV (input to prediction)
COMBINED_CSV = Path(os.path.normpath(
    os.path.join(_PROJECT_ROOT, "data", "combined_sensor_context.csv")
))

FLOOD_LOG_PATH = os.path.join(_PROJECT_ROOT, "data", "flood_event_log.csv")

# State tracking
STATE_FILE = os.path.join(SPEEDTEST_OUTPUT_DIR, "speedtest_state.json")

# ML pipeline path
_ML_PIPELINE = os.path.join(_PROJECT_ROOT, "ml_pipeline")
sys.path.insert(0, _ML_PIPELINE)

# Deployment scripts path (for Sat_SensorData_proxy and merge_sensor)
_DEPLOYMENT = os.path.join(_PROJECT_ROOT, "deployment")
sys.path.insert(0, _DEPLOYMENT)

# Scripts path
_SCRIPTS = os.path.join(_PROJECT_ROOT, "scripts")
sys.path.insert(0, _SCRIPTS)

# ===========================================================================
# TERMINAL STYLE
# ===========================================================================

class C:
    """ANSI color codes. Auto-disabled if not a TTY."""
    _on = sys.stdout.isatty()
    RESET  = "\033[0m"        if _on else ""
    BOLD   = "\033[1m"        if _on else ""
    DIM    = "\033[2m"        if _on else ""
    GREEN  = "\033[92m"       if _on else ""
    YELLOW = "\033[93m"       if _on else ""
    ORANGE = "\033[38;5;208m" if _on else ""
    RED    = "\033[91m"       if _on else ""
    BLUE   = "\033[94m"       if _on else ""
    CYAN   = "\033[96m"       if _on else ""
    GRAY   = "\033[90m"       if _on else ""
    WHITE  = "\033[97m"       if _on else ""

PHASE_COLORS = {
    1: C.BLUE,
    2: C.YELLOW,
    3: C.CYAN,
    4: C.RED,
}

TIER_COLORS = {
    "CLEAR":   C.GREEN,
    "WATCH":   C.YELLOW,
    "WARNING": C.ORANGE,
    "DANGER":  C.RED,
}

TIER_ICONS = {
    "CLEAR":   "●",
    "WATCH":   "◆",
    "WARNING": "▲",
    "DANGER":  "■",
}


def _short(path: str) -> str:
    """…parent/filename  (max ~42 chars, always shows filename)"""
    p = Path(path)
    try:
        rel = p.relative_to(Path.cwd())
        parts = rel.parts
    except ValueError:
        parts = p.parts
    if len(parts) == 0:
        return path
    if len(parts) == 1:
        return parts[0]
    slug = f"…/{'/'.join(parts[-2:])}" if len(parts) > 2 else str(Path(*parts))
    return slug[:42] + "…" if len(slug) > 42 else slug


def _link(full: str, label: str | None = None, *, tty: bool | None = None) -> str:
    """OSC 8 hyperlink — short display label in terminal, full path copy-pastable.

    Supported terminals: iTerm2, VS Code terminal, Windows Terminal, GNOME Terminal.
    Falls back to the full path string when stdout is not a TTY (CI, pipes, SSH).
    """
    if tty is None:
        tty = sys.stdout.isatty()
    if not tty:
        return full
    short = label or _short(full)
    return f"\033]8;;{full}\033\\{short}\033]8;;\033\\"


# ===========================================================================
# LOGGING
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("speed_test_runner_v3")

# ── Suppress noisy HTTP request logs from httpx / httpcore ──
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Prediction config
# ---------------------------------------------------------------------------
DEFAULT_ALERT_THRESHOLD = 0.50
MIN_CONSECUTIVE_DAYS    = 2
ROLLING_MEAN_WINDOW     = 7
ROLLING_SUM_WINDOW      = 14
LAST_TRAINING_DATE      = "2024-12-31"

# Daily prediction timing (UTC)
PREDICTION_HOUR_UTC       = 0   # 00:00 UTC = 08:00 PHT
PREDICTION_WINDOW_MINUTES = 30  # window: 00:00–00:30 UTC

SENSOR_TIMEZONE = os.getenv("SENSOR_TIMEZONE", "UTC")

# ---------------------------------------------------------------------------
# Timestamp normalization
# ---------------------------------------------------------------------------
OUTPUT_TZ        = "Asia/Manila"
ASSUME_NAIVE_AS  = "UTC"

# ---------------------------------------------------------------------------
# Facebook
# ---------------------------------------------------------------------------
FB_PAGE_ID     = os.getenv("FB_PAGE_ID")
FB_PAGE_TOKEN  = os.getenv("FB_PAGE_TOKEN")
FB_API_VERSION = "v23.0"
FB_STATE_PATH  = os.path.join(SCRIPT_DIR, "speedtest_last_fb_prediction_ts.json")

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

# Session ID
_SESSION_ID = str(uuid.uuid4())[:8]


# ===========================================================================
# TIMESTAMP NORMALIZATION HELPERS
# ===========================================================================

def _to_pht(val, base_date=None) -> pd.Timestamp | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    val_str = str(val).strip()
    if not val_str or val_str.lower() == "nat":
        return None
    if len(val_str) <= 8 and ":" in val_str:
        if base_date is None:
            base_date = pd.Timestamp.now(tz="UTC").date()
        val_str = f"{base_date} {val_str}"
    ts = pd.to_datetime(val_str, errors="coerce")
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        if ASSUME_NAIVE_AS == "UTC":
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_localize(OUTPUT_TZ)
    ts = ts.tz_convert(OUTPUT_TZ).round("s")
    return ts


def pht_iso(val, base_date=None) -> str | None:
    ts = _to_pht(val, base_date)
    if ts is None:
        return None
    s = ts.strftime("%Y-%m-%dT%H:%M:%S%z")
    import re
    s = re.sub(r"(\+|-)(\d{2})(\d{2})$", r"\1\2:\3", s)
    return s


def now_pht() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_convert(OUTPUT_TZ).round("s")


def now_iso() -> str:
    return pht_iso(pd.Timestamp.now(tz="UTC")) or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def compute_time_diff_seconds(sensor_ts_str: str | None, created_at_str: str | None) -> float | None:
    if not sensor_ts_str or not created_at_str:
        return None
    try:
        s = pd.to_datetime(sensor_ts_str, utc=True)
        c = pd.to_datetime(created_at_str, utc=True)
        return round((c - s).total_seconds(), 3)
    except Exception:
        return None


def normalize_predictions_csv(df: pd.DataFrame) -> pd.DataFrame:
    ts_cols = ["sensor_timestamp", "prediction_created_at", "row_created_at"]
    base_date = None
    for col in ts_cols:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            valid = parsed.dropna()
            if not valid.empty:
                base_date = valid.iloc[0].date()
                break
    if base_date is None:
        base_date = pd.Timestamp.now().date()
    for col in ts_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: pht_iso(v, base_date))
    if "sensor_timestamp" in df.columns and "row_created_at" in df.columns:
        df["sensor_minus_row_created_at_sec"] = df.apply(
            lambda r: compute_time_diff_seconds(
                r.get("sensor_timestamp"), r.get("row_created_at")
            ),
            axis=1,
        )
    if "test_no" not in df.columns:
        df.insert(0, "test_no", range(1, len(df) + 1))
    return df


# ===========================================================================
# TERMINAL UI HELPERS
# ===========================================================================

def sep(title="", phase: int | None = None) -> None:
    width = 70
    color = PHASE_COLORS.get(phase, C.GRAY) if phase else C.GRAY
    if title:
        phase_tag = f"{C.BOLD}{color}[{phase:02d}]{C.RESET} " if phase else ""
        bar = f"{C.DIM}{'─' * width}{C.RESET}"
        print(f"\n{bar}")
        print(f"  {phase_tag}{C.BOLD}{color}{title}{C.RESET}")
        print(bar)
    else:
        print(f"{C.DIM}{'═' * width}{C.RESET}")


def print_startup_banner() -> None:
    try:
        import pyfiglet
        art = pyfiglet.figlet_format("Rapid Relay", font="slant").rstrip()
    except Exception:
        art = "  Rapid Relay"

    print()
    for line in art.split("\n"):
        print(f"{C.CYAN}{line}{C.RESET}")
    print(f"  {C.DIM}Flood Early Warning System  ·  Obando, Bulacan{C.RESET}")
    print()

    rows = [
        ("Sensor CSV",     _link(str(SENSOR_CSV))),
        ("Combined CSV",   _link(str(COMBINED_CSV))),
        ("Ingestion log",  _link(INGESTION_AUDIT_CSV)),
        ("Predictions",    _link(SPEEDTEST_PRED_CSV)),
        ("Predict window", f"{PREDICTION_HOUR_UTC:02d}:00–{PREDICTION_HOUR_UTC:02d}:{PREDICTION_WINDOW_MINUTES:02d} UTC  (08:00–08:30 PHT)"),
        ("Output TZ",      f"{OUTPUT_TZ}  (UTC+8 / PHT)"),
        ("Session ID",     _SESSION_ID),
    ]
    for k, v in rows:
        print(f"  {C.GRAY}{k:<18}{C.RESET}{C.BLUE}{v}{C.RESET}")
    print()
    print(f"  {C.DIM}Press Ctrl+C to stop.{C.RESET}\n")


def print_phase(num: int, title: str, subtitle: str = "") -> None:
    color = PHASE_COLORS.get(num, C.GRAY)
    sub   = f"  {C.DIM}{subtitle}{C.RESET}" if subtitle else ""
    bar   = f"{C.DIM}{'─' * 70}{C.RESET}"
    print(f"\n{bar}")
    print(f"  {C.BOLD}{color}[{num:02d}]{C.RESET}  {C.BOLD}{C.WHITE}{title}{C.RESET}{sub}")
    print(bar)


def print_tick_header(run_number: int, run_ts: str) -> None:
    w = 70
    print(f"\n{C.DIM}{'═' * w}{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}TICK #{run_number}{C.RESET}  {C.DIM}|{C.RESET}  {C.CYAN}{run_ts}{C.RESET}")
    print(f"{C.DIM}{'═' * w}{C.RESET}")


def print_prediction_result(
    tier: str,
    prob: float,
    prediction: dict,
    proxy_ran: bool,
    merge_rows: int,
    fb_posted: bool,
) -> None:
    tier_color = TIER_COLORS.get(tier, C.WHITE)
    tier_icon  = TIER_ICONS.get(tier, "●")
    w = 70
    print(f"\n{C.DIM}{'─' * w}{C.RESET}")
    print(
        f"  {C.BOLD}{tier_color}{tier_icon}  {tier}{C.RESET}"
        f"  {C.DIM}flood probability{C.RESET}"
        f"  {C.BOLD}{C.WHITE}{prob:.1%}{C.RESET}"
    )
    print(f"{C.DIM}{'─' * w}{C.RESET}")
    yes = f"{C.GREEN}yes{C.RESET}"
    no  = f"{C.GRAY}no{C.RESET}"
    # Resolve input_csv basename back to a full path for the OSC 8 link
    input_csv_name = prediction.get("input_csv", "-")
    if input_csv_name and input_csv_name != "-":
        if input_csv_name == COMBINED_CSV.name:
            input_csv_full = str(COMBINED_CSV)
        elif input_csv_name == SENSOR_CSV.name:
            input_csv_full = str(SENSOR_CSV)
        else:
            input_csv_full = input_csv_name
    else:
        input_csv_full = input_csv_name
    fields = [
        ("History rows",  f"{C.CYAN}{prediction['rows_in_history']}{C.RESET}"),
        ("Input CSV",     f"{C.GRAY}{_link(input_csv_full, input_csv_name)}{C.RESET}"),
        ("Proxy fetched", yes if proxy_ran  else no),
        ("Merge rows",    f"{C.CYAN}{merge_rows}{C.RESET}"),
        ("FB posted",     yes if fb_posted  else no),
    ]
    for k, v in fields:
        print(f"  {C.GRAY}{k:<16}{C.RESET}{v}")


def print_tick_footer(
    new_ingested: int,
    prediction_ran: bool,
    elapsed: float,
    status: str,
) -> None:
    # Availability bar
    avail_str = ""
    if os.path.exists(AVAILABILITY_CSV):
        try:
            _df = pd.read_csv(AVAILABILITY_CSV)
            _s  = _df[_df["session_id"] == _SESSION_ID]
            if not _s.empty:
                _a     = float(_s.iloc[-1].get("availability_pct", 0))
                filled = int(_a / 100 * 20)
                bar    = f"{C.GREEN}{'█' * filled}{C.DIM}{'░' * (20 - filled)}{C.RESET}"
                avail_str = f"  {C.GRAY}Availability  {C.RESET}{bar} {C.GREEN}{_a:.2f}%{C.RESET}"
        except Exception:
            pass

    status_color = {
        "ACTIVE":  C.GREEN,
        "SKIPPED": C.GRAY,
        "FAILED":  C.RED,
    }.get(status, C.GRAY)

    w = 70
    print(f"\n{C.DIM}{'─' * w}{C.RESET}")
    print(
        f"  {C.BOLD}{C.WHITE}Tick complete{C.RESET}"
        f"  {C.GRAY}elapsed{C.RESET} {C.CYAN}{elapsed:.2f}s{C.RESET}"
        f"  {C.GRAY}status{C.RESET} {C.BOLD}{status_color}{status}{C.RESET}"
    )
    print(
        f"  {C.GRAY}Rows ingested{C.RESET}  "
        f"{(C.GREEN if new_ingested > 0 else C.GRAY)}{new_ingested}{C.RESET}"
        f"  {C.GRAY}Prediction{C.RESET}  "
        f"{C.GREEN + 'RAN' + C.RESET if prediction_ran else C.GRAY + 'skipped' + C.RESET}"
    )
    if avail_str:
        print(avail_str)
    print(f"{C.DIM}{'═' * w}{C.RESET}\n")


def countdown(seconds: int, label: str = "Next run in") -> None:
    try:
        for remaining in range(seconds, 0, -1):
            mins, secs = divmod(remaining, 60)
            time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
            print(f"\r  {C.DIM}⏳  {label}: {C.CYAN}{time_str}{C.RESET}   ", end="", flush=True)
            time.sleep(1)
        print(f"\r  {C.GREEN}✓{C.RESET}  Starting next run...{' ' * 20}", flush=True)
        time.sleep(0.3)
        print(f"\r{' ' * 55}\r", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\r{' ' * 55}\r", end="", flush=True)
        raise


# ===========================================================================
# GENERAL HELPERS
# ===========================================================================

def is_prediction_time() -> bool:
    now_utc = datetime.now(pytz.UTC)
    return (now_utc.hour == PREDICTION_HOUR_UTC
            and now_utc.minute < PREDICTION_WINDOW_MINUTES)


def get_last_prediction_timestamp() -> str | None:
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f).get("last_prediction_timestamp")
    except Exception:
        return None


def save_state(last_ingest_id: int | None = None, last_predict_ts: str | None = None) -> None:
    os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
    state = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
        except Exception:
            pass
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
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f).get("last_ingested_id")
    except Exception:
        return None


def get_supabase():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            f"Supabase credentials missing.\n"
            f"  .env checked         : {_link(_ENV_PATH)}\n"
            f"  SUPABASE_URL         : {'SET' if SUPABASE_URL else 'MISSING'}\n"
            f"  SUPABASE_SERVICE_KEY : {'SET' if SUPABASE_KEY else 'MISSING'}\n"
        )
    from supabase import create_client
    _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# ===========================================================================
# SUPABASE FETCH
# ===========================================================================

def fetch_latest_row() -> list | None:
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


def fetch_all_rows() -> list | None:
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
    date_val = row.get(COL_DATE)
    time_val = row.get(COL_TIME)
    if not date_val or not time_val:
        return None
    raw = f"{date_val}T{time_val}"
    try:
        ts = pd.to_datetime(raw, utc=False, errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(raw, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            try:
                tz = pytz.timezone(SENSOR_TIMEZONE)
                ts = tz.localize(ts)
            except Exception:
                ts = ts.replace(tzinfo=pytz.UTC)
        return ts.astimezone(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


# ===========================================================================
# AVAILABILITY TRACKING
# ===========================================================================

def get_availability_state() -> tuple[int, int]:
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


def write_availability_row(status: str, supabase_row_id=None, dry_run=False):
    os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)

    session_uptime = 0
    session_total  = 0
    tick_number    = 1

    if os.path.exists(AVAILABILITY_CSV):
        try:
            df = pd.read_csv(AVAILABILITY_CSV)
            session_rows = df[df["session_id"] == _SESSION_ID]
            tick_number    = len(session_rows) + 1
            session_total  = len(session_rows)
            session_uptime = session_rows["status"].isin(["ACTIVE", "SKIPPED"]).sum()
        except Exception:
            pass

    total_ticks  = session_total + 1
    uptime_ticks = session_uptime + (1 if status in ("ACTIVE", "SKIPPED") else 0)
    avail_pct    = round((uptime_ticks / total_ticks) * 100, 4)

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

    df_row = pd.DataFrame([row])
    if dry_run:
        log.info("[DRY RUN] Availability row: %s", row)
        return

    if not os.path.exists(AVAILABILITY_CSV):
        df_row.to_csv(AVAILABILITY_CSV, index=False)
    else:
        df_row.to_csv(AVAILABILITY_CSV, mode="a", header=False, index=False)

    log.info(
        "Availability → tick=%d  status=%-7s  Aₒ=%.2f%%  (%d/%d ticks)",
        tick_number, status, avail_pct, uptime_ticks, total_ticks,
    )


# ===========================================================================
# SENSOR CSV APPEND
# ===========================================================================

def append_to_sensor_csv(row: dict, cal: dict, dry_run: bool = False) -> bool:
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
        log.debug("Created sensor CSV: %s", _link(str(SENSOR_CSV)))
        return True

    try:
        existing = pd.read_csv(SENSOR_CSV, usecols=["timestamp"])["timestamp"].astype(str)
        existing_norm = pd.to_datetime(existing, utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if ts_iso in existing_norm.values:
            log.debug("Timestamp %s already in sensor CSV — skipping.", ts_iso)
            return False
    except Exception as e:
        log.debug("Could not read existing sensor CSV timestamps (%s) — proceeding.", e)

    new_row.to_csv(SENSOR_CSV, mode="a", header=False, index=False)
    log.debug("Appended to sensor CSV: %s", ts_iso)
    return True


def _append_ingestion_audit_single(
    run_number: int,
    rows_fetched: int,
    new_rows_found: int,
    new_rows_ingested: int,
    last_ingested_id: int | None,
    last_ingested_timestamp: str | None,
    dry_run: bool = False,
) -> None:
    audit = {
        "run_number":             run_number,
        "run_timestamp":          now_iso(),
        "rows_fetched":           rows_fetched,
        "new_rows_found":         new_rows_found,
        "new_rows_ingested":      new_rows_ingested,
        "last_ingested_id":       last_ingested_id,
        "last_ingested_timestamp": last_ingested_timestamp,
        "ingestion_errors":       new_rows_found - new_rows_ingested,
    }
    df = pd.DataFrame([audit])
    os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
    if dry_run:
        log.info("[DRY RUN] Ingestion audit: %s", audit)
        return
    if not os.path.exists(INGESTION_AUDIT_CSV):
        df.to_csv(INGESTION_AUDIT_CSV, index=False)
    else:
        df.to_csv(INGESTION_AUDIT_CSV, mode="a", header=False, index=False)


# ===========================================================================
# STEP 0b — GEE SATELLITE PROXY FETCH
# ===========================================================================

def run_proxy_fetch(dry_run: bool = False) -> bool:
    try:
        import Sat_SensorData_proxy
    except ImportError as e:
        log.warning("Could not import Sat_SensorData_proxy.py — skipping proxy fetch. (%s)", e)
        return False

    log.info("Running GEE satellite proxy fetch (incremental)...")
    if dry_run:
        log.info("[DRY RUN] Would call Sat_SensorData_proxy.run_pipeline(force_full=False)")
        return False

    try:
        updated = Sat_SensorData_proxy.run_pipeline(force_full=False)
        if updated:
            log.info("✓ Proxy CSV updated with new GEE data.")
        else:
            log.info("Proxy CSV already up to date — no new rows from GEE.")
        return bool(updated)
    except Exception as e:
        log.error("GEE proxy fetch failed: %s", e)
        traceback.print_exc()
        return False


# ===========================================================================
# STEP 0c — SENSOR-PROXY MERGE
# ===========================================================================

def run_merge(dry_run: bool = False) -> int:
    try:
        import merge_sensor
    except ImportError as e:
        log.warning("Could not import merge_sensor.py — skipping merge. (%s)", e)
        return 0

    log.info("Running sensor-proxy merge (incremental)...")
    if dry_run:
        log.info("[DRY RUN] Would call merge_sensor.run_pipeline()")
        return 0

    try:
        new_rows = merge_sensor.run_pipeline()
        if new_rows is not None and len(new_rows) > 0:
            log.info("✓ Merged %d new row(s) into %s.", len(new_rows), _link(str(COMBINED_CSV)))
            return len(new_rows)
        else:
            log.info("%s already up to date — no new rows to merge.", _link(str(COMBINED_CSV)))
            return 0
    except Exception as e:
        log.error("Merge failed: %s", e)
        traceback.print_exc()
        return 0


# ===========================================================================
# PREDICTION
# ===========================================================================

_model_cache = None


def load_model():
    global _model_cache, LAST_TRAINING_DATE
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model not found: {_link(MODEL_FILE)}")

    artifact     = joblib.load(MODEL_FILE)
    model        = artifact["model"]
    feature_cols = artifact["feature_columns"]
    threshold    = artifact.get("threshold", DEFAULT_ALERT_THRESHOLD)
    watch_t      = artifact.get("watch_threshold", threshold)
    warn_t       = artifact.get("warning_threshold", threshold + 0.10)
    LAST_TRAINING_DATE = artifact.get("last_training_date", LAST_TRAINING_DATE)

    log.info("Model loaded: %s", _link(MODEL_FILE))
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


def run_prediction(use_combined: bool = True) -> dict | None:
    from prepare_dataset import load_sensor
    from feature_engineering import build_features

    if use_combined and COMBINED_CSV.exists():
        input_csv = COMBINED_CSV
        log.info("Predicting from merged combined CSV: %s", _link(str(input_csv)))
    elif SENSOR_CSV.exists():
        input_csv = SENSOR_CSV
        log.warning("combined_sensor_context.csv not found — falling back to %s.", _link(str(SENSOR_CSV)))
    else:
        log.error("No input CSV available for prediction.")
        return None

    try:
        model, feature_cols, watch_t, warn_t = load_model()
    except FileNotFoundError as e:
        log.error("Cannot load model: %s", e)
        return None

    try:
        sensor_df, freq = load_sensor(sensor_path=str(input_csv))
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

    pred_ts_pht = now_iso()

    log.info(
        "Prediction → %s %s  (%.1f%%)  on %d rows  [input: %s]",
        RISK_TIERS[latest_tier]["emoji"], latest_tier, latest_prob * 100,
        len(features), _link(str(input_csv)),
    )

    return {
        "tier":                  latest_tier,
        "probability":           latest_prob,
        "timestamp":             pred_ts_pht,
        "prediction_created_at": pred_ts_pht,
        "pred_epoch":            time.time(),
        "watch_threshold":       watch_t,
        "warning_threshold":     warn_t,
        "danger_threshold":      danger_t,
        "rows_in_history":       len(features),
        "input_csv":             os.path.basename(str(input_csv)),
    }


# ===========================================================================
# PREDICTION CSV (with PST normalization)
# ===========================================================================

def save_speedtest_prediction_row(
    pred: dict,
    sensor_timestamp: str | None = None,
    dry_run: bool = False,
) -> None:
    created_at_pht = now_iso()

    row = {
        "test_no":               None,
        "sensor_timestamp":      pht_iso(sensor_timestamp) if sensor_timestamp else None,
        "prediction_created_at": pht_iso(pred.get("prediction_created_at")),
        "row_created_at":        created_at_pht,
        "flood_probability":     pred.get("probability"),
        "risk_tier":             pred.get("tier"),
        "watch_threshold":       pred.get("watch_threshold"),
        "warning_threshold":     pred.get("warning_threshold"),
        "danger_threshold":      pred.get("danger_threshold"),
        "rows_in_history":       pred.get("rows_in_history"),
        "input_csv":             pred.get("input_csv"),
    }

    row["sensor_minus_row_created_at_sec"] = compute_time_diff_seconds(
        sensor_timestamp, pred.get("prediction_created_at")
    )

    os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)

    if os.path.exists(SPEEDTEST_PRED_CSV):
        try:
            existing = pd.read_csv(SPEEDTEST_PRED_CSV)
            row["test_no"] = len(existing) + 1
        except Exception:
            row["test_no"] = 1
    else:
        row["test_no"] = 1

    df_row = pd.DataFrame([row])

    if dry_run:
        log.info("[DRY RUN] Would append to predictions CSV:\n%s", df_row.to_string(index=False))
        return

    if not os.path.exists(SPEEDTEST_PRED_CSV):
        df_row.to_csv(SPEEDTEST_PRED_CSV, index=False)
    else:
        df_row.to_csv(SPEEDTEST_PRED_CSV, mode="a", header=False, index=False)

    log.info("Prediction row saved (test_no=%d, latency=%ss)", row["test_no"], row["sensor_minus_row_created_at_sec"])


def sync_prediction_to_supabase(pred: dict, dry_run: bool = False) -> bool:
    try:
        supabase = get_supabase()
    except Exception as e:
        log.error("Supabase client unavailable: %s", e)
        return False

    ts = pred.get("prediction_created_at") or pred.get("timestamp")
    if not ts:
        ts = datetime.now(pytz.UTC).isoformat()

    record = {
        "timestamp":         ts,
        "flood_probability": float(pred.get("probability", 0.0)),
        "risk_tier":         str(pred.get("tier", "")),
    }

    if dry_run:
        log.info("[DRY RUN] Would upsert prediction to Supabase: %s", record)
        return True

    try:
        res = supabase.table(SUPABASE_PRED_TABLE).upsert([record], on_conflict="timestamp").execute()
        if getattr(res, "error", None):
            log.error("Supabase upsert error: %s", res.error)
            return False
        log.info("Synced prediction to Supabase (timestamp=%s)", ts)
        return True
    except Exception as e:
        log.error("Supabase upsert failed: %s", e)
        return False


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
    if not tok:
        return "<missing>"
    s = str(tok)
    if len(s) <= 10:
        return s[:4] + "..." + s[-2:]
    return s[:6] + "..." + s[-4:]


def fb_token_info(dry_run: bool = False) -> dict | None:
    if not FB_PAGE_TOKEN:
        log.error("FB token missing — cannot run token diagnostics.")
        return None

    url    = f"https://graph.facebook.com/{FB_API_VERSION}/me"
    params = {"access_token": FB_PAGE_TOKEN, "fields": "id,name"}

    if dry_run:
        log.info("[DRY RUN] FB token diagnostics: GET %s", url)
        return None

    try:
        res = requests.get(url, params=params, timeout=10)
        try:
            data = res.json()
        except ValueError:
            data = {"_raw": res.text, "status_code": res.status_code}

        if isinstance(data, dict) and data.get("error"):
            log.error("FB token diagnostics failed: status=%s body=%s", res.status_code, data)
        else:
            log.info("FB token valid: status=%s body=%s", res.status_code, data)
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
    if not FB_PAGE_ID or not FB_PAGE_TOKEN:
        log.error(
            "FB credentials missing. FB_PAGE_ID=%s FB_PAGE_TOKEN=%s",
            "SET" if FB_PAGE_ID else "MISSING",
            "SET" if FB_PAGE_TOKEN else "MISSING",
        )
        return False

    last  = _load_fb_last_posted()
    today = datetime.now().strftime("%Y-%m-%d")
    if not force and last and last.startswith(today):
        log.info("FB: already posted today — skipping.")
        return True
    if force:
        log.info("FB: force post enabled — bypassing once-per-day guard.")

    template = FB_TIER_MESSAGE.get(tier, FB_TIER_MESSAGE["WARNING"])
    message  = template.format(prob_pct=f"{probability:.1%}", timestamp=timestamp)
    url      = f"https://graph.facebook.com/{FB_API_VERSION}/{FB_PAGE_ID}/feed"

    log.info("Posting daily %s alert to Facebook...", tier)

    if dry_run:
        log.info("[DRY RUN] Would post to Facebook:\n%s", message)
        _save_fb_last_posted(now_iso())
        return True

    try:
        log.debug("FB POST url=%s token=%s len=%d", url, _mask_token(FB_PAGE_TOKEN), len(message))

        try:
            fb_token_info(dry_run=False)
        except Exception:
            pass

        payload = {"message": message, "access_token": FB_PAGE_TOKEN}
        res     = requests.post(url, data=payload, timeout=15)

        try:
            data = res.json()
        except ValueError:
            data = {"_raw": res.text, "status_code": res.status_code}

        if res.status_code >= 400 or (isinstance(data, dict) and data.get("error")):
            err      = data.get("error") if isinstance(data, dict) else data
            err_type = err.get("type") if isinstance(err, dict) else None
            err_code = err.get("code") if isinstance(err, dict) else None
            err_msg  = err.get("message") if isinstance(err, dict) else str(err)
            trace    = res.headers.get("x-fb-trace-id") or "-"
            log.error(
                "FB API error status=%s type=%s code=%s trace=%s message=%s",
                res.status_code, err_type, err_code, trace, err_msg
            )
            return False

        post_id = data.get("id") if isinstance(data, dict) else None
        log.info("FB post successful — post id: %s", post_id or "<unknown>")
        _save_fb_last_posted(now_iso())
        return True

    except requests.Timeout:
        log.error("FB request timed out.")
        return False
    except Exception as e:
        log.exception("FB unexpected error: %s", e)
        return False


# ===========================================================================
# MAIN PIPELINE TICK
# ===========================================================================

def run_once(
    run_number:    int,
    ingest_number: int,
    force_predict: bool = False,
    dry_run:       bool = False,
    force_fb:      bool = False,
    skip_proxy:    bool = False,
    skip_merge:    bool = False,
) -> tuple[int, bool]:
    tick_start = time.time()
    run_ts     = now_iso()

    print_tick_header(run_number, run_ts)

    # ------------------------------------------------------------------ #
    # PHASE 1: REAL-TIME HARDWARE INGEST
    # ------------------------------------------------------------------ #
    print_phase(1, "Real-time Hardware Ingest", "Supabase")

    all_rows = fetch_latest_row()
    if all_rows is None:
        log.error("Supabase fetch failed.")
        write_availability_row("FAILED", dry_run=dry_run)
        return 0, False

    total_fetched  = len(all_rows)
    last_ingest_id = get_last_ingested_id()

    if total_fetched == 0:
        log.info("Scanned Supabase: 0 rows returned.")
        new_rows = []
    else:
        new_rows = (
            all_rows
            if last_ingest_id is None
            else [r for r in all_rows if (r.get(COL_ID) or 0) > (last_ingest_id or 0)]
        )
        sample_newest = _row_timestamp_to_utc_iso(all_rows[0]) or "<no-ts>"
        missing_ts    = sum(1 for r in new_rows if not r.get(COL_DATE) or not r.get(COL_TIME))
        log.info(
            "Scanned: fetched=%d  last_ingest_id=%s  new=%d  missing_ts=%d  newest=%s",
            total_fetched, last_ingest_id, len(new_rows), missing_ts, sample_newest
        )

    new_ingested  = 0
    last_row_data = None

    for row in new_rows:
        row_id = row.get(COL_ID)
        cal    = calibrate_row(row)
        if cal is None:
            continue
        if append_to_sensor_csv(row, cal, dry_run=dry_run):
            new_ingested += 1
            last_row_data = {
                "row_id": row_id,
                "date":   row.get(COL_DATE),
                "time":   row.get(COL_TIME),
                "cal":    cal,
            }

    if new_ingested > 0:
        log.info("✓ Ingested %d new hardware row(s).", new_ingested)
        if not dry_run:
            save_state(last_ingest_id=last_row_data["row_id"])

    if not dry_run:
        ingestion_audit = {
            "run_number":             run_number,
            "run_timestamp":          run_ts,
            "rows_fetched":           total_fetched,
            "new_rows_found":         len(new_rows),
            "new_rows_ingested":      new_ingested,
            "last_ingested_id":       last_row_data["row_id"] if last_row_data else None,
            "last_ingested_timestamp": (
                f"{last_row_data['date']} {last_row_data['time']}" if last_row_data else None
            ),
            "ingestion_errors":       len(new_rows) - new_ingested,
        }
        os.makedirs(SPEEDTEST_OUTPUT_DIR, exist_ok=True)
        audit_df = pd.DataFrame([ingestion_audit])
        if not os.path.exists(INGESTION_AUDIT_CSV):
            audit_df.to_csv(INGESTION_AUDIT_CSV, index=False)
        else:
            audit_df.to_csv(INGESTION_AUDIT_CSV, mode="a", header=False, index=False)
        log.info("Ingestion audit recorded.")

    # ------------------------------------------------------------------ #
    # DAILY PIPELINE GATE
    # ------------------------------------------------------------------ #
    last_pred_ts_raw = get_last_prediction_timestamp()
    last_pred_date   = None
    if last_pred_ts_raw:
        try:
            parsed = pd.to_datetime(last_pred_ts_raw, utc=True, errors="coerce")
            last_pred_date = (
                parsed.strftime("%Y-%m-%d") if not pd.isna(parsed)
                else str(last_pred_ts_raw)[:10]
            )
        except Exception:
            last_pred_date = str(last_pred_ts_raw)[:10]

    last_row_date_utc = None
    if last_row_data:
        row_iso = _row_timestamp_to_utc_iso(
            {COL_DATE: last_row_data.get("date"), COL_TIME: last_row_data.get("time")}
        )
        if row_iso:
            try:
                last_row_date_utc = pd.to_datetime(row_iso, utc=True).strftime("%Y-%m-%d")
            except Exception:
                last_row_date_utc = str(row_iso)[:10]

    should_run_daily = (
        force_predict
        or is_prediction_time()
        or (
            new_ingested > 0
            and last_row_date_utc
            and last_row_date_utc != last_pred_date
        )
    )

    if (
        should_run_daily
        and not force_predict
        and last_pred_date
        and last_row_date_utc == last_pred_date
    ):
        log.info("Already predicted for UTC date %s — skipping daily pipeline.", last_pred_date)
        should_run_daily = False

    prediction_ran = False
    proxy_ran      = False
    merge_rows     = 0
    fb_posted      = False

    if should_run_daily:
        # -------------------------------------------------------------- #
        # PHASE 2: GEE SATELLITE PROXY FETCH
        # -------------------------------------------------------------- #
        print_phase(2, "GEE Satellite Proxy Fetch")
        if skip_proxy:
            log.info("Skipped — --skip-proxy flag set.")
        else:
            proxy_ran = run_proxy_fetch(dry_run=dry_run)

        # -------------------------------------------------------------- #
        # PHASE 3: SENSOR-PROXY MERGE
        # -------------------------------------------------------------- #
        print_phase(3, "Sensor-Proxy Merge")
        if skip_merge:
            log.info("Skipped — --skip-merge flag set.")
        else:
            merge_rows = run_merge(dry_run=dry_run)
            if merge_rows == 0 and not COMBINED_CSV.exists():
                log.warning(
                    "Merge produced no output and combined CSV missing. "
                    "Prediction will fall back to sensor-only CSV."
                )

        # -------------------------------------------------------------- #
        # PHASE 4: DAILY PREDICTION
        # -------------------------------------------------------------- #
        print_phase(4, "Daily Prediction")
        log.info("Running daily prediction on accumulated history...")

        prediction = run_prediction(use_combined=(not skip_merge))

        if prediction is None:
            log.error("Prediction failed.")
        else:
            prediction_ran = True
            tier = prediction["tier"]
            prob = prediction["probability"]

            sensor_iso = None
            if last_row_data:
                sensor_iso = _row_timestamp_to_utc_iso(
                    {COL_DATE: last_row_data.get("date"), COL_TIME: last_row_data.get("time")}
                )

            try:
                save_speedtest_prediction_row(
                    prediction,
                    sensor_timestamp=sensor_iso,
                    dry_run=dry_run,
                )
            except Exception as e:
                log.debug("Could not save prediction row: %s", e)

            try:
                sync_prediction_to_supabase(prediction, dry_run=dry_run)
            except Exception as e:
                log.debug("Could not sync prediction to Supabase: %s", e)

            fb_posted = send_fb_alert(
                tier=tier,
                probability=prob,
                timestamp=prediction["timestamp"],
                dry_run=dry_run,
                force=force_fb,
            )

            if not dry_run:
                pred_audit = {
                    "prediction_number":     ingest_number,
                    "prediction_timestamp":  prediction["timestamp"],
                    "rows_in_history":       prediction["rows_in_history"],
                    "flood_probability":     prob,
                    "risk_tier":             tier,
                    "watch_threshold":       prediction.get("watch_threshold"),
                    "warning_threshold":     prediction.get("warning_threshold"),
                    "danger_threshold":      prediction.get("danger_threshold"),
                    "fb_posted":             fb_posted,
                    "prediction_created_at": prediction["prediction_created_at"],
                    "proxy_fetched":         proxy_ran,
                    "merge_rows":            merge_rows,
                    "input_csv":             prediction.get("input_csv"),
                }
                pred_df = pd.DataFrame([pred_audit])
                if not os.path.exists(PREDICTION_AUDIT_CSV):
                    pred_df.to_csv(PREDICTION_AUDIT_CSV, index=False)
                else:
                    pred_df.to_csv(PREDICTION_AUDIT_CSV, mode="a", header=False, index=False)

                try:
                    save_state(last_predict_ts=datetime.now(pytz.UTC).strftime("%Y-%m-%d"))
                except Exception:
                    save_state(last_predict_ts=prediction.get("prediction_created_at", ""))

            print_prediction_result(
                tier=tier,
                prob=prob,
                prediction=prediction,
                proxy_ran=proxy_ran,
                merge_rows=merge_rows,
                fb_posted=fb_posted,
            )

    # ------------------------------------------------------------------ #
    # AVAILABILITY + TICK FOOTER
    # ------------------------------------------------------------------ #
    elapsed = time.time() - tick_start

    if all_rows is None:
        status = "FAILED"
    elif new_ingested > 0:
        if should_run_daily:
            status = "ACTIVE" if prediction_ran else "FAILED"
        else:
            status = "ACTIVE"
    else:
        status = "SKIPPED"

    supabase_row_id = last_row_data["row_id"] if last_row_data else None
    try:
        write_availability_row(status=status, supabase_row_id=supabase_row_id, dry_run=dry_run)
    except Exception as e:
        log.debug("Could not write availability row: %s", e)

    print_tick_footer(
        new_ingested=new_ingested,
        prediction_ran=prediction_ran,
        elapsed=elapsed,
        status=status,
    )

    return new_ingested, prediction_ran


# ===========================================================================
# REALTIME LISTENER
# ===========================================================================

def start_realtime_listener(
    poll_interval:     int  = 5,
    dry_run:           bool = False,
    predict_on_insert: bool = False,
    skip_proxy:        bool = False,
    skip_merge:        bool = False,
    force_fb:          bool = False,
) -> None:
    supabase = get_supabase()

    def handle_payload(payload):
        new = None
        if isinstance(payload, dict):
            new = payload.get("new") or payload.get("record")
            if new is None:
                inner = payload.get("payload")
                if isinstance(inner, dict):
                    new = inner.get("new") or inner.get("record")

        if not new:
            log.debug("Realtime payload without record: %s", payload)
            return

        row_id = new.get(COL_ID)
        log.info("Realtime INSERT id=%s", row_id)

        cal = calibrate_row(new)
        if cal is None:
            log.debug("Realtime row id=%s calibration failed.", row_id)
            write_availability_row("FAILED", supabase_row_id=row_id, dry_run=dry_run)
            _append_ingestion_audit_single(1, 1, 1, 0, None, None, dry_run=dry_run)
            return

        appended = append_to_sensor_csv(new, cal, dry_run=dry_run)
        if appended and not dry_run:
            try:
                save_state(last_ingest_id=row_id)
            except Exception:
                pass

        pred_ok = False
        if predict_on_insert:
            if not skip_proxy:
                run_proxy_fetch(dry_run=dry_run)
            if not skip_merge:
                run_merge(dry_run=dry_run)

            pred = run_prediction(use_combined=(not skip_merge))
            if pred is not None:
                pred_ok = True
                sensor_iso = _row_timestamp_to_utc_iso(new)
                try:
                    save_speedtest_prediction_row(pred, sensor_timestamp=sensor_iso, dry_run=dry_run)
                except Exception as e:
                    log.debug("Could not save realtime prediction row: %s", e)
                try:
                    sync_prediction_to_supabase(pred, dry_run=dry_run)
                except Exception as e:
                    log.debug("Could not sync realtime prediction: %s", e)
                try:
                    fb_ok = send_fb_alert(
                        tier=pred.get("tier"),
                        probability=pred.get("probability"),
                        timestamp=pred.get("prediction_created_at") or pred.get("timestamp"),
                        dry_run=dry_run,
                        force=(force_fb or ALWAYS_POST_FB),
                    )
                    log.info("Realtime FB post: %s", "posted" if fb_ok else "not-posted")
                except Exception as e:
                    log.exception("Realtime FB post failed: %s", e)

        try:
            _append_ingestion_audit_single(
                1, 1, 1, 1 if appended else 0,
                row_id if appended else None,
                f"{new.get(COL_DATE)} {new.get(COL_TIME)}" if appended else None,
                dry_run=dry_run,
            )
        except Exception:
            pass

        status = (
            "ACTIVE" if appended and (not predict_on_insert or pred_ok)
            else ("SKIPPED" if not appended else "FAILED")
        )
        try:
            write_availability_row(status=status, supabase_row_id=row_id, dry_run=dry_run)
        except Exception as e:
            log.debug("Could not write realtime availability row: %s", e)

    try:
        log.info("Starting Supabase realtime listener on '%s'...", TABLE_ENV_DATA)
        supabase.from_(TABLE_ENV_DATA).on("INSERT", lambda p: handle_payload(p)).subscribe()
        while True:
            time.sleep(poll_interval)
    except Exception as e:
        log.warning("Realtime subscribe failed (%s). Falling back to polling every %ds.", e, poll_interval)
        last_seen = get_last_ingested_id() or 0
        try:
            while True:
                rows = fetch_latest_row() or []
                new_rows = [r for r in rows if (r.get(COL_ID) or 0) > (last_seen or 0)]
                if new_rows:
                    log.info("Realtime poll: %d new row(s) found.", len(new_rows))
                    for r in reversed(new_rows):
                        handle_payload({"new": r})
                        last_seen = max(last_seen, r.get(COL_ID) or last_seen)
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            log.info("Realtime polling stopped by user.")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Rapid Relay — Full Hybrid Pipeline\n"
            "Phases: hardware ingest → GEE proxy → merge → daily prediction → FB alert\n"
            "Prediction always uses combined_sensor_context.csv (sensor + proxy merged)."
        )
    )
    parser.add_argument("--interval",          type=int, default=30,
                        help="Loop interval in seconds (default: 30)")
    parser.add_argument("--once",              action="store_true",
                        help="Run one tick and exit")
    parser.add_argument("--force-predict",     action="store_true",
                        help="Force prediction now (ignore time window)")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Preview only, no writes")
    parser.add_argument("--skip-proxy",        action="store_true",
                        help="Skip GEE satellite proxy fetch (Phase 2)")
    parser.add_argument("--skip-merge",        action="store_true",
                        help="Skip sensor-proxy merge (Phase 3)")
    parser.add_argument("--realtime",          action="store_true",
                        help="Run Supabase realtime listener instead of polling loop")
    parser.add_argument("--predict-on-insert", action="store_true",
                        help="Run prediction on every INSERT (realtime mode only, expensive)")
    parser.add_argument("--realtime-interval", type=int, default=5,
                        help="Polling fallback interval in seconds (default: 5)")
    parser.add_argument("--test-fb",           action="store_true",
                        help="Test Facebook posting once and exit")
    parser.add_argument("--fb-debug",          action="store_true",
                        help="Run FB token diagnostics and verbose test post")
    parser.add_argument("--force-fb",          action="store_true",
                        help="Bypass once-per-day FB guard on every prediction")
    args = parser.parse_args()

    print_startup_banner()

    if args.test_fb:
        print(f"  {C.CYAN}Testing Facebook posting{C.RESET}  {C.DIM}dry_run={args.dry_run}{C.RESET}")
        ok = send_fb_alert("WARNING", 0.5, now_iso(), dry_run=args.dry_run, force=(args.force_fb or ALWAYS_POST_FB))
        result_color = C.GREEN if ok else C.RED
        print(f"\n  Facebook test: {C.BOLD}{result_color}{'OK' if ok else 'FAILED'}{C.RESET}\n")
        sys.exit(0)

    if args.fb_debug:
        print(f"  {C.CYAN}FB token diagnostics{C.RESET}  {C.DIM}dry_run={args.dry_run}{C.RESET}")
        info = fb_token_info(dry_run=args.dry_run)
        print(f"  {C.GRAY}token_present{C.RESET}  {'yes' if FB_PAGE_TOKEN else C.RED + 'no' + C.RESET}")
        if info is not None:
            print(f"  {C.GRAY}token_info   {C.RESET}  {info}")
        ok = send_fb_alert("WARNING", 0.5, now_iso(), dry_run=args.dry_run, force=(args.force_fb or ALWAYS_POST_FB))
        result_color = C.GREEN if ok else C.RED
        print(f"\n  FB post attempt: {C.BOLD}{result_color}{'OK' if ok else 'FAILED'}{C.RESET}\n")
        sys.exit(0)

    if args.once:
        new_rows, pred_ran = run_once(
            run_number=1, ingest_number=1,
            force_predict=args.force_predict,
            dry_run=args.dry_run,
            force_fb=(args.force_fb or ALWAYS_POST_FB),
            skip_proxy=args.skip_proxy,
            skip_merge=args.skip_merge,
        )
        ingested_color = C.GREEN if new_rows > 0 else C.GRAY
        pred_color     = C.GREEN if pred_ran else C.GRAY
        print(
            f"  {C.GRAY}Rows ingested{C.RESET} {ingested_color}{new_rows}{C.RESET}  "
            f"{C.GRAY}Prediction{C.RESET} {pred_color}{'RAN' if pred_ran else 'SKIPPED'}{C.RESET}\n"
        )
        sys.exit(0)

    if args.realtime:
        start_realtime_listener(
            poll_interval=args.realtime_interval,
            dry_run=args.dry_run,
            predict_on_insert=args.predict_on_insert,
            skip_proxy=args.skip_proxy,
            skip_merge=args.skip_merge,
            force_fb=(args.force_fb or ALWAYS_POST_FB),
        )
        sys.exit(0)

    # ── Continuous polling loop ──────────────────────────────────────────
    run_number    = 1
    ingest_number = 1

    while True:
        try:
            new_rows, pred_ran = run_once(
                run_number=run_number,
                ingest_number=ingest_number,
                force_predict=args.force_predict,
                dry_run=args.dry_run,
                force_fb=(args.force_fb or ALWAYS_POST_FB),
                skip_proxy=args.skip_proxy,
                skip_merge=args.skip_merge,
            )
            if new_rows > 0 or pred_ran:
                ingest_number += 1
            run_number += 1
        except KeyboardInterrupt:
            print(f"\n  {C.YELLOW}Stopped by user.{C.RESET}\n")
            sys.exit(0)
        except Exception as exc:
            log.error("Unhandled error: %s", exc, exc_info=True)

        try:
            countdown(args.interval, label="Next run in")
        except KeyboardInterrupt:
            print(f"\n  {C.YELLOW}Stopped by user.{C.RESET}\n")
            sys.exit(0)