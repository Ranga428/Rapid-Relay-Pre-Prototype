"""
showcase_sensor_ingest.py
=========================
SHOWCASE VERSION of sensor_ingest.py

Changes from original:
  - Output CSV  : showcase_sensor.csv  (instead of obando_sensor_data.csv)
  - Output RAW  : showcase_sensor_raw.csv  (instead of obando_sensor_data_raw.csv)
  - Trigger mode: ingest_on_new_row() — two modes available:
      * REALTIME (default): Supabase Realtime websocket subscription — fires
        immediately on INSERT with zero polling delay.
      * POLLING (fallback): polls Supabase row count every N seconds.
  - Null date/time guard in _rows_to_df to prevent ValueError crashes.
  - All script references updated to the showcase/ folder.
  - All other calibration logic is identical to the original.
  - ingest_latest() updated to full-sync mode: fetches ALL Supabase rows and
    appends any that are missing from the CSV, regardless of insertion order.

Usage
-----
    python showcase_sensor_ingest.py                       # one-shot incremental ingest
    python showcase_sensor_ingest.py --watch               # realtime watch (default)
    python showcase_sensor_ingest.py --watch --poll        # fallback polling mode
    python showcase_sensor_ingest.py --watch --interval 30 # poll every 30s (poll mode)
    python showcase_sensor_ingest.py --date 2025-04-01     # backfill a specific date
    python showcase_sensor_ingest.py --dry-run             # preview without writing
    python showcase_sensor_ingest.py --check-calibration   # print calibration summary
"""

import os
import time
import argparse
import logging
import threading
from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths  — all outputs go to data/sensor/, same as the originals
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root is one level up from showcase/
_PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
_ENV_PATH     = os.path.normpath(os.path.join(_PROJECT_ROOT, "..", ".env"))

# Showcase-specific output CSVs (same data/sensor/ folder as originals)
SENSOR_CSV = Path(os.getenv(
    "SHOWCASE_SENSOR_CSV_PATH",
    os.path.normpath(os.path.join(_PROJECT_ROOT, "data", "sensor", "showcase_sensor.csv"))
))

SENSOR_CSV_RAW = Path(os.getenv(
    "SHOWCASE_SENSOR_CSV_RAW_PATH",
    os.path.normpath(os.path.join(_PROJECT_ROOT, "data", "sensor", "showcase_sensor_raw.csv"))
))

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
_loaded = load_dotenv(_ENV_PATH, override=False)

from supabase import create_client, Client

# ===========================================================================
# HARDWARE CALIBRATION CONSTANTS  (identical to original sensor_ingest.py)
# ===========================================================================

DIKE_HEIGHT_M = 4.039

HW_WL_DRY    = 3.819
HW_WL_WET    = 3.999
PROXY_WL_DRY = 0.718
PROXY_WL_WET = 2.197

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
# CONFIG
# ===========================================================================

USE_HARDWARE = True

TABLE_ENV_DATA    = "obando_environmental_data"
TABLE_PREDICTIONS = "predictions"

COL_DATE     = "Date"
COL_TIME     = "Time"
COL_SOIL     = "Soil Moisture"
COL_HUMIDITY = "Humidity"
COL_DISTANCE = "Final Distance"

_SELECT = '"Date", "Time", "Soil Moisture", "Humidity", "Final Distance"'

# Default polling interval in seconds for --watch --poll fallback mode
DEFAULT_POLL_INTERVAL = 60

log = logging.getLogger("showcase_sensor_ingest")

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
    log.debug("Supabase client initialised.")
    return _supabase_client


# ---------------------------------------------------------------------------
# Calibration functions  (identical to original)
# ---------------------------------------------------------------------------

def _calibrate_waterlevel(distance_m: float) -> float | None:
    if distance_m <= DISTANCE_MIN_VALID_M or distance_m >= DISTANCE_MAX_VALID_M:
        return None
    hw_wl_m = DIKE_HEIGHT_M - distance_m
    t = (hw_wl_m - HW_WL_DRY) / (HW_WL_WET - HW_WL_DRY)
    t = max(0.0, min(1.0, t))
    return round(PROXY_WL_DRY + t * (PROXY_WL_WET - PROXY_WL_DRY), 6)


def _calibrate_soil_moisture(hw_pct: float) -> float:
    t = (hw_pct - SOIL_HW_DRY) / (SOIL_HW_WET - SOIL_HW_DRY)
    t = max(0.0, min(1.0, t))
    return round(SOIL_PROXY_DRY + t * (SOIL_PROXY_WET - SOIL_PROXY_DRY), 6)


def _calibrate_humidity(hw_rh: float) -> float:
    t = (hw_rh - HUMIDITY_HW_MIN) / (HUMIDITY_HW_MAX - HUMIDITY_HW_MIN)
    t = max(0.0, min(1.0, t))
    return round(HUMIDITY_PROXY_MIN + t * (HUMIDITY_PROXY_MAX - HUMIDITY_PROXY_MIN), 6)


def calibrate_df(df: pd.DataFrame) -> pd.DataFrame:
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
    print()
    print("=" * 78)
    print("  SENSOR CALIBRATION SUMMARY — Showcase (Obando Flood EWS)")
    print("=" * 78)
    print(f"\n  Output CSV (calibrated) : {SENSOR_CSV}")
    print(f"  Output CSV (raw)        : {SENSOR_CSV_RAW}")
    print(f"\n  Dike height             : {DIKE_HEIGHT_M} m")
    print(f"  Dry anchor              : hw_wl={HW_WL_DRY} m  →  proxy={PROXY_WL_DRY} m")
    print(f"  Wet anchor (est.)       : hw_wl={HW_WL_WET} m  →  proxy={PROXY_WL_WET} m")
    print()


# ---------------------------------------------------------------------------
# Supabase fetch helpers
# ---------------------------------------------------------------------------

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "waterlevel", "soil_moisture", "humidity"])


def _rows_to_df(data: list[dict]) -> pd.DataFrame:
    """
    Convert raw Supabase rows to a normalised DataFrame.

    FIX: Rows where Date or Time is None / "None" / NaN are dropped
    before pd.to_datetime is called, preventing the crash:
        ValueError: time data "None None" doesn't match format "%Y-%m-%d %H:%M:%S"
    """
    if not data:
        return _empty_df()

    df = pd.DataFrame(data)

    # ── Guard: drop rows with null or string-"None" Date / Time ───────────
    for col in (COL_DATE, COL_TIME):
        if col not in df.columns:
            log.warning("Expected column '%s' not found in Supabase response.", col)
            return _empty_df()

    # Replace Python None and the string "None" with pd.NA, then drop
    df[COL_DATE] = df[COL_DATE].replace("None", pd.NA)
    df[COL_TIME] = df[COL_TIME].replace("None", pd.NA)
    null_mask = df[COL_DATE].isna() | df[COL_TIME].isna()
    n_null = null_mask.sum()
    if n_null:
        log.warning(
            "Dropped %d row(s) with null Date or Time before timestamp parse.", n_null
        )
        df = df[~null_mask].copy()

    if df.empty:
        log.warning("All rows were dropped due to null Date/Time — returning empty DataFrame.")
        return _empty_df()

    # ── Parse timestamps ───────────────────────────────────────────────────
    df["timestamp"] = (
        pd.to_datetime(
            df[COL_DATE].astype(str) + " " + df[COL_TIME].astype(str),
            format="mixed",      # handles slight format variation across rows
            utc=True,
        )
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
    )

    df = df.rename(columns={
        COL_DISTANCE: "waterlevel",
        COL_SOIL:     "soil_moisture",
        COL_HUMIDITY: "humidity",
    })
    return df[["timestamp", "waterlevel", "soil_moisture", "humidity"]]


def fetch_rows_since(after_timestamp: str | None) -> pd.DataFrame:
    """
    Fetch rows from Supabase.

    If after_timestamp is None, fetches ALL rows (full sync).
    If after_timestamp is provided, fetches only rows after that date
    (legacy cutoff behaviour — still used by ingest_date).
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
            # Only apply date filter if a cutoff was explicitly given
            if cutoff_date:
                query = query.gt(COL_DATE, cutoff_date)
            response = query.execute()
        except Exception as exc:
            log.error("Supabase fetch failed at offset %d: %s", offset, exc)
            break

        batch = response.data or []
        all_rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    log.info("Total raw rows fetched from Supabase: %d", len(all_rows))
    return _rows_to_df(all_rows)


def fetch_rows_for_date(target_date: str) -> pd.DataFrame:
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
            log.error("Supabase fetch failed: %s", exc)
            break

        batch = response.data or []
        all_rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    return _rows_to_df(all_rows)


def get_supabase_row_count() -> int:
    """Return the current total row count in the Supabase table."""
    try:
        response = (
            get_client()
            .table(TABLE_ENV_DATA)
            .select("id", count="exact")
            .execute()
        )
        return response.count or 0
    except Exception as e:
        log.warning("Could not get row count: %s", e)
        return -1


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


def _existing_timestamps(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["timestamp"])
        return set(df["timestamp"].astype(str).tolist())
    except Exception:
        return set()


def append_rows_to_csv(rows: pd.DataFrame, dry_run: bool = False) -> int:
    if rows.empty:
        return 0

    rows = rows[["timestamp", "waterlevel", "soil_moisture", "humidity"]].copy()
    existing = _existing_timestamps(SENSOR_CSV)
    before   = len(rows)
    rows     = rows[~rows["timestamp"].astype(str).isin(existing)]
    skipped  = before - len(rows)
    if skipped:
        log.warning("Skipped %d row(s) already present in showcase_sensor.csv.", skipped)

    if rows.empty:
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
    if rows.empty:
        return 0

    raw = rows[["timestamp", "waterlevel", "soil_moisture", "humidity"]].copy()
    raw["waterlevel"] = (DIKE_HEIGHT_M - raw["waterlevel"]).round(6)
    raw["date"] = raw["timestamp"].str[:10]
    daily_raw = (
        raw.groupby("date", sort=True)[["waterlevel", "soil_moisture", "humidity"]]
        .mean()
        .round(6)
        .reset_index()
    )
    daily_raw["timestamp"] = daily_raw["date"] + "T00:00:00"
    raw = daily_raw[["timestamp", "waterlevel", "soil_moisture", "humidity"]]

    existing = _existing_timestamps(SENSOR_CSV_RAW)
    before   = len(raw)
    raw      = raw[~raw["timestamp"].astype(str).isin(existing)]
    skipped  = before - len(raw)
    if skipped:
        log.warning("Skipped %d raw row(s) already present in showcase_sensor_raw.csv.", skipped)

    if raw.empty:
        return 0

    SENSOR_CSV_RAW.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.info("[DRY RUN] Would append %d raw row(s):", len(raw))
        print(raw.to_string(index=False))
        return len(raw)

    if not SENSOR_CSV_RAW.exists():
        raw.to_csv(SENSOR_CSV_RAW, index=False)
    else:
        raw.to_csv(SENSOR_CSV_RAW, mode="a", header=False, index=False)
        log.info("Appended %d raw row(s) to %s", len(raw), SENSOR_CSV_RAW)

    return len(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_latest(dry_run: bool = False) -> bool:
    """
    Full-sync ingest. Fetches ALL rows from Supabase, calibrates them to
    daily aggregates, then appends any rows not already present in
    showcase_sensor.csv — regardless of when they were inserted.

    This means late-arriving or backfilled historical rows in Supabase will
    always be picked up, not just rows newer than the CSV's latest timestamp.

    Returns True if at least one new row was written.
    """
    if not USE_HARDWARE:
        log.info("USE_HARDWARE=False — ingestion skipped.")
        return False

    # Get all timestamps already in the CSV
    existing_ts = _existing_timestamps(SENSOR_CSV)
    log.info("CSV currently has %d row(s).", len(existing_ts))

    # Fetch ALL rows from Supabase (no date cutoff)
    raw = fetch_rows_since(after_timestamp=None)

    if raw.empty:
        log.info("No rows in Supabase — nothing to do.")
        return False

    # Calibrate to daily aggregates
    calibrated = calibrate_df(raw)

    if calibrated.empty:
        log.warning("All fetched rows were discarded as invalid sensor readings.")
        return False

    # Find which calibrated rows are missing from the CSV
    new_rows = calibrated[~calibrated["timestamp"].astype(str).isin(existing_ts)]

    if new_rows.empty:
        log.info(
            "CSV already has all %d Supabase row(s) — up to date.", len(calibrated)
        )
        return False

    log.info(
        "Found %d new/missing row(s) not in CSV (Supabase has %d, CSV has %d).",
        len(new_rows), len(calibrated), len(existing_ts),
    )

    append_raw_rows_to_csv(raw, dry_run=dry_run)
    written = append_rows_to_csv(new_rows, dry_run=dry_run)
    return written > 0


# ---------------------------------------------------------------------------
# REALTIME watch mode  (event-driven, zero polling latency)
# ---------------------------------------------------------------------------

def ingest_on_new_row(poll_interval: int = DEFAULT_POLL_INTERVAL,
                      dry_run: bool = False,
                      use_realtime: bool = True) -> None:
    """
    Watch mode — triggers ingest the moment a new row is inserted into
    obando_environmental_data in Supabase.

    Two strategies (selected by use_realtime):

    REALTIME (default, use_realtime=True):
        Uses the Supabase Realtime websocket subscription via
        supabase-py's channel API.  The callback fires immediately
        on every INSERT event — no polling interval is needed.
        Falls back to polling mode automatically if the Realtime
        subscription cannot be established.

    POLLING (use_realtime=False):
        Polls the Supabase row count every poll_interval seconds
        and fires ingest_latest() whenever the count increases.
        Use this if your Supabase plan does not include Realtime,
        or if you are running behind a firewall that blocks websockets.

    Parameters
    ----------
    poll_interval : seconds between polls in POLLING mode (default 60)
    dry_run       : if True, preview output without writing to disk
    use_realtime  : True = Realtime subscription, False = polling fallback
    """
    if use_realtime:
        _watch_realtime(dry_run=dry_run, poll_interval=poll_interval)
    else:
        _watch_polling(poll_interval=poll_interval, dry_run=dry_run)


def _watch_realtime(dry_run: bool = False,
                    poll_interval: int = DEFAULT_POLL_INTERVAL) -> None:
    """
    Supabase Realtime watch.

    The supabase-py Realtime client fires a Python callback synchronously
    on the same thread that processes websocket frames.  We do the heavy
    ingest work on a separate daemon thread so we never block the socket.
    """
    print(f"\n  [showcase_sensor_ingest] Realtime watch mode started.")
    print(f"  Listening for INSERT events on '{TABLE_ENV_DATA}' via Supabase Realtime.")
    print(f"  Output → {SENSOR_CSV}")
    print(f"  Press Ctrl+C to stop.\n")

    client = get_client()

    # Guard against concurrent ingest calls fired from rapid bursts of inserts
    _ingest_lock = threading.Lock()

    def _on_insert(payload):
        """Called by the Realtime client on every INSERT."""
        ts_str = pd.Timestamp.now().strftime("%H:%M:%S")
        record = payload.get("record") or payload.get("new") or {}
        date_val = record.get(COL_DATE, "?")
        time_val = record.get(COL_TIME, "?")
        print(f"\n  [realtime] {ts_str}  INSERT detected — "
              f"Date={date_val}  Time={time_val}")

        if not _ingest_lock.acquire(blocking=False):
            print(f"  [realtime] Ingest already running — skipping duplicate trigger.")
            return

        def _do_ingest():
            try:
                wrote = ingest_latest(dry_run=dry_run)
                if wrote:
                    print(f"  [realtime] ✅  showcase_sensor.csv updated.")
                else:
                    print(f"  [realtime] No new calibrated rows (all invalid or duplicates).")
            except Exception as exc:
                print(f"  [realtime] ⚠️  Ingest error: {exc}")
                log.exception("Ingest error after realtime trigger")
            finally:
                _ingest_lock.release()

        t = threading.Thread(target=_do_ingest, daemon=True)
        t.start()

    # Try to establish a Realtime subscription
    try:
        channel = (
            client.channel(f"showcase-sensor-{TABLE_ENV_DATA}")
            .on_postgres_changes(
                event="INSERT",
                schema="public",
                table=TABLE_ENV_DATA,
                callback=_on_insert,
            )
            .subscribe()
        )
        print(f"  Realtime subscription active. Waiting for inserts...\n")
    except Exception as exc:
        print(f"  [realtime] ⚠️  Could not start Realtime subscription: {exc}")
        print(f"  [realtime] Falling back to polling mode (interval={poll_interval}s).\n")
        _watch_polling(poll_interval=poll_interval, dry_run=dry_run)
        return

    # Keep the main thread alive; the Realtime client runs on a background thread
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n  [realtime] Stopped by user.")
        try:
            client.remove_channel(channel)
        except Exception:
            pass


def _watch_polling(poll_interval: int = DEFAULT_POLL_INTERVAL,
                   dry_run: bool = False) -> None:
    """
    Polling fallback — checks Supabase row count every poll_interval seconds
    and fires ingest_latest() whenever the count increases.
    """
    print(f"\n  [showcase_sensor_ingest] Polling watch mode started.")
    print(f"  Polling Supabase every {poll_interval}s for new rows.")
    print(f"  Output → {SENSOR_CSV}")
    print(f"  Press Ctrl+C to stop.\n")

    last_count = get_supabase_row_count()
    print(f"  Initial Supabase row count: {last_count}")

    try:
        while True:
            time.sleep(poll_interval)
            current_count = get_supabase_row_count()

            if current_count < 0:
                print(f"  [poll] Could not read row count — will retry.")
                continue

            if current_count > last_count:
                new_rows_detected = current_count - last_count
                print(f"\n  [poll] ✅ {new_rows_detected} new row(s) detected "
                      f"(count: {last_count} → {current_count})")
                last_count = current_count
                wrote = ingest_latest(dry_run=dry_run)
                if wrote:
                    print(f"  [poll] showcase_sensor.csv updated.")
                else:
                    print(f"  [poll] No calibrated rows written (all invalid or duplicates).")
            else:
                ts = pd.Timestamp.now().strftime("%H:%M:%S")
                print(f"  [poll] {ts}  No new rows (count={current_count})", end="\r")

    except KeyboardInterrupt:
        print(f"\n  [poll] Stopped by user.")


def ingest_date(target_date: str, dry_run: bool = False) -> bool:
    if not USE_HARDWARE:
        return False
    if date_already_in_csv(target_date):
        log.info("%s already in showcase_sensor.csv — skipping.", target_date)
        return False
    raw = fetch_rows_for_date(target_date)
    if raw.empty:
        log.warning("No Supabase rows found for %s.", target_date)
        return False
    append_raw_rows_to_csv(raw, dry_run=dry_run)
    calibrated = calibrate_df(raw)
    written = append_rows_to_csv(calibrated, dry_run=dry_run)
    return written > 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Showcase sensor ingest — writes to showcase_sensor.csv.\n"
            "Default: one-shot full-sync ingest (fetches all Supabase rows,\n"
            "appends any missing from CSV regardless of insertion order).\n"
            "--watch         : Realtime websocket (fires on INSERT, zero latency).\n"
            "--watch --poll  : Polling fallback (checks row count every N seconds)."
        )
    )
    parser.add_argument("--watch",    action="store_true",
                        help="Watch for new Supabase rows (Realtime by default).")
    parser.add_argument("--poll",     action="store_true",
                        help="Use polling mode instead of Realtime (requires --watch).")
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL,
                        help=f"Polling interval in seconds for --poll mode (default: {DEFAULT_POLL_INTERVAL}).")
    parser.add_argument("--date",     type=str, default=None,
                        help="Backfill a specific date (YYYY-MM-DD).")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview calibrated output without writing to disk.")
    parser.add_argument("--check-calibration", action="store_true",
                        help="Print calibration summary and exit.")
    args = parser.parse_args()

    if args.check_calibration:
        print_calibration_summary()
    elif args.watch:
        ingest_on_new_row(
            poll_interval=args.interval,
            dry_run=args.dry_run,
            use_realtime=not args.poll,
        )
    elif args.date:
        ingest_date(args.date, dry_run=args.dry_run)
    else:
        ingest_latest(dry_run=args.dry_run)