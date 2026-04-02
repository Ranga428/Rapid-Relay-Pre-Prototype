"""
sensor_ingest.py
================
Daily Hardware Sensor Ingestion → ML Pipeline CSV
Flood Early Warning System — Obando, Bulacan

Reads directly from the `obando_environmental_data` table in Supabase and
writes only new rows into the local ML-pipeline CSV (incremental sync).

On first run (CSV missing or empty) → full bootstrap from Supabase.
On subsequent runs → only rows newer than the latest CSV timestamp are fetched.

Integrate with Start.py:
    from sensor_ingest import ingest_latest, log_prediction
    ingest_latest()                  # before RF_Predict.run_pipeline()
    log_prediction(tier, prob, ts)   # after predictions are made

Required environment variables (.env at project root):
    SUPABASE_URL         = https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY = <service_role key>

Supabase tables used
--------------------
obando_environmental_data
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
    timestamp     TIMESTAMPTZ NOT NULL
    waterlevel    NUMERIC NOT NULL
    soil_moisture NUMERIC NOT NULL
    humidity      NUMERIC NOT NULL

predictions
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
    timestamp     TEXT        NOT NULL
    probability   REAL        NOT NULL
    risk_tier     TEXT        NOT NULL
    model         TEXT        NOT NULL
    created_at    TIMESTAMPTZ DEFAULT now()

Usage
-----
    python sensor_ingest.py                     # ingest all new rows (default)
    python sensor_ingest.py --date 2025-04-01   # backfill a specific date
    python sensor_ingest.py --show 7            # print last 7 days from Supabase
    python sensor_ingest.py --dry-run           # preview without writing
"""

import os
import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file, not the working directory
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# .env lives 3 levels above deployment/  →  project root
_ENV_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".env"))

# ML pipeline CSV output path
SENSOR_CSV = Path(os.getenv(
    "SENSOR_CSV_PATH",
    os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "sensor", "obando_sensor_data.csv"))
))

# ---------------------------------------------------------------------------
# Load .env before importing supabase so credentials are in os.environ
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
_loaded = load_dotenv(_ENV_PATH, override=False)

from supabase import create_client, Client

# ===========================================================================
# CONFIG
# ===========================================================================

USE_HARDWARE = True          # Set False to disable all ingestion (safe no-op)

TABLE_ENV_DATA    = "obando_environmental_data"
TABLE_PREDICTIONS = "predictions"

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
            f"Ensure both variables exist in your .env file."
        )

    _supabase_client = create_client(url, key)
    log.debug("Supabase client initialised — %s", url)
    return _supabase_client


# ---------------------------------------------------------------------------
# Supabase fetch helpers
# ---------------------------------------------------------------------------

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "waterlevel", "soil_moisture", "humidity"])


def fetch_rows_since(after_timestamp: str | None) -> pd.DataFrame:
    """
    Fetch all rows strictly newer than `after_timestamp`.
    If None, returns the full table (bootstrap mode).
    """
    try:
        query = (
            get_client()
            .table(TABLE_ENV_DATA)
            .select("timestamp, waterlevel, soil_moisture, humidity")
            .order("timestamp", desc=False)
        )
        if after_timestamp:
            query = query.gt("timestamp", after_timestamp)
        response = query.execute()
    except Exception as exc:
        log.error("Supabase fetch failed: %s", exc)
        return _empty_df()

    return pd.DataFrame(response.data) if response.data else _empty_df()


def fetch_rows_for_date(target_date: str) -> pd.DataFrame:
    """Fetch all rows whose timestamp falls on `target_date` (YYYY-MM-DD, UTC)."""
    start = f"{target_date}T00:00:00+00:00"
    end   = f"{target_date}T23:59:59.999999+00:00"
    try:
        response = (
            get_client()
            .table(TABLE_ENV_DATA)
            .select("timestamp, waterlevel, soil_moisture, humidity")
            .gte("timestamp", start)
            .lte("timestamp", end)
            .order("timestamp", desc=False)
            .execute()
        )
    except Exception as exc:
        log.error("Supabase fetch failed: %s", exc)
        return _empty_df()

    return pd.DataFrame(response.data) if response.data else _empty_df()


def fetch_rows_for_range(days: int) -> pd.DataFrame:
    """Fetch all rows from the last `days` calendar days (used by --show)."""
    cutoff = (date.today() - timedelta(days=days)).isoformat() + "T00:00:00+00:00"
    try:
        response = (
            get_client()
            .table(TABLE_ENV_DATA)
            .select("timestamp, waterlevel, soil_moisture, humidity")
            .gte("timestamp", cutoff)
            .order("timestamp", desc=False)
            .execute()
        )
    except Exception as exc:
        log.error("Supabase fetch failed: %s", exc)
        return _empty_df()

    return pd.DataFrame(response.data) if response.data else _empty_df()


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
    """
    Return the max timestamp already stored in the CSV.
    Returns None if the CSV does not exist or is empty.
    """
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
    """Return True if any row for `target_date` already exists in the CSV."""
    if not SENSOR_CSV.exists():
        return False
    try:
        df = pd.read_csv(SENSOR_CSV)
        return df["timestamp"].astype(str).str.startswith(target_date).any()
    except Exception:
        return False


def append_rows_to_csv(rows: pd.DataFrame, dry_run: bool = False) -> None:
    """Write new rows to the ML pipeline CSV, creating the file if needed."""
    if rows.empty:
        log.warning("Nothing to write — DataFrame is empty.")
        return

    rows = rows[["timestamp", "waterlevel", "soil_moisture", "humidity"]].copy()
    SENSOR_CSV.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.info("[DRY RUN] Would append %d row(s):", len(rows))
        print(rows.to_string(index=False))
        return

    if not SENSOR_CSV.exists():
        rows.to_csv(SENSOR_CSV, index=False)
        log.info("Created %s with %d row(s)", SENSOR_CSV, len(rows))
    else:
        rows.to_csv(SENSOR_CSV, mode="a", header=False, index=False)
        log.info("Appended %d new row(s) to %s", len(rows), SENSOR_CSV)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_latest(dry_run: bool = False) -> bool:
    """
    Default deployment mode — incremental sync.

    Reads the latest timestamp in the local CSV and fetches only rows
    newer than that from Supabase. On first run (no CSV yet), pulls
    the full table to bootstrap the dataset.

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

    rows = fetch_rows_since(after_timestamp=latest_ts)

    if rows.empty:
        log.info("No new rows since %s — CSV is already up to date.", latest_ts)
        return False

    log.info("Ingesting %d new row(s) from '%s'.", len(rows), TABLE_ENV_DATA)
    append_rows_to_csv(rows, dry_run=dry_run)
    return True


def ingest_date(target_date: str, dry_run: bool = False) -> bool:
    """
    Backfill a specific date. Skips if already present in the CSV.
    Returns True if at least one row was written.
    """
    if not USE_HARDWARE:
        log.info("USE_HARDWARE=False — ingestion skipped.")
        return False

    if date_already_in_csv(target_date):
        log.info("%s already in CSV — skipping.", target_date)
        return False

    rows = fetch_rows_for_date(target_date)

    if rows.empty:
        log.warning("No Supabase rows found for %s.", target_date)
        return False

    log.info("Ingesting %d row(s) for %s.", len(rows), target_date)
    append_rows_to_csv(rows, dry_run=dry_run)
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

    # Always show where credentials are being loaded from
    log.info(".env path  : %s", _ENV_PATH)
    log.info(".env found : %s", _loaded)

    parser = argparse.ArgumentParser(
        description="Sync obando_environmental_data from Supabase into the ML pipeline CSV."
    )
    parser.add_argument("--date",    type=str, default=None,
                        help="Backfill a specific date (YYYY-MM-DD).")
    parser.add_argument("--show",    type=int, metavar="DAYS", default=None,
                        help="Print the last N days of rows from Supabase.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing to disk.")
    args = parser.parse_args()

    if args.show:
        df = fetch_rows_for_range(days=args.show)
        print(f"\nLast {args.show} day(s) from Supabase ({TABLE_ENV_DATA}):\n")
        print(df.to_string(index=False) if not df.empty else "No data found.")
    elif args.date:
        ingest_date(args.date, dry_run=args.dry_run)
    else:
        ingest_latest(dry_run=args.dry_run)