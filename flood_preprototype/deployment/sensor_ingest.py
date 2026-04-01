"""
sensor_ingest.py
================
Daily Hardware Sensor Ingestion → ML Pipeline CSV
Flood Early Warning System — Obando, Bulacan

Reads sensor_daily rows directly from Supabase (no local SQLite / db.py).
Logs each ML prediction back to Supabase for remote monitoring.

Integrate with Start.py:
    from sensor_ingest import ingest_latest, log_prediction
    ingest_latest()                  # before RF_Predict.run_pipeline()
    log_prediction(tier, prob, ts)   # after predictions are made

USE_HARDWARE = False by default — safe no-op until calibration done.

Required environment variables (in deployment/.env):
    SUPABASE_URL      = https://<project-ref>.supabase.co
    SUPABASE_KEY      = <service_role or anon key>

Required Supabase tables
------------------------
sensor_daily
    date                TEXT  PRIMARY KEY   -- 'YYYY-MM-DD'
    waterlevel_cm_mean  REAL
    soil_pct_mean       REAL
    humidity_pct_mean   REAL

predictions
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
    timestamp           TEXT   NOT NULL
    probability         REAL   NOT NULL
    risk_tier           TEXT   NOT NULL
    model               TEXT   NOT NULL
    created_at          TIMESTAMPTZ DEFAULT now()

Usage
-----
    python3 sensor_ingest.py               # ingest yesterday
    python3 sensor_ingest.py --date 2025-04-01
    python3 sensor_ingest.py --show 7      # last 7 days from Supabase
    python3 sensor_ingest.py --dry-run
"""

import os
import sys
import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from supabase import create_client, Client

load_dotenv(Path(__file__).parent / ".env")

log = logging.getLogger("sensor_ingest")

# ===========================================================================
# CONFIG
# ===========================================================================

SENSOR_CSV = Path(os.getenv(
    "SENSOR_CSV_PATH",
    str(Path(__file__).parent / "data" / "obando_environmental_data.csv")
))

# Set True ONLY after calibration validated.
# False = this script is a safe no-op.
USE_HARDWARE = False

# Soil conversion constants
VWC_DRY = 0.05
VWC_WET = 0.48

# Supabase table names
TABLE_SENSOR_DAILY = "sensor_daily"
TABLE_PREDICTIONS  = "predictions"

# ===========================================================================
# END CONFIG
# ===========================================================================


# ---------------------------------------------------------------------------
# Supabase client (lazy singleton)
# ---------------------------------------------------------------------------

_supabase_client: Client | None = None


def get_client() -> Client:
    """
    Return a cached Supabase client.
    Reads SUPABASE_URL and SUPABASE_KEY from the environment / .env file.
    Raises RuntimeError if either variable is missing.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "Supabase credentials not found.\n"
            "Set SUPABASE_URL and SUPABASE_KEY in deployment/.env or your environment."
        )

    _supabase_client = create_client(url, key)
    log.debug("Supabase client initialised — %s", url)
    return _supabase_client


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def fetch_daily_rows(days: int) -> pd.DataFrame:
    """
    Pull up to `days` most-recent rows from the sensor_daily table.

    Returns a DataFrame with columns:
        date, waterlevel_cm_mean, soil_pct_mean, humidity_pct_mean

    Returns an empty DataFrame on error.
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    try:
        response = (
            get_client()
            .table(TABLE_SENSOR_DAILY)
            .select("date, waterlevel_cm_mean, soil_pct_mean, humidity_pct_mean")
            .gte("date", cutoff)
            .order("date", desc=False)
            .execute()
        )
    except Exception as exc:
        log.error("Supabase fetch failed: %s", exc)
        return pd.DataFrame(
            columns=["date", "waterlevel_cm_mean", "soil_pct_mean", "humidity_pct_mean"]
        )

    if not response.data:
        return pd.DataFrame(
            columns=["date", "waterlevel_cm_mean", "soil_pct_mean", "humidity_pct_mean"]
        )

    return pd.DataFrame(response.data)


def insert_prediction_row(record: dict) -> bool:
    """
    Insert a single prediction record into the predictions table.

    Expected keys: timestamp, probability, risk_tier, model

    Returns True on success, False on failure.
    """
    try:
        get_client().table(TABLE_PREDICTIONS).insert(record).execute()
        return True
    except Exception as exc:
        log.error("Failed to insert prediction: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Sensor CSV helpers (unchanged from original)
# ---------------------------------------------------------------------------

def soil_pct_to_vwc(soil_pct: float) -> float:
    return round(VWC_DRY + (soil_pct / 100.0) * (VWC_WET - VWC_DRY), 4)


def row_already_in_csv(target_date: str) -> bool:
    if not SENSOR_CSV.exists():
        return False
    try:
        df = pd.read_csv(SENSOR_CSV)
        return df["timestamp"].astype(str).str.startswith(target_date).any()
    except Exception:
        return False


def build_pipeline_row(daily: dict) -> dict:
    """
    Convert a sensor_daily Supabase row to ML pipeline CSV format:
        timestamp, waterlevel, soil_moisture, humidity
    """
    soil_pct = daily.get("soil_pct_mean") or 50.0
    return {
        "timestamp":     daily["date"] + "T00:00:00",
        # Raw cm — sensor_normalisation.py z-scores in RF_Predict.load_live_features()
        "waterlevel":    daily.get("waterlevel_cm_mean"),
        "soil_moisture": soil_pct_to_vwc(soil_pct),
        "humidity":      daily.get("humidity_pct_mean"),
    }


def append_to_csv(row: dict, dry_run: bool = False) -> None:
    SENSOR_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if dry_run:
        log.info("[DRY RUN] Would append:")
        print(df.to_string(index=False))
        return
    if not SENSOR_CSV.exists():
        df.to_csv(SENSOR_CSV, index=False)
        log.info("Created %s", SENSOR_CSV)
    else:
        df.to_csv(SENSOR_CSV, mode="a", header=False, index=False)
        log.info("Appended to %s", SENSOR_CSV)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_latest(dry_run: bool = False) -> bool:
    """
    Ingest yesterday's hardware reading into the ML pipeline CSV.
    Reads from Supabase — requires internet connectivity.
    Returns True if a row was written.
    """
    if not USE_HARDWARE:
        log.info("USE_HARDWARE=False — hardware ingestion skipped")
        return False

    yesterday = (date.today() - timedelta(days=1)).isoformat()

    if row_already_in_csv(yesterday):
        log.info("%s already in sensor CSV", yesterday)
        return False

    daily_df = fetch_daily_rows(days=2)

    if daily_df.empty or yesterday not in daily_df["date"].values:
        log.warning("No Supabase row for %s", yesterday)
        return False

    daily = daily_df[daily_df["date"] == yesterday].iloc[0].to_dict()
    row   = build_pipeline_row(daily)

    log.info(
        "Ingesting %s: wl=%s cm  soil=%s m³/m³  hum=%s%%",
        yesterday,
        row["waterlevel"],
        row["soil_moisture"],
        row["humidity"],
    )
    append_to_csv(row, dry_run=dry_run)
    return True


def ingest_date(target_date: str, dry_run: bool = False) -> bool:
    """Ingest a specific date (backfill)."""
    if not USE_HARDWARE:
        log.info("USE_HARDWARE=False")
        return False

    if row_already_in_csv(target_date):
        log.info("%s already in CSV", target_date)
        return False

    daily_df = fetch_daily_rows(days=365)

    if daily_df.empty or target_date not in daily_df["date"].values:
        log.warning("No Supabase row for %s", target_date)
        return False

    daily = daily_df[daily_df["date"] == target_date].iloc[0].to_dict()
    row   = build_pipeline_row(daily)
    append_to_csv(row, dry_run=dry_run)
    return True


def log_prediction(risk_tier: str, probability: float, timestamp: str) -> None:
    """
    Called by Start.py after each prediction run.
    Stores the ML result in Supabase for remote monitoring.
    """
    record = {
        "timestamp":   timestamp,
        "probability": round(probability, 4),
        "risk_tier":   risk_tier,
        "model":       "RF",
    }
    ok = insert_prediction_row(record)
    if ok:
        log.info("Prediction logged: %s %s (%.1f%%)", timestamp, risk_tier, probability * 100)
    else:
        log.warning("Prediction NOT logged (Supabase insert failed): %s %s", timestamp, risk_tier)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ingest hardware sensor data into ML pipeline CSV (Supabase backend)."
    )
    parser.add_argument("--date",    type=str, default=None,
                        help="Ingest a specific date (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument("--show",    type=int, metavar="DAYS", default=None,
                        help="Print the last N days of sensor rows from Supabase.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be appended without writing.")
    args = parser.parse_args()

    if args.show:
        df = fetch_daily_rows(days=args.show)
        print(f"\nLast {args.show} days from Supabase ({TABLE_SENSOR_DAILY}):\n")
        print(df.to_string(index=False) if not df.empty else "No data found.")
    elif args.date:
        ingest_date(args.date, dry_run=args.dry_run)
    else:
        ingest_latest(dry_run=args.dry_run)