"""
sensor_ingest.py
================
Daily Hardware Sensor Ingestion → ML Pipeline CSV
Flood Early Warning System — Obando, Bulacan

Reads from local SQLite (via db.py) — works offline, no internet needed.
Also logs each ML prediction back to the DB for remote monitoring.

Integrate with Start.py:
    from sensor_ingest import ingest_latest, log_prediction
    ingest_latest()                  # before RF_Predict.run_pipeline()
    log_prediction(tier, prob, ts)   # after predictions are made

USE_HARDWARE = False by default — safe no-op until calibration done.

Usage
-----
    python3 sensor_ingest.py               # ingest yesterday
    python3 sensor_ingest.py --date 2025-04-01
    python3 sensor_ingest.py --show 7      # last 7 days from local DB
    python3 sensor_ingest.py --dry-run
    python3 sensor_ingest.py --sync        # retry Supabase pending
"""

import os
import sys
import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd

from db import DB

load_dotenv(Path(__file__).parent / ".env")

log = logging.getLogger("sensor_ingest")

# ===========================================================================
# CONFIG
# ===========================================================================

SENSOR_CSV   = Path(os.getenv(
    "SENSOR_CSV_PATH",
    str(Path(__file__).parent / "data" / "obando_environmental_data.csv")
))

# Set True ONLY after calibration validated.
# False = this script is a safe no-op.
USE_HARDWARE = False

# Soil conversion constants
VWC_DRY = 0.05
VWC_WET = 0.48

# ===========================================================================
# END CONFIG
# ===========================================================================


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
    Convert sensor_daily DB row to ML pipeline CSV format:
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
        log.info(f"Created {SENSOR_CSV}")
    else:
        df.to_csv(SENSOR_CSV, mode="a", header=False, index=False)
        log.info(f"Appended to {SENSOR_CSV}")


def ingest_latest(dry_run: bool = False) -> bool:
    """
    Ingest yesterday's hardware reading into the ML pipeline CSV.
    Reads from LOCAL SQLite — works without internet.
    Returns True if a row was written.
    """
    if not USE_HARDWARE:
        log.info("USE_HARDWARE=False — hardware ingestion skipped")
        return False

    yesterday = (date.today() - timedelta(days=1)).isoformat()

    if row_already_in_csv(yesterday):
        log.info(f"{yesterday} already in sensor CSV")
        return False

    db    = DB()
    daily_df = db.get_daily(days=2)
    db.close()

    if daily_df.empty or yesterday not in daily_df["date"].values:
        log.warning(f"No local DB row for {yesterday}")
        return False

    daily = daily_df[daily_df["date"] == yesterday].iloc[0].to_dict()
    row   = build_pipeline_row(daily)

    log.info(
        f"Ingesting {yesterday}: "
        f"wl={row['waterlevel']}cm "
        f"soil={row['soil_moisture']}m³/m³ "
        f"hum={row['humidity']}%"
    )
    append_to_csv(row, dry_run=dry_run)
    return True


def ingest_date(target_date: str, dry_run: bool = False) -> bool:
    """Ingest a specific date (backfill)."""
    if not USE_HARDWARE:
        log.info("USE_HARDWARE=False")
        return False

    if row_already_in_csv(target_date):
        log.info(f"{target_date} already in CSV")
        return False

    db       = DB()
    daily_df = db.get_daily(days=365)
    db.close()

    if daily_df.empty or target_date not in daily_df["date"].values:
        log.warning(f"No DB row for {target_date}")
        return False

    daily = daily_df[daily_df["date"] == target_date].iloc[0].to_dict()
    row   = build_pipeline_row(daily)
    append_to_csv(row, dry_run=dry_run)
    return True


def log_prediction(risk_tier: str, probability: float, timestamp: str) -> None:
    """
    Called by Start.py after each prediction run.
    Stores the ML result in SQLite + Supabase for remote monitoring.
    """
    db = DB()
    db.insert_prediction({
        "timestamp":   timestamp,
        "probability": round(probability, 4),
        "risk_tier":   risk_tier,
        "model":       "RF",
    })
    db.close()
    log.info(f"Prediction logged: {timestamp} {risk_tier} ({probability:.1%})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest hardware sensor data into ML pipeline CSV.")
    parser.add_argument("--date",    type=str, default=None)
    parser.add_argument("--show",    type=int, metavar="DAYS", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync",    action="store_true", help="Retry Supabase pending syncs")
    args = parser.parse_args()

    if args.show:
        db = DB()
        df = db.get_daily(days=args.show)
        db.close()
        print(f"\nLast {args.show} days from local DB:\n")
        print(df.to_string(index=False) if not df.empty else "No data")
    elif args.sync:
        db = DB()
        n  = db.sync_pending()
        log.info(f"Synced {n} rows to Supabase")
        db.close()
    elif args.date:
        ingest_date(args.date, dry_run=args.dry_run)
    else:
        ingest_latest(dry_run=args.dry_run)