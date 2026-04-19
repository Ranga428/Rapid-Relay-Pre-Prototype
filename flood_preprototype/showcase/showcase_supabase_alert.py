"""
showcase_supabase_alert.py
==========================
SHOWCASE VERSION of Supabase_Alert.py

Changes from original:
  - Reads from  : showcase_predict.csv  (instead of flood_rf_sensor_predictions.csv)
  - Upsert target table : flood_predictions  (same as original)
  - All Supabase sync logic is identical to the original.

Usage
-----
    import showcase_supabase_alert
    showcase_supabase_alert.append_new_predictions_to_supabase()

    python showcase_supabase_alert.py           # append new rows
    python showcase_supabase_alert.py --sync    # full upsert
"""

import os
import pandas as pd
import argparse
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Showcase: reads showcase_predict.csv
CSV_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "predictions", "showcase_predict.csv")
)

BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Full sync (UPSERT)
# ---------------------------------------------------------------------------

def sync_predictions_to_supabase():
    try:
        print("  🚀 [Showcase Supabase] Full sync operation")
        print("  📂 CSV PATH:", CSV_PATH)

        if not os.path.exists(CSV_PATH):
            print("  ❌ showcase_predict.csv not found")
            return

        df = pd.read_csv(CSV_PATH)
        if df.empty:
            print("  ⚠️ CSV is empty")
            return

        print(f"  📊 CSV rows loaded: {len(df)}")

        required_cols = ["timestamp", "flood_probability", "risk_tier"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        records = []
        for _, row in df.iterrows():
            records.append({
                "timestamp":       row["timestamp"].isoformat(),
                "flood_probability": float(row["flood_probability"]),
                "risk_tier":       row["risk_tier"],
            })

        print(f"  🚀 Syncing {len(records)} rows")

        total_inserted = 0
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            supabase.table("flood_predictions_showcase") \
                .upsert(batch, on_conflict="timestamp") \
                .execute()
            total_inserted += len(batch)
            print(f"  ✓ Batch synced: {total_inserted}/{len(records)}")

        print(f"  ✅ Sync complete: {total_inserted} rows processed")

    except Exception as e:
        print(f"  ❌ Sync failed: {e}")


# ---------------------------------------------------------------------------
# Append new rows only
# ---------------------------------------------------------------------------

def append_new_predictions_to_supabase():
    try:
        print("  🚀 [Showcase Supabase] Append new rows only")
        print("  📂 CSV PATH:", CSV_PATH)

        if not os.path.exists(CSV_PATH):
            print("  ❌ showcase_predict.csv not found")
            return

        df = pd.read_csv(CSV_PATH)
        if df.empty:
            print("  ⚠️ CSV is empty")
            return

        print(f"  📊 CSV rows loaded: {len(df)}")

        required_cols = ["timestamp", "flood_probability", "risk_tier"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df["timestamp"]     = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp_iso"] = df["timestamp"].apply(lambda ts: ts.isoformat())

        existing = supabase.table("flood_predictions_showcase").select("timestamp").execute()
        existing_timestamps = {row["timestamp"] for row in existing.data}
        print(f"  ℹ️ Existing in Supabase: {len(existing_timestamps)}")

        new_rows = df[~df["timestamp_iso"].isin(existing_timestamps)]
        if new_rows.empty:
            print("  ✅ No new rows to insert")
            return

        print(f"  🚀 New rows to insert: {len(new_rows)}")

        records = []
        for _, row in new_rows.iterrows():
            records.append({
                "timestamp":       row["timestamp_iso"],
                "flood_probability": float(row["flood_probability"]),
                "risk_tier":       row["risk_tier"],
            })

        total_inserted = 0
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            supabase.table("flood_predictions_showcase").insert(batch).execute()
            total_inserted += len(batch)
            print(f"  ✓ Batch inserted: {total_inserted}/{len(records)}")

        print(f"  ✅ Append complete: {total_inserted} new rows added")

    except Exception as e:
        print(f"  ❌ Append failed: {e}")


# ---------------------------------------------------------------------------
# Standard channel interface
# ---------------------------------------------------------------------------

def send(tier: str, probability: float, timestamp: str) -> bool:
    """Standard send() wrapper for showcase_alert.py integration."""
    try:
        append_new_predictions_to_supabase()
        return True
    except Exception as e:
        print(f"  [showcase_supabase_alert] send() raised: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Showcase Supabase Alert Utility")
    parser.add_argument("--sync", action="store_true",
                        help="Full CSV → Supabase sync (UPSERT)")
    args = parser.parse_args()

    if args.sync:
        sync_predictions_to_supabase()
    else:
        append_new_predictions_to_supabase()
