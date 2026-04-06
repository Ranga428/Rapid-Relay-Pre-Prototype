"""
Supabase_Alert.py
=================
Handles:
1. Syncing ML prediction CSV → Supabase (FULL SYNC via UPSERT)
2. Sending alert records → Supabase

Used by:
Start.py → Alert.py → this file
"""

import os
import pandas as pd
import argparse
from supabase import create_client
from dotenv import load_dotenv

# =========================================================
# LOAD ENV VARIABLES
# =========================================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================================================
# CONFIG
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..", "..",
        "predictions",
        "flood_rf_sensor_predictions.csv"
    )
)

BATCH_SIZE = 500  # safe batch size for Supabase

# =========================================================
# SYNC CSV → SUPABASE (FULL UPSERT)
# =========================================================
def sync_predictions_to_supabase():
    try:
        print("  🚀 Starting full sync operation")
        print("  📂 CSV PATH:", CSV_PATH)

        if not os.path.exists(CSV_PATH):
            print("  ❌ CSV not found")
            return

        df = pd.read_csv(CSV_PATH)

        if df.empty:
            print("  ⚠️ CSV is empty")
            return

        print(f"  📊 CSV rows loaded: {len(df)}")

        # Validate columns
        required_cols = ["timestamp", "flood_probability", "risk_tier"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Convert timestamp (timezone-safe)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Build records
        records = []
        for _, row in df.iterrows():
            records.append({
                "timestamp": row["timestamp"].isoformat(),
                "flood_probability": float(row["flood_probability"]),
                "risk_tier": row["risk_tier"]
            })

        print(f"  🚀 Preparing to sync {len(records)} rows")

        # Batch UPSERT
        total_inserted = 0

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]

            supabase.table("flood_predictions") \
                .upsert(batch, on_conflict="timestamp") \
                .execute()

            total_inserted += len(batch)
            print(f"  ✓ Batch inserted: {total_inserted}/{len(records)}")

        print(f"  ✅ Sync complete: {total_inserted} rows processed")

    except Exception as e:
        print(f"  ❌ CSV sync failed: {e}")

# =========================================================
# SYNC CSV → SUPABASE (NEW ROW UPSERT)
# =========================================================

def append_new_predictions_to_supabase():
    try:
        print("  🚀 Starting append operation (new rows only)")
        print("  📂 CSV PATH:", CSV_PATH)

        if not os.path.exists(CSV_PATH):
            print("  ❌ CSV not found")
            return

        df = pd.read_csv(CSV_PATH)
        if df.empty:
            print("  ⚠️ CSV is empty")
            return

        print(f"  📊 CSV rows loaded: {len(df)}")

        # Validate columns
        required_cols = ["timestamp", "flood_probability", "risk_tier"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp_iso"] = df["timestamp"].apply(lambda ts: ts.isoformat())

        # Fetch existing timestamps from Supabase
        existing = supabase.table("flood_predictions") \
            .select("timestamp") \
            .execute()

        existing_timestamps = {row["timestamp"] for row in existing.data}
        print(f"  ℹ️ Existing records in Supabase: {len(existing_timestamps)}")

        # Filter new rows only
        new_rows = df[~df["timestamp_iso"].isin(existing_timestamps)]
        if new_rows.empty:
            print("  ✅ No new rows to insert")
            return

        print(f"  🚀 New rows to insert: {len(new_rows)}")

        # Build records
        records = []
        for _, row in new_rows.iterrows():
            records.append({
                "timestamp": row["timestamp_iso"],
                "flood_probability": float(row["flood_probability"]),
                "risk_tier": row["risk_tier"]
            })

        # Batch insert
        total_inserted = 0
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            supabase.table("flood_predictions").insert(batch).execute()
            total_inserted += len(batch)
            print(f"  ✓ Batch inserted: {total_inserted}/{len(records)}")

        print(f"  ✅ Append complete: {total_inserted} new rows added")

    except Exception as e:
        print(f"  ❌ CSV append failed: {e}")

# =========================================================
# TEST RUN
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Supabase Flood Alert Utility")

    parser.add_argument("--sync", action="store_true",
                        help="Perform full CSV → Supabase sync (UPSERT)")
    
    args = parser.parse_args()

    if args.sync:
        sync_predictions_to_supabase()
    else:
        append_new_predictions_to_supabase()

    #sync_predictions_to_supabase()
    #append_new_predictions_to_supabase()

    # Comment this out in production
    #send_to_supabase(
   #    tier="WARNING",
   #     probability=0.82,
   #     timestamp="2025-07-01"
   # )