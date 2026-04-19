"""
showcase_merge.py
=================
SHOWCASE VERSION of merge_sensor.py

Changes from original:
  - Proxy input   : showcase_proxy.csv   (instead of obando_environmental_data.csv)
  - Hardware input: showcase_sensor.csv  (instead of obando_sensor_data.csv)
  - Output CSV    : showcase_merge.csv   (instead of combined_sensor_context.csv)
  - All paths resolved relative to showcase/ folder.
  - Keeps ALL sub-daily rows (no daily flooring, no dropping duplicates).
  - Merges on exact raw timestamps.

Usage
-----
    python showcase_merge.py               # incremental (new rows only)
    python showcase_merge.py --full-rebuild  # reprocess all history
"""

import os
import sys
import argparse
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # showcase/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                  # flood_preprototype/
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
SENSOR_DIR   = os.path.join(DATA_ROOT, "sensor")

# Showcase-specific paths
PROXY_CSV    = os.path.join(SENSOR_DIR, "showcase_proxy.csv")
HARDWARE_CSV = os.path.join(SENSOR_DIR, "showcase_sensor.csv")
OUTPUT_CSV   = os.path.join(SENSOR_DIR, "showcase_merge.csv")

SENSOR_COLS = ["waterlevel", "soil_moisture", "humidity"]

HISTORY_START = pd.Timestamp("2025-07-01 00:00:00", tz="UTC")

INTERPOLATE_GAPS       = True
MAX_INTERPOLATION_GAP  = 3
OUTPUT_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


def separator(title=""):
    line = "=" * 55
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def ensure_sensor_dir() -> None:
    if not os.path.exists(SENSOR_DIR):
        os.makedirs(SENSOR_DIR, exist_ok=True)
        print(f"  Created directory : {SENSOR_DIR}")


def ensure_output_csv(output_path: str) -> bool:
    if not os.path.exists(output_path):
        empty_df = pd.DataFrame(columns=SENSOR_COLS)
        empty_df.index.name = "timestamp"
        empty_df.to_csv(output_path)
        print(f"  Created output CSV : {output_path}")
        return True
    return False


# ---------------------------------------------------------------------------
# Incremental cutoff
# ---------------------------------------------------------------------------

def get_cutoff_date(output_path: str) -> "pd.Timestamp | None":
    if not os.path.exists(output_path):
        return None
    try:
        existing = pd.read_csv(output_path, parse_dates=["timestamp"], index_col="timestamp")
        if len(existing) == 0:
            return None
        existing.index = pd.to_datetime(existing.index, utc=True)
        cutoff = existing.index.max()
        print(f"  showcase_merge.csv last row : {cutoff}  ({len(existing):,} rows)")
        return cutoff
    except Exception as e:
        print(f"  WARNING: Could not read output CSV ({e}) — full rebuild.")
        return None


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_csv(path: str, label: str,
             after: "pd.Timestamp | None" = None,
             history_start: "pd.Timestamp | None" = None) -> "pd.DataFrame | None":
    if not os.path.exists(path):
        print(f"  WARNING: {label} CSV not found → {path}")
        return None

    try:
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        if len(df) == 0:
            print(f"  WARNING: {label} CSV is empty → {path}")
            return None

        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        
        # REMOVED: df.index = df.index.floor("D") 
        # REMOVED: df = df[~df.index.duplicated(keep="last")]

        if history_start is not None:
            df = df[df.index >= history_start]

        if after is not None:
            df = df[df.index > after]
            print(f"  {label:<12}: {len(df):>4} new rows after {after}")
        else:
            if len(df) > 0:
                print(f"  {label:<12}: {len(df):>6,} rows  "
                      f"| {df.index.min()} → {df.index.max()}")
            else:
                print(f"  {label:<12}: 0 rows after HISTORY_START filter")

        if len(df) == 0:
            return None

        missing_cols = [c for c in SENSOR_COLS if c not in df.columns]
        if missing_cols:
            for c in missing_cols:
                df[c] = np.nan

        return df[SENSOR_COLS]

    except Exception as e:
        print(f"  ERROR loading {label} CSV: {e}")
        return None


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_sources(proxy: "pd.DataFrame | None",
                  hardware: "pd.DataFrame | None") -> pd.DataFrame:
    if proxy is None and hardware is None:
        sys.exit("\n  ERROR: Both proxy and hardware CSVs are missing or empty.\n")

    if proxy is None:
        print("  NOTE: No proxy data — output is hardware-only.")
        return hardware[SENSOR_COLS].copy()

    if hardware is None:
        print("  NOTE: No hardware data — output is proxy-only.")
        return proxy[SENSOR_COLS].copy()

    # combine_first automatically aligns the exact sub-daily timestamps from both frames.
    # Where both exist at the exact same time, hardware takes precedence.
    merged = hardware.combine_first(proxy)
    merged.index.name = "timestamp"

    print(f"  Hardware rows provided : {len(hardware):,}")
    print(f"  Proxy rows provided    : {len(proxy):,}")
    print(f"  Total merged timeline  : {len(merged):,} rows")

    return merged[SENSOR_COLS]


# ---------------------------------------------------------------------------
# Gap interpolation
# ---------------------------------------------------------------------------

def interpolate_gaps(merged: pd.DataFrame) -> pd.DataFrame:
    if not INTERPOLATE_GAPS:
        return merged

    n_gaps = merged[SENSOR_COLS[0]].isna().sum()
    if n_gaps == 0:
        print("  Gap interpolation : no gaps found.")
        return merged

    merged = merged.copy()
    
    # Method="time" crashes if there are literal duplicate timestamps in the index.
    # We fallback to standard linear interpolation if you have exact timestamp duplicates.
    try:
        merged[SENSOR_COLS] = merged[SENSOR_COLS].interpolate(
            method="time", limit=MAX_INTERPOLATION_GAP
        )
    except ValueError:
        merged[SENSOR_COLS] = merged[SENSOR_COLS].interpolate(
            method="linear", limit=MAX_INTERPOLATION_GAP
        )
        
    still_empty = merged[SENSOR_COLS[0]].isna().sum()
    filled      = n_gaps - still_empty
    print(f"  Gap interpolation : {n_gaps} gaps — {filled} filled, {still_empty} left as NaN")
    return merged


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def format_index_iso(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.strftime(OUTPUT_TIMESTAMP_FORMAT)
    df.index.name = "timestamp"
    return df


def save_output(merged: pd.DataFrame, output_path: str, overwrite: bool = False) -> None:
    separator("Saving showcase_merge.csv")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if overwrite:
        combined = merged[SENSOR_COLS].copy()
        print("  Full rebuild — overwriting output CSV.")
    elif os.path.exists(output_path):
        try:
            existing = pd.read_csv(output_path, parse_dates=["timestamp"], index_col="timestamp")
            if len(existing) > 0:
                existing.index = pd.to_datetime(existing.index, utc=True)
                existing = existing[[c for c in SENSOR_COLS if c in existing.columns]]
                existing = existing.dropna(subset=SENSOR_COLS, how="any")
                
                # Append new rows and sort. 
                # REMOVED: combined = combined[~combined.index.duplicated(keep="last")]
                combined = pd.concat([existing, merged[SENSOR_COLS]]).sort_index()
                
                print(f"  Existing rows    : {len(existing):,}")
                print(f"  Appended rows    : {len(merged):,}")
                print(f"  Final row count  : {len(combined):,}")
            else:
                combined = merged[SENSOR_COLS]
        except Exception as e:
            print(f"  WARNING: Could not read existing CSV ({e}) — overwriting.")
            combined = merged[SENSOR_COLS]
    else:
        combined = merged[SENSOR_COLS]
        print("  Output CSV did not exist — creating from scratch.")

    before_drop = len(combined)
    combined = combined.dropna(subset=SENSOR_COLS, how="any")
    dropped = before_drop - len(combined)
    if dropped > 0:
        print(f"  Dropped blank rows : {dropped:,}")

    format_index_iso(combined).to_csv(output_path)
    print(f"  Saved  →  {output_path}  ({len(combined):,} rows total)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    proxy_path:   str  = PROXY_CSV,
    hardware_path: str = HARDWARE_CSV,
    output_path:  str  = OUTPUT_CSV,
    full_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Merge showcase_proxy.csv + showcase_sensor.csv → showcase_merge.csv.
    Returns the merged DataFrame of NEW rows only.
    """
    separator("showcase_merge.py — Showcase Sensor Context Merger")
    print(f"  Proxy CSV    : {proxy_path}")
    print(f"  Hardware CSV : {hardware_path}")
    print(f"  Output CSV   : {output_path}")
    print(f"  Mode         : {'FULL REBUILD' if full_rebuild else 'INCREMENTAL'}")

    separator("Checking Directories")
    ensure_sensor_dir()
    newly_created = ensure_output_csv(output_path)
    if not newly_created:
        print(f"  Output CSV exists : {output_path}")

    separator("Incremental Cutoff Check")
    cutoff = None if full_rebuild else get_cutoff_date(output_path)
    if cutoff is None and not full_rebuild:
        print("  No existing rows — processing all rows (first run).")

    separator("Loading Sources")
    proxy    = load_csv(proxy_path,    "proxy",    after=cutoff, history_start=HISTORY_START)
    hardware = load_csv(hardware_path, "hardware", after=cutoff, history_start=HISTORY_START)

    if proxy is None and hardware is None:
        separator("DONE — already up to date")
        print("  No new rows in either source CSV.")
        separator()
        return pd.DataFrame(columns=SENSOR_COLS)

    separator("Merging New Rows")
    new_rows = merge_sources(proxy, hardware)

    if INTERPOLATE_GAPS:
        new_rows = interpolate_gaps(new_rows)

    if full_rebuild:
        save_output(new_rows, output_path, overwrite=True)
    else:
        save_output(new_rows, output_path, overwrite=False)

    separator("DONE")
    print(f"  showcase_merge.csv ready at: {output_path}")
    separator()
    return new_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="showcase_merge — merges showcase_proxy.csv + showcase_sensor.csv → showcase_merge.csv"
    )
    parser.add_argument("--full-rebuild", action="store_true",
                        help="Reprocess all rows and overwrite the output CSV.")
    args = parser.parse_args()
    run_pipeline(full_rebuild=args.full_rebuild)