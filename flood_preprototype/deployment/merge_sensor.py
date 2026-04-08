"""
merge_sensor.py
=======================
Flood Prediction — Sensor Context Merger

PURPOSE
-------
Combines proxy (satellite-derived historical) data with real hardware sensor
data into a single unified CSV: combined_sensor_context.csv

Output format is identical to the source CSVs:
    timestamp | waterlevel | soil_moisture | humidity

No extra columns. The result is a clean, gap-filled sensor history that
RF_Predict.py can consume directly — just point LIVE_SENSOR_FILE at it.

MERGE STRATEGY
--------------
Priority: hardware > proxy

For any given date:
  - Hardware reading exists → use hardware value
  - No hardware reading     → fall back to proxy value
  - Neither source has data → interpolate if gap ≤ MAX_INTERPOLATION_GAP days,
                              otherwise leave as NaN

INCREMENTAL MODE (default)
---------------------------
On every normal run the script checks the latest timestamp already in the
output CSV and only processes rows AFTER that date from both sources.
Daily runs touch only 1-2 new rows instead of reprocessing the entire
proxy history every time.

  First run (output CSV empty or missing) : processes everything
  Subsequent runs                         : only new rows appended

Use --full-rebuild to force a complete reprocess (e.g. after correcting
a source CSV or changing interpolation settings).

FOLDER / FILE AUTO-CREATION
----------------------------
If the output folder does not exist it is created automatically.
If the output CSV does not exist it is created from scratch.
No manual folder setup needed before first run.

TIMESTAMP FORMAT
----------------
All timestamps in the output CSV are written in ISO 8601 format:
    2025-07-01T00:00:00
This ensures RF_Predict.py / prepare_dataset.py can parse them without
hitting the slow infer_datetime_format path.

PROJECT LAYOUT
--------------
  Script   : .../flood_preprototype/deployment/merge_sensor.py
  Proxy    : .../flood_preprototype/data/sensor/obando_environmental_data.csv
  Hardware : .../flood_preprototype/data/sensor/obando_sensor_data.csv
  Output   : .../flood_preprototype/data/sensor/combined_sensor_context.csv

USAGE
-----
    # Normal daily run — incremental (only new rows)
    python merge_sensor.py

    # Force full reprocess of all history
    python merge_sensor.py --full-rebuild

    # Custom paths
    python merge_sensor.py \
        --proxy    "../data/sensor/obando_environmental_data.csv" \
        --hardware "../data/sensor/obando_sensor_data.csv" \
        --output   "../data/sensor/combined_sensor_context.csv"

    # From Start.py (incremental by default)
    from merge_sensor import run_pipeline
    run_pipeline()

INTEGRATION WITH Start.py
--------------------------
    Step 0a  Sat_SensorData_proxy.py  — fetch/append proxy data
    Step 0b  hardware sensor CSV      — appended by IoT ingest
    Step 0c  merge_sensor.py          — merge → combined_sensor_context.csv
    Step 1   RF_Predict.py            — reads combined_sensor_context.csv

In RF_Predict.py change:
    LIVE_SENSOR_FILE = r"..\data\sensor\combined_sensor_context.csv"

CHANGES FROM PREVIOUS VERSION
------------------------------
FIX — Timestamps in the output CSV are now written in ISO 8601 format
      (2025-07-01T00:00:00) instead of the default pandas space-separated
      format (2025-07-01 00:00:00). This prevents RF_Predict.py /
      prepare_dataset.py from hitting the slow infer_datetime_format path
      which caused the terminal to hang on startup.
"""

import os
import sys
import argparse
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Resolve project root from this script's location
#
#   Script location : .../flood_preprototype/deployment/merge_sensor.py
#   Project root    : .../flood_preprototype/
#   Data root       : .../flood_preprototype/data/
# ---------------------------------------------------------------------------

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))   # .../deployment/
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR)                  # .../flood_preprototype/
DATA_ROOT     = os.path.join(PROJECT_ROOT, "data")           # .../flood_preprototype/data/
SENSOR_DIR    = os.path.join(DATA_ROOT, "sensor")            # .../data/sensor/


# ===========================================================================
# CONFIG — paths are relative to the project root, resolved at runtime.
#          Change only if you rename folders.
# ===========================================================================

PROXY_CSV    = os.path.join(SENSOR_DIR, "obando_environmental_data.csv")
HARDWARE_CSV = os.path.join(SENSOR_DIR, "obando_sensor_data.csv")
OUTPUT_CSV   = os.path.join(SENSOR_DIR, "combined_sensor_context.csv")

# Sensor columns present in both source CSVs
SENSOR_COLS = ["waterlevel", "soil_moisture", "humidity"]

# Earliest date to include in the combined output.
# Rows before this date are ignored from both sources.
# Change this if you want to extend or shrink the history window.
HISTORY_START = pd.Timestamp("2025-07-01 00:00:00", tz="UTC")

# Fill isolated missing dates via linear interpolation.
INTERPOLATE_GAPS = True

# Maximum consecutive missing days to interpolate across.
# Gaps longer than this are left as NaN (not interpolated).
MAX_INTERPOLATION_GAP = 3

# Timestamp format written to the output CSV.
# ISO 8601 with T separator — ensures fast parsing in RF_Predict.py.
# Do NOT change unless you also update parse_timestamps_robust().
OUTPUT_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

# ===========================================================================
# END CONFIG
# ===========================================================================


def separator(title=""):
    line = "=" * 55
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ---------------------------------------------------------------------------
# Folder + file bootstrap
# ---------------------------------------------------------------------------

def ensure_sensor_dir() -> None:
    """
    Create the sensor data directory if it does not exist.
    Also creates any intermediate folders (e.g. data/ if missing).
    Called once at the top of run_pipeline() before any file I/O.
    """
    if not os.path.exists(SENSOR_DIR):
        os.makedirs(SENSOR_DIR, exist_ok=True)
        print(f"  Created directory : {SENSOR_DIR}")
    else:
        print(f"  Sensor dir OK     : {SENSOR_DIR}")


def ensure_output_csv(output_path: str) -> bool:
    """
    If the output CSV does not exist, create an empty one with the correct
    headers so downstream reads never fail on a missing file.

    Returns True if the file was newly created, False if it already existed.
    """
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
    """
    Read the latest timestamp already present in the output CSV.
    Returns a UTC Timestamp to use as the incremental cutoff, or None if
    the output CSV is empty / missing (triggers a full rebuild).
    """
    if not os.path.exists(output_path):
        return None
    try:
        existing = pd.read_csv(
            output_path,
            parse_dates=["timestamp"],
            index_col="timestamp",
        )
        if len(existing) == 0:
            return None
        existing.index = pd.to_datetime(existing.index, utc=True)
        cutoff = existing.index.max()
        print(f"  Output CSV last row : {cutoff.date()}  "
              f"({len(existing):,} rows already present)")
        return cutoff
    except Exception as e:
        print(f"  WARNING: Could not read output CSV for cutoff check ({e}) "
              f"— will do full rebuild.")
        return None


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_csv(path: str, label: str,
             after: "pd.Timestamp | None" = None,
             history_start: "pd.Timestamp | None" = None) -> "pd.DataFrame | None":
    """
    Load a sensor CSV (timestamp, waterlevel, soil_moisture, humidity).
    Returns a DataFrame with a UTC DatetimeIndex, or None if missing/empty.

    history_start: ignore rows before this date (overrides HISTORY_START global).
    after: ignore rows on or before this date (incremental cutoff).
    """
    if not os.path.exists(path):
        print(f"  WARNING: {label} CSV not found → {path}")
        return None

    try:
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")

        # Handle empty file (headers only)
        if len(df) == 0:
            print(f"  WARNING: {label} CSV is empty (headers only) → {path}")
            return None

        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        # Normalise to daily midnight UTC — drop intra-day duplicates
        df.index = df.index.floor("D")
        df = df[~df.index.duplicated(keep="last")]

        total_rows = len(df)

        # Hard start date — ignore anything before history_start
        if history_start is not None:
            df = df[df.index >= history_start]
            if len(df) < total_rows:
                print(f"  {label:<12}: trimmed to history_start "
                      f"({total_rows - len(df):,} rows before "
                      f"{history_start.date()} dropped)")

        # Incremental filter — only keep rows after the cutoff
        if after is not None:
            df = df[df.index > after]
            print(f"  {label:<12}: {len(df):>4} new rows after {after.date()}  "
                  f"(total in file: {total_rows:,})")
        else:
            if len(df) > 0:
                print(f"  {label:<12}: {len(df):>6,} rows  "
                      f"| {df.index.min().date()} → {df.index.max().date()}")
            else:
                print(f"  {label:<12}: 0 rows after HISTORY_START filter")

        if len(df) == 0:
            return None

        # Ensure all expected sensor columns exist
        missing_cols = [c for c in SENSOR_COLS if c not in df.columns]
        if missing_cols:
            print(f"  WARNING: {label} CSV missing columns {missing_cols} — filling NaN")
            for c in missing_cols:
                df[c] = np.nan

        return df[SENSOR_COLS]

    except Exception as e:
        print(f"  ERROR loading {label} CSV: {e}")
        return None


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def merge_sources(
    proxy:    "pd.DataFrame | None",
    hardware: "pd.DataFrame | None",
) -> pd.DataFrame:
    """
    Merge proxy and hardware DataFrames. Priority: hardware > proxy.
    Output contains only: timestamp, waterlevel, soil_moisture, humidity.
    """
    if proxy is None and hardware is None:
        sys.exit(
            "\n  ERROR: Both proxy and hardware CSVs are missing or empty.\n"
            "  Make sure at least one sensor CSV has data before merging.\n"
        )

    # ── Only one source available ─────────────────────────────────────────
    if proxy is None:
        print("  NOTE: No proxy data available — output will be hardware-only.")
        return hardware[SENSOR_COLS].copy()

    if hardware is None:
        print("  NOTE: No hardware data available — output will be proxy-only.")
        return proxy[SENSOR_COLS].copy()

    # ── Both sources present ──────────────────────────────────────────────
    # Continuous daily index spanning both sources
    all_dates = pd.date_range(
        start = min(proxy.index.min(), hardware.index.min()),
        end   = max(proxy.index.max(), hardware.index.max()),
        freq  = "D",
        tz    = "UTC",
    )

    proxy_r    = proxy.reindex(all_dates)
    hardware_r = hardware.reindex(all_dates)

    # Hardware wins where it has data; proxy fills the rest
    merged = hardware_r.combine_first(proxy_r)
    merged.index.name = "timestamp"

    hw_count    = hardware_r.notna().all(axis=1).sum()
    proxy_count = (~hardware_r.notna().all(axis=1) & proxy_r.notna().all(axis=1)).sum()
    gap_count   = (~hardware_r.notna().all(axis=1) & ~proxy_r.notna().all(axis=1)).sum()

    print(f"  Hardware rows used : {hw_count:,}")
    print(f"  Proxy fill rows    : {proxy_count:,}")
    print(f"  Remaining gaps     : {gap_count:,}  (will interpolate if ≤{MAX_INTERPOLATION_GAP}d)")

    return merged[SENSOR_COLS]


# ---------------------------------------------------------------------------
# Gap interpolation
# ---------------------------------------------------------------------------

def interpolate_gaps(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN rows via linear interpolation, but only across runs of
    <= MAX_INTERPOLATION_GAP consecutive missing days.
    Longer gaps are left as NaN.
    """
    if not INTERPOLATE_GAPS:
        return merged

    n_gaps = merged[SENSOR_COLS[0]].isna().sum()
    if n_gaps == 0:
        print("  Gap interpolation : no gaps found.")
        return merged

    merged = merged.copy()
    merged[SENSOR_COLS] = merged[SENSOR_COLS].interpolate(
        method="time", limit=MAX_INTERPOLATION_GAP
    )

    still_empty = merged[SENSOR_COLS[0]].isna().sum()
    filled      = n_gaps - still_empty
    print(f"  Gap interpolation : {n_gaps} gap rows — "
          f"{filled} filled (≤{MAX_INTERPOLATION_GAP}d), "
          f"{still_empty} left as NaN (run too long)")

    return merged


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(merged: pd.DataFrame) -> None:
    separator("Merge Summary")
    total = len(merged)
    nans  = merged[SENSOR_COLS[0]].isna().sum()

    print(f"  Total rows       : {total:,}")
    print(f"  Date range       : {merged.index.min().date()} → {merged.index.max().date()}")
    print(f"  Complete rows    : {total - nans:,}")
    print(f"  Remaining NaNs   : {nans:,}  "
          f"{'(gaps > ' + str(MAX_INTERPOLATION_GAP) + 'd — too long to interpolate)' if nans > 0 else ''}")

    # Rolling window readiness for the most recent row
    print(f"\n  Rolling context at most recent row ({merged.index.max().date()}):")
    print(f"  {'Window':<20} {'Rows available':>15}  Status")
    print(f"  {'-'*20} {'-'*15}  {'-'*10}")
    for window, label in [(7, "7d mean"), (14, "14d cumrise"), (30, "30d context")]:
        avail  = min(window, total)
        status = "✅ OK" if avail >= window else f"⚠️  only {avail}d"
        print(f"  {label:<20} {avail:>15}  {status}")


# ---------------------------------------------------------------------------
# Save output (create or update)
# ---------------------------------------------------------------------------

def format_index_iso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the DatetimeIndex to ISO 8601 strings (2025-07-01T00:00:00)
    before writing to CSV. This ensures RF_Predict.py / prepare_dataset.py
    parse timestamps at C speed without triggering infer_datetime_format.
    """
    df = df.copy()
    df.index = df.index.strftime(OUTPUT_TIMESTAMP_FORMAT)
    df.index.name = "timestamp"
    return df


def save_output(merged: pd.DataFrame, output_path: str,
                overwrite: bool = False) -> None:
    separator("Saving Combined CSV")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if overwrite:
        # Full rebuild — write new_rows as the entire file
        combined = merged[SENSOR_COLS].copy()
        print(f"  Full rebuild — overwriting output CSV.")
    elif os.path.exists(output_path):
        # Incremental — append new rows to existing file
        try:
            existing = pd.read_csv(
                output_path,
                parse_dates=["timestamp"],
                index_col="timestamp",
            )
            if len(existing) > 0:
                existing.index = pd.to_datetime(existing.index, utc=True)
                existing = existing[[c for c in SENSOR_COLS if c in existing.columns]]
                existing = existing.dropna(subset=SENSOR_COLS, how="any")
                combined = pd.concat([existing, merged[SENSOR_COLS]])
                combined = (
                    combined[~combined.index.duplicated(keep="last")]
                    .sort_index()
                )
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
        print(f"  Output CSV did not exist — creating from scratch.")

    # Write with ISO 8601 timestamps (2025-07-01T00:00:00)
        # Drop rows where all sensor columns are NaN
    before_drop = len(combined)
    combined = combined.dropna(subset=SENSOR_COLS, how="any")
    dropped = before_drop - len(combined)
    if dropped > 0:
        print(f"  Dropped blank rows : {dropped:,}  (all sensor columns NaN)")

    # Write with ISO 8601 timestamps (2025-07-01T00:00:00)
    format_index_iso(combined).to_csv(output_path)
    print(f"  Timestamp format : {OUTPUT_TIMESTAMP_FORMAT}  (ISO 8601)")
    print(f"  Saved  →  {output_path}  ({len(combined):,} rows total)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    proxy_path:    str  = PROXY_CSV,
    hardware_path: str  = HARDWARE_CSV,
    output_path:   str  = OUTPUT_CSV,
    full_rebuild:  bool = False,
) -> pd.DataFrame:
    """
    Full merge pipeline. Returns the merged DataFrame of NEW rows only.

    full_rebuild=False (default): incremental — only rows after the latest
        timestamp already in the output CSV are processed and appended.
    full_rebuild=True: reprocesses all rows and overwrites the output CSV.

    To change HISTORY_START permanently:
        python merge_sensor.py --history-start 2024-01-01

    Called from Start.py as:
        from merge_sensor import run_pipeline
        run_pipeline()
    """
    effective_start = HISTORY_START

    separator("merge_sensor.py — Sensor Context Merger")
    print(f"  Script dir   : {SCRIPT_DIR}")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Data root    : {DATA_ROOT}")
    print()
    print(f"  Proxy CSV    : {proxy_path}")
    print(f"  Hardware CSV : {hardware_path}")
    print(f"  Output CSV   : {output_path}")
    print(f"  Mode         : {'FULL REBUILD — reprocessing all rows' if full_rebuild else 'INCREMENTAL — new rows only'}")
    print(f"  History from : {effective_start.date() if effective_start else 'all time'}"
          + ("  (--history-start override)" if effective_start != HISTORY_START else "  (config default)"))
    print(f"  Interpolate  : "
          f"{'yes (gaps ≤' + str(MAX_INTERPOLATION_GAP) + 'd)' if INTERPOLATE_GAPS else 'no'}")
    print(f"  Timestamp fmt: {OUTPUT_TIMESTAMP_FORMAT}  (ISO 8601)")

    # ── Step A: Ensure folder structure exists ────────────────────────────
    separator("Checking / Creating Directories")
    ensure_sensor_dir()

    # ── Step B: Bootstrap empty output CSV if first run ───────────────────
    newly_created = ensure_output_csv(output_path)
    if not newly_created:
        print(f"  Output CSV exists : {output_path}")

    # ── Step C: Determine incremental cutoff ─────────────────────────────
    separator("Incremental Cutoff Check")
    if full_rebuild:
        cutoff = None
        print("  --full-rebuild flag set — cutoff ignored, processing all rows.")
    else:
        cutoff = get_cutoff_date(output_path)
        if cutoff is None:
            print("  No existing rows found — processing all rows (first run).")
        else:
            print(f"  Incremental cutoff : {cutoff.date()} — "
                  f"only rows after this date will be merged.")

    # ── Step D: Load sources (filtered to new rows only) ─────────────────
    separator("Loading Sources")
    proxy    = load_csv(proxy_path,    "proxy",    after=cutoff, history_start=effective_start)
    hardware = load_csv(hardware_path, "hardware", after=cutoff, history_start=effective_start)

    # Nothing new in either source — exit early, output CSV unchanged
    if proxy is None and hardware is None:
        separator("DONE — already up to date")
        print("  No new rows in either source CSV.")
        print(f"  Output CSV unchanged : {output_path}")
        separator()
        return pd.DataFrame(columns=SENSOR_COLS)

    # ── Step E: Merge new rows ────────────────────────────────────────────
    separator("Merging New Rows")
    new_rows = merge_sources(proxy, hardware)

    # ── Step F: Gap interpolation on new rows ────────────────────────────
    if INTERPOLATE_GAPS:
        new_rows = interpolate_gaps(new_rows)

    # ── Step G: Summary of new rows ───────────────────────────────────────
    print_summary(new_rows)

    # ── Step H: Append to (or rebuild) output CSV ────────────────────────
    if full_rebuild:
        save_output(new_rows, output_path, overwrite=True)
    else:
        save_output(new_rows, output_path, overwrite=False)

    separator("DONE")
    print(f"  RF_Predict.py should read: {output_path}")
    separator()

    return new_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Merge proxy and hardware sensor CSVs into a unified context CSV.\n"
            "Default: incremental — only rows newer than the output CSV's latest "
            "timestamp are processed and appended.\n"
            "Use --full-rebuild to reprocess everything from scratch."
        )
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=PROXY_CSV,
        help=f"Proxy/historical sensor CSV (default: {PROXY_CSV})",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=HARDWARE_CSV,
        help=f"Hardware sensor CSV (default: {HARDWARE_CSV})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_CSV,
        help=f"Output combined CSV (default: {OUTPUT_CSV})",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Reprocess all rows from both source CSVs and overwrite the output CSV.",
    )
    parser.add_argument(
        "--history-start",
        type=str,
        default=None,
        help=(
            "Permanently update HISTORY_START in this script's config. "
            "Format: YYYY-MM-DD  e.g. --history-start 2024-01-01  "
            "The new date is written into the script file and used for all future runs."
        ),
    )
    args = parser.parse_args()

    # --history-start: permanently rewrite HISTORY_START in this script file
    if args.history_start:
        try:
            new_start = pd.Timestamp(args.history_start, tz="UTC")
        except Exception:
            raise ValueError(
                f"Could not parse --history-start '{args.history_start}'. "
                f"Use YYYY-MM-DD format, e.g. 2024-01-01"
            )

        old_line = f'HISTORY_START = pd.Timestamp("{HISTORY_START.strftime("%Y-%m-%d %H:%M:%S")}", tz="UTC")'
        new_line = f'HISTORY_START = pd.Timestamp("{new_start.strftime("%Y-%m-%d %H:%M:%S")}", tz="UTC")'

        script_path = os.path.abspath(__file__)
        with open(script_path, "r") as f:
            source = f.read()

        if old_line not in source:
            raise RuntimeError(
                f"Could not find HISTORY_START line to update in {script_path}.\n"
                f"Expected: {old_line}\n"
                f"Edit HISTORY_START manually in the CONFIG section."
            )

        updated_source = source.replace(old_line, new_line, 1)
        with open(script_path, "w") as f:
            f.write(updated_source)

        print(f"  HISTORY_START updated : {HISTORY_START.date()} → {new_start.date()}")
        print(f"  Written to            : {script_path}")
        print(f"  All future runs will use {new_start.date()} as the history start.")
        print(f"  Re-run with --full-rebuild if you want to rebuild the output CSV now.")
        sys.exit(0)

    run_pipeline(
        proxy_path    = args.proxy,
        hardware_path = args.hardware,
        output_path   = args.output,
        full_rebuild  = args.full_rebuild,
    )