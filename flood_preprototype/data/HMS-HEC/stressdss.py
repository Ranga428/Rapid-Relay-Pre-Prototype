"""
write_stress_dss.py
====================
Reads CHIRPS PRECIP-INC from sensor_data.dss for every year 2017-2026,
multiplies by 7 scale factors, and writes SCALE_* pathnames for ALL years.

Previously only 01Jan2018 had SCALE_* records. This script fills in every
other year so HEC-HMS FullRecord_Control (2017-2026) can run all 7 scenarios
across the full period.

REQUIREMENTS
------------
  conda activate floodenv
  python write_stress_dss.py

pydsstools must be installed in floodenv. This script will not run correctly
outside that environment.

WHAT IT WRITES
--------------
For each year in YEARS and each scale in SCALES:
  /YOUR-RIVER/STATION1/PRECIP-INC/01JanYYYY/1Day/SCALE_Xp0X/

2018 records are also rewritten to ensure consistency (safe to overwrite).

OUTPUT
------
Appends to sensor_data.dss in-place. HEC-HMS reads the updated DSS immediately
— no HMS restart needed, just re-run the simulation.
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    from pydsstools.heclib.dss import HecDss
    from pydsstools.core import TimeSeriesContainer
except ImportError:
    sys.exit(
        "\nERROR: pydsstools not found.\n"
        "Run this script inside the floodenv conda environment:\n"
        "  conda activate floodenv\n"
        "  python write_stress_dss.py\n"
    )

# ===========================================================================
# CONFIGURATION
# ===========================================================================

DSS_FILE = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\HMS-HEC\sensor_data.dss"

# All years present in your FullRecord_Control (2017-2026)
YEARS = list(range(2017, 2027))

# Scale factors — must match your Met_Model_1 scenario names exactly
SCALES = {
    "SCALE_1P0X": 0.50,
    "SCALE_1P3X": 0.71,
    "SCALE_1P6X": 1.00,
    "SCALE_1P9X": 1.41,
    "SCALE_2P2X": 2.00,
    "SCALE_2P5X": 2.83,
    "SCALE_3P0X": 4.00,
}

# DSS pathname parts — must match your existing CHIRPS records exactly
PART_A = "YOUR-RIVER"
PART_B = "STATION1"
PART_C = "PRECIP-INC"
PART_E = "1Day"

# Undefined/missing value sentinel used by HEC-DSS
DSS_UNDEFINED = -3.4028234663852886e+38

# ===========================================================================
# HELPERS
# ===========================================================================

def part_d(year: int) -> str:
    """Returns the DSS Part D start-date string for a given year."""
    return f"01Jan{year}"


def chirps_pathname(year: int) -> str:
    return f"/{PART_A}/{PART_B}/{PART_C}/{part_d(year)}/{PART_E}/CHIRPS/"


def scale_pathname(year: int, scale_label: str) -> str:
    return f"/{PART_A}/{PART_B}/{PART_C}/{part_d(year)}/{PART_E}/{scale_label}/"


def start_datetime(year: int) -> str:
    """HEC-DSS start datetime string."""
    return f"01Jan{year} 00:00:00"


def days_in_year(year: int) -> int:
    import calendar
    return 366 if calendar.isleap(year) else 365


def clean_values(vals: np.ndarray) -> np.ndarray:
    """Replace DSS undefined sentinel and any negatives with 0.0."""
    v = vals.astype(float).copy()
    v[v < -1e+20] = 0.0   # undefined sentinel
    v[~np.isfinite(v)] = 0.0
    v[v < 0] = 0.0
    return v


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("  write_stress_dss.py — full-record stress scaling")
    print("=" * 60)

    if not os.path.exists(DSS_FILE):
        sys.exit(
            f"\nERROR: DSS file not found:\n  {DSS_FILE}\n"
            f"Update DSS_FILE in the CONFIG block."
        )

    print(f"\n  DSS file : {DSS_FILE}")
    print(f"  Years    : {YEARS[0]} – {YEARS[-1]}")
    print(f"  Scales   : {list(SCALES.keys())}\n")

    total_written = 0
    total_skipped = 0

    with HecDss.Open(DSS_FILE) as dss:

        for year in YEARS:
            src_path = chirps_pathname(year)
            n_days   = days_in_year(year)
            start    = start_datetime(year)

            # ── Read source CHIRPS record ────────────────────────────────────
            try:
                ts = dss.read_ts(src_path)
            except Exception as e:
                print(f"  {year}  CHIRPS read FAILED — {e}")
                print(f"         Skipping all scales for {year}.")
                total_skipped += len(SCALES)
                continue

            raw = np.array(ts.values, dtype=float)
            chirps = clean_values(raw)

            # Trim or pad to exact number of days in this year
            if len(chirps) > n_days:
                chirps = chirps[:n_days]
            elif len(chirps) < n_days:
                chirps = np.pad(chirps, (0, n_days - len(chirps)))

            rain_days  = int((chirps > 0.5).sum())
            annual_sum = chirps.sum()
            peak_day   = chirps.max()

            print(f"  {year}  CHIRPS: {rain_days} rain days, "
                  f"annual={annual_sum:.0f}mm, peak={peak_day:.1f}mm")

            # ── Write each scaled record ─────────────────────────────────────
            for label, factor in SCALES.items():
                dst_path   = scale_pathname(year, label)
                scaled     = chirps * factor

                # Cap at physically extreme but not impossible daily value
                # 600mm/day is the absolute upper bound for any known storm
                scaled = np.clip(scaled, 0.0, 600.0)

                tsc = TimeSeriesContainer()
                tsc.pathname      = dst_path
                tsc.startDateTime = start
                tsc.numberValues  = n_days
                tsc.units         = "MM"
                tsc.type          = "PER-CUM"
                tsc.interval      = 1440        # 1 day in minutes
                tsc.values        = scaled.tolist()

                try:
                    dss.put(tsc)
                    total_written += 1
                    peak_scaled = scaled.max()
                    print(f"    ✓  {label:<12}  factor={factor:.1f}x  "
                          f"peak={peak_scaled:.1f}mm  → {dst_path}")
                except Exception as e:
                    print(f"    ✗  {label}  WRITE FAILED — {e}")
                    total_skipped += 1

            print()

    print("=" * 60)
    print(f"  Written : {total_written} records")
    print(f"  Skipped : {total_skipped} records")
    print("=" * 60)

    # ── Verification read-back ───────────────────────────────────────────────
    print("\n  Verification — reading back sample records...")
    check_years  = [2017, 2019, 2022, 2025]
    check_scales = ["SCALE_1P0X", "SCALE_1P9X", "SCALE_3P0X"]

    with HecDss.Open(DSS_FILE) as dss:
        print(f"\n  {'Pathname':<65} {'Days':>5}  {'Sum':>8}  {'Peak':>7}")
        print(f"  {'-'*65} {'-'*5}  {'-'*8}  {'-'*7}")
        for yr in check_years:
            for sl in check_scales:
                p = scale_pathname(yr, sl)
                try:
                    ts   = dss.read_ts(p)
                    vals = clean_values(np.array(ts.values))
                    print(f"  {p:<65} {len(vals):>5}  "
                          f"{vals.sum():>8.0f}  {vals.max():>7.1f}")
                except Exception as e:
                    print(f"  {p:<65}  READ FAILED: {e}")

    print("\n  Done.")
    print("  Next step: In HEC-HMS, assign each SCALE_* pathname as the")
    print("  precipitation gage for each scenario's Met_Model, then")
    print("  run FullRecord_Control for all 7 scenarios.")
    print("=" * 60)


if __name__ == "__main__":
    main()