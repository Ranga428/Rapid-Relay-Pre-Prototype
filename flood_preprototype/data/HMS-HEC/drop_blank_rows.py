"""
drop_blank_rows.py
==================
Reads a CSV file and removes any rows that contain one or more blank
(NaN, None, empty string, or whitespace-only) values.

Input and output paths are set directly in this script.

USAGE
-----
    # Run with hardcoded paths
    python drop_blank_rows.py

    # Only check specific columns for blanks (ignore blanks in others)
    python drop_blank_rows.py --columns col1 col2 col3

    # Preview how many rows would be dropped without saving
    python drop_blank_rows.py --dry-run
"""

import argparse
import os
import sys
import pandas as pd

INPUT_PATH  = r"D:\Rapid-Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\HMS-HEC\HEC-HMS-Calibration_Data.csv"
OUTPUT_PATH = r"D:\Rapid-Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\HMS-HEC\HEC-HMS-Calibration_Data_cleaned.csv"


def drop_blank_rows(
    columns: list[str] | None = None,
    dry_run: bool = False,
) -> None:

    if not os.path.exists(INPUT_PATH):
        sys.exit(f"ERROR: File not found — {INPUT_PATH}")

    print(f"  Input  : {INPUT_PATH}")
    print(f"  Output : {OUTPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype=str)  # read as str to catch whitespace-only cells
    total_rows = len(df)
    print(f"  Rows (before) : {total_rows:,}")
    print(f"  Columns       : {list(df.columns)}")

    # Normalise: treat whitespace-only strings as NaN
    df = df.apply(lambda col: col.str.strip().replace("", pd.NA))

    # Decide which columns to check
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            sys.exit(f"ERROR: Column(s) not found in CSV: {missing}")
        check_cols = columns
        print(f"  Checking blanks in : {check_cols}")
    else:
        check_cols = df.columns.tolist()
        print(f"  Checking blanks in : ALL columns")

    mask_blank = df[check_cols].isna().any(axis=1)
    n_dropped  = int(mask_blank.sum())
    n_kept     = total_rows - n_dropped

    print(f"\n  Rows with blanks  : {n_dropped:,}")
    print(f"  Rows kept         : {n_kept:,}")

    if n_dropped == 0:
        print("\n  ✅  No blank rows found — nothing to remove.")
        return

    if dry_run:
        print("\n  [DRY RUN] No file written. Remove --dry-run to save.")
        sample = df[mask_blank].head(10)
        print(f"\n  Sample of rows that would be dropped (up to 10):")
        print(sample.to_string(index=True))
        return

    cleaned_df = df[~mask_blank]
    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  ✅  Done. {n_dropped:,} blank row(s) removed. "
          f"{n_kept:,} row(s) saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove rows with blank/null values from a CSV file."
    )
    parser.add_argument(
        "--columns", "-c",
        nargs="+",
        default=None,
        help="Only drop rows that have blanks in these specific column(s). "
             "If omitted, any blank in any column triggers a drop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a summary and sample of rows to be dropped without saving.",
    )

    args = parser.parse_args()
    drop_blank_rows(
        columns=args.columns,
        dry_run=args.dry_run,
    )