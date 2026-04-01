"""
extract_test_set.py
===================
Extracts the test split rows from flood_dataset.csv into a separate CSV.
The test split is the held-out period never seen during training or
threshold tuning:

    Test : 2025-07-01 → present

This CSV can be used to:
    - Inspect what the model was evaluated on
    - Run manual prediction checks
    - Feed into predict_*.py for batch prediction
    - Validate against known flood events in 2025–2026

Test set contains flood events across the Jul 2025 – Mar 2026 window:
    2025 : Jul        — strong flood events
    2025 : Aug–Nov    — sensor-invisible flood cluster
    2026 : Jan–Mar    — quiet / dry period

Note: The test set (~265 rows as of Mar 2026) is smaller than ideal for
robust per-regime evaluation. Defer final conclusions until mid-2026 when
the second dry season is more complete and row count reaches ~350–400.

Usage
-----
    python extract_test_set.py

Output
------
    ..\data\flood_dataset_test.csv
"""

import os
import sys
import pandas as pd


# ===========================================================================
# CONFIG
# ===========================================================================

INPUT_FILE  = r"..\data\flood_dataset.csv"
OUTPUT_FILE = r"..\data\flood_dataset_test.csv"

# Must match TRAIN_END and VAL_END in the training scripts
TRAIN_END = "2024-06-30"   # UPDATED — was "2022-12-31"
VAL_END   = "2025-06-30"   # UPDATED — was "2023-12-31"

# ===========================================================================
# END CONFIG
# ===========================================================================


def main():
    print("=" * 55)
    print("  Extract Test Set from flood_dataset.csv")
    print("=" * 55)

    # --- Load ---
    if not os.path.exists(INPUT_FILE):
        sys.exit(f"\n  ERROR: Input file not found.\n  Expected: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()

    print(f"\n  Input file    : {INPUT_FILE}")
    print(f"  Total rows    : {len(df):,}")
    print(f"  Date range    : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Flood=1       : {int(df['flood_label'].sum())}  ({100*df['flood_label'].mean():.1f}%)")
    print(f"  Flood=0       : {int((df['flood_label']==0).sum())}  ({100*(1-df['flood_label'].mean()):.1f}%)")

    # --- Split boundaries ---
    tz           = df.index.tz
    train_end_ts = pd.Timestamp(TRAIN_END, tz=tz)
    val_end_ts   = pd.Timestamp(VAL_END,   tz=tz)

    train_df = df[df.index <= train_end_ts]
    val_df   = df[(df.index > train_end_ts) & (df.index <= val_end_ts)]
    test_df  = df[df.index > val_end_ts]

    # --- Report all splits for context ---
    print(f"\n  Split breakdown:")
    print(f"    Train : {len(train_df):>4} rows  "
          f"({train_df.index.min().date()} -> {train_df.index.max().date()})  "
          f"flood={int(train_df['flood_label'].sum())}  "
          f"no-flood={int((train_df['flood_label']==0).sum())}")
    print(f"    Val   : {len(val_df):>4} rows  "
          f"({val_df.index.min().date()} -> {val_df.index.max().date()})  "
          f"flood={int(val_df['flood_label'].sum())}  "
          f"no-flood={int((val_df['flood_label']==0).sum())}")
    print(f"    Test  : {len(test_df):>4} rows  "
          f"({test_df.index.min().date()} -> {test_df.index.max().date()})  "
          f"flood={int(test_df['flood_label'].sum())}  "
          f"no-flood={int((test_df['flood_label']==0).sum())}")

    if len(test_df) == 0:
        sys.exit("\n  ERROR: Test split is empty. Check VAL_END date.")

    # --- Flood clusters in test set ---
    test_df_copy = test_df.copy()
    test_df_copy['group'] = (
        test_df_copy['flood_label'] != test_df_copy['flood_label'].shift()
    ).cumsum()
    clusters = (
        test_df_copy[test_df_copy['flood_label'] == 1]
        .groupby('group')
        .agg(
            start=('flood_label', lambda x: x.index.min()),
            end=('flood_label', lambda x: x.index.max()),
            days=('flood_label', 'count'),
        )
        .reset_index(drop=True)
    )
    print(f"\n  Flood clusters in test set ({len(clusters)} total):")
    for _, row in clusters.iterrows():
        print(f"    {str(row['start'].date()):<12} -> {str(row['end'].date()):<12}  "
              f"{row['days']:>2} days")

    if len(test_df) < 300:
        print(f"\n  ⚠️  Test set has {len(test_df)} rows — smaller than the 300–400 row")
        print(f"      target for robust per-regime evaluation. Consider deferring")
        print(f"      final conclusions until mid-2026.")

    # --- Save ---
    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    test_df.index.name = "timestamp"
    test_df.to_csv(OUTPUT_FILE)

    print(f"\n  Test set saved : {OUTPUT_FILE}")
    print(f"  Rows saved     : {len(test_df):,}")
    print(f"  Flood rate     : {100*test_df['flood_label'].mean():.1f}%")

    print(f"\n  Columns ({len(test_df.columns)}):")
    for col in test_df.columns:
        print(f"    {col}")

    print(f"\n  Sample rows:")
    print(test_df.head(5).to_string())

    print(f"\n  Note: These rows were NEVER seen during training or threshold tuning.")
    print(f"        The model's performance on this set is the honest prediction result.")
    print(f"        TRAIN_END = {TRAIN_END}  |  VAL_END = {VAL_END}")
    print(f"        Test starts {(val_end_ts + pd.Timedelta(days=1)).date()}  "
          f"(~13-month gap from train cutoff)")
    print("=" * 55)


if __name__ == "__main__":
    main()