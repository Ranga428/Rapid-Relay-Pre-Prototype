"""
hms_dss_to_rf_csv.py
=====================
Reads hourly Junction-1 FLOW time series directly from the HMS output .dss
files, converts discharge → stage via Manning's equation, resamples to daily,
merges with observed soil_moisture and humidity, and writes RF-ready CSVs.

OUTPUT columns: timestamp, waterlevel, soil_moisture, humidity, flood_label

USAGE
-----
    conda activate floodenv
    python hms_dss_to_rf_csv.py

REQUIRES
--------
    pydsstools  (already in floodenv)
    pandas, numpy (already in floodenv)
"""

import os
import sys
import math
import re
import numpy as np
import pandas as pd

try:
    from pydsstools.heclib.dss import HecDss
except ImportError:
    sys.exit(
        "\nERROR: pydsstools not found.\n"
        "Run inside floodenv:  conda activate floodenv\n"
    )

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Directory containing Stress_1P0X.dss … Stress_3P0X.dss
DSS_DIR = r"D:\RapidRelayHEC-HMS\RapidRelay"

# Observed sensor CSV — for soil_moisture and humidity columns
OBSERVED_CSV = (
    r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype"
    r"\data\HMS-HEC\full_results.csv"
)

# Output directory
OUTPUT_DIR = (
    r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype"
    r"\data\HMS-HEC\stress_inputs"
)

# ── Manning's channel geometry ────────────────────────────────────────────────
MANNING_N     = 0.035
CHANNEL_SLOPE = 0.0005
BOTTOM_WIDTH  = 20.0    # m
SIDE_SLOPE    = 1.5     # H:V
DRY_BASELINE  = 0.65    # m  (minimum stage)

# ── RF label threshold ────────────────────────────────────────────────────────
FLOOD_THRESHOLD = 2.0   # m
DIKE_HEIGHT     = 4.039 # m

# ── DSS pathname template for Junction-1 hourly FLOW ─────────────────────────
# Pattern: //Junction-1/FLOW/01MonYYYY/1Hour/RUN:Stress_XXXXX/
FLOW_PART_B     = "Junction-1"
FLOW_PART_C     = "FLOW"
FLOW_PART_E     = "1Hour"

# Scenarios: DSS filename stem → Part-F label in pathnames
SCENARIOS = {
    "Stress_1P0X": "RUN:Stress_1P0X",
    "Stress_1P3X": "RUN:Stress_1P3X",
    "Stress_1P6X": "RUN:Stress_1P6X",
    "Stress_1P9X": "RUN:Stress_1P9X",
    "Stress_2P2X": "RUN:Stress_2P2X",
    "Stress_2P5X": "RUN:Stress_2P5X",
    "Stress_3P0X": "RUN:Stress_3P0X",
}

# Month abbreviations used in DSS Part-D
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ===========================================================================
# MANNING'S INVERSE  (Q → stage)
# ===========================================================================

def stage_from_Q(Q: float) -> float:
    if Q <= 0:
        return DRY_BASELINE

    def Q_m(y):
        A  = (BOTTOM_WIDTH + SIDE_SLOPE * y) * y
        P  = BOTTOM_WIDTH + 2 * y * math.sqrt(1 + SIDE_SLOPE ** 2)
        return (1.0 / MANNING_N) * A * (A / P) ** (2/3) * CHANNEL_SLOPE ** 0.5

    def dQ(y, e=1e-6):
        return (Q_m(y + e) - Q_m(y - e)) / (2 * e)

    y = max(0.5, Q / 80.0)
    for _ in range(300):
        f  = Q_m(y) - Q
        df = dQ(y)
        if abs(df) < 1e-14:
            break
        yn = max(0.01, y - f / df)
        if abs(yn - y) < 1e-7:
            y = yn
            break
        y = yn
    return round(y, 4)

# Vectorise for speed
stage_from_Q_vec = np.vectorize(stage_from_Q)


# ===========================================================================
# READ ALL MONTHLY FLOW BLOCKS FROM ONE DSS FILE
# ===========================================================================

def read_junction1_flow(dss_path: str, part_f: str) -> pd.Series:
    """
    Reads all //Junction-1/FLOW/01MonYYYY/1Hour/<part_f>/ blocks,
    stitches them into a continuous hourly Series indexed by datetime.
    """
    all_records = []

    # Build list of all monthly Part-D values from 2016-2026
    part_d_list = []
    for year in range(2016, 2027):
        for mon in MONTHS:
            part_d_list.append(f"01{mon}{year}")

    with HecDss.Open(dss_path) as dss:
        for part_d in part_d_list:
            pathname = f"//{FLOW_PART_B}/{FLOW_PART_C}/{part_d}/{FLOW_PART_E}/{part_f}/"
            try:
                ts = dss.read_ts(pathname)
                if ts is None or ts.numberValues == 0:
                    continue

                # Parse start datetime from HMS format e.g. "01Jan2017 00:00:00"
                start_str = ts.startDateTime          # e.g. "01Jan2017, 00:00"
                # Normalise to pandas-parseable format
                start_str = start_str.replace(",", "")
                try:
                    start_dt = pd.to_datetime(start_str, format="%d%b%Y %H:%M")
                except Exception:
                    start_dt = pd.to_datetime(start_str)

                n      = ts.numberValues
                freq   = pd.Timedelta(hours=1)
                index  = pd.date_range(start=start_dt, periods=n, freq=freq)
                vals   = np.array(ts.values, dtype=float)

                # Clean undefined sentinel
                vals[vals < -1e20] = np.nan
                vals[vals < 0]     = 0.0

                s = pd.Series(vals, index=index)
                all_records.append(s)

            except Exception:
                continue  # block doesn't exist for this month — skip

    if not all_records:
        return pd.Series(dtype=float)

    combined = pd.concat(all_records)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


# ===========================================================================
# LOAD OBSERVED SENSOR DATA
# ===========================================================================

def load_observed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()
    if "soilmoisture" in df.columns and "soil_moisture" not in df.columns:
        df = df.rename(columns={"soilmoisture": "soil_moisture"})
    for col in ["soil_moisture", "humidity"]:
        if col not in df.columns:
            sys.exit(f"\nERROR: Column '{col}' missing in observed CSV.")
    df = df.resample("1D").mean()
    return df[["soil_moisture", "humidity"]]


# ===========================================================================
# BUILD RF CSV FOR ONE SCENARIO
# ===========================================================================

def build_rf_csv(flow_hourly: pd.Series,
                 observed:    pd.DataFrame) -> pd.DataFrame:
    """
    1. Convert hourly Q → hourly stage.
    2. Resample to daily max stage (worst-case each day, matching sensor logic).
    3. Merge with observed soil_moisture and humidity (DOY mean).
    4. Add flood_label.
    """
    # Step 1: Q → stage
    stage_hourly = pd.Series(
        stage_from_Q_vec(flow_hourly.fillna(0).values),
        index=flow_hourly.index,
        name="waterlevel"
    )

    # Step 2: daily max stage
    stage_daily = stage_hourly.resample("1D").max()
    stage_daily = stage_daily.clip(lower=DRY_BASELINE)

    # Step 3: soil moisture and humidity from observed DOY means
    sm_doy = observed["soil_moisture"].groupby(observed.index.day_of_year).mean()
    hm_doy = observed["humidity"].groupby(observed.index.day_of_year).mean()

    sm_vals = [sm_doy.get(d.day_of_year, observed["soil_moisture"].mean())
               for d in stage_daily.index]
    hm_vals = [hm_doy.get(d.day_of_year, observed["humidity"].mean())
               for d in stage_daily.index]

    df = pd.DataFrame({
        "waterlevel":    stage_daily.round(4).values,
        "soil_moisture": np.round(sm_vals, 4),
        "humidity":      np.round(hm_vals, 4),
    }, index=stage_daily.index)

    df.index.name = "timestamp"
    df["flood_label"] = (df["waterlevel"] >= FLOOD_THRESHOLD).astype(int)
    return df


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 65)
    print("  hms_dss_to_rf_csv.py  —  real DSS flow → RF CSV")
    print("=" * 65)

    if not os.path.isdir(DSS_DIR):
        sys.exit(f"\nERROR: DSS_DIR not found:\n  {DSS_DIR}")
    if not os.path.exists(OBSERVED_CSV):
        sys.exit(f"\nERROR: OBSERVED_CSV not found:\n  {OBSERVED_CSV}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    observed = load_observed(OBSERVED_CSV)
    print(f"\n  Observed data : {len(observed)} rows  "
          f"({observed.index.min().date()} → {observed.index.max().date()})")
    print(f"  Output dir    : {OUTPUT_DIR}\n")

    print(f"  {'Scenario':<14} {'Hours':>7}  {'Days':>6}  "
          f"{'Peak Q':>9}  {'Peak Stage':>11}  {'vs Dike':>15}  "
          f"{'Flood days':>10}  Output")
    print(f"  {'-'*14} {'-'*7}  {'-'*6}  {'-'*9}  {'-'*11}  "
          f"{'-'*15}  {'-'*10}  {'-'*30}")

    for stem, part_f in SCENARIOS.items():
        dss_path = os.path.join(DSS_DIR, f"{stem}.dss")
        if not os.path.exists(dss_path):
            print(f"  {stem:<14}  MISSING: {dss_path}")
            continue

        flow_hourly = read_junction1_flow(dss_path, part_f)

        if flow_hourly.empty:
            print(f"  {stem:<14}  WARNING: no FLOW data found in DSS")
            continue

        df = build_rf_csv(flow_hourly, observed)

        peak_Q     = float(flow_hourly.max())
        peak_stage = float(df["waterlevel"].max())
        flood_days = int(df["flood_label"].sum())
        overtop    = (f"OVERTOPS +{peak_stage - DIKE_HEIGHT:.3f}m"
                      if peak_stage >= DIKE_HEIGHT
                      else f"safe  -{DIKE_HEIGHT - peak_stage:.3f}m")

        label    = stem.lower().replace("stress_", "")
        out_name = f"stress_{label}_rf.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        df.to_csv(out_path)

        print(f"  {stem:<14} {len(flow_hourly):>7}  {len(df):>6}  "
              f"{peak_Q:>9.2f}  {peak_stage:>10.4f}m  "
              f"{overtop:>15}  {flood_days:>10}  {out_name}")

    print(f"\n  Done. Feed CSVs to RF_Predict:")
    print(f"    python RF_Predict.py --data <path>\\stress_1p0x_rf.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()