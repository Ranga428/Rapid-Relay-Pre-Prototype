"""
stress_test_fixed.py
Extracts the Aug 2018 storm window from sensor_data.dss (CHIRPS rainfall),
scales it by factors [1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 3.0],
and writes each scaled scenario as a new DSS pathname.

Run AFTER calibration is confirmed.
Run from: D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\HMS-HEC\
Conda env: floodenv
"""

import numpy as np
import pandas as pd
from pydsstools.heclib.dss import HecDss
from pydsstools.core import TimeSeriesContainer

DSS_FILE      = r"D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\HMS-HEC\sensor_data.dss"
BASE_PATH     = "/YOUR-RIVER/STATION1/PRECIP-INC/01JAN2017/1DAY/CHIRPS/"
STORM_START   = "2018-08-01"
STORM_END     = "2018-08-31"
SCALE_FACTORS = [1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 3.0]
DIKE_HEIGHT_M = 4.039

# ── Read full rainfall record using explicit window ────────────────────────────
print("Reading base CHIRPS rainfall ...")
with HecDss.Open(DSS_FILE) as dss:
    ts = dss.read_ts(
        BASE_PATH,
        window=("01Jan2017 00:00:00", "28Feb2026 00:00:00"),
        trim_missing=True
    )
    all_times  = pd.to_datetime(ts.pytimes)
    all_values = np.array(ts.values, dtype=float)
    all_values[all_values < -1e30] = np.nan

df = pd.DataFrame({"precip_mm": all_values}, index=all_times)
df.index.name = "datetime"
print(f"  Total records read: {len(df)}")
print(f"  Date range: {df.index[0].date()} -> {df.index[-1].date()}")

# ── Extract Aug 2018 storm window ──────────────────────────────────────────────
storm = df.loc[STORM_START:STORM_END].copy()

if len(storm) == 0:
    print("\nERROR: Storm window is empty after slicing.")
    print("Available date range in DSS:", df.index[0].date(), "->", df.index[-1].date())
    raise SystemExit(1)

print(f"\nStorm window: {storm.index[0].date()} -> {storm.index[-1].date()}")
print(f"  Days         : {len(storm)}")
print(f"  Total rain   : {storm['precip_mm'].sum():.1f} mm")
print(f"  Peak 1-day   : {storm['precip_mm'].max():.1f} mm")

# ── Write scaled scenarios to DSS ─────────────────────────────────────────────
print(f"\nWriting {len(SCALE_FACTORS)} scaled scenarios to DSS ...")

with HecDss.Open(DSS_FILE) as dss:
    for factor in SCALE_FACTORS:
        scaled_values = (storm["precip_mm"].values * factor)
        scaled_values = np.nan_to_num(scaled_values, nan=0.0)
        scaled_peak   = scaled_values.max()
        scaled_total  = scaled_values.sum()

        label    = f"SCALE_{factor:.1f}X".replace(".", "P")
        pathname = f"/YOUR-RIVER/STATION1/PRECIP-INC/01AUG2018/1DAY/{label}/"

        tsc = TimeSeriesContainer()
        tsc.pathname      = pathname
        tsc.startDateTime = "01Aug2018 00:00:00"
        tsc.numberValues  = len(scaled_values)
        tsc.values        = scaled_values.tolist()
        tsc.units         = "MM"
        tsc.type          = "PER-CUM"
        tsc.interval      = 1440

        dss.put_ts(tsc)
        print(f"  {factor:.1f}x  peak={scaled_peak:.1f} mm  total={scaled_total:.0f} mm  [{pathname}]")

print("\nAll stress-test rainfall scenarios written.")
print(f"\nNext step in HEC-HMS:")
print(f"  1. Duplicate Calibration_Run_1 for each factor")
print(f"  2. Change Met Model hyetograph to the scaled pathname")
print(f"  3. Set Control window: 01 Aug 2018 - 31 Aug 2018")
print(f"  4. Run and record peak stage at Junction-1")
print(f"  Target: find factor where peak stage >= {DIKE_HEIGHT_M} m")

print(f"\n{'Factor':>8}  {'Peak Rain (mm)':>16}  {'Total Rain (mm)':>16}")
print("-" * 46)
for factor in SCALE_FACTORS:
    print(f"  {factor:.1f}x  {storm['precip_mm'].max() * factor:>16.1f}  {storm['precip_mm'].sum() * factor:>16.0f}")