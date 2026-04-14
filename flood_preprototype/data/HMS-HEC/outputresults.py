"""
export_results_fixed.py
Reads all years from sensor_data.dss and exports to CSV with headers:
    timestamp, waterlevel, soilmoisture, humidity

Run with HEC-HMS CLOSED.
Run from: D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\HMS-HEC\
Conda env: floodenv
"""

import numpy as np
import pandas as pd
from pydsstools.heclib.dss import HecDss

SENSOR_DSS = r'D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\HMS-HEC\sensor_data.dss'
OUT_CSV    = 'full_results.csv'

# Replace the read_series function with this version
def read_series(dss_path, pathname, name):
    with HecDss.Open(dss_path) as dss:
        # Explicitly request the full date window
        ts = dss.read_ts(
            pathname,
            window=("01Jan2017 00:00:00", "10Apr2026 00:00:00"),
            trim_missing=True
        )
        vals = np.array(ts.values, dtype=float)
        vals[vals < -1e30] = np.nan
        dates = pd.to_datetime(ts.pytimes)
        return pd.Series(vals, index=dates, name=name)

print("Reading sensor_data.dss ...")

obs_stage    = read_series(SENSOR_DSS,
                   '/YOUR-RIVER/STATION1/STAGE/01JAN2017/1DAY/OBS/',
                   'waterlevel')

obs_soil     = read_series(SENSOR_DSS,
                   '/YOUR-RIVER/STATION1/SOIL MOISTURE/01JAN2017/1DAY/OBS/',
                   'soilmoisture')

obs_humidity = read_series(SENSOR_DSS,
                   '/YOUR-RIVER/STATION1/RELATIVE HUMIDITY/01JAN2017/1DAY/OBS/',
                   'humidity')

# Merge on the actual date index from DSS — outer join keeps all dates
df = pd.concat([obs_stage, obs_soil, obs_humidity], axis=1)
df.index.name = 'timestamp'
df = df.sort_index()

# Drop rows where all three are NaN
df.dropna(how='all', inplace=True)

print(f"Date range : {df.index[0].date()} → {df.index[-1].date()}")
print(f"Total rows : {len(df)}")
print(f"Columns    : {list(df.columns)}")
print(f"\nHead:\n{df.head(3)}")
print(f"\nTail:\n{df.tail(3)}")
print(f"\nNaN counts:\n{df.isna().sum()}")

df.to_csv(OUT_CSV, index=True)
print(f"\nSaved → {OUT_CSV}")