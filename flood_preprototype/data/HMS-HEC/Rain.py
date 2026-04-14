import netCDF4 as nc
import numpy as np
import pandas as pd
import urllib.request
import os
from pydsstools.heclib.dss import HecDss
from pydsstools.core import TimeSeriesContainer

# ── CONFIG ────────────────────────────────────────────────────────────────────
LAT        = 14.706
LON        = 120.936
START_YEAR = 2017
END_YEAR   = 2026
DSS_FILE   = 'sensor_data.dss'
NC_DIR     = 'chirps_nc'   # folder to store downloaded .nc files
os.makedirs(NC_DIR, exist_ok=True)

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"

# ── DOWNLOAD + EXTRACT ────────────────────────────────────────────────────────
all_rows = []

for year in range(START_YEAR, END_YEAR + 1):
    fname    = f"chirps-v2.0.{year}.days_p05.nc"
    fpath    = os.path.join(NC_DIR, fname)
    url      = BASE_URL + fname

    if os.path.exists(fpath):
        print(f"{year}: already downloaded, skipping.")
    else:
        print(f"{year}: downloading {fname} (~40–80 MB)...", end=' ', flush=True)
        try:
            urllib.request.urlretrieve(url, fpath)
            print("done.")
        except Exception as e:
            print(f"\n  FAILED: {e}")
            print(f"  Skipping {year} — you can re-run later to fill the gap.")
            continue

    # Extract point value nearest to LAT/LON
    try:
        ds   = nc.Dataset(fpath)
        lats = ds.variables['latitude'][:]
        lons = ds.variables['longitude'][:]

        lat_idx = int(np.argmin(np.abs(lats - LAT)))
        lon_idx = int(np.argmin(np.abs(lons - LON)))

        # Confirm we landed on the right cell
        if year == START_YEAR:
            print(f"  Grid cell: lat={float(lats[lat_idx]):.3f}°N  "
                  f"lon={float(lons[lon_idx]):.3f}°E  "
                  f"(target: {LAT}°N {LON}°E)")

        time_var  = ds.variables['time']
        time_vals = nc.num2date(time_var[:], time_var.units,
                                only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True)

        precip_raw = ds.variables['precip'][:, lat_idx, lon_idx]
        precip     = np.ma.filled(precip_raw, np.nan).astype(float)

        for t, p in zip(time_vals, precip):
            val = 0.0 if (np.isnan(p) or p < 0) else float(p)
            all_rows.append({'date': pd.Timestamp(t.year, t.month, t.day),
                             'precip_mm': val})

        days_extracted = len(time_vals)
        max_p = float(np.nanmax(precip[precip >= 0])) if np.any(precip >= 0) else 0
        print(f"  {year}: {days_extracted} days extracted, "
              f"max={max_p:.1f} mm, "
              f"annual≈{float(np.nansum(precip[precip>=0])):.0f} mm")
        ds.close()

    except Exception as e:
        print(f"  ERROR reading {fname}: {e}")
        continue

# ── ASSEMBLE DATAFRAME ────────────────────────────────────────────────────────
if not all_rows:
    raise RuntimeError("No data extracted. Check your internet connection and re-run.")

df_rain = (pd.DataFrame(all_rows)
             .sort_values('date')
             .drop_duplicates('date')
             .reset_index(drop=True))

print(f"\n=== CHIRPS SUMMARY ===")
print(f"Total days : {len(df_rain)}")
print(f"Date range : {df_rain['date'].iloc[0].date()} → {df_rain['date'].iloc[-1].date()}")
print(f"Max 1-day  : {df_rain['precip_mm'].max():.1f} mm")
print(f"Mean/day   : {df_rain['precip_mm'].mean():.2f} mm")
annual_mm = df_rain['precip_mm'].mean() * 365
print(f"Annual est : {annual_mm:.0f} mm/yr  (Manila expected: 1800–2400 mm)")

if not (1200 < annual_mm < 3500):
    print("WARNING: Annual total outside expected range. "
          "Check that LAT/LON are correct.")

# ── WRITE TO DSS ──────────────────────────────────────────────────────────────
part_d    = df_rain['date'].iloc[0].strftime('%d%b%Y').upper()
start_str = df_rain['date'].iloc[0].strftime('%d%b%Y %H:%M:%S').upper()
pathname  = f'/YOUR-RIVER/STATION1/PRECIP-INC/{part_d}/1DAY/CHIRPS/'

tsc = TimeSeriesContainer()
tsc.pathname      = pathname
tsc.startDateTime = start_str
tsc.numberValues  = len(df_rain)
tsc.units         = 'MM'
tsc.type          = 'PER-CUM'
tsc.interval      = 1440
tsc.values        = df_rain['precip_mm'].values.astype(float)

with HecDss.Open(DSS_FILE) as dss:
    dss.put(tsc)
    print(f"\nWritten to DSS: {pathname}")

# ── VERIFY ────────────────────────────────────────────────────────────────────
with HecDss.Open(DSS_FILE) as dss:
    ts    = dss.read_ts(pathname)
    vals  = np.array(ts.values)
    valid = vals[vals > -1e+30]
    rainy = np.sum(valid > 0)
    print(f"Verified  : {len(valid)} values  "
          f"min={valid.min():.1f}  max={valid.max():.1f}  "
          f"mean={valid.mean():.2f}  "
          f"rainy days={rainy}  nulls={len(vals)-len(valid)}")

print(f"\n=== NEXT STEP IN HEC-HMS ===")
print(f"Components → Time-Series Data Manager → Precipitation Gage → Rainfall_Gage")
print(f"  DSS File : {DSS_FILE}")
print(f"  Pathname : {pathname}")