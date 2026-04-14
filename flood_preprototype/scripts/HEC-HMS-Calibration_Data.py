"""
HEC-HMS-Calibration_Data.py
=======================
GEE Data Extractor — Obando Environmental Data  (Incremental Append Mode)

PURPOSE
-------
Fetches the latest missing sensor readings and APPENDS them to the existing
HEC-HMS-Calibration_Data.csv. On first run it pulls the full history
(2017-01-01 → today). On subsequent runs it only pulls from the last
recorded date + 1 day → today, making daily runs fast.

ROLE IN PIPELINE
----------------
This script is Step 0 of Start.py's daily scheduled run:
    Step 0  HEC-HMS-Calibration_Data.py  ← fetch today's sensor reading
    Step 1  RF_Predict.run_pipeline()    ← predict on updated CSV
    Step 2  FB_Post (if WATCH+)

WATERLEVEL UNITS
----------------
waterlevel is output in RAW METERS (m above chart datum).
UHSLC sea_level field is in millimetres — divided by 1000 to get metres.
CMEMS zos field is already in metres — used as-is.
No z-score normalisation is applied.

DATA SOURCES
------------
- waterlevel    : UHSLC Station 370 Manila South Harbor (primary)
                  CMEMS zos SSH anomaly (gap-fill)
                  Output is RAW METRES (m)
- soil_moisture : ERA5-Land volumetric_soil_water_layer_1 (0-7cm), daily aggr
- humidity      : Aqua MODIS MCD19A2_GRANULES Column_WV
- rainfall_1d   : GPM IMERG daily precipitation (mm), 1-day accumulation    [NEW]
- rainfall_7d   : GPM IMERG daily precipitation (mm), 7-day rolling sum     [NEW]
- era5_runoff_1d: ERA5-Land surface+subsurface runoff (mm), 1-day           [NEW]
- era5_runoff_7d: ERA5-Land surface+subsurface runoff (mm), 7-day sum       [NEW]

CHANGES FROM PREVIOUS VERSION
------------------------------
+ Added get_gpm_rainfall()  — GPM IMERG V07 daily rainfall via GEE
+ Added get_era5_runoff()   — ERA5-Land hourly runoff aggregated to daily mm
+ Output CSV now has 8 columns instead of 4:
      timestamp, waterlevel, soil_moisture, humidity,
      rainfall_1d, rainfall_7d, era5_runoff_1d, era5_runoff_7d
+ NaN backfill check extended to cover all 6 sensor columns
+ GPM and ERA5 runoff use the same HMS-HEC AOI as all other sources
  (no separate AOI file needed)

NaN BACKFILL BEHAVIOUR
----------------------
On every incremental run, the script inspects the last NAN_LOOKBACK_DAYS
rows. If any sensor column is NaN (expected for the most recent ~5 days
due to ERA5/MODIS/GPM publication lag), it re-fetches from the earliest
NaN row so those values get filled in once upstream providers publish them.

Requirements:
    pip install earthengine-api geemap pandas requests copernicusmarine
    earthengine authenticate
    copernicusmarine login   ← one-time CMEMS credential setup (free account)

Usage
-----
    python HEC-HMS-Calibration_Data.py              # incremental
    python HEC-HMS-Calibration_Data.py --full       # force full history re-pull
"""

import gc
import ee
import io
import os
import json
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

# ─── CONFIG ───────────────────────────────────────────────────────────────────

GEE_PROJECT       = "jenel-466709"
HISTORY_START     = "2017-01-01"
NAN_LOOKBACK_DAYS = 14          # ERA5 lag ~5-7d, MODIS ~2-4d, GPM ~1-2d

_SCRIPT_DIR = Path(__file__).resolve().parent
_AOI_PATH   = _SCRIPT_DIR.parent / "config" / "aoi.geojson"   # unchanged
OUTPUT_CSV  = _SCRIPT_DIR.parent / "data" / "HMS-HEC" / "HEC-HMS-Calibration_Data.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

_CMEMS_LON_MIN = 120.5
_CMEMS_LON_MAX = 121.2
_CMEMS_LAT_MIN = 14.5
_CMEMS_LAT_MAX = 15.0

_UHSLC_ERDDAP_URL = (
    "https://uhslc.soest.hawaii.edu/erddap/tabledap/global_daily_fast.csv"
    "?sea_level,time"
    "&uhslc_id=370"
    "&time>={start}T00:00:00Z"
    "&time<={end}T00:00:00Z"
)

# All new columns added in this version
_SENSOR_COLS = ["soil_moisture", "humidity", "rainfall_1d", "rainfall_7d",
                "era5_runoff_1d", "era5_runoff_7d"]

# ─── DYNAMIC DATE RANGE ───────────────────────────────────────────────────────

def get_date_range(force_full: bool = False) -> tuple[str, str]:
    """
    Determine the date range to fetch.
    Returns (start_date_str, end_date_str) or (None, None) if up to date.
    End date is tomorrow (exclusive upper bound for GEE filterDate).
    """
    today    = date.today()
    end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if force_full or not OUTPUT_CSV.exists():
        mode = "full history (forced)" if force_full else "full history (no CSV found)"
        print(f"  CSV pull mode: {mode}")
        print(f"  Range: {HISTORY_START} → {today}")
        return HISTORY_START, end_date

    try:
        existing = pd.read_csv(OUTPUT_CSV, parse_dates=["timestamp"])
        if len(existing) == 0:
            print(f"  CSV pull mode: full history (CSV empty)")
            return HISTORY_START, end_date

        existing["timestamp"] = pd.to_datetime(existing["timestamp"])
        last_date = existing["timestamp"].max()
        if hasattr(last_date, "date"):
            last_date = last_date.date()

        # NaN backfill: check all sensor columns that exist in the CSV
        present_cols = [c for c in _SENSOR_COLS if c in existing.columns]
        tail         = existing.tail(NAN_LOOKBACK_DAYS).copy()
        nan_mask     = tail[present_cols].isnull().any(axis=1) if present_cols else pd.Series(False, index=tail.index)

        if nan_mask.any():
            earliest_nan_ts = tail.loc[nan_mask, "timestamp"].min()
            earliest_nan    = earliest_nan_ts.date() if hasattr(earliest_nan_ts, "date") else pd.Timestamp(earliest_nan_ts).date()
            start_date      = earliest_nan.strftime("%Y-%m-%d")
            n_nan_rows      = int(nan_mask.sum())
            print(f"  CSV pull mode: incremental (+ NaN backfill)")
            print(f"  Last row      : {last_date}")
            print(f"  NaN rows      : {n_nan_rows} in last {NAN_LOOKBACK_DAYS} rows")
            print(f"  Pulling from  : {start_date} → {today}")
        else:
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            if start_date >= end_date:
                print(f"  CSV already up to date (last: {last_date}). Nothing to fetch.")
                return None, None
            n_days = (today - (last_date + timedelta(days=1))).days + 1
            print(f"  CSV pull mode : incremental")
            print(f"  Last row      : {last_date}")
            print(f"  Pulling       : {start_date} → {today} ({n_days} days)")

        return start_date, end_date

    except Exception as e:
        print(f"  WARNING: Could not read existing CSV ({e}). Falling back to full history.")
        return HISTORY_START, end_date


# ─── INIT ─────────────────────────────────────────────────────────────────────

def init_gee():
    print("Authenticating with Google Earth Engine...")
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
    print("✓ GEE initialized")


def load_aoi() -> ee.Geometry:
    with open(_AOI_PATH, "r") as f:
        _geojson = json.load(f)
    if _geojson["type"] == "FeatureCollection":
        _geometry = _geojson["features"][0]["geometry"]
    elif _geojson["type"] == "Feature":
        _geometry = _geojson["geometry"]
    else:
        _geometry = _geojson
    aoi = ee.Geometry(_geometry)
    del _geojson, _geometry
    print(f"✓ AOI loaded from: {_AOI_PATH}")
    return aoi


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _quarter_ranges(start_str: str, end_str: str):
    start = pd.Timestamp(start_str)
    end   = pd.Timestamp(end_str)
    cur   = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(months=3), end)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt


def _year_ranges(start_str: str, end_str: str):
    start = pd.Timestamp(start_str)
    end   = pd.Timestamp(end_str)
    cur   = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(years=1), end)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt


def _fc_to_rows(fc_info: dict, props: list) -> list:
    return [
        {k: f["properties"].get(k) for k in props}
        for f in fc_info["features"]
    ]


# ─── 1. SOIL MOISTURE — ERA5-Land daily ───────────────────────────────────────
# Unchanged from previous version.

def get_soil_moisture(start_date: str, end_date: str) -> pd.DataFrame:
    print("Fetching ERA5-Land soil moisture data...")

    BAND     = "volumetric_soil_water_layer_1"
    LAND_AOI = ee.Geometry.Point([120.9333, 14.8333]).buffer(5000)

    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start_date, end_date)
        .select(BAND)
    )

    def _map_fn(img):
        date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        val = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=LAND_AOI,
            scale=11132,
            maxPixels=1e9,
        ).get(BAND)
        return ee.Feature(None, {"timestamp": date_str, "soil_moisture": val})

    all_rows = []
    for y_start, y_end in _year_ranges(start_date, end_date):
        batch = (
            collection
            .filterDate(y_start, y_end)
            .map(_map_fn)
            .filter(ee.Filter.notNull(["soil_moisture"]))
        )
        n = batch.size().getInfo()
        if n == 0:
            print(f"  {y_start[:7]}: 0 images, skipping")
            continue
        info = batch.getInfo()
        rows = _fc_to_rows(info, ["timestamp", "soil_moisture"])
        all_rows.extend(rows)
        print(f"  {y_start[:7]} → {y_end[:7]}: {len(rows)} records")
        del info, rows, batch
        gc.collect()

    if not all_rows:
        print("  WARNING: No soil moisture data retrieved.")
        return pd.DataFrame(columns=["timestamp", "soil_moisture"])

    df = pd.DataFrame(all_rows)
    del all_rows
    gc.collect()

    df["timestamp"]     = pd.to_datetime(df["timestamp"])
    df["soil_moisture"] = pd.to_numeric(df["soil_moisture"], errors="coerce").astype("float32")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  → {len(df)} soil moisture records fetched")
    return df[["timestamp", "soil_moisture"]]


# ─── 2. ATMOSPHERIC HUMIDITY — Aqua MODIS MCD19A2 ────────────────────────────
# Unchanged from previous version.

def get_humidity(start_date: str, end_date: str, aoi: ee.Geometry) -> pd.DataFrame:
    print("Fetching Aqua MODIS humidity data...")

    BAND = "Column_WV"
    collection = (
        ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
        .filterDate(start_date, end_date)
        .select(BAND)
        .filterBounds(aoi)
    )

    def _map_fn(img):
        date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        val = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9,
        ).get(BAND)
        return ee.Feature(None, {"timestamp": date_str, "humidity_raw": val})

    daily_frames = []
    for q_start, q_end in _quarter_ranges(start_date, end_date):
        batch = (
            collection
            .filterDate(q_start, q_end)
            .map(_map_fn)
            .filter(ee.Filter.notNull(["humidity_raw"]))
        )
        n = batch.size().getInfo()
        if n == 0:
            continue
        info = batch.getInfo()
        rows = _fc_to_rows(info, ["timestamp", "humidity_raw"])
        del info, batch
        gc.collect()

        if not rows:
            del rows
            continue

        qdf = pd.DataFrame(rows)
        del rows
        qdf["timestamp"]    = pd.to_datetime(qdf["timestamp"])
        qdf["humidity_raw"] = pd.to_numeric(qdf["humidity_raw"], errors="coerce")
        qdf = qdf.groupby("timestamp", as_index=False)["humidity_raw"].mean()
        daily_frames.append(qdf)
        del qdf
        gc.collect()

    if not daily_frames:
        print("  WARNING: No humidity data retrieved.")
        return pd.DataFrame(columns=["timestamp", "humidity"])

    df = pd.concat(daily_frames, ignore_index=True)
    del daily_frames
    gc.collect()

    df = df.groupby("timestamp", as_index=False)["humidity_raw"].mean()
    df["humidity"] = (df["humidity_raw"] * 0.001).astype("float32")
    df.drop(columns=["humidity_raw"], inplace=True)

    n_bad = (df["humidity"] <= 0.15).sum()
    if n_bad > 0:
        df.loc[df["humidity"] <= 0.15, "humidity"] = float("nan")
        df["humidity"] = df["humidity"].ffill(limit=3).astype("float32")
        print(f"  Humidity floor filter: {n_bad} near-zero values replaced")

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  → {len(df)} humidity records fetched")
    return df[["timestamp", "humidity"]]


# ─── 3. TIDAL WATER LEVEL — UHSLC + CMEMS gap-fill ───────────────────────────
# Unchanged from previous version.

def _get_uhslc(start_date: str, end_date: str) -> pd.DataFrame:
    print("Fetching UHSLC Manila tide gauge (ERDDAP)...")

    url = _UHSLC_ERDDAP_URL.format(start=start_date, end=end_date)

    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        raw_bytes = resp.content
    except requests.RequestException as e:
        print(f"  UHSLC ERDDAP fetch failed: {e}")
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    chunks = pd.read_csv(
        io.BytesIO(raw_bytes),
        skiprows=[1],
        on_bad_lines="skip",
        dtype={"sea_level": "float32"},
        chunksize=500,
    )
    del raw_bytes
    gc.collect()

    frames = []
    for chunk in chunks:
        chunk = chunk.rename(columns={"time": "timestamp", "sea_level": "sea_level_mm"})
        chunk["timestamp"]    = pd.to_datetime(chunk["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
        chunk["sea_level_mm"] = pd.to_numeric(chunk["sea_level_mm"], errors="coerce")
        chunk = chunk[chunk["sea_level_mm"] > -9000].dropna(subset=["sea_level_mm"])
        chunk["waterlevel"]   = (chunk["sea_level_mm"] / 1000.0).astype("float32")
        frames.append(chunk[["timestamp", "waterlevel"]])

    if not frames:
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    df    = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    print(f"  → {len(df)} UHSLC records fetched")
    return df


def _get_cmems(missing_dates: pd.DatetimeIndex) -> pd.DataFrame:
    if missing_dates.empty:
        print("  No UHSLC gaps to fill — skipping CMEMS.")
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    import copernicusmarine

    print(f"  Gap-filling {len(missing_dates)} missing days via CMEMS...")

    REANALYSIS_CUTOFF = pd.Timestamp("2021-07-01")
    gaps_rean = missing_dates[missing_dates <  REANALYSIS_CUTOFF]
    gaps_nrt  = missing_dates[missing_dates >= REANALYSIS_CUTOFF]

    common_kwargs = dict(
        variables         = ["zos"],
        minimum_longitude = _CMEMS_LON_MIN,
        maximum_longitude = _CMEMS_LON_MAX,
        minimum_latitude  = _CMEMS_LAT_MIN,
        maximum_latitude  = _CMEMS_LAT_MAX,
        username          = os.getenv("CMEMS_USERNAME"),
        password          = os.getenv("CMEMS_PASSWORD"),
    )

    frames = []

    if len(gaps_rean) > 0:
        try:
            ds = copernicusmarine.open_dataset(
                dataset_id     = "cmems_mod_glo_phy_my_0.083deg_P1D-m",
                start_datetime = str(gaps_rean.min().date()),
                end_datetime   = str(gaps_rean.max().date()),
                **common_kwargs,
            )
            df_r = (
                ds["zos"].mean(dim=["latitude", "longitude"])
                .to_dataframe().reset_index()[["time", "zos"]]
                .rename(columns={"time": "timestamp", "zos": "waterlevel"})
            )
            ds.close()
            del ds
            df_r["timestamp"] = pd.to_datetime(df_r["timestamp"]).dt.normalize()
            df_r = df_r[df_r["timestamp"].isin(gaps_rean)]
            df_r["waterlevel"] = df_r["waterlevel"].astype("float32")
            frames.append(df_r)
            del df_r
        except Exception as e:
            print(f"    WARNING: CMEMS reanalysis failed: {e}")

    if len(gaps_nrt) > 0:
        try:
            ds = copernicusmarine.open_dataset(
                dataset_id     = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
                start_datetime = str(gaps_nrt.min().date()),
                end_datetime   = str(gaps_nrt.max().date()),
                **common_kwargs,
            )
            df_n = (
                ds["zos"].mean(dim=["latitude", "longitude"])
                .to_dataframe().reset_index()[["time", "zos"]]
                .rename(columns={"time": "timestamp", "zos": "waterlevel"})
            )
            ds.close()
            del ds
            df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.normalize()
            df_n = df_n[df_n["timestamp"].isin(gaps_nrt)]
            df_n["waterlevel"] = df_n["waterlevel"].astype("float32")
            frames.append(df_n)
            del df_n
        except Exception as e:
            print(f"    WARNING: CMEMS NRT failed: {e}")

    gc.collect()

    if not frames:
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    df_cmems = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"    → {len(df_cmems)} CMEMS gap-fill records")
    return df_cmems[["timestamp", "waterlevel"]].reset_index(drop=True)


def get_tidal_level(start_date: str, end_date: str) -> pd.DataFrame:
    print("Fetching tidal water level data (UHSLC + CMEMS)...")

    df_uhslc  = _get_uhslc(start_date, end_date)
    all_dates = pd.date_range(start_date, end=pd.Timestamp(end_date) - pd.Timedelta(days=1), freq="D")
    covered   = set(df_uhslc["timestamp"].dt.normalize()) if not df_uhslc.empty else set()
    missing   = pd.DatetimeIndex([d for d in all_dates if d not in covered])
    df_cmems  = _get_cmems(missing)
    del missing, covered
    gc.collect()

    df = pd.concat([df_uhslc, df_cmems], ignore_index=True)
    del df_uhslc, df_cmems
    gc.collect()

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["waterlevel"] = pd.to_numeric(df["waterlevel"], errors="coerce").astype("float32")
    print(f"  → {len(df)} tidal records total (UHSLC + CMEMS gap-fill)")
    print(f"  Waterlevel range: {df['waterlevel'].min():.3f}m → {df['waterlevel'].max():.3f}m")
    return df[["timestamp", "waterlevel"]]


# ─── 4. GPM IMERG RAINFALL  [NEW] ────────────────────────────────────────────
# Ported from sentinel1_GEE.py. Uses HMS-HEC AOI instead of Sentinel AOI.
# Produces rainfall_1d and rainfall_7d columns.
# GPM IMERG V07 half-hourly → summed to daily mm (multiply by 0.5 for mm/hr→mm/30min).
# 7-day rolling sum is computed in pandas after the full series is built,
# so the fetch window is extended by 7 days before start_date to ensure
# rolling windows are populated from day 1.

def get_gpm_rainfall(start_date: str, end_date: str, aoi: ee.Geometry) -> pd.DataFrame:
    """
    Fetch GPM IMERG V07 daily rainfall for the given date range.

    To guarantee rainfall_7d is fully populated from start_date onward,
    the caller should pass a start_date that is already 7 days before the
    earliest date needed (this is handled in run_pipeline).

    Returns DataFrame with columns: timestamp, rainfall_1d, rainfall_7d
    Timestamps are daily, normalised to midnight.
    """
    print("Fetching GPM IMERG daily rainfall data...")

    start      = pd.Timestamp(start_date)
    end        = pd.Timestamp(end_date)
    total_days = (end - start).days
    chunk_size = 90            # matches sentinel1_GEE.py chunk size
    rain_dict  = {}

    for offset in range(0, total_days, chunk_size):
        chunk_start = (start + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        chunk_end   = (start + pd.Timedelta(days=min(offset + chunk_size, total_days))).strftime("%Y-%m-%d")

        def _fetch_chunk(cs=chunk_start, ce=chunk_end):
            gpm = (
                ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
                .filterDate(cs, ce)
                .filterBounds(aoi)
                .select("precipitation")
            )
            n_days  = (pd.Timestamp(ce) - pd.Timestamp(cs)).days
            cs_date = ee.Date(cs)

            def daily_sum(day_offset):
                day       = cs_date.advance(day_offset, "day")
                daily_img = gpm.filterDate(day, day.advance(1, "day")).sum().multiply(0.5)
                mean_val  = daily_img.reduceRegion(
                    reducer  = ee.Reducer.mean(),
                    geometry = aoi,
                    scale    = 10000,
                    maxPixels= 1e8,
                    bestEffort=True,
                ).get("precipitation")
                return ee.Feature(None, {"date": day.format("YYYY-MM-dd"), "rainfall_mm": mean_val})

            day_list = ee.List.sequence(0, n_days - 1)
            features = ee.FeatureCollection(day_list.map(daily_sum))
            return features.toList(features.size()).getInfo()

        chunk_data = _safe_gee_compute(_fetch_chunk, max_retries=3)
        if chunk_data:
            for feat in chunk_data:
                props = feat.get("properties", {})
                d     = props.get("date")
                val   = props.get("rainfall_mm")
                if d and val is not None:
                    rain_dict[d] = round(float(val), 3)

        print(f"  GPM days {offset}–{min(offset + chunk_size, total_days) - 1} done")

    if not rain_dict:
        print("  WARNING: No GPM rainfall data retrieved.")
        return pd.DataFrame(columns=["timestamp", "rainfall_1d", "rainfall_7d"])

    df = pd.DataFrame([
        {"timestamp": pd.Timestamp(d), "rainfall_1d": v}
        for d, v in sorted(rain_dict.items())
    ])
    df["rainfall_1d"] = df["rainfall_1d"].astype("float32")

    # 7-day rolling sum — window=7, min_periods=1 so edge rows are still populated
    df["rainfall_7d"] = df["rainfall_1d"].rolling(window=7, min_periods=1).sum().astype("float32")

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  → {len(df)} GPM daily records fetched")
    print(f"  Rainfall range  : {df['rainfall_1d'].min():.1f} – {df['rainfall_1d'].max():.1f} mm/day")
    return df[["timestamp", "rainfall_1d", "rainfall_7d"]]


# ─── 5. ERA5-LAND RUNOFF  [NEW] ───────────────────────────────────────────────
# Ported from sentinel1_GEE.py (get_era5_daily), rainfall-runoff columns only.
# era5_soil_water is NOT included here — it is already covered by soil_moisture
# (ERA5-Land DAILY_AGGR, same variable, fetched in get_soil_moisture above).
# Runoff = surface_runoff + sub_surface_runoff, converted m → mm (*1000).
# era5_runoff_7d is a 7-day rolling sum computed in pandas.
# Fetch window is extended by 7 days before start_date in run_pipeline.

def get_era5_runoff(start_date: str, end_date: str, aoi: ee.Geometry) -> pd.DataFrame:
    """
    Fetch ERA5-Land hourly surface + subsurface runoff, aggregated to daily mm.

    Returns DataFrame with columns: timestamp, era5_runoff_1d, era5_runoff_7d
    """
    print("Fetching ERA5-Land runoff data...")

    # Use the centroid of the HMS-HEC AOI buffered to 5 km.
    # ERA5 native resolution is ~9 km so a 5 km buffer is appropriate.
    era5_aoi = aoi.centroid(maxError=100).buffer(5000)

    start      = pd.Timestamp(start_date)
    end        = pd.Timestamp(end_date)
    total_days = (end - start).days
    chunk_size = 30            # matches sentinel1_GEE.py ERA5 chunk size
    runoff_dict: dict[str, float] = {}

    for offset in range(0, total_days, chunk_size):
        chunk_dates = [
            (start + pd.Timedelta(days=offset + d)).strftime("%Y-%m-%d")
            for d in range(min(chunk_size, total_days - offset))
        ]

        for date_str in chunk_dates:
            def _fetch_one_day(ds=date_str):
                day_start = ee.Date(ds)
                day_end   = day_start.advance(1, "day")
                era5_day  = (
                    ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                    .filterDate(day_start, day_end)
                    .select(["surface_runoff", "sub_surface_runoff"])
                )
                # Mean of hourly values, then ×1000 to convert m → mm
                runoff_img = era5_day.mean().multiply(1000)

                sro_val = runoff_img.select("surface_runoff").reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=era5_aoi,
                    scale=9000, maxPixels=1e6, bestEffort=True,
                ).getInfo().get("surface_runoff")

                ssro_val = runoff_img.select("sub_surface_runoff").reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=era5_aoi,
                    scale=9000, maxPixels=1e6, bestEffort=True,
                ).getInfo().get("sub_surface_runoff")

                total = 0.0
                if sro_val  is not None: total += float(sro_val)
                if ssro_val is not None: total += float(ssro_val)
                return round(total, 4)

            result = _safe_gee_compute(_fetch_one_day, max_retries=3)
            runoff_dict[date_str] = result if result is not None else 0.0

        print(f"  ERA5 runoff days {offset}–{min(offset + chunk_size, total_days) - 1} done")

    if not runoff_dict:
        print("  WARNING: No ERA5 runoff data retrieved.")
        return pd.DataFrame(columns=["timestamp", "era5_runoff_1d", "era5_runoff_7d"])

    df = pd.DataFrame([
        {"timestamp": pd.Timestamp(d), "era5_runoff_1d": v}
        for d, v in sorted(runoff_dict.items())
    ])
    df["era5_runoff_1d"] = df["era5_runoff_1d"].astype("float32")

    # 7-day rolling sum — consistent with GPM treatment above
    df["era5_runoff_7d"] = df["era5_runoff_1d"].rolling(window=7, min_periods=1).sum().astype("float32")

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  → {len(df)} ERA5 runoff records fetched")
    print(f"  Runoff range    : {df['era5_runoff_1d'].min():.3f} – {df['era5_runoff_1d'].max():.3f} mm/day")
    return df[["timestamp", "era5_runoff_1d", "era5_runoff_7d"]]


# ─── SAFE GEE WRAPPER  [NEW] ──────────────────────────────────────────────────
# Shared retry wrapper used by get_gpm_rainfall and get_era5_runoff.
# Not needed by the original three sources (they handle errors inline).

import time

def _safe_gee_compute(func, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"    GEE attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    GEE max retries reached: {e}")
                return None


# ─── MERGE & APPEND / OVERWRITE ───────────────────────────────────────────────

# Output column order — 4 original + 4 new
_OUTPUT_COLS = [
    "timestamp",
    "waterlevel",
    "soil_moisture",
    "humidity",
    "rainfall_1d",      # NEW
    "rainfall_7d",      # NEW
    "era5_runoff_1d",   # NEW
    "era5_runoff_7d",   # NEW
]


def save_to_csv(new_df: pd.DataFrame, force_full: bool = False) -> None:
    """
    Incremental mode (force_full=False):
        Appends new_df to the existing CSV.
        Deduplicates on timestamp — new row wins on conflict.
        Sorts the full CSV by timestamp before saving.

    Full mode (force_full=True):
        Overwrites the CSV entirely with new_df.
    """
    def _cast(df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        for col in _OUTPUT_COLS[1:]:          # all numeric columns
            if col in df.columns:
                df[col] = df[col].astype("float32")
        # Ensure all output columns exist (fill missing with NaN for graceful upgrades)
        for col in _OUTPUT_COLS:
            if col not in df.columns:
                df[col] = float("nan")
        return df[_OUTPUT_COLS]

    if force_full:
        df_out = _cast(new_df.copy())
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✓ Saved (full overwrite) → {OUTPUT_CSV}")
        print(f"  Total rows written : {len(df_out)}")
        return

    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV, parse_dates=["timestamp"])
        combined = pd.concat([existing, new_df], ignore_index=True)
        del existing
    else:
        combined = new_df.copy()

    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined = _cast(combined)
    combined.to_csv(OUTPUT_CSV, index=False)

    nan_after = combined[[c for c in _SENSOR_COLS if c in combined.columns]].isnull().any(axis=1).sum()
    print(f"\n✓ Saved (incremental append) → {OUTPUT_CSV}")
    print(f"  Total rows now     : {len(combined)}")
    print(f"  Rows still NaN     : {nan_after} (upstream data not yet published)")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def run_pipeline(force_full: bool = False) -> bool:
    """
    Main entry point. Called directly or imported by Start.py.

    force_full=True  — re-pull everything from HISTORY_START, overwrite CSV.
    force_full=False — incremental append + NaN backfill (default).

    Returns True if new data was fetched and written, False if already up to date.

    NOTE on fetch window extension:
        GPM and ERA5 rolling 7-day windows require 7 days of data *before*
        start_date. To avoid NaN at the leading edge, the rainfall/runoff
        fetch is extended 7 days earlier than start_date. The extra rows are
        included in the merge but will be deduplicated against existing CSV
        rows on save, so no double-counting occurs.
    """
    print("\n" + "=" * 60)
    print("  HEC-HMS-Calibration_Data — Sensor Update")
    print(f"  Mode: {'FULL HISTORY RE-PULL' if force_full else 'incremental append'}")
    print("=" * 60)

    start_date, end_date = get_date_range(force_full=force_full)
    if start_date is None:
        return False

    init_gee()
    aoi = load_aoi()

    # Rolling-window lookback: extend GPM + ERA5 fetch 7 days earlier
    rolling_start = (pd.Timestamp(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    # Fetch all sources
    df_sm     = get_soil_moisture(start_date, end_date)
    df_humid  = get_humidity(start_date, end_date, aoi)
    df_tide   = get_tidal_level(start_date, end_date)
    df_rain   = get_gpm_rainfall(rolling_start, end_date, aoi)   # NEW — extended window
    df_runoff = get_era5_runoff(rolling_start, end_date, aoi)     # NEW — extended window

    print("\nMerging new data...")

    # Trim GPM + ERA5 back to start_date before merging so the 7-day
    # lookback rows don't inflate the append (they already exist in CSV).
    start_ts  = pd.Timestamp(start_date)
    df_rain   = df_rain[df_rain["timestamp"] >= start_ts].reset_index(drop=True)
    df_runoff = df_runoff[df_runoff["timestamp"] >= start_ts].reset_index(drop=True)

    df = (
        df_tide
        .merge(df_sm,     on="timestamp", how="outer")
        .merge(df_humid,  on="timestamp", how="outer")
        .merge(df_rain,   on="timestamp", how="outer")   # NEW
        .merge(df_runoff, on="timestamp", how="outer")   # NEW
    )
    del df_tide, df_sm, df_humid, df_rain, df_runoff
    gc.collect()

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(how="all")

    for col in _OUTPUT_COLS[1:]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    print(f"  New/updated rows   : {len(df)}")
    print(f"  Date range         : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\nPreview (last 5 rows):\n{df[_OUTPUT_COLS].tail(5).to_string(index=False)}")

    save_to_csv(df, force_full=force_full)
    return True


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "HEC-HMS-Calibration_Data — GEE sensor data extractor for Obando.\n"
            "Default: incremental append (pulls only missing days + NaN backfill).\n"
            "Use --full to re-pull the entire history from 2017-01-01 and overwrite the CSV."
        )
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Force a full history re-pull from 2017-01-01 → today. "
            "Overwrites HEC-HMS-Calibration_Data.csv entirely."
        ),
    )
    args = parser.parse_args()
    run_pipeline(force_full=args.full)