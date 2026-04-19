"""
showcase_proxy.py
=================
SHOWCASE VERSION of Sat_SensorData_proxy.py

Changes from original:
  - Output CSV  : showcase_proxy.csv  (instead of obando_environmental_data.csv)
  - All script references updated to showcase/ folder.
  - All data-fetch logic is identical to the original.

Usage
-----
    python showcase_proxy.py              # incremental append
    python showcase_proxy.py --full       # force full history re-pull
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

GEE_PROJECT    = "jenel-466709"
HISTORY_START  = "2017-01-01"
NAN_LOOKBACK_DAYS = 14

_SCRIPT_DIR = Path(__file__).resolve().parent

# AOI config is one level up from showcase/ (same project config/)
_AOI_PATH   = _SCRIPT_DIR.parent / "config" / "aoi.geojson"

# Output goes to the same data/sensor/ folder as other CSVs
OUTPUT_CSV  = _SCRIPT_DIR.parent / "data" / "sensor" / "showcase_proxy.csv"
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

# ─── DYNAMIC DATE RANGE ───────────────────────────────────────────────────────

def get_date_range(force_full: bool = False) -> tuple[str, str]:
    today    = date.today()
    end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if force_full or not OUTPUT_CSV.exists():
        mode = "full history (forced)" if force_full else "full history (no CSV found)"
        print(f"  Proxy CSV pull mode: {mode}")
        print(f"  Range: {HISTORY_START} → {today}")
        return HISTORY_START, end_date

    try:
        existing = pd.read_csv(OUTPUT_CSV, parse_dates=["timestamp"])
        if len(existing) == 0:
            return HISTORY_START, end_date

        existing["timestamp"] = pd.to_datetime(existing["timestamp"])
        last_date = existing["timestamp"].max()
        if hasattr(last_date, "date"):
            last_date = last_date.date()

        tail     = existing.tail(NAN_LOOKBACK_DAYS).copy()
        nan_mask = tail[["soil_moisture", "humidity"]].isnull().any(axis=1)

        if nan_mask.any():
            earliest_nan_ts = tail.loc[nan_mask, "timestamp"].min()
            earliest_nan    = earliest_nan_ts.date() if hasattr(earliest_nan_ts, "date") else pd.Timestamp(earliest_nan_ts).date()
            start_date      = earliest_nan.strftime("%Y-%m-%d")
            print(f"  Proxy CSV pull mode: incremental (+ NaN backfill from {start_date})")
        else:
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            if start_date >= end_date:
                print(f"  showcase_proxy.csv already up to date (last: {last_date}).")
                return None, None
            print(f"  Proxy CSV pull mode: incremental ({start_date} → {today})")

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


# ─── 1. SOIL MOISTURE ─────────────────────────────────────────────────────────

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
            continue
        info = batch.getInfo()
        rows = _fc_to_rows(info, ["timestamp", "soil_moisture"])
        all_rows.extend(rows)
        print(f"  {y_start[:7]} → {y_end[:7]}: {len(rows)} records")
        del info, rows, batch
        gc.collect()

    if not all_rows:
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


# ─── 2. HUMIDITY ──────────────────────────────────────────────────────────────

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

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  → {len(df)} humidity records fetched")
    return df[["timestamp", "humidity"]]


# ─── 3. TIDAL WATER LEVEL ─────────────────────────────────────────────────────

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
            df_r["timestamp"] = pd.to_datetime(df_r["timestamp"]).dt.normalize()
            df_r = df_r[df_r["timestamp"].isin(gaps_rean)]
            df_r["waterlevel"] = df_r["waterlevel"].astype("float32")
            frames.append(df_r)
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
            df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.normalize()
            df_n = df_n[df_n["timestamp"].isin(gaps_nrt)]
            df_n["waterlevel"] = df_n["waterlevel"].astype("float32")
            frames.append(df_n)
        except Exception as e:
            print(f"    WARNING: CMEMS NRT failed: {e}")

    gc.collect()

    if not frames:
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    df_cmems = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
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
    print(f"  → {len(df)} tidal records total")
    return df[["timestamp", "waterlevel"]]


# ─── SAVE ─────────────────────────────────────────────────────────────────────

def save_to_csv(new_df: pd.DataFrame, force_full: bool = False) -> None:
    if force_full:
        df_out = new_df.copy()
        df_out["timestamp"] = pd.to_datetime(df_out["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        for col in ["waterlevel", "soil_moisture", "humidity"]:
            if col in df_out.columns:
                df_out[col] = df_out[col].astype("float32")
        df_out = df_out[["timestamp", "waterlevel", "soil_moisture", "humidity"]]
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✓ Saved (full overwrite) → {OUTPUT_CSV}  ({len(df_out)} rows)")
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
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    for col in ["waterlevel", "soil_moisture", "humidity"]:
        if col in combined.columns:
            combined[col] = combined[col].astype("float32")
    combined = combined[["timestamp", "waterlevel", "soil_moisture", "humidity"]]
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved (incremental) → {OUTPUT_CSV}  ({len(combined)} rows total)")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def run_pipeline(force_full: bool = False) -> bool:
    """
    Main entry point. Called by showcase_start.py or directly.
    Returns True if new data was fetched and written.
    """
    print("\n" + "=" * 55)
    print("  showcase_proxy — Satellite Sensor Update")
    print(f"  Output: showcase_proxy.csv")
    print(f"  Mode  : {'FULL HISTORY RE-PULL' if force_full else 'incremental append'}")
    print("=" * 55)

    start_date, end_date = get_date_range(force_full=force_full)

    if start_date is None:
        return False

    init_gee()
    aoi = load_aoi()

    df_sm    = get_soil_moisture(start_date, end_date)
    df_humid = get_humidity(start_date, end_date, aoi)
    df_tide  = get_tidal_level(start_date, end_date)

    print("\nMerging new data...")
    df = (
        df_tide
        .merge(df_sm,    on="timestamp", how="outer")
        .merge(df_humid, on="timestamp", how="outer")
    )
    del df_tide, df_sm, df_humid
    gc.collect()

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(how="all")
    for col in ["waterlevel", "soil_moisture", "humidity"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    df = df[["timestamp", "waterlevel", "soil_moisture", "humidity"]]

    print(f"  New/updated rows : {len(df)}")
    save_to_csv(df, force_full=force_full)
    return True


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="showcase_proxy — GEE sensor extractor (writes showcase_proxy.csv)"
    )
    parser.add_argument("--full", action="store_true",
                        help="Force full history re-pull from 2017-01-01.")
    args = parser.parse_args()
    run_pipeline(force_full=args.full)
