"""
Sat_SensorData_proxy.py
=======================
GEE Data Extractor — Obando Environmental Data  (Incremental Append Mode)

PURPOSE
-------
Fetches the latest missing sensor readings and APPENDS them to the existing
obando_environmental_data.csv. On first run it pulls the full history
(2017-01-01 → today). On subsequent runs it only pulls from the last
recorded date + 1 day → today, making daily runs fast (1 day of data
instead of 9 years).

ROLE IN PIPELINE
----------------
This script is Step 0 of Start.py's daily scheduled run:
    Step 0  Sat_SensorData_proxy.py  ← fetch today's sensor reading
    Step 1  RF_Predict.run_pipeline() ← predict on updated CSV
    Step 2  FB_Post (if WATCH+)

NORMALIZATION NOTE
------------------
This script will also serve as the normalization reference baseline when
real hardware sensors are deployed. The z-scored waterlevel values from
UHSLC/CMEMS establish the distribution that hardware readings will be
aligned to during the transition period (Phase 1 parallel running).
When hardware sensors take over, their readings should be z-scored against
the same 2017-2021 reference window (uhslc_mu / uhslc_std) to maintain
model compatibility.

DATA SOURCES
------------
- waterlevel : UHSLC Station 370 Manila South Harbor (primary)
               CMEMS zos SSH anomaly (gap-fill)
               Output is z-score normalised (dimensionless)
- soil_moisture : ERA5-Land volumetric_soil_water_layer_1 (0-7cm)
- humidity      : Aqua MODIS MCD19A2_GRANULES Column_WV

OUTPUT
------
    data/sensor/obando_environmental_data.csv
    Columns: timestamp, waterlevel, soil_moisture, humidity
    New rows are APPENDED. Existing rows are never modified.

Requirements:
    pip install earthengine-api geemap pandas requests copernicusmarine
    earthengine authenticate
    copernicusmarine login   ← one-time CMEMS credential setup (free account)
"""

import gc
import ee
import io
import os
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

# ─── CONFIG ───────────────────────────────────────────────────────────────────

GEE_PROJECT    = "jenel-466709"
HISTORY_START  = "2017-01-01"   # only used on first-ever run (empty CSV)

_SCRIPT_DIR = Path(__file__).resolve().parent
_AOI_PATH   = _SCRIPT_DIR.parent / "config" / "aoi.geojson"
OUTPUT_CSV  = _SCRIPT_DIR.parent / "data" / "sensor" / "obando_environmental_data.csv"
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

def get_date_range() -> tuple[str, str]:
    """
    Determine the incremental date range to fetch.

    If the output CSV exists and has data, pull from last_date + 1 day → today.
    If the CSV is empty or missing, pull full history from HISTORY_START → today.

    Returns (start_date_str, end_date_str) as 'YYYY-MM-DD' strings.
    End date is tomorrow (exclusive upper bound for GEE filterDate).
    """
    today     = date.today()
    end_date  = (today + timedelta(days=1)).strftime("%Y-%m-%d")  # exclusive

    if OUTPUT_CSV.exists():
        try:
            existing = pd.read_csv(OUTPUT_CSV, usecols=["timestamp"], parse_dates=["timestamp"])
            if len(existing) > 0:
                last_date  = existing["timestamp"].max()
                # Normalize to date (strip time/tz if present)
                if hasattr(last_date, "date"):
                    last_date = last_date.date()
                start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

                if start_date >= end_date:
                    print(f"  Sensor CSV already up to date (last: {last_date}). Nothing to fetch.")
                    return None, None

                print(f"  Incremental mode: {start_date} → {today} ({(today - (last_date + timedelta(days=1))).days + 1} days)")
                return start_date, end_date
        except Exception as e:
            print(f"  WARNING: Could not read existing CSV ({e}). Falling back to full history.")

    print(f"  Full history mode: {HISTORY_START} → {today}")
    return HISTORY_START, end_date


# ─── NORMALIZATION REFERENCE ──────────────────────────────────────────────────

def get_normalization_reference() -> tuple[float, float]:
    """
    Load the UHSLC normalization reference (mu, std) from the existing CSV.
    Uses the 2017-01-01 → 2021-07-01 window, consistent with the original
    full-history run.

    Returns (mu, std). If CSV is missing or too small, returns (0.0, 1.0)
    which is a safe no-op normalisation.

    NOTE: These values must remain stable across runs so that new appended
    rows are on the same z-score scale as historical rows. Hardware sensor
    readings should also be normalised with these same constants during
    the parallel-running transition period.
    """
    REF_START = pd.Timestamp("2017-01-01")
    REF_END   = pd.Timestamp("2021-07-01")

    if not OUTPUT_CSV.exists():
        return 0.0, 1.0

    try:
        df = pd.read_csv(OUTPUT_CSV, usecols=["timestamp", "waterlevel"], parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        ref = df[(df["timestamp"] >= REF_START) & (df["timestamp"] < REF_END)]
        if len(ref) < 30:
            print("  WARNING: Reference window too small — using full series for normalisation.")
            ref = df
        mu  = float(ref["waterlevel"].mean())
        std = float(ref["waterlevel"].std())
        std = std if std > 0 else 1.0
        print(f"  Normalisation reference loaded — μ={mu:.4f}, σ={std:.4f} (from existing CSV)")
        return mu, std
    except Exception as e:
        print(f"  WARNING: Could not load normalisation reference ({e}). Using (0, 1).")
        return 0.0, 1.0


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


# ─── 1. SOIL MOISTURE — ERA5-Land ─────────────────────────────────────────────

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
        val  = img.reduceRegion(
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
        val  = img.reduceRegion(
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


def get_tidal_level(
    start_date: str,
    end_date: str,
    norm_mu: float,
    norm_std: float,
) -> pd.DataFrame:
    """
    Fetch tidal water level for the given date range and normalise using
    the provided reference mu/std (loaded from existing CSV).

    On first run (full history), mu/std are computed from the 2017-2021
    reference window of the fetched data itself. On incremental runs,
    mu/std come from the existing CSV so new rows are on the same scale.
    """
    print("Fetching tidal water level data (UHSLC + CMEMS)...")

    df_uhslc  = _get_uhslc(start_date, end_date)
    all_dates = pd.date_range(start_date, end=pd.Timestamp(end_date) - pd.Timedelta(days=1), freq="D")
    covered   = set(df_uhslc["timestamp"].dt.normalize()) if not df_uhslc.empty else set()
    missing   = pd.DatetimeIndex([d for d in all_dates if d not in covered])
    df_cmems  = _get_cmems(missing)
    del missing, covered
    gc.collect()

    # ── Normalise ─────────────────────────────────────────────────────────
    # If this is a full-history first run (norm_mu=0, norm_std=1 default),
    # compute the reference from this batch's 2017-2021 window.
    REF_START = pd.Timestamp("2017-01-01")
    REF_END   = pd.Timestamp("2021-07-01")

    if norm_mu == 0.0 and norm_std == 1.0 and not df_uhslc.empty:
        ref_mask  = (df_uhslc["timestamp"] >= REF_START) & (df_uhslc["timestamp"] < REF_END)
        if ref_mask.sum() > 30:
            norm_mu  = float(df_uhslc.loc[ref_mask, "waterlevel"].mean())
            norm_std = float(df_uhslc.loc[ref_mask, "waterlevel"].std())
            norm_std = norm_std if norm_std > 0 else 1.0
            print(f"  First-run normalisation computed — μ={norm_mu:.4f}, σ={norm_std:.4f}")

    if not df_uhslc.empty:
        df_uhslc["waterlevel"] = (
            (df_uhslc["waterlevel"] - norm_mu) / norm_std
        ).astype("float32")

    if not df_cmems.empty:
        # CMEMS: use its own ref window if available, else use UHSLC constants
        ref_mask = (df_cmems["timestamp"] >= REF_START) & (df_cmems["timestamp"] < REF_END)
        if ref_mask.sum() > 10:
            cmems_mu  = float(df_cmems.loc[ref_mask, "waterlevel"].mean())
            cmems_std = float(df_cmems.loc[ref_mask, "waterlevel"].std())
            cmems_std = cmems_std if cmems_std > 0 else 1.0
        else:
            # Incremental run — not enough CMEMS history in this batch.
            # Use UHSLC reference constants (same scale).
            cmems_mu, cmems_std = norm_mu, norm_std
        df_cmems["waterlevel"] = (
            (df_cmems["waterlevel"] - cmems_mu) / cmems_std
        ).astype("float32")

    df = pd.concat([df_uhslc, df_cmems], ignore_index=True)
    del df_uhslc, df_cmems
    gc.collect()

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["waterlevel"] = pd.to_numeric(df["waterlevel"], errors="coerce").astype("float32")
    print(f"  → {len(df)} tidal records total (UHSLC + CMEMS gap-fill)")
    return df[["timestamp", "waterlevel"]]


# ─── MERGE & APPEND ───────────────────────────────────────────────────────────

def append_to_csv(new_df: pd.DataFrame) -> None:
    """
    Append new_df rows to the existing CSV.
    Deduplicates on timestamp so re-running never creates duplicate rows.
    Sorts the full CSV by timestamp before saving.
    """
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV, parse_dates=["timestamp"])
        combined = pd.concat([existing, new_df], ignore_index=True)
        del existing
    else:
        combined = new_df.copy()

    # Deduplicate — keep last (new row wins if same timestamp)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Format timestamp for CSV
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    for col in ["waterlevel", "soil_moisture", "humidity"]:
        if col in combined.columns:
            combined[col] = combined[col].astype("float32")

    combined = combined[["timestamp", "waterlevel", "soil_moisture", "humidity"]]
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved → {OUTPUT_CSV}")
    print(f"  Total rows now : {len(combined)}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def run_pipeline() -> bool:
    """
    Main entry point. Called directly or imported by Start.py.
    Returns True if new data was fetched and appended, False if already up to date.
    """
    print("\n" + "=" * 55)
    print("  Sat_SensorData_proxy — Incremental Sensor Update")
    print("=" * 55)

    start_date, end_date = get_date_range()

    if start_date is None:
        # Already up to date
        return False

    # Load normalisation constants from existing CSV before touching GEE
    norm_mu, norm_std = get_normalization_reference()

    # Init GEE and AOI
    init_gee()
    aoi = load_aoi()

    # Fetch each source for the incremental window
    df_sm    = get_soil_moisture(start_date, end_date)
    df_humid = get_humidity(start_date, end_date, aoi)
    df_tide  = get_tidal_level(start_date, end_date, norm_mu, norm_std)

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

    print(f"  New rows to append : {len(df)}")
    print(f"  Date range         : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\nPreview (last 5 rows):\n{df.tail(5).to_string(index=False)}")

    append_to_csv(df)
    return True


if __name__ == "__main__":
    run_pipeline()