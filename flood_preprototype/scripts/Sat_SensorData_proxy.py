"""
GEE Data Extractor — Obando Environmental Data
Extracts: Soil Moisture (ERA5-Land), Atmospheric Humidity (Aqua/MODIS), Tidal Level (Obando)
Output: CSV with columns: timestamp, waterlevel, soil_moisture, humidity

Requirements:
    pip install earthengine-api geemap pandas requests copernicusmarine
    earthengine authenticate
    copernicusmarine login   ← one-time CMEMS credential setup (free account)

CMEMS account (free): https://marine.copernicus.eu/
Or pass credentials via environment variables: CMEMS_USERNAME / CMEMS_PASSWORD
"""

import ee
import io
import os
import json
import requests
import pandas as pd
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

GEE_PROJECT = "jenel-466709"
START_DATE  = "2017-01-01"
END_DATE    = "2026-03-01"

# Paths — relative to this script's location
# Project structure:
#   flood_preprototype/
#     config/aoi.geojson
#     data/sensor/           ← CSV output
#     scripts/               ← this file
_SCRIPT_DIR = Path(__file__).resolve().parent
_AOI_PATH   = _SCRIPT_DIR.parent / "config" / "aoi.geojson"
OUTPUT_CSV  = _SCRIPT_DIR.parent / "data" / "sensor" / "obando_environmental_data.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Manila Bay bounding box for CMEMS spatial subset (covers Obando coastline)
_CMEMS_LON_MIN = 120.5
_CMEMS_LON_MAX = 121.2
_CMEMS_LAT_MIN = 14.5
_CMEMS_LAT_MAX = 15.0

# UHSLC Manila South Harbor Station 370 — ERDDAP API
# Station 111 (old Manila gauge) had a multi-year outage around 2021-2022 and is gap-prone.
_UHSLC_ERDDAP_URL = (
    "https://uhslc.soest.hawaii.edu/erddap/tabledap/global_daily_fast.csv"
    "?sea_level,time"
    "&uhslc_id=370"
    "&time>={start}T00:00:00Z"
    "&time<={end}T00:00:00Z"
)

# ─── INIT ─────────────────────────────────────────────────────────────────────

print("Authenticating with Google Earth Engine...")
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
print("✓ GEE initialized")

# ─── LOAD AOI (must be after ee.Initialize) ───────────────────────────────────

with open(_AOI_PATH, "r") as f:
    _geojson = json.load(f)

if _geojson["type"] == "FeatureCollection":
    _geometry = _geojson["features"][0]["geometry"]
elif _geojson["type"] == "Feature":
    _geometry = _geojson["geometry"]
else:
    _geometry = _geojson

AOI = ee.Geometry(_geometry)
print(f"✓ AOI loaded from: {_AOI_PATH}")


# ─── 1. SOIL MOISTURE — ERA5-Land (m³/m³) ────────────────────────────────────
# ERA5-Land reanalysis: volumetric soil water, layer 1 (0-7cm depth).
# Daily, continuous coverage, standard WGS84 projection — no projection issues.
# Values typically 0.05–0.45 m³/m³ for this region.

def get_soil_moisture() -> pd.DataFrame:
    """
    ERA5-Land Daily Aggregated — volumetric_soil_water_layer_1 (0-7cm, m³/m³).
    Dataset: ECMWF/ERA5_LAND/DAILY_AGGR

    NOTE: ERA5-Land masks ocean pixels. The full AOI spans Manila Bay (ocean),
    so we use a small land-only buffer around Obando town center instead of
    the full AOI for reduceRegion, to guarantee land pixel coverage.
    Processed one year at a time to stay within GEE memory limits.
    """
    print("Fetching ERA5-Land soil moisture data...")

    BAND = "volumetric_soil_water_layer_1"

    # Obando town center — guaranteed land pixel, avoids Manila Bay ocean masking
    OBANDO_POINT = ee.Geometry.Point([120.9333, 14.8333])
    LAND_AOI     = OBANDO_POINT.buffer(5000)   # 5km radius land buffer

    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(START_DATE, END_DATE)
        .select(BAND)
    )

    start_year = int(START_DATE[:4])
    end_year   = int(END_DATE[:4])

    def fetch_year(year):
        yr_start = f"{year}-01-01"
        yr_end   = f"{year + 1}-01-01"

        col = collection.filterDate(yr_start, yr_end)
        n   = col.size().getInfo()
        if n == 0:
            print(f"  {year}: 0 images, skipping")
            return []

        def img_to_feature(img):
            date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            val  = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=LAND_AOI,
                scale=11132,      # ERA5-Land native resolution
                maxPixels=1e9
            ).get(BAND)
            return ee.Feature(None, {"timestamp": date, "soil_moisture": val})

        fc   = col.map(img_to_feature).filter(ee.Filter.notNull(["soil_moisture"]))
        info = fc.getInfo()
        rows = [f["properties"] for f in info["features"]]
        print(f"  {year}: {len(rows)} records")
        return rows

    all_rows = []
    for yr in range(start_year, end_year + 1):
        all_rows.extend(fetch_year(yr))

    if not all_rows:
        print("  WARNING: No soil moisture data retrieved.")
        return pd.DataFrame(columns=["timestamp", "soil_moisture"])

    df = pd.DataFrame(all_rows)
    df["timestamp"]     = pd.to_datetime(df["timestamp"])
    df["soil_moisture"] = pd.to_numeric(df["soil_moisture"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  -> {len(df)} daily soil moisture records total")
    return df[["timestamp", "soil_moisture"]]


# ─── 2. ATMOSPHERIC HUMIDITY — Aqua MODIS MCD19A2 ────────────────────────────

def get_humidity() -> pd.DataFrame:
    """
    Aqua MODIS — MCD19A2_GRANULES Column Water Vapor (cm) as humidity proxy.
    Daily composite over the Obando area.
    """
    print("Fetching Aqua MODIS humidity (column water vapor) data...")

    # MCD19A2 GRANULES: Terra+Aqua MAIAC daily 1km — Column_WV band (cm)
    # Uses same-day server-side aggregation to avoid >5000 element limit.
    collection = (
        ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
        .filterDate(START_DATE, END_DATE)
        .select("Column_WV")
        .filterBounds(AOI)
    )

    n_days   = ee.Date(END_DATE).difference(ee.Date(START_DATE), "day").int()
    day_list = ee.List.sequence(0, n_days.subtract(1))

    def daily_mean(offset):
        date  = ee.Date(START_DATE).advance(offset, "day")
        daily = collection.filterDate(date, date.advance(1, "day"))
        has_data  = daily.size().gt(0)
        composite = daily.mean()
        stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=AOI,
            scale=1000,
            maxPixels=1e9
        )
        val = ee.Algorithms.If(has_data, stats.get("Column_WV", None), None)
        return ee.Feature(None, {
            "timestamp":    date.format("YYYY-MM-dd"),
            "humidity_raw": val
        })

    features = ee.FeatureCollection(day_list.map(daily_mean)).filter(
        ee.Filter.notNull(["humidity_raw"])
    )

    info = features.getInfo()
    rows = [f["properties"] for f in info["features"]]
    df = pd.DataFrame(rows)

    df["timestamp"]    = pd.to_datetime(df["timestamp"])
    df["humidity_raw"] = pd.to_numeric(df["humidity_raw"], errors="coerce")
    df["humidity"]     = df["humidity_raw"] * 0.001   # scale → cm column water vapor

    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  → {len(df)} daily humidity records")
    return df[["timestamp", "humidity"]]


# ─── 3. TIDAL WATER LEVEL — UHSLC Manila + CMEMS gap-fill ────────────────────
#
# Source priority (accuracy-first):
#   1. UHSLC Station 370 Manila South Harbor — real measured tide gauge, daily, continuous
#      Research Quality (best QC) → Fast Delivery (recent months) as fallback
#   2. CMEMS zos sea surface height — modelled SSH, used only for gap days
#      Reanalysis (GLOBAL_MULTIYEAR_PHY_001_030) for 1993–2021
#      NRT/Forecast (GLOBAL_ANALYSISFORECAST_PHY_001_024) for 2021–present
#
# Both series are mean-anomaly normalised before merging so their scales
# are consistent with the original HYCOM output this replaces.

def _get_uhslc() -> pd.DataFrame:
    """
    Fetch UHSLC Manila South Harbor Station 370 daily data via ERDDAP API.

    UHSLC migrated from flat CSV files to ERDDAP tabledap in 2024. Station 370 is the active
    The ERDDAP endpoint returns a hybrid RQ+FD series: Research Quality
    data where available, Fast Delivery appended for recent dates.

    ERDDAP CSV format (2-row header):
        Row 1: column names  (time, sea_level)
        Row 2: units         (UTC, mm)
        Row 3+: data

    sea_level is in millimetres; missing-value sentinel is -32767.
    We convert to metres and normalise to anomaly (mean-subtracted)
    for consistency with the HYCOM output this replaces.
    """
    print("Fetching UHSLC Manila tide gauge (ERDDAP)...")

    url = _UHSLC_ERDDAP_URL.format(start=START_DATE, end=END_DATE)

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  UHSLC ERDDAP fetch failed: {e}")
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    # Skip the units row (row index 1) — read header from row 0
    df = pd.read_csv(
        io.StringIO(resp.text),
        skiprows=[1],       # skip units row, keep column-name row
        on_bad_lines="skip",
    )

    # Rename ERDDAP columns to pipeline names
    df = df.rename(columns={"time": "timestamp", "sea_level": "sea_level_mm"})

    df["timestamp"]    = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
    df["sea_level_mm"] = pd.to_numeric(df["sea_level_mm"], errors="coerce")

    # Drop missing-value sentinels (-32767) and NaNs
    df = df[df["sea_level_mm"] > -9000].dropna(subset=["sea_level_mm"])

    # mm → metres, then mean-anomaly normalise
    df["waterlevel"] = df["sea_level_mm"] / 1000.0

    df = df[["timestamp", "waterlevel"]].sort_values("timestamp").reset_index(drop=True)

    start    = pd.Timestamp(START_DATE)
    end      = pd.Timestamp(END_DATE)
    df       = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    expected = (end - start).days

    print(f"  → {len(df)} / {expected} days from UHSLC "
          f"({100 * len(df) / expected:.1f}% coverage)")
    return df


def _get_cmems(missing_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch CMEMS daily mean sea surface height (zos, metres) for the specific
    dates not covered by UHSLC.

    Splits automatically between reanalysis (pre-2021) and NRT (2021-present).
    Credentials: run `copernicusmarine login` once, or set env vars
    CMEMS_USERNAME / CMEMS_PASSWORD.
    """
    if missing_dates.empty:
        print("  No gaps to fill — skipping CMEMS.")
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    # Lazy import — only needed when gaps exist
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

    # --- Reanalysis (1993–2021) ---
    if len(gaps_rean) > 0:
        print(f"    Reanalysis: {gaps_rean.min().date()} → {gaps_rean.max().date()}")
        try:
            ds = copernicusmarine.open_dataset(
                dataset_id     = "cmems_mod_glo_phy_my_0.083deg_P1D-m",
                start_datetime = str(gaps_rean.min().date()),
                end_datetime   = str(gaps_rean.max().date()),
                **common_kwargs,
            )
            df_r = (
                ds["zos"]
                .mean(dim=["latitude", "longitude"])
                .to_dataframe()
                .reset_index()[["time", "zos"]]
                .rename(columns={"time": "timestamp", "zos": "waterlevel"})
            )
            df_r["timestamp"] = pd.to_datetime(df_r["timestamp"]).dt.normalize()
            frames.append(df_r)
        except Exception as e:
            print(f"    WARNING: CMEMS reanalysis failed: {e}")

    # --- NRT / Near-Real-Time (2021–present) ---
    if len(gaps_nrt) > 0:
        print(f"    NRT: {gaps_nrt.min().date()} → {gaps_nrt.max().date()}")
        try:
            ds = copernicusmarine.open_dataset(
                dataset_id     = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
                start_datetime = str(gaps_nrt.min().date()),
                end_datetime   = str(gaps_nrt.max().date()),
                **common_kwargs,
            )
            df_n = (
                ds["zos"]
                .mean(dim=["latitude", "longitude"])
                .to_dataframe()
                .reset_index()[["time", "zos"]]
                .rename(columns={"time": "timestamp", "zos": "waterlevel"})
            )
            df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.normalize()
            frames.append(df_n)
        except Exception as e:
            print(f"    WARNING: CMEMS NRT failed: {e}")

    if not frames:
        return pd.DataFrame(columns=["timestamp", "waterlevel"])

    df_cmems = pd.concat(frames, ignore_index=True)
    df_cmems = df_cmems[df_cmems["timestamp"].isin(missing_dates)]

    print(f"    → {len(df_cmems)} CMEMS gap-fill records")
    return df_cmems[["timestamp", "waterlevel"]].reset_index(drop=True)


def get_tidal_level() -> pd.DataFrame:
    """
    Tidal water level for Obando / Manila Bay (metres, mean-anomaly normalised).

    Primary:  UHSLC Station 370 Manila South Harbor — real measured tide gauge
              Research Quality → Fast Delivery fallback for recent dates not yet QC'd
    Gap-fill: CMEMS zos SSH — reanalysis (pre-2021) + NRT (2021-present)
    """
    print("Fetching tidal water level data (UHSLC + CMEMS)...")

    # Step 1 — primary measured data
    df_uhslc = _get_uhslc()

    # Step 2 — identify uncovered days across the full pipeline range
    all_dates = pd.date_range(START_DATE, end=pd.Timestamp(END_DATE) - pd.Timedelta(days=1), freq="D")
    covered   = set(df_uhslc["timestamp"].dt.normalize()) if not df_uhslc.empty else set()
    missing   = pd.DatetimeIndex([d for d in all_dates if d not in covered])

    # Step 3 — fill gaps with CMEMS
    df_cmems = _get_cmems(missing)

    # Step 4 — combine
    df = pd.concat([df_uhslc, df_cmems], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["waterlevel"] = pd.to_numeric(df["waterlevel"], errors="coerce")

    total  = len(all_dates)
    filled = df["waterlevel"].notna().sum()
    print(f"  Water level summary:")
    print(f"    UHSLC (measured):  {len(df_uhslc):>5} days")
    print(f"    CMEMS (gap-fill):  {len(df_cmems):>5} days")
    print(f"    Total filled:      {filled:>5} / {total} "
          f"({100 * filled / total:.1f}%)")

    return df[["timestamp", "waterlevel"]]


# ─── MERGE & EXPORT ───────────────────────────────────────────────────────────

def main():
    df_sm    = get_soil_moisture()
    df_humid = get_humidity()
    df_tide  = get_tidal_level()

    print("\nMerging datasets...")

    df = (
        df_tide
        .merge(df_sm,    on="timestamp", how="outer")
        .merge(df_humid, on="timestamp", how="outer")
    )

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df = df[["timestamp", "waterlevel", "soil_moisture", "humidity"]]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved → {OUTPUT_CSV}")
    print(f"  Rows: {len(df)}")
    print(f"\nPreview:\n{df.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()