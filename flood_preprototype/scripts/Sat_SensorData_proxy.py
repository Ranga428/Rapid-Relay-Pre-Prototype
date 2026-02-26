"""
GEE Data Extractor — Obando Environmental Data
Extracts: Rainfall (GPM), Atmospheric Humidity (Aqua/MODIS), Tidal Level (Obando)
Output: CSV with columns: timestamp, waterlevel, rainfall, humidity

Requirements:
    pip install earthengine-api geemap pandas
    earthengine authenticate
"""

import ee
import json
import pandas as pd
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Paths — relative to this script's location
# Project structure:
#   flood_preprototype/
#     config/aoi.geojson
#     data/sensor/           ← CSV output
#     scripts/               ← this file
_SCRIPT_DIR = Path(__file__).resolve().parent
_AOI_PATH   = _SCRIPT_DIR.parent / "config" / "aoi.geojson"

# Date range — adjust as needed
START_DATE = "2017-01-01"
END_DATE   = "2026-02-26"

# Output CSV — flood_preprototype/data/sensor/
OUTPUT_CSV = _SCRIPT_DIR.parent / "data" / "sensor" / "obando_environmental_data.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)  # create folders if needed

# ─── INIT ─────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Set your Google Cloud project ID (must have Earth Engine API enabled)
# Find it: https://console.cloud.google.com → select project → copy the ID
GEE_PROJECT = "jenel-466709"   # ← REPLACE THIS
# ──────────────────────────────────────────────────────────────────────────────

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

# Handle FeatureCollection, Feature, or bare Geometry
if _geojson["type"] == "FeatureCollection":
    _geometry = _geojson["features"][0]["geometry"]
elif _geojson["type"] == "Feature":
    _geometry = _geojson["geometry"]
else:
    _geometry = _geojson

AOI = ee.Geometry(_geometry)
print(f"✓ AOI loaded from: {_AOI_PATH}")


# ─── 1. RAINFALL — GPM IMERG (mm/hr) ─────────────────────────────────────────

def get_rainfall() -> pd.DataFrame:
    """
    GPM IMERG V07 — aggregated to daily total (mm/day) on the GEE side.
    Uses a date sequence to avoid the 5000-element getInfo() limit.
    """
    print("Fetching GPM rainfall data...")

    collection = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
        .filterDate(START_DATE, END_DATE)
        .select("precipitation")
        .filterBounds(AOI)
    )

    # Build a list of days and sum 30-min images per day server-side
    n_days = ee.Date(END_DATE).difference(ee.Date(START_DATE), "day").int()
    day_list = ee.List.sequence(0, n_days.subtract(1))

    def daily_sum(offset):
        date  = ee.Date(START_DATE).advance(offset, "day")
        daily = collection.filterDate(date, date.advance(1, "day"))
        # sum of (mm/hr * 0.5 hr) = mm per 30-min interval, summed = daily mm
        total = daily.map(lambda img: img.multiply(0.5)).sum()
        val   = total.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=AOI,
            scale=11132,
            maxPixels=1e9
        ).get("precipitation")
        return ee.Feature(None, {
            "timestamp": date.format("YYYY-MM-dd"),
            "rainfall": val
        })

    features = ee.FeatureCollection(day_list.map(daily_sum)).filter(
        ee.Filter.notNull(["rainfall"])
    )

    info = features.getInfo()
    rows = [f["properties"] for f in info["features"]]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["rainfall"]  = pd.to_numeric(df["rainfall"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  → {len(df)} daily rainfall records")
    return df[["timestamp", "rainfall"]]


# ─── 2. ATMOSPHERIC HUMIDITY — Aqua MODIS MOD07 ───────────────────────────────

def get_humidity() -> pd.DataFrame:
    """
    Aqua MODIS — MOD07_L2 Total Precipitable Water (cm) as humidity proxy.
    Daily composite over the Obando area.
    """
    print("Fetching Aqua MODIS humidity (column water vapor) data...")

    # MCD19A2 GRANULES: Terra+Aqua MAIAC daily 1km — Column_WV band (cm)
    # This is the confirmed GEE-hosted daily atmospheric water vapor product.
    # Uses same-day server-side aggregation to avoid >5000 element limit.
    collection = (
        ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
        .filterDate(START_DATE, END_DATE)
        .select("Column_WV")
        .filterBounds(AOI)
    )

    n_days = ee.Date(END_DATE).difference(ee.Date(START_DATE), "day").int()
    day_list = ee.List.sequence(0, n_days.subtract(1))

    def daily_mean(offset):
        date  = ee.Date(START_DATE).advance(offset, "day")
        daily = collection.filterDate(date, date.advance(1, "day"))
        composite = daily.mean()

        # Compute region stats for all bands present in composite; returns a dictionary
        stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=AOI,
            scale=1000,
            maxPixels=1e9
        )
        stats_dict = ee.Dictionary(stats)

        # Try a set of candidate band names (choose first available)
        # Adjust this list after inspecting bandNames if needed
        def _first_available():
            # Nested ee.Algorithms.If chain to pick first present key
            return ee.Algorithms.If(
                stats_dict.contains('Column_WV'), stats_dict.get('Column_WV'),
                ee.Algorithms.If(
                    stats_dict.contains('Column_WV_cm'), stats_dict.get('Column_WV_cm'),
                    ee.Algorithms.If(
                        stats_dict.contains('Total_Column_Water_Vapor'), stats_dict.get('Total_Column_Water_Vapor'),
                        ee.Algorithms.If(
                            stats_dict.contains('ColumnWV'), stats_dict.get('ColumnWV'),
                            None
                        )
                    )
                )
            )

        val = _first_available()

        return ee.Feature(None, {
            "timestamp": date.format("YYYY-MM-dd"),
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

    # Scale factor: 0.001 → result in cm of column water vapor
    df["humidity"] = df["humidity_raw"] * 0.001

    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  → {len(df)} daily humidity records")
    return df[["timestamp", "humidity"]]


# ─── 3. TIDAL WATER LEVEL — Obando (UHSLC / Derived) ─────────────────────────

def get_tidal_level() -> pd.DataFrame:
    """
    GEE does not host direct tide gauge data.
    We use ECMWF ERA5 Sea Surface Height (SSH) as a tidal-level proxy
    for the coastal Obando area (Manila Bay adjacent).

    Dataset: ECMWF/ERA5_LAND/DAILY_AGGR — surface_pressure as proxy,
    or alternatively use Copernicus Global Ocean tide model via
    HYCOM sea surface height.

    Here we use HYCOM: sea_surface_elevation (m).
    """
    print("Fetching tidal/sea surface elevation data (HYCOM)...")

    # HYCOM sea surface elevation band
    try:
        collection_ssh = (
            ee.ImageCollection("HYCOM/sea_surface_elevation")
            .filterDate(START_DATE, END_DATE)
            .filterBounds(AOI)
        )

        def extract_ssh(image):
            bands = image.bandNames().getInfo()
            band = bands[0] if bands else "surf_el"
            val = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=AOI,
                scale=8905,
                maxPixels=1e9
            ).get(band)
            return ee.Feature(None, {
                "timestamp": image.date().format("YYYY-MM-dd"),
                "water_level": val
            })

        features = collection_ssh.map(extract_ssh).filter(
            ee.Filter.notNull(["water_level"])
        )

        info = features.getInfo()
        rows = [f["properties"] for f in info["features"]]
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")

        daily = (
            df.groupby("timestamp")["water_level"]
            .mean()
            .reset_index()
        )
        print(f"  → {len(daily)} daily tidal records")
        return daily[["timestamp", "water_level"]]

    except Exception as e:
        print(f"  ⚠ HYCOM SSH failed ({e}), falling back to ERA5 surface pressure proxy")
        return _get_era5_pressure_proxy()


def _get_era5_pressure_proxy() -> pd.DataFrame:
    """Fallback: ERA5 surface_pressure normalized as water-level proxy."""
    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(START_DATE, END_DATE)
        .select("surface_pressure")
        .filterBounds(AOI)
    )

    def extract(image):
        val = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=AOI,
            scale=11132,
            maxPixels=1e9
        ).get("surface_pressure")
        return ee.Feature(None, {
            "timestamp": image.date().format("YYYY-MM-dd"),
            "pressure_pa": val
        })

    features = collection.map(extract).filter(ee.Filter.notNull(["pressure_pa"]))
    info = features.getInfo()
    rows = [f["properties"] for f in info["features"]]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["pressure_pa"] = pd.to_numeric(df["pressure_pa"], errors="coerce")
    # Rough inverse barometer: ~1 hPa ≈ 1 cm water level change
    p_mean = df["pressure_pa"].mean()
    df["water_level"] = (df["pressure_pa"] - p_mean) / -100.0  # cm deviation
    return df[["timestamp", "water_level"]]


# ─── MERGE & EXPORT ───────────────────────────────────────────────────────────

def main():
    df_rain  = get_rainfall()
    df_humid = get_humidity()
    df_tide  = get_tidal_level()

    print("\nMerging datasets...")

    # Merge on daily timestamp
    df = (
        df_tide
        .merge(df_rain,  on="timestamp", how="outer")
        .merge(df_humid, on="timestamp", how="outer")
    )

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Reorder columns
    df = df[["timestamp", "water_level", "rainfall", "humidity"]]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved → {OUTPUT_CSV}")
    print(f"  Rows: {len(df)}")
    print(f"\nPreview:\n{df.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()