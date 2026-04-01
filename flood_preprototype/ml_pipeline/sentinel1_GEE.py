"""
sentinel1_GEE.py
================
Sentinel-1 SAR + GPM + ERA5-Land Flood Label Pipeline  (Incremental Mode)

INCREMENTAL PULL
----------------
On every run, this script reads the last timestamp in sentinel1_timeseries.csv
and pulls only new Sentinel-1 passes from that date onward. New rows are
APPENDED — the full 2017-2026 history is never re-pulled.

On first run (empty or missing CSV) it falls back to full history from
START_DATE_DEFAULT = '2017-01-01'.

This script is called by Retraining_Pipeline.py, NOT by Start.py's daily loop.
It should be run when new Sentinel-1 passes are expected (~every 12 days),
or manually before retraining.

LABELING STRATEGY — TRIPLE CONDITION (ERA5 runoff + soil water + GPM)
----------------------------------------------------------------------
    flood_label = 1  iff ALL THREE conditions met:
        A: era5_runoff_7d  >= ERA5_RUNOFF_7D_THRESHOLD   (rivers overwhelmed)
        B: era5_soil_water >= ERA5_SOIL_WATER_THRESHOLD   (ground near-saturated)
        C: rainfall_7d     >= GPM_7D_FLOOD_THRESHOLD      (sustained heavy rain)

    ERA5_RUNOFF_7D_THRESHOLD  = 8.0   mm
    ERA5_SOIL_WATER_THRESHOLD = 0.46  m³/m³
    GPM_7D_FLOOD_THRESHOLD    = 75.0  mm

KNOWN OBANDO FLOOD EVENTS (validation anchors)
    2018-08  Typhoon Karding / southwest monsoon
    2019-07  Typhoon Falcon + LPA
    2020-10  Typhoon Quinta
    2021-10  Typhoon Maring + monsoon
    2022-07  Southwest monsoon — Bulacan
    2024-07  Typhoon Carina + southwest monsoon

OUTPUTS
-------
    sentinel1_timeseries.csv  (APPENDED, not overwritten)
        timestamp, orbit, orbit_flag, vv_mean, vh_mean,
        soil_saturation, flood_extent, wetness_trend,
        rainfall_1d, rainfall_3d, rainfall_7d,
        era5_runoff_1d, era5_runoff_7d, era5_soil_water,
        flood_label

Usage
-----
    python sentinel1_GEE.py               # incremental pull (auto-detects last date)
    python sentinel1_GEE.py --full        # force full history re-pull (rare)
"""

import argparse
import ee
import json
import numpy as np
import pandas as pd
import os
import time
from datetime import date, timedelta

# =========================
# CONFIGURATION
# =========================
_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR       = os.path.join(_SCRIPT_DIR, "..", "data", "sentinel1", "GEE-Processing")
OUTPUT_CSV       = os.path.join(OUTPUT_DIR, "sentinel1_timeseries.csv")
AOI_GEOJSON      = os.path.join(_SCRIPT_DIR, "..", "config", "aoi.geojson")
START_DATE_DEFAULT = "2017-01-01"   # used only when CSV is empty/missing

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Labeling thresholds — triple condition
ERA5_RUNOFF_7D_THRESHOLD  = 8.0
ERA5_SOIL_WATER_THRESHOLD = 0.46
GPM_7D_FLOOD_THRESHOLD    = 75.0

ERA5_LAT = 14.71
ERA5_LON  = 120.85

FLOOD_EXTENT_MIN = -5.0
FLOOD_EXTENT_MAX = -0.3


# =========================
# DYNAMIC DATE RANGE
# =========================

def get_date_range(force_full: bool = False) -> tuple[str, str]:
    """
    Determine incremental start/end dates.

    Reads the last timestamp from sentinel1_timeseries.csv.
    Pulls from that date onward (overlap by 1 day to catch late-arriving passes).
    End date = tomorrow (exclusive GEE upper bound).

    Returns (start_date, end_date) as 'YYYY-MM-DD' strings.
    Returns (None, None) if already up to date.
    """
    today    = date.today()
    end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if force_full or not os.path.exists(OUTPUT_CSV):
        mode = "full history (forced)" if force_full else "full history (no CSV found)"
        print(f"  Sentinel-1 pull mode: {mode}")
        print(f"  Range: {START_DATE_DEFAULT} → {today}")
        return START_DATE_DEFAULT, end_date

    try:
        df_existing = pd.read_csv(OUTPUT_CSV, usecols=["timestamp"], parse_dates=["timestamp"])
        if len(df_existing) == 0:
            print(f"  Sentinel-1 pull mode: full history (CSV empty)")
            return START_DATE_DEFAULT, end_date

        last_ts    = df_existing["timestamp"].max()
        last_date  = last_ts.date() if hasattr(last_ts, "date") else pd.Timestamp(last_ts).date()

        # Overlap by 1 day — catches any passes that arrived late from Copernicus
        start_date = (last_date - timedelta(days=1)).strftime("%Y-%m-%d")

        if start_date >= end_date:
            print(f"  Sentinel-1 CSV already up to date (last pass: {last_date}). Nothing to pull.")
            return None, None

        n_days = (today - last_date).days
        print(f"  Sentinel-1 pull mode: incremental")
        print(f"  Last pass in CSV : {last_date}")
        print(f"  Pulling          : {start_date} → {today} (~{n_days} days)")
        return start_date, end_date

    except Exception as e:
        print(f"  WARNING: Could not read existing CSV ({e}). Falling back to full history.")
        return START_DATE_DEFAULT, end_date


def check_new_passes_available(start_date: str) -> bool:
    """
    Quick GEE check: are there any new Sentinel-1 passes since start_date?
    Returns True if passes found, False if nothing new.
    """
    print(f"  Checking for new Sentinel-1 passes since {start_date}...")
    try:
        aoi = load_aoi_ee(AOI_GEOJSON)
        end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        count = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .size()
            .getInfo()
        )
        print(f"  Found {count} new image(s).")
        return count > 0
    except Exception as e:
        print(f"  WARNING: Could not check pass count ({e}). Proceeding anyway.")
        return True


# =========================
# INITIALIZE EARTH ENGINE
# =========================
def initialize_gee():
    try:
        ee.Initialize(project="jenel-466709")
        print("[+] Google Earth Engine initialized")
    except Exception as e:
        print(f"[!] GEE initialization failed: {e}")
        exit()


# =========================
# LOAD AOI
# =========================
def load_aoi_ee(geojson_path):
    if not os.path.exists(geojson_path):
        print(f"[!] AOI file not found: {geojson_path}")
        return None
    try:
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)
        if "features" in geojson_data and len(geojson_data["features"]) > 0:
            geometry = geojson_data["features"][0]["geometry"]
        elif "geometry" in geojson_data:
            geometry = geojson_data["geometry"]
        else:
            geometry = geojson_data
        ee_geometry = ee.Geometry(geometry)
        print(f"[+] Loaded AOI from {geojson_path}")
        return ee_geometry
    except Exception as e:
        print(f"[!] Error loading AOI: {e}")
        return None


# =========================
# LOAD SENTINEL-1 COLLECTION
# =========================
def load_sentinel1_collection(aoi, start_date, end_date, orbit):
    print(f"[+] Loading Sentinel-1 GRD ({orbit}) from {start_date} to {end_date}")
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
    )
    count = collection.size().getInfo()
    print(f"    Found {count} images")
    return collection


# =========================
# PREPROCESSING
# =========================
def preprocess_sentinel1(image):
    vv_filtered = image.select("VV").focal_median(radius=50, kernelType="circle", units="meters")
    vh_filtered = image.select("VH").focal_median(radius=50, kernelType="circle", units="meters")
    return (
        image
        .addBands(vv_filtered.rename("VV_filtered"))
        .addBands(vh_filtered.rename("VH_filtered"))
    )


def calculate_soil_saturation(collection):
    def add_soil_saturation(image):
        vv  = image.select("VV_filtered")
        vh  = image.select("VH_filtered")
        sat = vv.subtract(vh).divide(vv.abs().add(vh.abs())).rename("soil_saturation")
        return image.addBands(sat)
    return collection.map(add_soil_saturation)


def calculate_flood_extent(collection):
    def add_flood_extent(image):
        vv         = image.select("VV_filtered")
        vh         = image.select("VH_filtered")
        cr         = vh.divide(vv).multiply(-1)
        cr_clamped = cr.clamp(FLOOD_EXTENT_MIN, FLOOD_EXTENT_MAX).rename("flood_extent")
        return image.addBands(cr_clamped)
    return collection.map(add_flood_extent)


def calculate_wetness_trend_per_image(collection, aoi):
    image_list = collection.toList(collection.size())
    n          = image_list.size().getInfo()
    trends     = {}

    print(f"[+] Calculating per-image wetness trend ({n} images)...")

    if n > 0:
        first_img  = ee.Image(image_list.get(0))
        first_time = int(first_img.get("system:time_start").getInfo())
        trends[first_time] = 0

    for i in range(1, n):
        current_time_int = None
        try:
            current          = ee.Image(image_list.get(i))
            current_time_int = int(current.get("system:time_start").getInfo())
            current_date     = pd.Timestamp(current_time_int, unit="ms")
            window_start     = (current_date - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            window_end       = current_date.strftime("%Y-%m-%d")
            window_col       = collection.filterDate(window_start, window_end)

            if window_col.size().getInfo() < 2:
                trends[current_time_int] = 0
                continue

            window_list = window_col.toList(window_col.size())
            n_window    = window_list.size().getInfo()

            first_stats = ee.Image(window_list.get(0)).select("VV_filtered").reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e8, bestEffort=True
            ).getInfo()
            last_stats  = ee.Image(window_list.get(n_window - 1)).select("VV_filtered").reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e8, bestEffort=True
            ).getInfo()

            first_vv = first_stats.get("VV_filtered")
            last_vv  = last_stats.get("VV_filtered")

            if first_vv is None or last_vv is None:
                trends[current_time_int] = 0
                continue

            mean_change = float(last_vv) - float(first_vv)
            if   mean_change < -0.5: trends[current_time_int] =  1
            elif mean_change >  0.5: trends[current_time_int] = -1
            else:                    trends[current_time_int] =  0

        except Exception as e:
            print(f"    [!] Trend failed for image {i}: {e}")
            if current_time_int is not None:
                trends[current_time_int] = 0

    print(f"    Computed trends for {len(trends)} images")
    return trends


# =========================
# GPM IMERG RAINFALL
# =========================
def get_gpm_rainfall(aoi, start_date, end_date):
    print(f"\n[+] Fetching GPM IMERG daily rainfall ({start_date} → {end_date})...")

    start      = pd.Timestamp(start_date)
    end        = pd.Timestamp(end_date)
    total_days = (end - start).days
    chunk_size = 90

    rainfall_dict = {}

    for offset in range(0, total_days, chunk_size):
        chunk_start = (start + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        chunk_end   = (start + pd.Timedelta(days=min(offset + chunk_size, total_days))).strftime("%Y-%m-%d")

        def fetch_chunk(cs=chunk_start, ce=chunk_end):
            gpm    = (
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
                    reducer=ee.Reducer.mean(), geometry=aoi, scale=10000, maxPixels=1e8, bestEffort=True
                ).get("precipitation")
                return ee.Feature(None, {"date": day.format("YYYY-MM-dd"), "rainfall_mm": mean_val})

            day_list = ee.List.sequence(0, n_days - 1)
            features = ee.FeatureCollection(day_list.map(daily_sum))
            return features.toList(features.size()).getInfo()

        chunk_data = safe_compute(fetch_chunk, max_retries=3)
        if chunk_data:
            for feat in chunk_data:
                props = feat.get("properties", {})
                d     = props.get("date")
                val   = props.get("rainfall_mm")
                if d and val is not None:
                    rainfall_dict[d] = round(float(val), 3)

        print(f"    Days {offset}–{min(offset + chunk_size, total_days) - 1} done")

    print(f"    Complete — {len(rainfall_dict)} days fetched")
    return rainfall_dict


# =========================
# ERA5-LAND RUNOFF + SOIL WATER
# =========================
def get_era5_daily(start_date, end_date):
    print(f"\n[+] Fetching ERA5-Land runoff + soil water ({start_date} → {end_date})...")

    era5_point = ee.Geometry.Point([ERA5_LON, ERA5_LAT]).buffer(5000)
    start      = pd.Timestamp(start_date)
    end        = pd.Timestamp(end_date)
    total_days = (end - start).days
    chunk_size = 30
    era5_dict  = {}

    for offset in range(0, total_days, chunk_size):
        chunk_dates = [
            (start + pd.Timedelta(days=offset + d)).strftime("%Y-%m-%d")
            for d in range(min(chunk_size, total_days - offset))
        ]

        for date_str in chunk_dates:
            def fetch_one_day(ds=date_str):
                day_start = ee.Date(ds)
                day_end   = day_start.advance(1, "day")
                era5_day  = (
                    ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                    .filterDate(day_start, day_end)
                    .select(["surface_runoff", "sub_surface_runoff", "volumetric_soil_water_layer_1"])
                )
                runoff_img = era5_day.select(["surface_runoff", "sub_surface_runoff"]).mean().multiply(1000)

                sro_val  = runoff_img.select("surface_runoff").reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=era5_point, scale=9000, maxPixels=1e6, bestEffort=True
                ).getInfo().get("surface_runoff")

                ssro_val = runoff_img.select("sub_surface_runoff").reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=era5_point, scale=9000, maxPixels=1e6, bestEffort=True
                ).getInfo().get("sub_surface_runoff")

                sw_val = era5_day.select("volumetric_soil_water_layer_1").mean().reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=era5_point, scale=9000, maxPixels=1e6, bestEffort=True
                ).getInfo().get("volumetric_soil_water_layer_1")

                return sro_val, ssro_val, sw_val

            result = safe_compute(fetch_one_day, max_retries=3)
            if result is not None:
                sro, ssro, sw = result
                runoff_mm = 0.0
                if sro  is not None: runoff_mm += float(sro)
                if ssro is not None: runoff_mm += float(ssro)
                era5_dict[date_str] = {
                    "runoff_mm":  round(runoff_mm, 4),
                    "soil_water": round(float(sw), 5) if sw is not None else None,
                }
            else:
                era5_dict[date_str] = {"runoff_mm": 0.0, "soil_water": None}

        print(f"    Days {offset}–{min(offset + chunk_size, total_days) - 1} done")

    print(f"    Complete — {total_days} days fetched")
    return era5_dict


# =========================
# ATTACH SIGNALS
# =========================
def attach_signals_to_results(results, rainfall_dict, era5_dict):
    for row in results:
        pass_date = pd.Timestamp(row["timestamp"]).normalize()
        keys = [(pass_date - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

        gpm_daily          = [rainfall_dict.get(k, 0.0) or 0.0 for k in keys]
        row["rainfall_1d"] = round(gpm_daily[0], 3)
        row["rainfall_3d"] = round(sum(gpm_daily[:3]), 3)
        row["rainfall_7d"] = round(sum(gpm_daily), 3)

        era5_runoff           = [(era5_dict.get(k) or {}).get("runoff_mm", 0.0) or 0.0 for k in keys]
        row["era5_runoff_1d"] = round(era5_runoff[0], 4)
        row["era5_runoff_7d"] = round(sum(era5_runoff), 4)

        soil_vals              = [(era5_dict.get(k) or {}).get("soil_water") for k in keys]
        soil_vals              = [v for v in soil_vals if v is not None]
        row["era5_soil_water"] = round(np.mean(soil_vals), 5) if soil_vals else None

    return results


# =========================
# FLOOD LABEL
# =========================
def derive_flood_label(era5_runoff_7d, era5_soil_water, rainfall_7d):
    cond_a = (era5_runoff_7d  is not None) and (era5_runoff_7d  >= ERA5_RUNOFF_7D_THRESHOLD)
    cond_b = (era5_soil_water is not None) and (era5_soil_water >= ERA5_SOIL_WATER_THRESHOLD)
    cond_c = (rainfall_7d     is not None) and (rainfall_7d     >= GPM_7D_FLOOD_THRESHOLD)
    label  = 1 if (cond_a and cond_b and cond_c) else 0
    return label


# =========================
# PROCESS ONE ORBIT
# =========================
def process_orbit(aoi, start_date, end_date, orbit):
    print(f"\n{'='*60}")
    print(f"  Processing orbit: {orbit}")
    print(f"{'='*60}")

    collection = load_sentinel1_collection(aoi, start_date, end_date, orbit)
    if collection.size().getInfo() == 0:
        print(f"  [!] No images for {orbit} — skipping")
        return []

    collection = collection.map(preprocess_sentinel1)
    collection = calculate_soil_saturation(collection)
    collection = calculate_flood_extent(collection)

    def add_stats(image):
        stats = image.select(["VV_filtered", "VH_filtered", "soil_saturation", "flood_extent"]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e8, bestEffort=True,
        )
        return image.set({
            "vv_mean":        stats.get("VV_filtered"),
            "vh_mean":        stats.get("VH_filtered"),
            "soil_sat_mean":  stats.get("soil_saturation"),
            "flood_ext_mean": stats.get("flood_extent"),
        })

    collection_with_stats = collection.map(add_stats)

    def get_features():
        features = collection_with_stats.map(lambda img: ee.Feature(None, {
            "timestamp":    ee.Date(img.date().millis()).format("YYYY-MM-dd'T'HH:mm:ss'Z'"),
            "timestamp_ms": img.get("system:time_start"),
            "orbit":        img.get("orbitProperties_pass"),
            "vv_mean":      img.get("vv_mean"),
            "vh_mean":      img.get("vh_mean"),
            "soil_saturation": img.get("soil_sat_mean"),
            "flood_extent": img.get("flood_ext_mean"),
        }))
        return features.toList(collection.size()).getInfo()

    feature_list = safe_compute(get_features, max_retries=3)
    if feature_list is None:
        print(f"  [!] Failed to retrieve features for {orbit}")
        return []

    trend_map  = calculate_wetness_trend_per_image(collection, aoi)
    orbit_flag = 0 if orbit == "ASCENDING" else 1
    results    = []
    n_skipped  = 0
    n_outliers = 0

    for feature in feature_list:
        props     = feature["properties"]
        ts_ms     = int(props.get("timestamp_ms", 0))
        vv_mean   = props.get("vv_mean",         0) or 0
        vh_mean   = props.get("vh_mean",         0) or 0
        soil_sat  = props.get("soil_saturation", 0) or 0
        flood_ext = props.get("flood_extent",    0) or 0

        if vv_mean == 0 and vh_mean == 0:
            n_skipped += 1
            continue

        flood_ext_f = float(flood_ext)
        if not (FLOOD_EXTENT_MIN <= flood_ext_f <= FLOOD_EXTENT_MAX):
            n_outliers += 1
            flood_ext_f = None

        results.append({
            "timestamp":       props["timestamp"],
            "orbit":           props.get("orbit", orbit),
            "orbit_flag":      orbit_flag,
            "vv_mean":         round(float(vv_mean),  4),
            "vh_mean":         round(float(vh_mean),  4),
            "soil_saturation": round(float(soil_sat), 4),
            "flood_extent":    round(flood_ext_f, 4) if flood_ext_f is not None else None,
            "wetness_trend":   trend_map.get(ts_ms, 0),
            "rainfall_1d":     0.0,
            "rainfall_3d":     0.0,
            "rainfall_7d":     0.0,
            "era5_runoff_1d":  0.0,
            "era5_runoff_7d":  0.0,
            "era5_soil_water": None,
            "flood_label":     None,
        })

    print(f"  [+] {orbit} — {len(results)} valid passes ({n_skipped} skipped, {n_outliers} outliers)")
    return results


# =========================
# SAFE COMPUTATION WRAPPER
# =========================
def safe_compute(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"[!] Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[!] Max retries reached: {e}")
                return None


# =========================
# APPEND NEW ROWS
# =========================
def append_to_csv(new_df: pd.DataFrame) -> None:
    """
    Append new_df to the existing sentinel1_timeseries.csv.
    Deduplicates on (timestamp, orbit) — overlap day rows are replaced by new pull.
    """
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV, parse_dates=["timestamp"])
        combined = pd.concat([existing, new_df], ignore_index=True)
        del existing
    else:
        combined = new_df.copy()

    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.drop_duplicates(subset=["timestamp", "orbit"], keep="last")
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[+] Appended → {OUTPUT_CSV}")
    print(f"    Total rows now : {len(combined)}")
    print(f"    Flood labels   : {int(combined['flood_label'].sum())} "
          f"({100 * combined['flood_label'].mean():.1f}%)")


# =========================
# VALIDATION
# =========================
def validate_known_events(df):
    known_events = [
        ("2018-08", "Typhoon Karding / SW monsoon"),
        ("2019-07", "Typhoon Falcon + LPA"),
        ("2020-10", "Typhoon Quinta"),
        ("2021-10", "Typhoon Maring + monsoon"),
        ("2022-07", "SW monsoon — Bulacan"),
        ("2024-07", "Typhoon Carina + SW monsoon"),
    ]
    event_months = {e[0] for e in known_events}
    df["ym"]     = df["timestamp"].astype(str).str[:7]

    print("\n" + "=" * 100)
    print("  KNOWN EVENT VALIDATION")
    print("=" * 100)

    all_correct = True
    for ym, name in known_events:
        subset = df[df["ym"] == ym]
        if len(subset) == 0:
            print(f"  {ym}  {name:<32}  NO DATA")
            continue
        n_flood = int((subset["flood_label"] == 1).sum())
        ok      = "✅" if n_flood > 0 else "❌"
        if n_flood == 0:
            all_correct = False
        print(f"  {ym}  {name:<32}  passes={len(subset)}  flood={n_flood}  {ok}")

    non_event_df = df[~df["ym"].isin(event_months)]
    fp_rate      = non_event_df["flood_label"].mean() if len(non_event_df) > 0 else 0.0
    print(f"\n  False positive rate (non-event months): {100*fp_rate:.1f}%")
    if all_correct:
        print("  ✅  All known flood events correctly labeled.")
    else:
        print("  ❌  Some events missed — check thresholds.")
    print("=" * 100)
    df.drop(columns=["ym"], inplace=True, errors="ignore")


# =========================
# MAIN
# =========================
def main(force_full: bool = False):
    print("=" * 80)
    print("SENTINEL-1 + GPM + ERA5 — INCREMENTAL FLOOD LABEL PIPELINE")
    print("=" * 80)
    print(f"\n  Condition A : ERA5 runoff 7d >= {ERA5_RUNOFF_7D_THRESHOLD} mm")
    print(f"  Condition B : ERA5 soil water (0–7cm) >= {ERA5_SOIL_WATER_THRESHOLD} m³/m³")
    print(f"  Condition C : GPM 7d rainfall >= {GPM_7D_FLOOD_THRESHOLD} mm\n")

    initialize_gee()

    # Determine date range
    start_date, end_date = get_date_range(force_full=force_full)
    if start_date is None:
        print("\n  Nothing to do — sentinel1_timeseries.csv is already up to date.")
        return

    print("\n[+] Loading Area of Interest...")
    aoi = load_aoi_ee(AOI_GEOJSON)
    if not aoi:
        print("[!] Failed to load AOI")
        exit()

    # Quick pass check before running full pipeline
    if not force_full and not check_new_passes_available(start_date):
        print("\n  No new Sentinel-1 passes available yet. Try again in a few days.")
        print("  (Sentinel-1 revisit period over Obando is ~12 days)")
        return

    # Step 1 — GPM (need 7-day lookback before start_date for rolling windows)
    gpm_start = (pd.Timestamp(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    rainfall_dict = get_gpm_rainfall(aoi, gpm_start, end_date)

    # Step 2 — ERA5
    era5_dict = get_era5_daily(gpm_start, end_date)

    # Step 3 — Sentinel-1 passes
    all_results = []
    for orbit in ["ASCENDING", "DESCENDING"]:
        rows = process_orbit(aoi, start_date, end_date, orbit)
        all_results.extend(rows)

    if not all_results:
        print("[!] No new SAR passes found in this date range.")
        return

    # Step 4 — Attach signals
    print("\n[+] Attaching GPM and ERA5 signals to SAR passes...")
    all_results = attach_signals_to_results(all_results, rainfall_dict, era5_dict)

    # Step 5 — Build DataFrame
    df_new = pd.DataFrame(all_results)
    df_new = df_new.dropna(how="all")
    df_new = df_new.sort_values("timestamp").reset_index(drop=True)
    df_new = df_new.drop_duplicates(subset=["timestamp", "orbit"]).reset_index(drop=True)

    # Step 6 — Labels
    print("\n[+] Assigning flood labels (triple condition)...")
    df_new["flood_label"] = df_new.apply(
        lambda row: derive_flood_label(
            row["era5_runoff_7d"],
            row["era5_soil_water"],
            row["rainfall_7d"],
        ),
        axis=1,
    )

    # Step 7 — Summary
    print(f"\n[+] New passes to append : {len(df_new)}")
    vc = df_new["flood_label"].value_counts().sort_index()
    for label, count in vc.items():
        print(f"    label={label} : {count} ({100*count/len(df_new):.1f}%)")

    # Step 8 — Append
    append_to_csv(df_new)

    # Step 9 — Validate on full CSV
    full_df = pd.read_csv(OUTPUT_CSV, parse_dates=["timestamp"])
    validate_known_events(full_df.copy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel-1 incremental flood label pipeline")
    parser.add_argument("--full", action="store_true", help="Force full history re-pull (rare)")
    args = parser.parse_args()
    main(force_full=args.full)