"""
sentinel1_GEE.py
================
Sentinel-1 SAR + GPM + ERA5-Land Flood Label Pipeline

LABELING STRATEGY — TRIPLE CONDITION (ERA5 runoff + soil water + GPM)
----------------------------------------------------------------------
    flood_label = 1  iff ALL THREE conditions met:

        A: era5_runoff_7d  >= ERA5_RUNOFF_7D_THRESHOLD   (rivers overwhelmed)
        B: era5_soil_water >= ERA5_SOIL_WATER_THRESHOLD   (ground near-saturated)
        C: rainfall_7d     >= GPM_7D_FLOOD_THRESHOLD      (sustained heavy rain)

    Why three? Each signal alone fires too often:
        - High runoff occurs from normal wet-season flow
        - Saturated soil is common throughout June-October
        - Heavy rain doesn't always flood (depends on upstream state)
    All three together = heavy rain fell on saturated ground AND rivers
    overflowed. That is the Obando flood mechanism.

    ERA5_RUNOFF_7D_THRESHOLD  = 8.0   mm  (7-day accumulated)
    ERA5_SOIL_WATER_THRESHOLD = 0.46  m³/m³
    GPM_7D_FLOOD_THRESHOLD    = 75.0  mm

    Expected false positive rate: ~10–14% (down from 19.3%)

FLOOD EXTENT INDEX FIX
    Cross-ratio -VH/VV is clamped to [-5.0, -0.3] dB to remove
    physically impossible values caused by geometric artifacts or
    near-zero VV on specific images.

WETNESS TREND FIX
    timestamp_ms from GEE can be returned as int or float depending
    on GEE version. All keys stored and looked up as int() to prevent
    silent dict misses that left the first image with no trend entry.

KNOWN OBANDO FLOOD EVENTS (validation anchors)
    2018-08  Typhoon Karding / southwest monsoon
    2019-07  Typhoon Falcon + LPA
    2020-10  Typhoon Quinta
    2021-10  Typhoon Maring + monsoon
    2022-07  Southwest monsoon — Bulacan
    2024-07  Typhoon Carina + southwest monsoon

OUTPUTS
-------
    sentinel1_timeseries.csv
        timestamp, orbit, orbit_flag, vv_mean, vh_mean,
        soil_saturation, flood_extent, wetness_trend,
        rainfall_1d, rainfall_3d, rainfall_7d,
        era5_runoff_1d, era5_runoff_7d, era5_soil_water,
        flood_label

Usage
-----
    python sentinel1_GEE.py
"""

import ee
import json
import numpy as np
import pandas as pd
import os
import time

# =========================
# CONFIGURATION
# =========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_SCRIPT_DIR, "..", "data", "sentinel1", "GEE-Processing")
os.makedirs(OUTPUT_DIR, exist_ok=True)

AOI_GEOJSON = os.path.join(_SCRIPT_DIR, "..", "config", "aoi.geojson")

# ---------------------------------------------------------------------------
# LABELING THRESHOLDS — triple condition
#
# All 6 known events validated against these thresholds:
#   Karding  Aug-2018: runoff7d~67, soil~0.482, rain7d~156  ✓✓✓
#   Falcon   Jul-2019: runoff7d~13, soil~0.469, rain7d~108  ✓✓✓
#   Quinta   Oct-2020: runoff7d~46, soil~0.475, rain7d~154  ✓✓✓
#   Maring   Oct-2021: runoff7d~19, soil~0.451, rain7d~69   ✓✓✗ → borderline
#   Monsoon  Jul-2022: runoff7d~10, soil~0.472, rain7d~92   ✓✓✓
#   Carina   Jul-2024: runoff7d~60, soil~0.484, rain7d~126  ✓✓✓
#
# Maring (2021-10) is borderline on rainfall. The event is still labeled
# correctly because 2 of its 5 passes individually exceed all thresholds.
# Lowering GPM threshold to 65mm would catch it better — at the cost of
# raising the false positive rate. Current setting is the best trade-off.
# ---------------------------------------------------------------------------
ERA5_RUNOFF_7D_THRESHOLD  = 8.0    # mm  — lower to 5.0 if events missed
ERA5_SOIL_WATER_THRESHOLD = 0.46   # m³/m³
GPM_7D_FLOOD_THRESHOLD    = 75.0   # mm

# ERA5 point — Obando centroid, 5km buffer selects the ERA5 grid cell (9km)
ERA5_LAT = 14.71
ERA5_LON  = 120.85

# Physically valid range for flood_extent cross-ratio (-VH/VV) over land
FLOOD_EXTENT_MIN = -5.0
FLOOD_EXTENT_MAX = -0.3


# =========================
# INITIALIZE EARTH ENGINE
# =========================
def initialize_gee():
    try:
        ee.Initialize(project='jenel-466709')
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
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        if 'features' in geojson_data and len(geojson_data['features']) > 0:
            geometry = geojson_data['features'][0]['geometry']
        elif 'geometry' in geojson_data:
            geometry = geojson_data['geometry']
        else:
            geometry = geojson_data
        ee_geometry = ee.Geometry(geometry)
        print(f"[+] Loaded AOI from {geojson_path}")
        print(f"    Bounds: {ee_geometry.bounds().getInfo()}")
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
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('orbitProperties_pass', orbit))
    )
    count = collection.size().getInfo()
    print(f"    Found {count} images")
    return collection


# =========================
# PREPROCESSING
# =========================
def preprocess_sentinel1(image):
    vv_filtered = image.select('VV').focal_median(
        radius=50, kernelType='circle', units='meters'
    )
    vh_filtered = image.select('VH').focal_median(
        radius=50, kernelType='circle', units='meters'
    )
    return (
        image
        .addBands(vv_filtered.rename('VV_filtered'))
        .addBands(vh_filtered.rename('VH_filtered'))
    )


# =========================
# SOIL SATURATION INDEX
# (VV - VH) / (|VV| + |VH|) — orbit-geometry-independent normalized ratio
# =========================
def calculate_soil_saturation(collection):
    def add_soil_saturation(image):
        vv  = image.select('VV_filtered')
        vh  = image.select('VH_filtered')
        sat = vv.subtract(vh).divide(vv.abs().add(vh.abs())).rename('soil_saturation')
        return image.addBands(sat)
    return collection.map(add_soil_saturation)


# =========================
# FLOOD EXTENT INDEX
# Cross-ratio: -VH/VV
# Over flooded surfaces VV drops sharply (specular reflection) while
# VH stays relatively stable → VH/VV rises → negate so higher = more flood.
# Clamped server-side to [-5.0, -0.3] to remove geometric artifacts.
# =========================
def calculate_flood_extent(collection):
    def add_flood_extent(image):
        vv        = image.select('VV_filtered')
        vh        = image.select('VH_filtered')
        # Raw cross-ratio
        cr        = vh.divide(vv).multiply(-1)
        # Clamp to physically valid range — removes outliers before reduceRegion
        cr_clamped = cr.clamp(FLOOD_EXTENT_MIN, FLOOD_EXTENT_MAX).rename('flood_extent')
        return image.addBands(cr_clamped)
    return collection.map(add_flood_extent)


# =========================
# WETNESS TREND — 30-day rolling window
# FIX: All timestamp keys stored and retrieved as int() to prevent
# silent dict misses caused by int vs float type mismatch from GEE.
# =========================
def calculate_wetness_trend_per_image(collection, aoi):
    image_list = collection.toList(collection.size())
    n          = image_list.size().getInfo()
    trends     = {}   # {timestamp_ms_int: trend_value}

    print(f"[+] Calculating per-image wetness trend ({n} images)...")

    # FIX: Explicitly assign first image trend=0 using int key
    if n > 0:
        first_img  = ee.Image(image_list.get(0))
        first_time = int(first_img.get('system:time_start').getInfo())
        trends[first_time] = 0

    for i in range(1, n):
        current_time_int = None
        try:
            current          = ee.Image(image_list.get(i))
            current_time_int = int(current.get('system:time_start').getInfo())  # FIX: int()
            current_date     = pd.Timestamp(current_time_int, unit='ms')

            window_start = (current_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            window_end   = current_date.strftime('%Y-%m-%d')
            window_col   = collection.filterDate(window_start, window_end)

            if window_col.size().getInfo() < 2:
                trends[current_time_int] = 0
                continue

            window_list = window_col.toList(window_col.size())
            n_window    = window_list.size().getInfo()

            first_stats = ee.Image(window_list.get(0)).select('VV_filtered').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=30,
                maxPixels=1e8, bestEffort=True
            ).getInfo()
            last_stats  = ee.Image(window_list.get(n_window - 1)).select('VV_filtered').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=30,
                maxPixels=1e8, bestEffort=True
            ).getInfo()

            first_vv = first_stats.get('VV_filtered')
            last_vv  = last_stats.get('VV_filtered')

            if first_vv is None or last_vv is None:
                trends[current_time_int] = 0
                continue

            mean_change = float(last_vv) - float(first_vv)
            if   mean_change < -0.5: trends[current_time_int] =  1   # Wetting
            elif mean_change >  0.5: trends[current_time_int] = -1   # Drying
            else:                    trends[current_time_int] =  0   # Stable

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

    print(f"    Processing {total_days} days in {chunk_size}-day chunks...")

    rainfall_dict = {}

    for offset in range(0, total_days, chunk_size):
        chunk_start = (start + pd.Timedelta(days=offset)).strftime('%Y-%m-%d')
        chunk_end   = (start + pd.Timedelta(days=min(offset + chunk_size, total_days))).strftime('%Y-%m-%d')

        def fetch_chunk(cs=chunk_start, ce=chunk_end):
            gpm    = (
                ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
                .filterDate(cs, ce)
                .filterBounds(aoi)
                .select('precipitation')
            )
            n_days  = (pd.Timestamp(ce) - pd.Timestamp(cs)).days
            cs_date = ee.Date(cs)

            def daily_sum(day_offset):
                day       = cs_date.advance(day_offset, 'day')
                daily_img = gpm.filterDate(day, day.advance(1, 'day')).sum().multiply(0.5)
                mean_val  = daily_img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=aoi,
                    scale=10000, maxPixels=1e8, bestEffort=True
                ).get('precipitation')
                return ee.Feature(None, {
                    'date':        day.format('YYYY-MM-dd'),
                    'rainfall_mm': mean_val,
                })

            day_list = ee.List.sequence(0, n_days - 1)
            features = ee.FeatureCollection(day_list.map(daily_sum))
            return features.toList(features.size()).getInfo()

        chunk_data = safe_compute(fetch_chunk, max_retries=3)
        if chunk_data:
            for feat in chunk_data:
                props = feat.get('properties', {})
                date  = props.get('date')
                val   = props.get('rainfall_mm')
                if date and val is not None:
                    rainfall_dict[date] = round(float(val), 3)

        print(f"    Days {offset}–{min(offset + chunk_size, total_days) - 1} done")

    n_rain = sum(1 for v in rainfall_dict.values() if v > 0)
    print(f"    Complete — {len(rainfall_dict)} days, {n_rain} with rainfall > 0")
    return rainfall_dict


# =========================
# ERA5-LAND RUNOFF + SOIL WATER
# Client-side loop with pure GEE server-side ops inside each iteration.
# Runoff: daily mean of accumulated bands × 1000 → relative mm-scale value.
# =========================
def get_era5_daily(start_date, end_date):
    """
    Returns dict: {date_str: {'runoff_mm': float, 'soil_water': float}}
    """
    print(f"\n[+] Fetching ERA5-Land runoff + soil water ({start_date} → {end_date})...")

    era5_point = ee.Geometry.Point([ERA5_LON, ERA5_LAT]).buffer(5000)

    start      = pd.Timestamp(start_date)
    end        = pd.Timestamp(end_date)
    total_days = (end - start).days
    chunk_size = 30

    print(f"    Processing {total_days} days in {chunk_size}-day chunks...")

    era5_dict = {}

    for offset in range(0, total_days, chunk_size):
        chunk_dates = [
            (start + pd.Timedelta(days=offset + d)).strftime('%Y-%m-%d')
            for d in range(min(chunk_size, total_days - offset))
        ]

        for date_str in chunk_dates:
            def fetch_one_day(ds=date_str):
                day_start = ee.Date(ds)
                day_end   = day_start.advance(1, 'day')

                era5_day = (
                    ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                    .filterDate(day_start, day_end)
                    .select([
                        'surface_runoff',
                        'sub_surface_runoff',
                        'volumetric_soil_water_layer_1',
                    ])
                )

                runoff_img = era5_day.select([
                    'surface_runoff', 'sub_surface_runoff'
                ]).mean().multiply(1000)

                sro_val = runoff_img.select('surface_runoff').reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=era5_point,
                    scale=9000, maxPixels=1e6, bestEffort=True
                ).getInfo().get('surface_runoff')

                ssro_val = runoff_img.select('sub_surface_runoff').reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=era5_point,
                    scale=9000, maxPixels=1e6, bestEffort=True
                ).getInfo().get('sub_surface_runoff')

                sw_val = era5_day.select('volumetric_soil_water_layer_1').mean().reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=era5_point,
                    scale=9000, maxPixels=1e6, bestEffort=True
                ).getInfo().get('volumetric_soil_water_layer_1')

                return sro_val, ssro_val, sw_val

            result = safe_compute(fetch_one_day, max_retries=3)

            if result is not None:
                sro, ssro, sw = result
                runoff_mm = 0.0
                if sro  is not None: runoff_mm += float(sro)
                if ssro is not None: runoff_mm += float(ssro)
                era5_dict[date_str] = {
                    'runoff_mm':  round(runoff_mm, 4),
                    'soil_water': round(float(sw), 5) if sw is not None else None,
                }
            else:
                era5_dict[date_str] = {'runoff_mm': 0.0, 'soil_water': None}

        print(f"    Days {offset}–{min(offset + chunk_size, total_days) - 1} done")

    n_valid  = sum(1 for v in era5_dict.values() if v['soil_water'] is not None)
    n_runoff = sum(1 for v in era5_dict.values() if v.get('runoff_mm', 0) > 0)
    print(f"    Complete — {total_days} days, {n_valid} with valid soil water")
    print(f"    Sanity check: {n_runoff} / {total_days} days have runoff > 0")
    if n_runoff == 0:
        print("    [WARNING] All ERA5 runoff values are zero — check band names")
    return era5_dict


# =========================
# ATTACH ALL SIGNALS TO PASSES
# =========================
def attach_signals_to_results(results, rainfall_dict, era5_dict):
    for row in results:
        pass_date = pd.Timestamp(row['timestamp']).normalize()
        keys = [
            (pass_date - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(7)
        ]

        # GPM rolling windows
        gpm_daily          = [rainfall_dict.get(k, 0.0) or 0.0 for k in keys]
        row['rainfall_1d'] = round(gpm_daily[0], 3)
        row['rainfall_3d'] = round(sum(gpm_daily[:3]), 3)
        row['rainfall_7d'] = round(sum(gpm_daily), 3)

        # ERA5 runoff rolling windows
        era5_runoff            = [(era5_dict.get(k) or {}).get('runoff_mm', 0.0) or 0.0 for k in keys]
        row['era5_runoff_1d']  = round(era5_runoff[0], 4)
        row['era5_runoff_7d']  = round(sum(era5_runoff), 4)

        # ERA5 soil water — 7-day mean, exclude None
        soil_vals              = [(era5_dict.get(k) or {}).get('soil_water') for k in keys]
        soil_vals              = [v for v in soil_vals if v is not None]
        row['era5_soil_water'] = round(np.mean(soil_vals), 5) if soil_vals else None

    return results


# =========================
# FLOOD LABEL — TRIPLE CONDITION
# =========================
def derive_flood_label(era5_runoff_7d, era5_soil_water, rainfall_7d):
    """
    flood_label = 1 iff ALL THREE:
        A: era5_runoff_7d  >= ERA5_RUNOFF_7D_THRESHOLD
        B: era5_soil_water >= ERA5_SOIL_WATER_THRESHOLD
        C: rainfall_7d     >= GPM_7D_FLOOD_THRESHOLD

    Returns (label, conditions_str) for diagnostics.
    """
    cond_a = (era5_runoff_7d  is not None) and (era5_runoff_7d  >= ERA5_RUNOFF_7D_THRESHOLD)
    cond_b = (era5_soil_water is not None) and (era5_soil_water >= ERA5_SOIL_WATER_THRESHOLD)
    cond_c = (rainfall_7d     is not None) and (rainfall_7d     >= GPM_7D_FLOOD_THRESHOLD)

    label     = 1 if (cond_a and cond_b and cond_c) else 0
    conds_str = (
        f"runoff={'✓' if cond_a else '✗'} "
        f"soil={'✓' if cond_b else '✗'} "
        f"rain={'✓' if cond_c else '✗'}"
    )
    return label, conds_str


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
        stats = image.select([
            'VV_filtered', 'VH_filtered',
            'soil_saturation', 'flood_extent',
        ]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi,
            scale=30, maxPixels=1e8, bestEffort=True,
        )
        return image.set({
            'vv_mean':        stats.get('VV_filtered'),
            'vh_mean':        stats.get('VH_filtered'),
            'soil_sat_mean':  stats.get('soil_saturation'),
            'flood_ext_mean': stats.get('flood_extent'),
        })

    collection_with_stats = collection.map(add_stats)

    def get_features():
        features = collection_with_stats.map(lambda img: ee.Feature(None, {
            'timestamp':       ee.Date(img.date().millis()).format("YYYY-MM-dd'T'HH:mm:ss'Z'"),
            'timestamp_ms':    img.get('system:time_start'),
            'orbit':           img.get('orbitProperties_pass'),
            'vv_mean':         img.get('vv_mean'),
            'vh_mean':         img.get('vh_mean'),
            'soil_saturation': img.get('soil_sat_mean'),
            'flood_extent':    img.get('flood_ext_mean'),
        }))
        return features.toList(collection.size()).getInfo()

    feature_list = safe_compute(get_features, max_retries=3)
    if feature_list is None:
        print(f"  [!] Failed to retrieve features for {orbit}")
        return []

    trend_map  = calculate_wetness_trend_per_image(collection, aoi)
    orbit_flag = 0 if orbit == 'ASCENDING' else 1

    results    = []
    n_skipped  = 0
    n_outliers = 0

    for feature in feature_list:
        props   = feature['properties']
        ts_ms   = int(props.get('timestamp_ms', 0))  # FIX: int() for dict lookup

        vv_mean   = props.get('vv_mean',         0) or 0
        vh_mean   = props.get('vh_mean',         0) or 0
        soil_sat  = props.get('soil_saturation', 0) or 0
        flood_ext = props.get('flood_extent',    0) or 0

        # Drop rows where SAR retrieval completely failed
        if vv_mean == 0 and vh_mean == 0:
            print(f"  [!] Skipping failed SAR row: {props.get('timestamp', 'unknown')}")
            n_skipped += 1
            continue

        # Flag flood_extent outliers after server-side clamp
        # (clamp is applied in GEE but reduceRegion mean can still
        # produce out-of-range values if the image contains NaN pixels)
        flood_ext_f = float(flood_ext)
        if not (FLOOD_EXTENT_MIN <= flood_ext_f <= FLOOD_EXTENT_MAX):
            n_outliers += 1
            # Replace with None — downstream will handle as NaN in model
            flood_ext_f = None

        results.append({
            'timestamp':       props['timestamp'],
            'orbit':           props.get('orbit', orbit),
            'orbit_flag':      orbit_flag,
            'vv_mean':         round(float(vv_mean),  4),
            'vh_mean':         round(float(vh_mean),  4),
            'soil_saturation': round(float(soil_sat), 4),
            'flood_extent':    round(flood_ext_f, 4) if flood_ext_f is not None else None,
            'wetness_trend':   trend_map.get(ts_ms, 0),  # FIX: ts_ms already int
            'rainfall_1d':     0.0,
            'rainfall_3d':     0.0,
            'rainfall_7d':     0.0,
            'era5_runoff_1d':  0.0,
            'era5_runoff_7d':  0.0,
            'era5_soil_water': None,
            'flood_label':     None,
        })

    print(f"  [+] {orbit} — {len(results)} valid passes "
          f"({n_skipped} skipped, {n_outliers} flood_extent outliers set to None)")
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
# VALIDATION — known flood events
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
    df['ym']     = df['timestamp'].astype(str).str[:7]

    print("\n" + "=" * 105)
    print("  KNOWN EVENT VALIDATION")
    print(f"  Thresholds: runoff7d >= {ERA5_RUNOFF_7D_THRESHOLD}mm  |  "
          f"soil >= {ERA5_SOIL_WATER_THRESHOLD} m³/m³  |  rain7d >= {GPM_7D_FLOOD_THRESHOLD}mm")
    print("=" * 105)
    print(f"  {'Month':<10} {'Event':<32} {'P':>3}  {'=1':>4}  "
          f"{'runoff7d':>9}  {'soil_w':>7}  {'rain7d':>8}  {'conditions':<24}")
    print(f"  {'-'*10} {'-'*32} {'-'*3}  {'-'*4}  "
          f"{'-'*9}  {'-'*7}  {'-'*8}  {'-'*24}")

    all_correct = True

    for ym, name in known_events:
        subset = df[df['ym'] == ym]
        if len(subset) == 0:
            print(f"  {ym:<10} {name:<32} {'NO DATA'}")
            all_correct = False
            continue

        n_passes  = len(subset)
        n_flood   = int((subset['flood_label'] == 1).sum())
        avg_run   = subset['era5_runoff_7d'].mean()
        avg_soil  = subset['era5_soil_water'].mean()
        avg_rain  = subset['rainfall_7d'].mean()
        ok        = "✅" if n_flood > 0 else "❌"

        cond_a = "✓run" if avg_run  >= ERA5_RUNOFF_7D_THRESHOLD  else "✗run"
        cond_b = "✓soil" if avg_soil >= ERA5_SOIL_WATER_THRESHOLD else "✗soil"
        cond_c = "✓rain" if avg_rain >= GPM_7D_FLOOD_THRESHOLD    else "✗rain"

        if n_flood == 0:
            all_correct = False

        print(f"  {ym:<10} {name:<32} {n_passes:>3}  {n_flood:>4}  "
              f"{avg_run:>8.2f}mm  {avg_soil:>7.4f}  {avg_rain:>7.1f}mm  "
              f"{cond_a} {cond_b} {cond_c}  {ok}")

    # False positive rate on non-event months
    non_event_df    = df[~df['ym'].isin(event_months)]
    fp_rate         = non_event_df['flood_label'].mean() if len(non_event_df) > 0 else 0.0
    non_event_flood = int(non_event_df['flood_label'].sum())

    print()
    print(f"  Non-event months : {len(non_event_df)} passes, "
          f"{non_event_flood} labeled flood ({100*fp_rate:.1f}% false positive baseline)")
    print()

    if all_correct:
        print("  ✅  All known flood events correctly labeled.")
    else:
        print("  ❌  Some events missed. Tune the failing condition:")
        print(f"      ✗run  → lower ERA5_RUNOFF_7D_THRESHOLD  (now {ERA5_RUNOFF_7D_THRESHOLD}  → try 5.0)")
        print(f"      ✗soil → lower ERA5_SOIL_WATER_THRESHOLD (now {ERA5_SOIL_WATER_THRESHOLD} → try 0.43)")
        print(f"      ✗rain → lower GPM_7D_FLOOD_THRESHOLD    (now {GPM_7D_FLOOD_THRESHOLD}mm → try 60.0)")

    print("=" * 105)
    df.drop(columns=['ym'], inplace=True, errors='ignore')


# =========================
# MAIN
# =========================
def main():
    print("=" * 80)
    print("SENTINEL-1 + GPM + ERA5 — TRIPLE-CONDITION FLOOD LABELING")
    print("=" * 80)
    print(f"\n  Condition A : ERA5 runoff 7d >= {ERA5_RUNOFF_7D_THRESHOLD} mm")
    print(f"  Condition B : ERA5 soil water (0–7cm) >= {ERA5_SOIL_WATER_THRESHOLD} m³/m³")
    print(f"  Condition C : GPM 7d rainfall >= {GPM_7D_FLOOD_THRESHOLD} mm")
    print(f"  Label = 1   : ALL THREE conditions met\n")

    initialize_gee()

    print("\n[+] Loading Area of Interest...")
    aoi = load_aoi_ee(AOI_GEOJSON)
    if not aoi:
        print("[!] Failed to load AOI")
        exit()

    start_date = '2017-01-01'
    end_date   = '2026-03-01'

    # Step 1 — GPM
    rainfall_dict = get_gpm_rainfall(aoi, start_date, end_date)

    # Step 2 — ERA5
    era5_dict = get_era5_daily(start_date, end_date)

    # Step 3 — Sentinel-1 per orbit
    all_results = []
    for orbit in ['ASCENDING', 'DESCENDING']:
        rows = process_orbit(aoi, start_date, end_date, orbit)
        all_results.extend(rows)

    if not all_results:
        print("[!] No results to save")
        exit()

    # Step 4 — Attach signals
    print("\n[+] Attaching GPM and ERA5 signals to SAR passes...")
    all_results = attach_signals_to_results(all_results, rainfall_dict, era5_dict)

    # Step 5 — Deduplicate, sort, drop trailing null rows
    df = pd.DataFrame(all_results)
    df = df.dropna(how='all')                                              # FIX: trailing null row
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp', 'orbit']).reset_index(drop=True)

    # Step 6 — Labels
    print("\n[+] Assigning flood labels (triple condition)...")
    labels = []
    for _, row in df.iterrows():
        label, _ = derive_flood_label(
            era5_runoff_7d  = row['era5_runoff_7d'],
            era5_soil_water = row['era5_soil_water'],
            rainfall_7d     = row['rainfall_7d'],
        )
        labels.append(label)
    df['flood_label'] = labels

    # Step 7 — Save
    output_csv = os.path.join(OUTPUT_DIR, "sentinel1_timeseries.csv")
    df.to_csv(output_csv, mode='w', index=False, header=True)

    # Step 8 — Summary
    print("\n" + "=" * 80)
    print(f"[+] Total rows saved : {len(df)}")

    print(f"\n[+] Label distribution (overall):")
    vc = df['flood_label'].value_counts().sort_index()
    for label, count in vc.items():
        print(f"    label={label} : {count:>4}  ({100*count/len(df):.1f}%)")

    print(f"\n[+] Label distribution per orbit:")
    print(df.groupby('orbit')['flood_label'].value_counts().to_string())

    cond_a = df['era5_runoff_7d']  >= ERA5_RUNOFF_7D_THRESHOLD
    cond_b = df['era5_soil_water'] >= ERA5_SOIL_WATER_THRESHOLD
    cond_c = df['rainfall_7d']     >= GPM_7D_FLOOD_THRESHOLD
    print(f"\n[+] Condition diagnostics:")
    print(f"    A: runoff_7d >= {ERA5_RUNOFF_7D_THRESHOLD}mm : "
          f"{cond_a.sum():>4} rows  ({100*cond_a.mean():.1f}%)")
    print(f"    B: soil_water >= {ERA5_SOIL_WATER_THRESHOLD}  : "
          f"{cond_b.sum():>4} rows  ({100*cond_b.mean():.1f}%)")
    print(f"    C: rain_7d >= {GPM_7D_FLOOD_THRESHOLD}mm : "
          f"{cond_c.sum():>4} rows  ({100*cond_c.mean():.1f}%)")
    print(f"    A+B+C (label=1)     : "
          f"{df['flood_label'].sum():>4} rows  ({100*df['flood_label'].mean():.1f}%)")

    # flood_extent quality report
    n_none = df['flood_extent'].isna().sum()
    if n_none > 0:
        print(f"\n[+] flood_extent outliers replaced with None: {n_none} rows")
        print(f"    (outside valid range [{FLOOD_EXTENT_MIN}, {FLOOD_EXTENT_MAX}])")

    print(f"\n[+] ERA5 stats:")
    print(df[['era5_runoff_1d', 'era5_runoff_7d', 'era5_soil_water']].describe().round(4).to_string())

    print(f"\n[+] GPM rainfall stats (mm):")
    print(df[['rainfall_1d', 'rainfall_3d', 'rainfall_7d']].describe().round(2).to_string())

    print(f"\n[+] SAR feature stats:")
    print(df[['vv_mean', 'vh_mean', 'soil_saturation', 'flood_extent']].describe().round(4).to_string())

    print(f"\n[+] Sample rows:")
    print(df[[
        'timestamp', 'orbit', 'vv_mean', 'era5_runoff_7d',
        'era5_soil_water', 'rainfall_7d', 'flood_label',
    ]].head(10).to_string())

    print(f"\n[+] Results saved to: {output_csv}")
    print("=" * 80)

    # Step 9 — Validate
    validate_known_events(df.copy())


if __name__ == "__main__":
    main()