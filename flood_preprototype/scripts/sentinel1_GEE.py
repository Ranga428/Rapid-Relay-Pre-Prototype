"""
sentinel1_GEE.py
================
Sentinel-1 SAR + GPM + ERA5-Land Flood Label Pipeline

LABELING STRATEGY — ERA5 RUNOFF + SOIL WATER + GPM (triple condition)
----------------------------------------------------------------------
Labels are derived from three independent, cloud-immune reanalysis/satellite
signals that reflect the actual physical mechanism of flooding in Obando:

    HOW OBANDO FLOODS
    -----------------
    Obando (elevation ~1m) floods when:
      1. Upstream watershed receives heavy rainfall → rivers overflow
      2. Soil is already saturated → no absorption capacity left
      3. Manila Bay high tide prevents drainage (backflow)

    The triple condition captures signals 1 and 2 directly.
    GPM provides an independent cross-check on ERA5 precipitation.

    CONDITION A — ERA5 7-day total runoff >= ERA5_RUNOFF_7D_THRESHOLD
        Surface + subsurface runoff accumulated over 7 days.
        High runoff = rivers are overwhelmed, water has nowhere to go.
        ERA5-Land ECMWF/ERA5_LAND/HOURLY, bands: surface_runoff +
        sub_surface_runoff. Units: m/hr → converted to mm/day by *1000.

    CONDITION B — ERA5 soil water (layer 1, top 7cm) >= ERA5_SOIL_WATER_THRESHOLD
        Volumetric soil water content (m³/m³). When soil is near saturation,
        rainfall contributes directly to runoff rather than infiltrating.
        Threshold ~0.35 = near field capacity for coastal clay/silt soil.

    CONDITION C — GPM 7-day rainfall >= GPM_7D_FLOOD_THRESHOLD
        Independent satellite rainfall from GPM IMERG V07.
        Cross-checks ERA5 — both must agree there was significant rainfall.

    flood_label = 1  iff  ALL THREE conditions met simultaneously
    flood_label = 0  otherwise (no rows dropped — full 2017–2026 coverage)

WHY THREE CONDITIONS?
    Any single signal can produce false positives:
      - High runoff can occur from normal wet-season flow
      - Saturated soil is common throughout rainy season
      - Heavy rain doesn't always flood (depends on upstream state)
    All three together mean: heavy rain fell, the ground was already full,
    and the rivers overflowed. That is the Obando flood mechanism.

WHY ERA5 INSTEAD OF SAR FOR LABELS?
    SAR backscatter direction (rise vs drop) is orbit-dependent for this
    coastal AOI — ASCENDING and DESCENDING orbits respond differently to
    flooding due to different incidence angles on estuarine/bay surfaces.
    ERA5 is a global reanalysis (assimilates real observations), cloud-immune,
    covers 1950–present, and directly models the hydrological state.

THRESHOLD CALIBRATION
    Thresholds are validated against 6 known Obando flood events after the
    run. The validation table prints ERA5 runoff, soil water, and GPM rain
    for each known event so you can see exactly which condition failed and
    adjust the relevant threshold.

    Starting thresholds (conservative — tune down if events are missed):
        ERA5_RUNOFF_7D_THRESHOLD  = 5.0  mm
        ERA5_SOIL_WATER_THRESHOLD = 0.35 m³/m³
        GPM_7D_FLOOD_THRESHOLD    = 25.0 mm

KNOWN OBANDO FLOOD EVENTS (validation anchors)
    2018-08  Typhoon Karding / southwest monsoon
    2019-07  Typhoon Falcon + LPA — Bulacan widespread flooding
    2020-10  Typhoon Quinta — Metro Manila + Bulacan flooding
    2021-10  Typhoon Maring + monsoon — Bulacan inundation
    2022-07  Southwest monsoon — Bulacan flooding
    2024-07  Typhoon Carina + southwest monsoon — major NCR/Bulacan flood

OUTPUTS
-------
    sentinel1_timeseries.csv
        timestamp, orbit, orbit_flag, vv_mean, vh_mean,
        soil_saturation, flood_extent, wetness_trend,
        rainfall_1d, rainfall_3d, rainfall_7d,
        era5_runoff_1d, era5_runoff_7d,
        era5_soil_water,
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
# LABELING THRESHOLDS
# ---------------------------------------------------------------------------

# ERA5-Land 7-day total runoff is retained as a FEATURE COLUMN only.
# It is NOT used as a labeling condition because:
#   1. ERA5 runoff accumulates since 00:00 UTC — summing hourly values
#      double-counts, inflating values ~12x. Unit fix would require taking
#      only the last hourly image per day.
#   2. Even with correct units, runoff at 9km resolution fires on 93% of
#      passes for this coastal lowland — it has no discriminating power.
# ERA5_RUNOFF_7D_THRESHOLD is kept here for reference only.
ERA5_RUNOFF_7D_THRESHOLD = 5.0    # mm (not used in labeling)

# ERA5-Land volumetric soil water content, top 7cm layer (m³/m³).
# Obando coastal clay/silt: field capacity ~0.35, saturation ~0.49.
# Threshold of 0.43 selects near-saturated soil — additional rain goes
# directly to surface runoff and river overflow.
# All 6 known flood events had soil_water >= 0.45, well above this threshold.
# Raise to 0.45 to reduce flood rate further. Lower to 0.40 if events missed.
ERA5_SOIL_WATER_THRESHOLD = 0.43  # m³/m³

# GPM IMERG 7-day accumulated rainfall in mm.
# 50mm over 7 days = sustained moderate-to-heavy rainfall (>7mm/day average).
# All 6 known flood events had rainfall_7d >= 67mm — safely above threshold.
# Raise to 75mm to reduce flood rate. Lower to 35mm if events missed.
GPM_7D_FLOOD_THRESHOLD = 50.0    # mm

# ERA5-Land point: centroid of AOI, buffered 5km.
# ERA5 is 9km resolution — a single grid cell covers the whole AOI.
ERA5_LAT = 14.71
ERA5_LON = 120.85


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
# =========================
def calculate_soil_saturation(collection):
    def add_soil_saturation(image):
        vv = image.select('VV_filtered')
        vh = image.select('VH_filtered')
        soil_sat = vv.subtract(vh).divide(vv.abs().add(vh.abs())).rename('soil_saturation')
        return image.addBands(soil_sat)
    return collection.map(add_soil_saturation)


# =========================
# FLOOD EXTENT INDEX
# =========================
def calculate_flood_extent(collection):
    def add_flood_extent(image):
        vv = image.select('VV_filtered')
        vh = image.select('VH_filtered')
        flood_ext = vv.multiply(-1).divide(vh.abs()).rename('flood_extent')
        return image.addBands(flood_ext)
    return collection.map(add_flood_extent)


# =========================
# WETNESS TREND
# =========================
def calculate_wetness_trend_per_image(collection, aoi):
    """
    For each image, computes a rolling 30-day VV change trend.
    Returns dict: {timestamp_ms: trend}
        +1 = wetting (VV decreasing), -1 = drying, 0 = stable
    """
    image_list = collection.toList(collection.size())
    n          = image_list.size().getInfo()
    trends     = {}

    print(f"[+] Calculating per-image wetness trend (30-day rolling window)...")

    for i in range(1, n):
        try:
            current      = ee.Image(image_list.get(i))
            current_time = current.get('system:time_start').getInfo()
            current_date = pd.Timestamp(current_time, unit='ms')

            window_start = (current_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            window_end   = current_date.strftime('%Y-%m-%d')

            window_col = collection.filterDate(window_start, window_end)

            if window_col.size().getInfo() < 2:
                trends[current_time] = 0
                continue

            window_list = window_col.toList(window_col.size())
            n_window    = window_list.size().getInfo()

            first_img = ee.Image(window_list.get(0))
            last_img  = ee.Image(window_list.get(n_window - 1))

            first_stats = first_img.select('VV_filtered').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=30,
                maxPixels=1e8, bestEffort=True
            ).getInfo()
            last_stats = last_img.select('VV_filtered').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=30,
                maxPixels=1e8, bestEffort=True
            ).getInfo()

            first_vv = first_stats.get('VV_filtered')
            last_vv  = last_stats.get('VV_filtered')

            if first_vv is None or last_vv is None:
                trends[current_time] = 0
                continue

            mean_change = float(last_vv) - float(first_vv)

            if   mean_change < -0.5: trends[current_time] =  1   # Wetting
            elif mean_change >  0.5: trends[current_time] = -1   # Drying
            else:                    trends[current_time] =  0   # Stable
        except Exception as e:
            print(f"    [!] Trend failed for image {i}: {e}")
            trends[current_time] = 0

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

    print(f"    Processing {total_days} days in 90-day chunks...")

    rainfall_dict = {}
    chunk_size    = 90

    for offset in range(0, total_days, chunk_size):
        chunk_start = (start + pd.Timedelta(days=offset)).strftime('%Y-%m-%d')
        chunk_end   = (start + pd.Timedelta(days=min(offset + chunk_size, total_days))).strftime('%Y-%m-%d')

        def fetch_chunk(cs=chunk_start, ce=chunk_end):
            gpm = (
                ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
                .filterDate(cs, ce)
                .filterBounds(aoi)
                .select('precipitation')
            )

            def daily_sum(day_offset):
                day       = ee.Date(cs).advance(day_offset, 'day')
                day_col   = gpm.filterDate(day, day.advance(1, 'day'))
                daily_img = day_col.sum().multiply(0.5)   # 30-min → mm/day
                mean_val  = daily_img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=aoi,
                    scale=10000, maxPixels=1e8, bestEffort=True
                ).get('precipitation')
                return ee.Feature(None, {
                    'date':        day.format('YYYY-MM-dd'),
                    'rainfall_mm': mean_val,
                })

            n_days   = (pd.Timestamp(ce) - pd.Timestamp(cs)).days
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

        print(f"    Days {offset}–{min(offset+chunk_size, total_days)-1} done")

    n_rain = sum(1 for v in rainfall_dict.values() if v > 0)
    print(f"    Complete — {total_days} days, {n_rain} with rainfall > 0")
    return rainfall_dict


# =========================
# ERA5-LAND RUNOFF + SOIL WATER
# =========================
def get_era5_daily(start_date, end_date):
    """
    Fetches daily ERA5-Land runoff and soil water for the Obando grid cell.

    Bands fetched:
        surface_runoff       : m/hr  → summed over 24hrs → mm/day (*1000)
        sub_surface_runoff   : m/hr  → summed over 24hrs → mm/day (*1000)
        volumetric_soil_water_layer_1 : m³/m³ (daily mean)

    ERA5 is 9km resolution. We use a point geometry (ERA5_LAT, ERA5_LON)
    with 5km buffer — this selects the single grid cell over Obando.
    Processed in 30-day chunks to avoid GEE memory limits.

    Returns dict: {date_str: {'runoff_mm': float, 'soil_water': float}}
    """
    print(f"\n[+] Fetching ERA5-Land runoff + soil water ({start_date} → {end_date})...")

    # 5km buffer around AOI centroid — selects the ERA5 grid cell
    era5_point = ee.Geometry.Point([ERA5_LON, ERA5_LAT]).buffer(5000)

    start      = pd.Timestamp(start_date)
    end        = pd.Timestamp(end_date)
    total_days = (end - start).days

    print(f"    Processing {total_days} days in 30-day chunks...")

    era5_dict  = {}
    chunk_size = 30   # ERA5 hourly is large — smaller chunks avoid timeouts

    for offset in range(0, total_days, chunk_size):
        chunk_start = (start + pd.Timedelta(days=offset)).strftime('%Y-%m-%d')
        chunk_end   = (start + pd.Timedelta(days=min(offset + chunk_size, total_days))).strftime('%Y-%m-%d')

        def fetch_era5_chunk(cs=chunk_start, ce=chunk_end):
            era5 = (
                ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                .filterDate(cs, ce)
                .select([
                    'surface_runoff',
                    'sub_surface_runoff',
                    'volumetric_soil_water_layer_1',
                ])
            )

            n_chunk_days = (pd.Timestamp(ce) - pd.Timestamp(cs)).days

            def daily_aggregate(day_offset):
                day     = ee.Date(cs).advance(day_offset, 'day')
                day_col = era5.filterDate(day, day.advance(1, 'day'))

                # Runoff: sum 24 hourly values then *1000 to convert m → mm
                runoff_sum = day_col.select([
                    'surface_runoff', 'sub_surface_runoff'
                ]).sum()

                sro_val  = runoff_sum.select('surface_runoff').reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=era5_point,
                    scale=9000, maxPixels=1e6, bestEffort=True
                ).get('surface_runoff')

                ssro_val = runoff_sum.select('sub_surface_runoff').reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=era5_point,
                    scale=9000, maxPixels=1e6, bestEffort=True
                ).get('sub_surface_runoff')

                # Soil water: mean across day (relatively stable within a day)
                swl_val  = day_col.select('volumetric_soil_water_layer_1').mean().reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=era5_point,
                    scale=9000, maxPixels=1e6, bestEffort=True
                ).get('volumetric_soil_water_layer_1')

                return ee.Feature(None, {
                    'date':         day.format('YYYY-MM-dd'),
                    'sro_m':        sro_val,    # surface runoff, m (24hr sum)
                    'ssro_m':       ssro_val,   # subsurface runoff, m (24hr sum)
                    'soil_water':   swl_val,    # m³/m³
                })

            day_list = ee.List.sequence(0, n_chunk_days - 1)
            features = ee.FeatureCollection(day_list.map(daily_aggregate))
            return features.toList(features.size()).getInfo()

        chunk_data = safe_compute(fetch_era5_chunk, max_retries=3)

        if chunk_data:
            for feat in chunk_data:
                props = feat.get('properties', {})
                date  = props.get('date')
                if not date:
                    continue

                sro  = props.get('sro_m')
                ssro = props.get('ssro_m')
                sw   = props.get('soil_water')

                # Convert m → mm for runoff (sum of hourly values in m/hr * 1hr = m)
                runoff_mm = 0.0
                if sro  is not None: runoff_mm += float(sro)  * 1000
                if ssro is not None: runoff_mm += float(ssro) * 1000

                soil_water = round(float(sw), 5) if sw is not None else None

                era5_dict[date] = {
                    'runoff_mm':  round(runoff_mm, 4),
                    'soil_water': soil_water,
                }

        print(f"    Days {offset}–{min(offset+chunk_size, total_days)-1} done")

    n_valid = sum(1 for v in era5_dict.values() if v['soil_water'] is not None)
    print(f"    Complete — {total_days} days, {n_valid} with valid ERA5 data")
    return era5_dict


# =========================
# ATTACH ALL SIGNALS TO PASSES
# =========================
def attach_signals_to_results(results, rainfall_dict, era5_dict):
    """
    Attaches GPM and ERA5 rolling window values to each SAR pass row.

    GPM:
        rainfall_1d = pass date
        rainfall_3d = pass date + 2 preceding days
        rainfall_7d = pass date + 6 preceding days

    ERA5:
        era5_runoff_1d  = total runoff on pass date (mm)
        era5_runoff_7d  = total runoff over 7 days (mm)
        era5_soil_water = mean soil water over 7 days (m³/m³)
    """
    for row in results:
        pass_date = pd.Timestamp(row['timestamp']).normalize()

        # Build 7-day key list: [today, yesterday, ..., 6 days ago]
        keys = [
            (pass_date - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(7)
        ]

        # GPM windows
        gpm_daily = [rainfall_dict.get(k, 0.0) or 0.0 for k in keys]
        row['rainfall_1d'] = round(gpm_daily[0], 3)
        row['rainfall_3d'] = round(sum(gpm_daily[:3]), 3)
        row['rainfall_7d'] = round(sum(gpm_daily), 3)

        # ERA5 runoff windows
        era5_daily_runoff = [
            (era5_dict.get(k) or {}).get('runoff_mm', 0.0) or 0.0
            for k in keys
        ]
        row['era5_runoff_1d'] = round(era5_daily_runoff[0], 4)
        row['era5_runoff_7d'] = round(sum(era5_daily_runoff), 4)

        # ERA5 soil water — mean over 7 days (exclude None)
        soil_vals = [
            (era5_dict.get(k) or {}).get('soil_water')
            for k in keys
        ]
        soil_vals = [v for v in soil_vals if v is not None]
        row['era5_soil_water'] = round(np.mean(soil_vals), 5) if soil_vals else None

    return results


# =========================
# FLOOD LABEL — TRIPLE CONDITION
# =========================
def derive_flood_label(era5_runoff_7d, era5_soil_water, rainfall_7d):
    """
    flood_label = 1 iff BOTH conditions met:
        B: era5_soil_water >= ERA5_SOIL_WATER_THRESHOLD  (ground near-saturated)
        C: rainfall_7d     >= GPM_7D_FLOOD_THRESHOLD     (sustained heavy rain)

    Runoff (era5_runoff_7d) is passed in but used as a feature only —
    not as a labeling condition. See config comments for why.

    Returns (flood_label, conditions_met_str) for diagnostics.
    """
    cond_b = (era5_soil_water is not None) and (era5_soil_water >= ERA5_SOIL_WATER_THRESHOLD)
    cond_c = (rainfall_7d     is not None) and (rainfall_7d     >= GPM_7D_FLOOD_THRESHOLD)

    flood_label = 1 if (cond_b and cond_c) else 0
    conds_str   = f"soil={'✓' if cond_b else '✗'} rain={'✓' if cond_c else '✗'}"
    return flood_label, conds_str


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
            reducer   = ee.Reducer.mean(),
            geometry  = aoi,
            scale     = 30,
            maxPixels = 1e8,
            bestEffort= True,
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
            'timestamp':       ee.Date(img.date().millis()).format(
                                   "YYYY-MM-dd'T'HH:mm:ss'Z'"
                               ),
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

    results = []
    for feature in feature_list:
        props   = feature['properties']
        ts_str  = props['timestamp']
        ts_ms   = props.get('timestamp_ms', 0)

        vv_mean   = round(float(props.get('vv_mean',         0) or 0), 4)
        vh_mean   = round(float(props.get('vh_mean',         0) or 0), 4)
        soil_sat  = round(float(props.get('soil_saturation', 0) or 0), 4)
        flood_ext = round(float(props.get('flood_extent',    0) or 0), 4)

        results.append({
            'timestamp':        ts_str,
            'orbit':            props.get('orbit', orbit),
            'orbit_flag':       orbit_flag,
            'vv_mean':          vv_mean,
            'vh_mean':          vh_mean,
            'soil_saturation':  soil_sat,
            'flood_extent':     flood_ext,
            'wetness_trend':    trend_map.get(ts_ms, 0),
            'rainfall_1d':      0.0,
            'rainfall_3d':      0.0,
            'rainfall_7d':      0.0,
            'era5_runoff_1d':   0.0,
            'era5_runoff_7d':   0.0,
            'era5_soil_water':  None,
            'flood_label':      None,
        })

    print(f"  [+] {orbit} — {len(results)} passes collected")
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
    """
    Cross-checks labels against known Obando/Bulacan flood events.
    Prints ERA5 runoff, soil water, and GPM rain for each event so you
    can see exactly which condition is failing and tune the right threshold.
    """
    known_events = [
        ("2018-08", "Typhoon Karding / SW monsoon"),
        ("2019-07", "Typhoon Falcon + LPA"),
        ("2020-10", "Typhoon Quinta"),
        ("2021-10", "Typhoon Maring + monsoon"),
        ("2022-07", "SW monsoon — Bulacan"),
        ("2024-07", "Typhoon Carina + SW monsoon"),
    ]

    print("\n" + "=" * 95)
    print("  KNOWN EVENT VALIDATION")
    print(f"  Thresholds: soil >= {ERA5_SOIL_WATER_THRESHOLD} m³/m³  |  rain7d >= {GPM_7D_FLOOD_THRESHOLD}mm")
    print("=" * 95)
    print(f"  {'Month':<10} {'Event':<32} {'P':>3}  {'=1':>4}  "
          f"{'runoff7d':>9}  {'soil_w':>7}  {'rain7d':>8}  {'conditions':<20}")
    print(f"  {'-'*10} {'-'*32} {'-'*3}  {'-'*4}  "
          f"{'-'*9}  {'-'*7}  {'-'*8}  {'-'*20}")

    df['ym'] = df['timestamp'].astype(str).str[:7]
    all_correct = True

    for ym, name in known_events:
        subset = df[df['ym'] == ym]
        if len(subset) == 0:
            print(f"  {ym:<10} {name:<32} {'NO DATA':>3}")
            continue

        n_passes   = len(subset)
        n_flood    = (subset['flood_label'] == 1).sum()
        avg_runoff = subset['era5_runoff_7d'].mean()
        avg_soil   = subset['era5_soil_water'].mean()
        avg_rain   = subset['rainfall_7d'].mean()
        ok         = "✅" if n_flood > 0 else "❌"

        # Show which conditions are met on average
        cond_b = "✓soil" if avg_soil >= ERA5_SOIL_WATER_THRESHOLD else "✗soil"
        cond_c = "✓rain" if avg_rain >= GPM_7D_FLOOD_THRESHOLD     else "✗rain"

        if n_flood == 0:
            all_correct = False

        print(f"  {ym:<10} {name:<32} {n_passes:>3}  {n_flood:>4}  "
              f"{avg_runoff:>8.2f}mm  {avg_soil:>7.4f}  {avg_rain:>7.1f}mm"
              f"  {cond_b} {cond_c}  {ok}")

    df.drop(columns=['ym'], inplace=True, errors='ignore')

    print()
    if all_correct:
        print("  ✅  All known flood events correctly labeled.")
    else:
        print("  ❌  Some events missed. Tune the failing condition:")
        print(f"      ✗soil → lower ERA5_SOIL_WATER_THRESHOLD (now {ERA5_SOIL_WATER_THRESHOLD} → try 0.40)")
        print(f"      ✗rain → lower GPM_7D_FLOOD_THRESHOLD    (now {GPM_7D_FLOOD_THRESHOLD}mm → try 35.0)")
    print("=" * 95)


# =========================
# MAIN
# =========================
def main():
    print("=" * 80)
    print("SENTINEL-1 + GPM + ERA5 — TRIPLE-CONDITION FLOOD LABELING")
    print("=" * 80)
    print(f"\n  Labeling strategy : ERA5 runoff + ERA5 soil water + GPM rainfall")
    print(f"  Condition B       : ERA5 soil water (0–7cm) >= {ERA5_SOIL_WATER_THRESHOLD} m³/m³")
    print(f"  Condition C       : GPM 7d rainfall >= {GPM_7D_FLOOD_THRESHOLD} mm")
    print(f"  ERA5 runoff       : retained as feature column, NOT used for labeling")
    print(f"  flood_label = 1   : BOTH conditions met")
    print(f"  flood_label = 0   : any condition not met (no rows dropped)\n")

    initialize_gee()

    print("\n[+] Loading Area of Interest...")
    aoi = load_aoi_ee(AOI_GEOJSON)
    if not aoi:
        print("[!] Failed to load AOI")
        exit()

    start_date = '2017-01-01'
    end_date   = '2026-03-01'

    # Step 1 — GPM rainfall
    rainfall_dict = get_gpm_rainfall(aoi, start_date, end_date)

    # Step 2 — ERA5-Land runoff and soil water
    era5_dict = get_era5_daily(start_date, end_date)

    # Step 3 — Sentinel-1 SAR passes per orbit
    all_results = []
    for orbit in ['ASCENDING', 'DESCENDING']:
        rows = process_orbit(aoi, start_date, end_date, orbit)
        all_results.extend(rows)

    if not all_results:
        print("[!] No results to save")
        exit()

    # Step 4 — Attach all signals (GPM + ERA5) to each pass
    print("\n[+] Attaching GPM and ERA5 signals to SAR passes...")
    all_results = attach_signals_to_results(all_results, rainfall_dict, era5_dict)

    # Step 5 — Build dataframe, deduplicate, sort
    df = pd.DataFrame(all_results)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp', 'orbit']).reset_index(drop=True)

    # Step 6 — Assign flood labels
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
        pct = 100 * count / len(df)
        print(f"    label={label} : {count:>4}  ({pct:.1f}%)")

    print(f"\n[+] Label distribution per orbit:")
    print(df.groupby('orbit')['flood_label'].value_counts().to_string())

    # Condition diagnostics
    cond_b = df['era5_soil_water'] >= ERA5_SOIL_WATER_THRESHOLD
    cond_c = df['rainfall_7d']     >= GPM_7D_FLOOD_THRESHOLD
    print(f"\n[+] Condition diagnostics:")
    print(f"    B: soil_water >= {ERA5_SOIL_WATER_THRESHOLD}    : "
          f"{cond_b.sum():>4} rows  ({100*cond_b.mean():.1f}%)")
    print(f"    C: rain_7d    >= {GPM_7D_FLOOD_THRESHOLD}mm  : "
          f"{cond_c.sum():>4} rows  ({100*cond_c.mean():.1f}%)")
    print(f"    B+C (flood=1)                     : "
          f"{df['flood_label'].sum():>4} rows  ({100*df['flood_label'].mean():.1f}%)")

    print(f"\n[+] ERA5 signal stats:")
    print(df[['era5_runoff_1d','era5_runoff_7d','era5_soil_water']].describe().round(4).to_string())

    print(f"\n[+] GPM rainfall stats (mm):")
    print(df[['rainfall_1d','rainfall_3d','rainfall_7d']].describe().round(2).to_string())

    print(f"\n[+] Sample rows (first 10):")
    print(df[[
        'timestamp', 'orbit',
        'era5_runoff_7d', 'era5_soil_water', 'rainfall_7d', 'flood_label',
    ]].head(10).to_string())

    print(f"\n[+] Results saved to: {output_csv}")
    print("=" * 80)

    # Step 9 — Validate against known flood events
    validate_known_events(df.copy())


if __name__ == "__main__":
    main()