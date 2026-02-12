import ee
import geemap
import json
import pandas as pd
from datetime import datetime
import os
import time

# =========================
# CONFIGURATION
# =========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "data", "sentinel1", "GEE-Processing")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AOI_GEOJSON = os.path.join(_SCRIPT_DIR, "..", "config", "aoi.geojson")

# =========================
# INITIALIZE EARTH ENGINE
# =========================
def initialize_gee():
    """
    Initialize Google Earth Engine
    First time: Run 'earthengine authenticate' in terminal
    """
    try:
        # Replace 'your-project-id' with your actual GEE project ID
        ee.Initialize(project='jenel-466709')
        print("[+] Google Earth Engine initialized")
    except Exception as e:
        print("[!] GEE initialization failed.")
        print(f"    Error: {e}")
        print("\n[!] Please follow these steps:")
        print("    1. Run: earthengine authenticate")
        print("    2. Sign up at: https://signup.earthengine.google.com/")
        print("    3. Get your project ID from: https://console.cloud.google.com/")
        print("    4. Update the script with: ee.Initialize(project='YOUR-PROJECT-ID')")
        exit()

# =========================
# LOAD AOI
# =========================
def load_aoi_ee(geojson_path):
    """
    Load AOI from GeoJSON and convert to Earth Engine geometry
    """
    if not os.path.exists(geojson_path):
        print(f"[!] AOI file not found: {geojson_path}")
        return None
    
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Extract geometry
        if 'features' in geojson_data and len(geojson_data['features']) > 0:
            geometry = geojson_data['features'][0]['geometry']
        elif 'geometry' in geojson_data:
            geometry = geojson_data['geometry']
        else:
            geometry = geojson_data
        
        # Convert to Earth Engine geometry
        ee_geometry = ee.Geometry(geometry)
        
        print(f"[+] Loaded AOI from {geojson_path}")
        print(f"    Bounds: {ee_geometry.bounds().getInfo()}")
        
        return ee_geometry
    except Exception as e:
        print(f"[!] Error loading AOI: {e}")
        return None

# =========================
# SEARCH AND LOAD SENTINEL-1
# =========================
def load_sentinel1_collection(aoi, start_date, end_date, orbit='ASCENDING'):
    """
    Load Sentinel-1 GRD collection from Google Earth Engine
    
    Parameters:
    - orbit: 'ASCENDING' or 'DESCENDING'
    """
    print(f"[+] Loading Sentinel-1 GRD data from {start_date} to {end_date}")
    
    collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', orbit))
    
    count = collection.size().getInfo()
    print(f"[+] Found {count} images in collection")
    
    return collection

# =========================
# PREPROCESSING
# =========================
def preprocess_sentinel1(image):
    """
    Preprocess Sentinel-1 image:
    1. Apply speckle filter (Lee filter)
    2. Clip to AOI
    """
    # Lee filter (reduce speckle)
    vv = image.select('VV')
    vh = image.select('VH')
    
    # Apply focal median filter (simple speckle reduction)
    vv_filtered = vv.focal_median(radius=50, kernelType='circle', units='meters')
    vh_filtered = vh.focal_median(radius=50, kernelType='circle', units='meters')
    
    return image.addBands(vv_filtered.rename('VV_filtered')) \
                .addBands(vh_filtered.rename('VH_filtered'))

# =========================
# CALCULATE SOIL SATURATION
# =========================
def calculate_soil_saturation(collection, aoi):
    """
    Calculate soil saturation from VV backscatter
    Uses empirical relationship: lower backscatter = higher moisture
    """
    print("[+] Calculating soil saturation...")
    
    # Get reference dry and wet conditions from collection
    vv_dry = collection.select('VV').max()  # Dry reference (highest backscatter)
    vv_wet = collection.select('VV').min()  # Wet reference (lowest backscatter)
    
    def calc_saturation(image):
        vv = image.select('VV_filtered')
        
        # Normalize: (dry - current) / (dry - wet)
        # Higher values = more saturated
        saturation = vv_dry.subtract(vv).divide(vv_dry.subtract(vv_wet))
        saturation = saturation.clamp(0, 1)  # Limit to 0-1 range
        
        return image.addBands(saturation.rename('soil_saturation'))
    
    return collection.map(calc_saturation)

# =========================
# CALCULATE FLOOD EXTENT
# =========================
def calculate_flood_extent(collection, aoi, threshold=-20):
    """
    Calculate flood extent probability
    Lower VV backscatter = higher flood probability
    """
    print("[+] Calculating flood extent...")
    
    # Reference dry image (median of collection)
    reference_dry = collection.select('VV').median()
    
    def calc_flood(image):
        vv = image.select('VV_filtered')
        
        # Change detection
        change = vv.subtract(reference_dry)
        
        # Thresholds for flood detection (wrap as ee.Image.constant)
        definitely_water = ee.Image.constant(-22)
        definitely_land = ee.Image.constant(-12)
        
        # Calculate probability based on VV value and change
        # Fuzzy membership function
        value_score = definitely_land.subtract(vv)
        value_score = value_score.divide(definitely_land.subtract(definitely_water))
        
        change_score = change.multiply(-1).subtract(1).divide(5)
        change_score = change_score.clamp(0, 1)
        
        # Combine scores
        flood_prob = value_score.add(change_score).divide(2)
        flood_prob = flood_prob.clamp(0, 1)
        
        return image.addBands(flood_prob.rename('flood_extent'))
    
    return collection.map(calc_flood)

# =========================
# CALCULATE WETNESS TREND
# =========================
def calculate_wetness_trend_simple(collection, aoi):
    """
    Simplified wetness trend: compare first and last images
    """
    print("[+] Calculating wetness trend (simple method)...")
    
    count = collection.size().getInfo()
    if count < 2:
        print("[!] Need at least 2 images")
        return None
    
    # Get first and last images
    sorted_collection = collection.sort('system:time_start')
    first_image = ee.Image(sorted_collection.first())
    last_image = ee.Image(sorted_collection.sort('system:time_start', False).first())
    
    # Get dates
    first_date = ee.Date(first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    last_date = ee.Date(last_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    
    print(f"    Comparing: {first_date} → {last_date}")
    
    # Calculate VV change
    vv_first = first_image.select('VV_filtered')
    vv_last = last_image.select('VV_filtered')
    vv_change = vv_last.subtract(vv_first)
    
    # Get mean change over AOI
    change_stats = vv_change.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30,
        maxPixels=1e8,
        bestEffort=True
    ).getInfo()
    
    mean_change = change_stats.get('VV_filtered', 0)
    print(f"    Mean VV change: {mean_change:.3f} dB")
    
    # Classify with 0.5 dB threshold
    # Your data shows -0.75 dB, so this should work!
    if mean_change < -0.5:
        trend_value = 1  # Wetting
        trend_label = "WETTING"
    elif mean_change > 0.5:
        trend_value = -1  # Drying
        trend_label = "DRYING"
    else:
        trend_value = 0  # Stable
        trend_label = "STABLE"
    
    print(f"    Trend classification: {trend_label} ({trend_value})")
    
    return trend_value  # Return scalar, not image

# =========================
# EXPORT TO CSV
# =========================
def export_timeseries_to_csv(collection, aoi, output_file):
    """
    Export time series data to CSV (optimized with error handling)
    """
    print(f"[+] Exporting time series to CSV...")
    
    # Define reducer for statistics
    def add_stats(image):
        stats = image.select(['soil_saturation', 'flood_extent']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e8,
            bestEffort=True
        )
        
        return image.set({
            'soil_sat_mean': stats.get('soil_saturation'),
            'flood_ext_mean': stats.get('flood_extent')
        })
    
    # Apply stats calculation
    print("[+] Computing statistics for all images...")
    collection_with_stats = collection.map(add_stats)
    
    # Batch retrieval with error handling
    def get_features():
        features = collection_with_stats.map(lambda img: ee.Feature(None, {
            # millis since epoch -> full ISO string, e.g. "2026-02-13T00:00:00+00:00"
            'timestamp': ee.Date(img.date().millis()).format("YYYY-MM-dd'T'HH:mm:ss'Z'"),
            'soil_saturation': img.get('soil_sat_mean'),
            'flood_extent':    img.get('flood_ext_mean'),
        }))
        return features.toList(collection.size()).getInfo()
    
    feature_list = safe_compute(get_features, max_retries=3)
    
    if feature_list is None:
        print("[!] Failed to retrieve statistics")
        return pd.DataFrame()
    
    # Convert to results
    results = []
    for feature in feature_list:
        props = feature['properties']
        results.append({
            'timestamp': props['timestamp'] + 'T00:00:00Z',
            'soil_saturation': round(float(props.get('soil_saturation', 0)), 2),
            'flood_extent': round(float(props.get('flood_extent', 0)), 2)
        })
    
    print(f"    Processed {len(results)} timestamps")
    
    # Calculate wetness trend with error handling
    print("[+] Calculating wetness trend...")
    
    def compute_trend():
        # Simple method returns scalar directly
        trend_value = calculate_wetness_trend_simple(collection, aoi)
        
        if trend_value is None:
            return 0
        
        return int(trend_value)
    
    trend_value = safe_compute(compute_trend, max_retries=3)
    
    if trend_value is None:
        print("[!] Using default trend value: 0")
        trend_value = 0
    
    print(f"    Final wetness trend: {trend_value}")
    
    # Add trend to all rows
    for row in results:
        row['wetness_trend'] = trend_value
    
    # Save to CSV
    df = pd.DataFrame(results)
    file_exists = os.path.exists(output_file)
    df.to_csv(output_file, mode='a', index=False, header=not file_exists)
    
    print(f"[+] Saved to {output_file}")
    print(f"\n{df.head(10)}")
    
    return df

# =========================
# EXPORT SPATIAL DATA (OPTIONAL)
# =========================
def export_spatial_data(image, aoi, output_file, description):
    """
    Export full spatial raster to Google Drive
    (Optional: Use this if you need full resolution maps)
    """
    print(f"[+] Exporting spatial data to Google Drive: {description}")
    
    task = ee.batch.Export.image.toDrive(
        image=image.clip(aoi),
        description=description,
        folder='Sentinel1_Exports',
        fileNamePrefix=output_file,
        region=aoi.bounds(),
        scale=10,
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )
    
    task.start()
    print(f"    Task started: {task.status()}")
    print(f"    Monitor at: https://code.earthengine.google.com/tasks")

# =========================
# SAFE COMPUTATION WRAPPER
# =========================
def safe_compute(func, max_retries=3, timeout=300):
    """
    Safely compute with retry logic
    Args:
        func: Function to execute (should return a value)
        max_retries: Number of retry attempts
        timeout: Not used directly, but indicates expected max wait time
    """
    for attempt in range(max_retries):
        try:
            result = func()
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30
                print(f"[!] Computation failed: {e}")
                print(f"    Retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"[!] Max retries reached. Error: {e}")
                return None

# =========================
# MAIN
# =========================

def main():
    print("=" * 80)
    print("SENTINEL-1 PROCESSING WITH GOOGLE EARTH ENGINE (ZERO CREDITS)")
    print("=" * 80 + "\n")
    
    # Initialize
    initialize_gee()
    
    # Load AOI
    print("\n[+] Loading Area of Interest...")
    aoi = load_aoi_ee(AOI_GEOJSON)
    
    if not aoi:
        print("[!] Failed to load AOI")
        exit()
    
    # Define time range
    start_date = '2019-06-01'
    end_date = '2020-01-15'
    
    # Load Sentinel-1 collection
    print(f"\n[+] Loading Sentinel-1 data...")
    collection = load_sentinel1_collection(aoi, start_date, end_date)
    
    # Safe check for collection size
    def check_collection_size():
        return collection.size().getInfo()
    
    collection_size = safe_compute(check_collection_size, max_retries=2)
    
    if collection_size is None or collection_size == 0:
        print("[!] No images found or failed to check collection")
        exit()
    
    print(f"[+] Collection contains {collection_size} images")
    
    # Preprocess
    print("\n[+] Preprocessing images...")
    collection = collection.map(preprocess_sentinel1)
    
    # Calculate metrics
    collection = calculate_soil_saturation(collection, aoi)
    collection = calculate_flood_extent(collection, aoi)
    
    # Export time series to CSV
    output_csv = os.path.join(OUTPUT_DIR, "sentinel1_timeseries.csv")
    df = export_timeseries_to_csv(collection, aoi, output_csv)
    
    # Optional: Export spatial maps
    export_choice = input("\nExport full spatial maps to Google Drive? (y/n): ").lower()
    if export_choice == 'y':
        # Export latest image with all bands
        latest = ee.Image(collection.sort('system:time_start', False).first())
        export_spatial_data(
            latest.select(['soil_saturation', 'flood_extent']),
            aoi,
            f"sentinel1_latest_{end_date}",
            "Sentinel1_Latest_Analysis"
        )
    
    print("\n" + "=" * 80)
    print("[+] Processing complete!")
    print(f"[+] Results saved to: {output_csv}")
    print("=" * 80)

if __name__ == "__main__":
    main()