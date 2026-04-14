import ee
import json
import os
import requests

# 1. Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# 2. File Paths
geojson_path = r'D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\config\HMS-HEC.geojson'
output_dir = r'D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\HMS-HEC'
output_filename = 'Valenzuela_DEM.tif'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Load GeoJSON and Convert to GEE Geometry
with open(geojson_path) as f:
    data = json.load(f)
    # The HMS-HEC.geojson contains a LineString, we create a bounding box (buffer) around it
    coords = data['features'][0]['geometry']['coordinates']
    line = ee.Geometry.LineString(coords)
    # We create a 500-meter buffer around the reach to capture the floodplain
    roi = line.buffer(500).bounds()

# 4. Fetch ALOS World 3D - 30m (JAXA)
# This is better for the Philippines than NASA SRTM
dem = ee.Image("JAXA/ALOS/AW3D30/V3_2").select('AVE_DSM')

# 5. Clip and Get Download URL
clipped_dem = dem.clip(roi)

print("Generating download link for Valenzuela Reach...")
url = clipped_dem.getDownloadURL({
    'scale': 30,
    'crs': 'EPSG:4326',
    'format': 'GEO_TIFF',
    'region': roi
})

print(f"\nSUCCESS! Download your .tif file here:\n{url}")
print(f"\nAfter downloading, move the file to: {output_dir}")