import os
import json
import requests
from datetime import datetime
from zipfile import ZipFile
from shapely.geometry import shape
from shapely import wkt

# =========================
# CONFIGURATION
# =========================
AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"

CLIENT_ID = "sh-4f6413ea-0fe8-49c0-8fb7-d68cf1b4d91c"
CLIENT_SECRET = "vgsiagBHBhqpjU4kLpVP9QWu0seEf7LW"

DOWNLOAD_DIR = os.path.join("..", "data", "sentinel1", "Sat-Storage-CDSE")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Path to AOI GeoJSON file
AOI_GEOJSON = os.path.join("..", "config", "aoi.geojson")

# =========================
# AUTHENTICATION
# =========================
def get_access_token():
    """
    Get OAuth2 access token using client credentials
    """
    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    r = requests.post(AUTH_URL, data=data)
    r.raise_for_status()
    token = r.json()["access_token"]
    print("[+] Access token obtained")
    return token

# =========================
# LOAD AOI FROM GEOJSON
# =========================
def load_aoi_wkt(geojson_path):
    """
    Load Area of Interest from GeoJSON file and convert to WKT
    Returns WKT string for OData query
    """
    if not os.path.exists(geojson_path):
        print(f"[!] AOI file not found: {geojson_path}")
        return None
    
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Extract geometry from first feature
        if 'features' in geojson_data and len(geojson_data['features']) > 0:
            geometry = geojson_data['features'][0]['geometry']
        elif 'geometry' in geojson_data:
            geometry = geojson_data['geometry']
        else:
            geometry = geojson_data
        
        # Convert to shapely geometry then to WKT
        geom = shape(geometry)
        wkt_string = geom.wkt
        
        print(f"[+] Loaded AOI from {geojson_path}")
        print(f"    Geometry type: {geom.geom_type}")
        print(f"    Bounds: {geom.bounds}")
        
        return wkt_string
    except Exception as e:
        print(f"[!] Error loading AOI: {e}")
        return None

# =========================
# SEARCH SENTINEL-1
# =========================
def search_sentinel1(token, aoi_wkt=None, start_date="2024-01-01", end_date="2024-01-15", 
                     product_type="GRD", max_results=5):
    """
    Search for Sentinel-1 products using OData API
    
    Parameters:
    - aoi_wkt: WKT string for area of interest
    - product_type: 'GRD' or 'SLC'
    - start_date/end_date: format 'YYYY-MM-DD'
    """
    headers = {"Authorization": f"Bearer {token}"}
    
    # Build OData filter query
    filters = [
        "Collection/Name eq 'SENTINEL-1'",
        f"contains(Name,'{product_type}')",
        f"ContentDate/Start ge {start_date}T00:00:00.000Z",
        f"ContentDate/Start le {end_date}T23:59:59.999Z"
    ]
    
    # Add spatial filter if AOI provided
    if aoi_wkt:
        # OData spatial query format
        spatial_filter = f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')"
        filters.append(spatial_filter)
        print("[+] Searching within AOI boundaries")
    else:
        print("[!] No AOI provided - searching globally")
    
    query = " and ".join(filters)
    
    params = {
        "$filter": query,
        "$orderby": "ContentDate/Start desc",
        "$top": max_results,
        "$expand": "Attributes"
    }
    
    print(f"[+] Executing search query...")
    r = requests.get(CATALOG_URL, headers=headers, params=params)
    r.raise_for_status()
    
    results = r.json()
    products = results.get("value", [])
    
    print(f"[+] Found {len(products)} products\n")
    
    if products:
        print("Products:")
        print("-" * 80)
        for i, p in enumerate(products, 1):
            print(f"{i}. {p['Name']}")
            print(f"   Date: {p['ContentDate']['Start']}")
            size_mb = int(p.get('ContentLength', 0)) / (1024 * 1024)
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   ID: {p['Id']}")
            print()
    
    return products

# =========================
# DOWNLOAD PRODUCT
# =========================
def download_product(product_id, product_name):
    """
    Download a single product
    Note: Downloads require a fresh token for each request
    """
    # Clean filename
    filename = product_name.replace('/', '_').replace('\\', '_') + ".zip"
    filepath = os.path.join(DOWNLOAD_DIR, filename)
    
    # Check if already downloaded
    if os.path.exists(filepath):
        print(f"[+] Already downloaded: {filename}")
        return filepath
    
    # Get fresh token for download
    print(f"[+] Getting fresh token for download...")
    token = get_access_token()
    
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{DOWNLOAD_URL}({product_id})/$value"
    
    print(f"[+] Downloading: {filename}")
    
    try:
        with requests.get(url, headers=headers, stream=True, timeout=300) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            total_mb = total_size / (1024 * 1024)
            
            with open(filepath, "wb") as f:
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress indicator
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            downloaded_mb = downloaded / (1024 * 1024)
                            print(f"\r    Progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='')
        
        print(f"\n[+] Downloaded to: {filepath}")
        return filepath
        
    except requests.exceptions.HTTPError as e:
        print(f"\n[!] HTTP Error downloading {filename}: {e}")
        print(f"    Status code: {e.response.status_code}")
        if e.response.status_code == 401:
            print("    Try regenerating your client credentials at https://dataspace.copernicus.eu/")
        if os.path.exists(filepath):
            os.remove(filepath)  # Remove partial download
        return None
    except Exception as e:
        print(f"\n[!] Error downloading {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)  # Remove partial download
        return None

# =========================
# EXTRACT PRODUCTS
# =========================
def extract_safe_files(filepaths):
    """
    Extract .zip files to .SAFE directories
    """
    for filepath in filepaths:
        if filepath and filepath.endswith('.zip'):
            extract_dir = filepath.replace('.zip', '.SAFE')
            if not os.path.exists(extract_dir):
                print(f"[+] Extracting {os.path.basename(filepath)}...")
                try:
                    with ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(os.path.dirname(filepath))
                    print(f"[+] Extracted to {extract_dir}")
                except Exception as e:
                    print(f"[!] Error extracting {filepath}: {e}")
            else:
                print(f"[+] Already extracted: {extract_dir}")

# =========================
# GET PRODUCT INFO
# =========================
def get_product_info(safe_path):
    """
    Extract basic information from SAFE directory
    """
    if os.path.isdir(safe_path):
        print(f"\n[+] Product: {os.path.basename(safe_path)}")
        
        # Look for measurement files
        measurement_dir = os.path.join(safe_path, 'measurement')
        if os.path.exists(measurement_dir):
            tiff_files = [f for f in os.listdir(measurement_dir) if f.endswith('.tiff')]
            print(f"    Measurement files: {len(tiff_files)}")
            for tiff in tiff_files:
                print(f"      - {tiff}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("[+] Getting OAuth access token...")
    token = get_access_token()
    
    print("\n[+] Loading Area of Interest...")
    aoi_wkt = load_aoi_wkt(AOI_GEOJSON)
    
    print("\n[+] Searching for Sentinel-1 GRD products...")
    products = search_sentinel1(
        token,
        aoi_wkt=aoi_wkt,
        start_date="2024-01-01",
        end_date="2024-01-15",
        product_type="GRD",
        max_results=5
    )
    
    if not products:
        print("[!] No products to download")
        exit()
    
    print("=" * 80)
    print("[+] Starting downloads...")
    print("=" * 80 + "\n")
    
    downloaded_files = []
    for i, product in enumerate(products, 1):
        print(f"\nProduct {i}/{len(products)}")
        print("-" * 80)
        filepath = download_product(product['Id'], product['Name'])
        if filepath:
            downloaded_files.append(filepath)
        print()
    
    print("=" * 80)
    print(f"[+] Successfully downloaded {len(downloaded_files)}/{len(products)} products")
    print("=" * 80 + "\n")
    
    # Optional: Extract the .zip files
    if downloaded_files:
        extract_choice = input("Extract .SAFE files? (y/n): ").lower()
        if extract_choice == 'y':
            print()
            extract_safe_files(downloaded_files)
            
            # Show info about extracted files
            print("\n[+] Extracted products info:")
            for filepath in downloaded_files:
                safe_path = filepath.replace('.zip', '.SAFE')
                if os.path.exists(safe_path):
                    get_product_info(safe_path)
    
    print("\n[+] All done!")
    print(f"[+] Files saved to: {DOWNLOAD_DIR}")