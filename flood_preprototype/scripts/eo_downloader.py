import os
from datetime import datetime, timedelta
from pystac_client import Client
import requests

EO_PATH = "data/sentinel1/"
AOI_PATH = "config/aoi.geojson"
UPDATE_INTERVAL_DAYS = 7

def internet_available():
    try:
        requests.get("https://earth-search.aws.element84.com", timeout=5)
        return True
    except:
        return False

def last_download_time():
    if not os.path.exists(EO_PATH):
        return datetime.min
    timestamps = [
        os.path.getmtime(os.path.join(EO_PATH, f))
        for f in os.listdir(EO_PATH)
    ]
    if timestamps:
        return datetime.fromtimestamp(max(timestamps))
    return datetime.min

def download_new_sentinel1():
    os.makedirs(EO_PATH, exist_ok=True)
    client = Client.open("https://earth-search.aws.element84.com/v1")
    with open(AOI_PATH) as f:
        aoi_geojson = f.read()

    items = client.search(
        collections=["sentinel-1-grd"],
        intersects=aoi_geojson,
        limit=1
    ).get_all_items()

    for item in items:
        for asset in item.assets.values():
            filename = os.path.join(EO_PATH, os.path.basename(asset.href))
            if not os.path.exists(filename):
                print(f"Downloading {filename}...")
                r = requests.get(asset.href)
                with open(filename, "wb") as f:
                    f.write(r.content)
            else:
                print(f"{filename} already exists.")

def update_eo_data():
    if not internet_available():
        print("No internet. Using existing EO data.")
        return
    last_time = last_download_time()
    if datetime.now() - last_time >= timedelta(days=UPDATE_INTERVAL_DAYS):
        download_new_sentinel1()
    else:
        print("EO data is up-to-date.")
