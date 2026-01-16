from pystac_client import Client
from datetime import datetime, timedelta
import os

EO_DIR = "data/sentinel1/"
COLLECTION = "sentinel-1-grd"

def download_recent_sentinel1(days=0.1, max_items=1):
    os.makedirs(EO_DIR, exist_ok=True)

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    catalog = Client.open("https://earth-search.aws.element84.com/v1")

    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start.isoformat()}Z/{end.isoformat()}Z",
        limit=max_items
    )

    files = []

    for item in search.items():
        fname = f"{item.id}.txt"
        path = os.path.join(EO_DIR, fname)

        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(f"ID: {item.id}\n")
                f.write(f"Datetime: {item.datetime}\n")

        files.append(path)

    return files
