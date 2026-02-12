import pandas as pd
from datetime import datetime

from scripts.eo_downloader import download_recent_sentinel1
from models import predictor
from alerts.notifier import notify
from models.calibrator import load_thresholds
from scripts import generate_mock_data
from scripts import sentinel1_GEE

def main():
    # -------------------------
    # 1. Generate mock data
    # -------------------------
    for _ in range(10):
        generate_mock_data.main()

    # -------------------------
    # 2. EO ingestion (offline / contextual)
    # -------------------------
    sentinel1_GEE.main()

    # -------------------------
    # 3. Load base thresholds
    # -------------------------
    thresholds = load_thresholds()

    # -------------------------
    # 4. Prediction Algorithm
    # -------------------------
    predictor.main()

    # -------------------------
    # 5. Notification
    # -------------------------
    notify()


if __name__ == "__main__":
    main()
