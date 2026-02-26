import pandas as pd
from datetime import datetime

from models import predictor
from alerts.notifier import notify
from models.calibrator import load_thresholds
from scripts import Sat_SensorData_proxy
from scripts import sentinel1_GEE

def main():
    # -------------------------
    # 1. Sat_SensorData_proxy 
    # -------------------------

    Sat_SensorData_proxy.main()
    # -------------------------
    # 2. EO ingestion (offline / contextual)
    # -------------------------
    sentinel1_GEE.main()

    # -------------------------
    # 3. Load base thresholds
    # -------------------------
    #thresholds = load_thresholds()

    # -------------------------
    # 4. Prediction Algorithm
    # -------------------------
    #predictor.main()

    # -------------------------
    # 5. Notification
    # -------------------------
    notify()


if __name__ == "__main__":
    main()
