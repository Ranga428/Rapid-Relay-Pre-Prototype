import pandas as pd
from datetime import datetime

from scripts.eo_downloader import download_recent_sentinel1
from scripts.eo_features import extract_eo_features
from models.predictor import compute_flood_risk
from alerts.notifier import notify
from scripts import sensor_data as sensor_module
from models.calibrator import load_thresholds

SENSOR_DATA_PATH = "data/sensor/simulated.csv"
LOG_PATH = "logs/events.csv"


def main():
    # -------------------------
    # 1. Generate sensor data
    # -------------------------
    for _ in range(20):
        d = sensor_module.generate_sensor_data()
        sensor_module.save_sensor_data(d)

    # -------------------------
    # 2. EO ingestion (offline / contextual)
    # -------------------------
    eo_files = download_recent_sentinel1(days=0.15, max_items=1)
    for _ in range(20):
        eo_features = extract_eo_features(eo_files)

    # -------------------------
    # 3. Load base thresholds
    # -------------------------
    thresholds = load_thresholds()

    # -------------------------
    # 4. Load sensor data
    # -------------------------
    df = pd.read_csv(SENSOR_DATA_PATH)

    for _, row in df.iterrows():
        sensor_record = {
            "timestamp": row["timestamp"],
            "water_level": row["water_level"],
            "rainfall": row["rainfall"],
            "humidity": row["humidity"],
        }

        # -------------------------
        # 5. Prediction (EO + sensor)
        # -------------------------
        alert = compute_flood_risk(sensor_record, eo_features, thresholds)

        # -------------------------
        # 6. Notify & log
        # -------------------------
        notify(alert, sensor_record)
        log_event(sensor_record, alert)


def log_event(sensor_record, alert):
    with open(LOG_PATH, "a") as f:
        f.write(
            f"{sensor_record['timestamp']},"
            f"{sensor_record['water_level']},"
            f"{sensor_record['rainfall']},"
            f"{sensor_record['humidity']},"
            f"{alert}\n"
        )


if __name__ == "__main__":
    main()
