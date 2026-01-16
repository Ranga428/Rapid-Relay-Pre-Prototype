import pandas as pd
from datetime import datetime
from scripts.eo_downloader import download_recent_sentinel1

from models.calibrator import load_thresholds, calibrate_thresholds
from models.predictor import predict
from alerts.notifier import notify
from scripts import sensor_data as sensor_module

SENSOR_DATA_PATH = "data/sensor/simulated.csv"
LOG_PATH = "logs/events.csv"

def main():
    for i in range(20):
        d = sensor_module.generate_sensor_data()
        sensor_module.save_sensor_data(d)

    
    download_recent_sentinel1()

    # Load and calibrate thresholds (EO-informed, offline)
    base_thresholds = load_thresholds()
    thresholds = calibrate_thresholds(base_thresholds)

    # Load sensor data (simulated or real)
    df = pd.read_csv(SENSOR_DATA_PATH)

    for _, row in df.iterrows():
        sensor_record = {
            "timestamp": row["timestamp"],
            "water_level": row["water_level"],
            "rainfall": row["rainfall"],
            "humidity": row["humidity"],
        }

        alert = predict(sensor_record, thresholds)
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
