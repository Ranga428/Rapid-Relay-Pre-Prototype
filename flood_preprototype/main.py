import pandas as pd
from datetime import datetime
from scripts.eo_downloader import update_eo_data

from models.calibrator import load_thresholds, calibrate_thresholds
from models.predictor import predict
from alerts.notifier import notify
from data.sensor import sensor_data

SENSOR_DATA_PATH = "data/sensor/simulated.csv"
LOG_PATH = "logs/events.csv"

def main():
    #for i in range(10):
    
    sensor_data.main()

    update_eo_data()

    # Load and calibrate thresholds (EO-informed, offline)
    base_thresholds = load_thresholds()
    thresholds = calibrate_thresholds(base_thresholds)

    # Load sensor data (simulated or real)
    df = pd.read_csv(SENSOR_DATA_PATH)

    for _, row in df.iterrows():
        sensor_data = {
            "timestamp": row["timestamp"],
            "water_level": row["water_level"],
            "rainfall": row["rainfall"],
            "humidity": row["humidity"],
        }

        alert = predict(sensor_data, thresholds)
        notify(alert, sensor_data)

        log_event(sensor_data, alert)

def log_event(sensor_data, alert):
    with open(LOG_PATH, "a") as f:
        f.write(
            f"{sensor_data['timestamp']},"
            f"{sensor_data['water_level']},"
            f"{sensor_data['rainfall']},"
            f"{sensor_data['humidity']},"
            f"{alert}\n"
        )

if __name__ == "__main__":
    main()
