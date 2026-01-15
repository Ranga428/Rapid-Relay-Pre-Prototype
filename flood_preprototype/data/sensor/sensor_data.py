import random
import csv
from pathlib import Path
from datetime import datetime, timezone

def generate_sensor_data():
    water_level = random.uniform(20, 80)
    rainfall = random.uniform(0, 30)
    humidity = random.uniform(50, 98)
    return {
        "water_level": water_level,
        "rainfall": rainfall,
        "humidity": humidity
    }

# CSV file to save simulated sensor readings (same folder as this script)
CSV_PATH = Path(__file__).resolve().parent / "simulated.csv"


def save_sensor_data(data, file_path=None):
    """Append a sensor data dict to CSV. Writes header if file is new or empty."""
    if file_path is None:
        file_path = CSV_PATH
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "water_level", "rainfall", "humidity"]
    write_header = not file_path.exists() or file_path.stat().st_size == 0

    # Ensure values include a timestamp and are numeric-friendly
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "water_level": round(float(data.get("water_level", 0)), 3),
        "rainfall": round(float(data.get("rainfall", 0)), 3),
        "humidity": round(float(data.get("humidity", 0)), 3)
    }

    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    d = generate_sensor_data()
    save_sensor_data(d)
    print(f"Saved sensor data to {CSV_PATH}")
