import random
import csv
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

def generate_sensor_data():
    water_level = random.uniform(20, 80)
    rainfall = random.uniform(0, 30)
    humidity = random.uniform(50, 98)
    return {
        "water_level": water_level,
        "rainfall": rainfall,
        "humidity": humidity
    }

# CSV file to save simulated sensor readings (in project's data/sensor folder)
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sensor" / "simulated.csv"


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


def load_sensor_records(file_path=None):
    """Read simulated sensor CSV and return a list of normalized sensor_record dicts.

    Normalizes column names (strip/lower), fills missing columns with defaults and
    converts numeric fields to floats.
    """
    if file_path is None:
        file_path = CSV_PATH
    else:
        file_path = Path(file_path)

    # If file missing, return empty list
    if not Path(file_path).exists():
        return []

    df = pd.read_csv(file_path)
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Ensure expected columns exist
    for col in ("timestamp", "water_level", "rainfall", "humidity"):
        if col not in df.columns:
            df[col] = None

    records = []
    for _, row in df.iterrows():
        records.append({
            "timestamp": row.get("timestamp"),
            "water_level": float(row.get("water_level") or 0),
            "rainfall": float(row.get("rainfall") or 0),
            "humidity": float(row.get("humidity") or 0),
        })

    return records

if __name__ == "__main__":
    d = generate_sensor_data()
    save_sensor_data(d)
    print(f"Saved sensor data to {CSV_PATH}")
