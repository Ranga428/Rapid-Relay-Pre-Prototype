import random
import csv
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

def generate_sensor_data(timestamp=None):
    """Generate a sensor data dict. If `timestamp` (ISO string) is provided it will
    be included in the returned dict so callers can synchronize rows across files.
    """
    water_level = random.uniform(20, 80)
    rainfall = random.uniform(0, 30)
    humidity = random.uniform(50, 98)
    return {
        "timestamp": timestamp,
        "water_level": water_level,
        "rainfall": rainfall,
        "humidity": humidity
    }


def extract_eo_features(save_csv=True, timestamp=None):
    """
    Placeholder EO-derived indicators.

    If `timestamp` is provided (ISO string) it will be used for the row so the
    caller can produce synchronized sensor + EO rows.
    """

    # Generate random proxies
    soil_saturation_index = round(random.uniform(0.6, 0.9), 2)
    flood_extent_ratio = round(random.uniform(0.0, 0.4), 2)
    wetness_trend = random.choice([-1, 0, 1])

    features = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "soil_saturation": soil_saturation_index,
        "flood_extent": flood_extent_ratio,
        "wetness_trend": wetness_trend,
    }

    if save_csv:
        save_eo_features(features)

    return features

# CSV file to save simulated sensor readings (in project's data/sensor folder)
sensor_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sensor" / "simulated.csv"
sentinel_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sentinel1" / "eo_features.csv"

def _ensure_trailing_newline(path: Path):
    """Ensure the file at `path` ends with a newline. Creates parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return
    try:
        size = path.stat().st_size
        if size == 0:
            return
        with path.open("rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
        if last not in (b"\n", b"\r"):
            # append a newline so CSV writer starts a fresh line
            with path.open("a", encoding="utf-8", newline="") as f:
                f.write("\n")
    except Exception:
        # best effort; if this fails we don't block writing
        return


def save_sensor_data(data, file_path=None):
    """Append a sensor data dict to CSV. Writes header if file is new or empty.

    If `data` contains a `timestamp` it will be used; otherwise current UTC time will be added.
    """
    if file_path is None:
        file_path = sensor_CSV_PATH
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "water_level", "rainfall", "humidity"]
    write_header = not file_path.exists() or file_path.stat().st_size == 0

    # Use provided timestamp if present otherwise generate one now
    ts = data.get("timestamp") or datetime.now(timezone.utc).isoformat()

    row = {
        "timestamp": ts,
        "water_level": round(float(data.get("water_level", 0)), 3),
        "rainfall": round(float(data.get("rainfall", 0)), 3),
        "humidity": round(float(data.get("humidity", 0)), 3)
    }

    # If file exists and is non-empty ensure it ends with a newline to avoid overlap
    if not write_header:
        _ensure_trailing_newline(file_path)

    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_eo_features(data, file_path=None):
    """Append an EO feature dict to CSV. Writes header if file is new or empty."""
    if file_path is None:
        file_path = sentinel_CSV_PATH
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "soil_saturation", "flood_extent", "wetness_trend"]
    write_header = not file_path.exists() or file_path.stat().st_size == 0

    row = {
        "timestamp": data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "soil_saturation": round(float(data.get("soil_saturation", 0)), 3),
        "flood_extent": round(float(data.get("flood_extent", 0)), 3),
        "wetness_trend": int(data.get("wetness_trend", 0)),
    }

    if not write_header:
        _ensure_trailing_newline(file_path)

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
        file_path = sensor_CSV_PATH
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

    sensor_records = []
    for _, row in df.iterrows():
        sensor_records.append({
            "timestamp": row.get("timestamp"),
            "water_level": float(row.get("water_level") or 0),
            "rainfall": float(row.get("rainfall") or 0),
            "humidity": float(row.get("humidity") or 0),
        })

    return sensor_records

def load_eo_records(file_path=None):
    """Read EO CSV and return a list of normalized feature dicts.
    Normalizes column names and converts numeric fields.
    """
    if file_path is None:
        file_path = sentinel_CSV_PATH
    else:
        file_path = Path(file_path)

    if not Path(file_path).exists():
        return []

    sentinel_records = []
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sentinel_records.append({
                "timestamp": r.get("timestamp"),
                "soil_saturation": try_float(r.get("soil_saturation")),
                "flood_extent": try_float(r.get("flood_extent")),
                "wetness_trend": try_int(r.get("wetness_trend")),
            })
    return sentinel_records

def try_float(x):
    try:
        return float(x)
    except Exception:
        return None


def try_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

def main():
    # create a single synchronized timestamp and use it for both outputs
    common_ts = datetime.now(timezone.utc).isoformat()
    d = generate_sensor_data(timestamp=common_ts)
    save_sensor_data(d)
    print(f"Saved sensor data to {sensor_CSV_PATH}: {d}")

    f = extract_eo_features(save_csv=True, timestamp=common_ts)
    print(f"Saved EO features to {sentinel_CSV_PATH}: {f}")

if __name__ == "__main__":
    main()
