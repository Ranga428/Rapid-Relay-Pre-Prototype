import random
import csv
from pathlib import Path
from datetime import datetime, timezone

# CSV file for EO features (data/sentinel1/eo_features.csv)
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sentinel1" / "eo_features.csv"


def extract_eo_features(save_csv=True):
    """
    Placeholder EO-derived indicators (replace later with real EO analytics).

    Generates soil_saturation, flood_extent and wetness_trend and optionally
    appends them to the project's EO CSV using the same pattern as sensor_data.
    """

    # Generate random proxies
    soil_saturation_index = round(random.uniform(0.6, 0.9), 2)
    flood_extent_ratio = round(random.uniform(0.0, 0.4), 2)
    wetness_trend = random.choice([-1, 0, 1])

    features = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "soil_saturation": soil_saturation_index,
        "flood_extent": flood_extent_ratio,
        "wetness_trend": wetness_trend,
    }

    if save_csv:
        save_eo_features(features)

    return features


def save_eo_features(data, file_path=None):
    """Append an EO feature dict to CSV. Writes header if file is new or empty."""
    if file_path is None:
        file_path = CSV_PATH
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

    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_eo_records(file_path=None):
    """Read EO CSV and return a list of normalized feature dicts.

    Normalizes column names and converts numeric fields.
    """
    if file_path is None:
        file_path = CSV_PATH
    else:
        file_path = Path(file_path)

    if not Path(file_path).exists():
        return []

    records = []
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append({
                "timestamp": r.get("timestamp"),
                "soil_saturation": try_float(r.get("soil_saturation")),
                "flood_extent": try_float(r.get("flood_extent")),
                "wetness_trend": try_int(r.get("wetness_trend")),
            })
    return records


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

if __name__ == "__main__":
    # quick test: generate one row
    f = extract_eo_features(save_csv=True)
    print(f"Saved EO features to {CSV_PATH}: {f}")

