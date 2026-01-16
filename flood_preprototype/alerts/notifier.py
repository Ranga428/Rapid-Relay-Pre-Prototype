from datetime import datetime
from pathlib import Path
import csv

# Try importing the EO loader from models.predictor to reuse matching/averaging logic.
try:
    from ..models.predictor import load_eo_features as _external_load_eo
except Exception:
    try:
        from models.predictor import load_eo_features as _external_load_eo
    except Exception:
        _external_load_eo = None


def _format_eo(eo_features):
    if not eo_features:
        return "soil_saturation=N/A | flood_extent=N/A | wetness_trend=N/A"
    soil = eo_features.get("soil_saturation")
    extent = eo_features.get("flood_extent")
    trend = eo_features.get("wetness_trend")

    soil_str = f"{round(float(soil) * 100, 1)}%" if soil is not None else "N/A"
    extent_str = f"{round(float(extent) * 100, 1)}%" if extent is not None else "N/A"
    if trend is None:
        trend_str = "N/A"
    else:
        trend_map = {1: "increasing", 0: "stable", -1: "decreasing"}
        trend_str = trend_map.get(int(trend), str(trend))

    return f"soil_saturation={soil_str} | flood_extent={extent_str} | wetness_trend={trend_str}"


def _load_latest_eo_features(csv_path=None):
    """Load the most recent EO features row from data/sentinel1/eo_features.csv.

    Returns a dict with keys soil_saturation (float), flood_extent (float), wetness_trend (int)
    or None values if the file/values are missing.
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "sentinel1" / "eo_features.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        return {"soil_saturation": None, "flood_extent": None, "wetness_trend": None}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        if not reader:
            return {"soil_saturation": None, "flood_extent": None, "wetness_trend": None}
        last = reader[-1]

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def _to_int(x):
        try:
            return int(float(x))
        except Exception:
            return None

    return {
        "soil_saturation": _to_float(last.get("soil_saturation")),
        "flood_extent": _to_float(last.get("flood_extent")),
        "wetness_trend": _to_int(last.get("wetness_trend")),
    }


def notify(alert, sensor_record, eo_features=None, n_eo_recent=3):
    """
    Handles alert notification.

    If `eo_features` is not provided, the notifier will attempt to load EO
    features. If the predictor loader is available it will request `n_eo_recent`
    rows matching the sensor timestamp (so notifier and predictor can be aligned).
    """

    # If EO features not supplied, attempt to load latest from disk or via predictor loader
    if eo_features is None:
        timestamp = sensor_record.get("timestamp")
        if _external_load_eo is not None:
            try:
                eo_features = _external_load_eo(match_timestamp=timestamp, n_recent=n_eo_recent)
            except Exception:
                eo_features = _load_latest_eo_features()
        else:
            eo_features = _load_latest_eo_features()

    timestamp = sensor_record.get("timestamp")

    eo_str = _format_eo(eo_features)

    if alert:
        print(
            f"[ALERT] {timestamp} | "
            f"Flood risk detected | "
            f"Water Level: {sensor_record.get('water_level')} m | "
            f"Rainfall: {sensor_record.get('rainfall')} mm | "
            f"Humidity: {sensor_record.get('humidity')} % | "
            f"{eo_str}"
        )
    else:
        print(
            f"[OK] {timestamp} | "
            f"Normal conditions | "
            f"Water Level: {sensor_record.get('water_level')} m | "
            f"Rainfall: {sensor_record.get('rainfall')} mm | "
            f"Humidity: {sensor_record.get('humidity')} % | "
            f"{eo_str}"
        )
