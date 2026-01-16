from pathlib import Path
import yaml
import csv
from datetime import datetime

# Load thresholds from config/thresholds.yaml relative to the project package root
_THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"
with _THRESHOLDS_PATH.open("r", encoding="utf-8") as _f:
    thresholds = yaml.safe_load(_f)


def _parse_iso(iso_str):
    """Parse ISO 8601 string to datetime, handle Z and +00:00 UTC offsets."""
    if iso_str is None:
        return None
    try:
        # Replace 'Z' with '+00:00' for consistent parsing
        if iso_str.endswith("Z"):
            iso_str = iso_str[:-1] + "+00:00"
        return datetime.fromisoformat(iso_str)
    except Exception:
        return None


def load_eo_features(csv_path=None, match_timestamp=None, n_recent=1, max_delta_seconds=60):
    """Load EO features from CSV.

    If match_timestamp is provided (ISO string or datetime), the function will return up to
    `n_recent` rows closest to that timestamp. Rows within `max_delta_seconds` are preferred;
    if none fall within the tolerance the nearest rows are returned.
    If match_timestamp is None, the function returns the last `n_recent` rows (averaged).
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

    # Parse timestamps and convert rows
    rows = []
    for r in reader:
        ts = r.get("timestamp")
        try:
            ts_dt = _parse_iso(ts)
        except Exception:
            ts_dt = None
        rows.append({
            "timestamp": ts_dt,
            "soil_saturation": _try_float(r.get("soil_saturation")),
            "flood_extent": _try_float(r.get("flood_extent")),
            "wetness_trend": _try_int(r.get("wetness_trend")),
        })

    def select_rows():
        if match_timestamp:
            # normalize match_timestamp to datetime
            if isinstance(match_timestamp, str):
                try:
                    target = _parse_iso(match_timestamp)
                except Exception:
                    target = None
            elif isinstance(match_timestamp, datetime):
                target = match_timestamp
            else:
                target = None

            if target is None:
                return rows[-n_recent:]

            # compute delta for rows with timestamps
            rows_with_delta = []
            for r in rows:
                if r["timestamp"] is None:
                    continue
                delta = abs((r["timestamp"] - target).total_seconds())
                rows_with_delta.append((delta, r))

            if not rows_with_delta:
                return rows[-n_recent:]

            # Prefer rows within tolerance
            within_tol = [r for d, r in rows_with_delta if d <= max_delta_seconds]
            if within_tol:
                # sort by proximity and return up to n_recent
                within_with_delta = [(d, r) for d, r in rows_with_delta if d <= max_delta_seconds]
                within_with_delta.sort(key=lambda x: x[0])
                return [r for _, r in within_with_delta[:n_recent]]

            # fallback: return nearest rows by delta
            rows_with_delta.sort(key=lambda x: x[0])
            return [r for _, r in rows_with_delta[:n_recent]]
        else:
            # last N rows
            return rows[-n_recent:]

    chosen = select_rows()
    if not chosen:
        return {"soil_saturation": None, "flood_extent": None, "wetness_trend": None}

    # average soil and flood, and take rounded mean for trend
    soil_vals = [r["soil_saturation"] for r in chosen if r["soil_saturation"] is not None]
    flood_vals = [r["flood_extent"] for r in chosen if r["flood_extent"] is not None]
    trend_vals = [r["wetness_trend"] for r in chosen if r["wetness_trend"] is not None]

    soil_avg = sum(soil_vals) / len(soil_vals) if soil_vals else None
    flood_avg = sum(flood_vals) / len(flood_vals) if flood_vals else None
    trend_avg = int(round(sum(trend_vals) / len(trend_vals))) if trend_vals else None

    return {"soil_saturation": soil_avg, "flood_extent": flood_avg, "wetness_trend": trend_avg}


# small helpers used above
def _try_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _try_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


def _delta_from_target(ts, target):
    try:
        return abs((ts - target).total_seconds())
    except Exception:
        return float('inf')


def load_latest_eo_features(csv_path=None):
    """Return the most recent EO features from data/sentinel1/eo_features.csv.

    If the CSV is missing or empty, returns a dict with None values.
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


def compute_flood_risk(sensor_data, eo_features=None, thresholds=thresholds):
    """
    Combines real-time sensor data with EO-derived context
    to compute a flood risk index between 0 and 1.

    If `eo_features` is not provided, the function will attempt to load
    EO features from `data/sentinel1/eo_features.csv` matching sensor timestamp
    (if available) or the most recent entry.
    """
    # If EO features not supplied, try to match by sensor timestamp
    if eo_features is None:
        eo_features = load_eo_features(match_timestamp=sensor_data.get("timestamp"), n_recent=1)

    # -------------------------
    # 1. Sensor-based baseline
    # -------------------------
    water_level_score = min(sensor_data["water_level"] / thresholds["water_level_m"], 1.0)
    rainfall_score = min(sensor_data["rainfall"] / thresholds["rainfall_mm"], 1.0)
    humidity_score = min(sensor_data["humidity"] / 100, 1.0)

    sensor_index = (
        0.4 * rainfall_score +
        0.3 * humidity_score +
        0.3 * water_level_score
    )

    # -------------------------
    # 2. EO-based scores (normalized to 0-1)
    # -------------------------
    # Safely extract EO values (None -> treated as 0)
    soil = eo_features.get("soil_saturation")
    flood = eo_features.get("flood_extent")
    trend = eo_features.get("wetness_trend")

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    # soil_saturation is already in 0-1 range
    soil_score = min(max(_safe_float(soil), 0.0), 1.0)

    # Normalize flood_extent using configured flood_index threshold as a reference
    flood_norm = thresholds.get("flood_index", 0.75) or 1.0
    flood_score = min(max(_safe_float(flood) / float(flood_norm), 0.0), 1.0)

    # Map wetness_trend (-1,0,1) to [0,1]
    if trend is None:
        trend_score = 0.0
    else:
        try:
            trend_int = int(float(trend))
            trend_score = {1: 1.0, 0: 0.5, -1: 0.0}.get(trend_int, 0.0)
        except Exception:
            trend_score = 0.0

    # Combine EO features into an eo_index using the same three-feature structure
    eo_index = (
        0.4 * soil_score +
        0.3 * flood_score +
        0.3 * trend_score
    )

    # -------------------------
    # 3. Final flood risk: combine sensor and EO indices
    # -------------------------
    # Give equal weight to sensor-derived index and EO-derived index so both
    # contribute equally (adjust weights here if needed)
    flood_risk = min(0.5 * sensor_index + 0.5 * eo_index, 1.0)

    return flood_risk
