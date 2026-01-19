from pathlib import Path
import yaml
import csv
from datetime import datetime, timezone

# Load thresholds from config/thresholds.yaml relative to the project package root
_THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"
with _THRESHOLDS_PATH.open("r", encoding="utf-8") as _f:
    thresholds = yaml.safe_load(_f)


def _parse_iso(iso_str):
    """Parse ISO 8601 string to an aware datetime in UTC.

    Returns None if parsing fails. If the string has no timezone info it is
    assumed to be UTC.
    """
    if iso_str is None:
        return None
    try:
        # Replace 'Z' with '+00:00' for consistent parsing
        if iso_str.endswith("Z"):
            iso_str = iso_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso_str)
        # If naive, assume UTC
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        # Convert aware datetimes to UTC
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def load_eo_features(csv_path=None, match_timestamp=None, n_recent=None, max_delta_seconds=60):
    """Load EO features from CSV.

    Behavior changes:
    - If `n_recent` is None (the default), the function will use the whole CSV when
      `match_timestamp` is None, or all rows within `max_delta_seconds` when a
      `match_timestamp` is provided (falls back to nearest rows if none within tol).
    - If `n_recent` is an int, it behaves as before (selects that many recent/nearest rows).
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
        # If no timestamp provided
        if not match_timestamp:
            if n_recent is None:
                # use whole CSV
                return rows
            # else last n_recent rows
            return rows[-n_recent:]

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
            # fallback to last n_recent or all
            return rows if n_recent is None else rows[-n_recent:]

        # compute delta for rows with timestamps
        rows_with_delta = []
        for r in rows:
            if r["timestamp"] is None:
                continue
            delta = abs((r["timestamp"] - target).total_seconds())
            rows_with_delta.append((delta, r))

        if not rows_with_delta:
            return rows if n_recent is None else rows[-n_recent:]

        # If n_recent is None - prefer all rows within tolerance, otherwise use nearest n
        if n_recent is None:
            within_tol = [r for d, r in rows_with_delta if d <= max_delta_seconds]
            if within_tol:
                # sort by proximity
                within_with_delta = [(d, r) for d, r in rows_with_delta if d <= max_delta_seconds]
                within_with_delta.sort(key=lambda x: x[0])
                return [r for _, r in within_with_delta]
            # fallback: return all rows sorted by proximity
            rows_with_delta.sort(key=lambda x: x[0])
            return [r for _, r in rows_with_delta]

        # n_recent is an int -> return nearest n_recent rows
        rows_with_delta.sort(key=lambda x: x[0])
        return [r for _, r in rows_with_delta[:n_recent]]

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


def compute_alert_level(flood_risk, thresholds=thresholds):
    """Map flood_risk (0-1) to an alert level string.

    Uses thresholds from config if available, otherwise defaults:
      - RED: risk >= 0.75
      - YELLOW: 0.4 <= risk < 0.75
      - GREEN: risk < 0.4
    """
    red_th = thresholds.get("alert_red", 0.75)
    yellow_th = thresholds.get("alert_yellow", 0.4)

    if flood_risk >= red_th:
        return "RED"
    if flood_risk >= yellow_th:
        return "YELLOW"
    return "GREEN"


def compute_flood_risk(sensor_data, eo_features=None, thresholds=thresholds):
    """
    Combines real-time sensor data with EO-derived context
    to compute a flood risk index between 0 and 1.

    If `eo_features` is not provided, the function will attempt to load
    EO features from `data/sentinel1/eo_features.csv` matching sensor timestamp
    (if available) or aggregate the whole file.
    """
    # If EO features not supplied, try to match by sensor timestamp and aggregate
    if eo_features is None:

        # Previously matched by sensor timestamp which often returned just the latest row.
        # Use the whole CSV by default so EO statistics reflect the dataset rather than a single row.
        eo_features = load_eo_features(match_timestamp=None, n_recent=None)

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

    # Normalize/prepare final sensor record (ensure timestamp parsed if possible)
    final_sensor = dict(sensor_data)
    try:
        parsed_ts = _parse_iso(sensor_data.get("timestamp"))
        final_sensor["timestamp"] = parsed_ts.isoformat() if parsed_ts is not None else sensor_data.get("timestamp")
    except Exception:
        final_sensor["timestamp"] = sensor_data.get("timestamp")

    # Prepare final EO features used in the decision (use values extracted above)
    final_eo = {
        "soil_saturation": soil,
        "flood_extent": flood,
        "wetness_trend": trend,
    }

    # return structured result with all finalized data; no printing here
    return {
        "flood_risk": flood_risk,
        "alert_level": compute_alert_level(flood_risk, thresholds),
        "eo_features": final_eo,
        "sensor_record": final_sensor,
    }


def log_event(result=None, log_path=None, sensor_csv=None, eo_csv=None):
    """Create a merged events CSV using all rows from the sensor and EO CSVs.

    - If called during runtime with a single `result` (previous behaviour), this
      function will still produce the merged CSV (it reads the source CSVs) so
      the `result` parameter is optional and ignored for merging.
    - Matching is done only on exact timestamp string (normalized to ISO when
      possible). If a timestamp exists only in one file the other columns are
      left blank for that row.
    """
    # defaults for source CSVs
    if sensor_csv is None:
        sensor_csv = Path(__file__).resolve().parents[1] / "data" / "sensor" / "simulated.csv"
    else:
        sensor_csv = Path(sensor_csv)
    if eo_csv is None:
        eo_csv = Path(__file__).resolve().parents[1] / "data" / "sentinel1" / "eo_features.csv"
    else:
        eo_csv = Path(eo_csv)

    if log_path is None:
        log_path = Path(__file__).resolve().parents[1] / "logs" / "events.csv"
    else:
        log_path = Path(log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # helper to read CSV and normalize timestamps
    def _read_rows(path):
        rows = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        for i, r in enumerate(reader):
            # Accept multiple possible timestamp column names and strip values
            raw_ts = (r.get("timestamp") or r.get("Timestamp") or r.get("time") or r.get("ts") or "").strip()
            # if the CSV writer accidentally wrote empty strings, normalize to empty
            if raw_ts == "":
                parsed = None
                iso_ts = ""
            else:
                parsed = _parse_iso(raw_ts)
                iso_ts = parsed.isoformat() if parsed is not None else raw_ts
            # normalize numeric fields to stripped strings (we'll parse later)
            normalized_raw = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            rows.append({"raw": normalized_raw, "iso_ts": iso_ts, "parsed": parsed, "index": i})
        return rows

    sensor_rows = _read_rows(sensor_csv)
    eo_rows = _read_rows(eo_csv)

    # build maps keyed by iso_ts (exact match). If iso_ts is empty string still use it as key
    sensor_map = {r["iso_ts"]: r for r in sensor_rows}
    eo_map = {r["iso_ts"]: r for r in eo_rows}

    # union of keys
    keys = set(sensor_map.keys()) | set(eo_map.keys())

    # sort keys: parsed datetimes first (by value), then the rest by string
    def _key_sort(k):
        s = sensor_map.get(k) or eo_map.get(k)
        dt = s.get("parsed") if s is not None else None
        return (0, dt) if dt is not None else (1, k)

    sorted_keys = sorted(keys, key=_key_sort)

    fieldnames = [
        "timestamp",
        "warning_level",
        "water_level",
        "rainfall",
        "humidity",
        "soil_saturation",
        "flood_extent",
        "wetness_trend",
        "risk",
    ]

    # write merged CSV (overwrite to keep consistent view)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for k in sorted_keys:
            srow = sensor_map.get(k)
            erow = eo_map.get(k)

            # prepare sensor values (convert where possible)
            if srow:
                sraw = srow["raw"]
                try:
                    water = float(sraw.get("water_level") or sraw.get("water") or "")
                except Exception:
                    water = ""
                try:
                    rainfall = float(sraw.get("rainfall") or "")
                except Exception:
                    rainfall = ""
                try:
                    humidity = float(sraw.get("humidity") or "")
                except Exception:
                    humidity = ""
                sensor_data = {
                    "timestamp": k,
                    "water_level": water if water != "" else None,
                    "rainfall": rainfall if rainfall != "" else None,
                    "humidity": humidity if humidity != "" else None,
                }
            else:
                sensor_data = None

            # prepare eo values
            if erow:
                eraw = erow["raw"]
                def _f(x):
                    try:
                        return float(x)
                    except Exception:
                        return None
                soil = _f(eraw.get("soil_saturation"))
                flood = _f(eraw.get("flood_extent"))
                try:
                    trend = int(float(eraw.get("wetness_trend")))
                except Exception:
                    trend = None
                eo_data = {"soil_saturation": soil, "flood_extent": flood, "wetness_trend": trend}
            else:
                eo_data = None

            # compute risk only if sensor data present
            if sensor_data is not None:
                # when eo_data is None, compute_flood_risk will load EO by default, but we pass eo_data so none is used
                res = compute_flood_risk({
                    "timestamp": sensor_data["timestamp"],
                    "water_level": sensor_data["water_level"] if sensor_data["water_level"] is not None else 0.0,
                    "rainfall": sensor_data["rainfall"] if sensor_data["rainfall"] is not None else 0.0,
                    "humidity": sensor_data["humidity"] if sensor_data["humidity"] is not None else 0.0,
                }, eo_features=eo_data, thresholds=thresholds)
                level = res.get("alert_level")
                risk = res.get("flood_risk")
            else:
                level = ""
                risk = ""

            out = {
                "timestamp": k,
                "warning_level": level or "",
                "water_level": sensor_data["water_level"] if sensor_data is not None and sensor_data.get("water_level") is not None else "",
                "rainfall": sensor_data["rainfall"] if sensor_data is not None and sensor_data.get("rainfall") is not None else "",
                "humidity": sensor_data["humidity"] if sensor_data is not None and sensor_data.get("humidity") is not None else "",
                "soil_saturation": eo_data["soil_saturation"] if eo_data is not None and eo_data.get("soil_saturation") is not None else "",
                "flood_extent": eo_data["flood_extent"] if eo_data is not None and eo_data.get("flood_extent") is not None else "",
                "wetness_trend": eo_data["wetness_trend"] if eo_data is not None and eo_data.get("wetness_trend") is not None else "",
                "risk": risk if risk != "" else "",
            }

            writer.writerow(out)


def main(sensor_csv: str | None = None, eo_csv: str | None = None, log_path: str | None = None):
    """Process existing sensor and EO CSVs and produce the merged events CSV.

    This function does NOT generate data. It reads the supplied CSV paths (or
    defaults under data/) and calls `log_event` to produce the merged output.
    """
    # Resolve default paths if not provided
    if sensor_csv is None:
        sensor_csv = Path(__file__).resolve().parents[1] / "data" / "sensor" / "simulated.csv"
    else:
        sensor_csv = Path(sensor_csv)

    if eo_csv is None:
        eo_csv = Path(__file__).resolve().parents[1] / "data" / "sentinel1" / "eo_features.csv"
    else:
        eo_csv = Path(eo_csv)

    if log_path is None:
        log_path = Path(__file__).resolve().parents[1] / "logs" / "events.csv"
    else:
        log_path = Path(log_path)

    # Perform processing (log_event will read the CSVs and write merged CSV)
    log_event(result=None, log_path=log_path, sensor_csv=sensor_csv, eo_csv=eo_csv)

    print(f"Processed sensor='{sensor_csv}' eo='{eo_csv}' -> merged events at '{log_path}'")


if __name__ == "__main__":
    main()
