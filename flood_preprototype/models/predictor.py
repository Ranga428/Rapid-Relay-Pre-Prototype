from pathlib import Path
import yaml
import csv
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths — all absolute, anchored to this file's location so they work
# regardless of where the script is launched from.
# ---------------------------------------------------------------------------
# This file lives at:
#   Rapid-Relay-Pre-Prototype/flood_preprototype/<package>/flood_processing.py
# So parents[1] = flood_preprototype/
_PACKAGE_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent   # flood_preprototype/

_THRESHOLDS_PATH = _PROJECT_ROOT / "config" / "thresholds.yaml"
_SENTINEL1_CSV   = _PROJECT_ROOT / "data" / "sentinel1" / "GEE-Processing" / "sentinel1_timeseries.csv"
_SENSOR_CSV      = _PROJECT_ROOT / "data" / "sensor" / "simulated.csv"
_LOG_PATH        = _PROJECT_ROOT / "logs" / "events.csv"

with _THRESHOLDS_PATH.open("r", encoding="utf-8") as _f:
    thresholds = yaml.safe_load(_f)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _to_canonical_ts(iso_str: str) -> str:
    """Normalize any ISO timestamp to 'YYYY-MM-DDTHH:MM:SSZ' (no microseconds, always Z)."""
    if not iso_str:
        return ""
    try:
        raw = iso_str.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return iso_str[:19] + "Z" if len(iso_str) >= 19 else iso_str


def _parse_iso(iso_str):
    """Parse ISO 8601 string to an aware datetime in UTC. Returns None on failure."""
    if iso_str is None:
        return None
    try:
        if iso_str.endswith("Z"):
            iso_str = iso_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Small type-coercion helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# EO feature loading
# ---------------------------------------------------------------------------

def load_eo_features(csv_path=None, match_timestamp=None, n_recent=None, max_delta_seconds=60):
    """Load and aggregate EO features from the Sentinel-1 timeseries CSV."""
    csv_path = Path(csv_path) if csv_path else _SENTINEL1_CSV

    if not csv_path.exists():
        return {"soil_saturation": None, "flood_extent": None, "wetness_trend": None}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    if not reader:
        return {"soil_saturation": None, "flood_extent": None, "wetness_trend": None}

    rows = []
    for r in reader:
        ts = r.get("timestamp")
        ts_dt = _parse_iso(_to_canonical_ts(ts)) if ts else None
        rows.append({
            "timestamp":       ts_dt,
            "soil_saturation": _try_float(r.get("soil_saturation")),
            "flood_extent":    _try_float(r.get("flood_extent")),
            "wetness_trend":   _try_int(r.get("wetness_trend")),
        })

    def select_rows():
        if not match_timestamp:
            return rows if n_recent is None else rows[-n_recent:]

        target = (
            _parse_iso(_to_canonical_ts(match_timestamp))
            if isinstance(match_timestamp, str)
            else (match_timestamp if isinstance(match_timestamp, datetime) else None)
        )
        if target is None:
            return rows if n_recent is None else rows[-n_recent:]

        rows_with_delta = [
            (abs((r["timestamp"] - target).total_seconds()), r)
            for r in rows if r["timestamp"] is not None
        ]
        if not rows_with_delta:
            return rows if n_recent is None else rows[-n_recent:]

        if n_recent is None:
            within = [(d, r) for d, r in rows_with_delta if d <= max_delta_seconds]
            pool = sorted(within or rows_with_delta, key=lambda x: x[0])
            return [r for _, r in pool]

        return [r for _, r in sorted(rows_with_delta, key=lambda x: x[0])[:n_recent]]

    chosen = select_rows()
    if not chosen:
        return {"soil_saturation": None, "flood_extent": None, "wetness_trend": None}

    soil_vals  = [r["soil_saturation"] for r in chosen if r["soil_saturation"] is not None]
    flood_vals = [r["flood_extent"]    for r in chosen if r["flood_extent"]    is not None]
    trend_vals = [r["wetness_trend"]   for r in chosen if r["wetness_trend"]   is not None]

    return {
        "soil_saturation": sum(soil_vals)  / len(soil_vals)  if soil_vals  else None,
        "flood_extent":    sum(flood_vals) / len(flood_vals) if flood_vals else None,
        "wetness_trend":   int(round(sum(trend_vals) / len(trend_vals))) if trend_vals else None,
    }


def load_latest_eo_features(csv_path=None):
    """Return EO features from the last row of the Sentinel-1 timeseries CSV."""
    csv_path = Path(csv_path) if csv_path else _SENTINEL1_CSV
    if not csv_path.exists():
        return {"soil_saturation": None, "flood_extent": None, "wetness_trend": None}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    if not reader:
        return {"soil_saturation": None, "flood_extent": None, "wetness_trend": None}
    last = reader[-1]
    return {
        "soil_saturation": _try_float(last.get("soil_saturation")),
        "flood_extent":    _try_float(last.get("flood_extent")),
        "wetness_trend":   _try_int(last.get("wetness_trend")),
    }


# ---------------------------------------------------------------------------
# Risk computation
# ---------------------------------------------------------------------------

def compute_alert_level(flood_risk, thresholds=thresholds):
    """Map flood_risk (0-1) to RED / YELLOW / GREEN."""
    if flood_risk >= thresholds.get("alert_red",    0.75):
        return "RED"
    if flood_risk >= thresholds.get("alert_yellow", 0.40):
        return "YELLOW"
    return "GREEN"


def compute_flood_risk(sensor_data, eo_features=None, thresholds=thresholds):
    """Combine sensor data and EO features into a flood risk index (0-1)."""
    if eo_features is None:
        eo_features = load_eo_features()

    # Sensor index
    water_level_score = min(sensor_data["water_level"] / thresholds["water_level_m"], 1.0)
    rainfall_score    = min(sensor_data["rainfall"]    / thresholds["rainfall_mm"],   1.0)
    humidity_score    = min(sensor_data["humidity"]    / 100,                         1.0)
    sensor_index = 0.4 * rainfall_score + 0.3 * humidity_score + 0.3 * water_level_score

    # EO index
    soil  = eo_features.get("soil_saturation")
    flood = eo_features.get("flood_extent")
    trend = eo_features.get("wetness_trend")

    def _sf(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    soil_score  = min(max(_sf(soil), 0.0), 1.0)
    flood_norm  = thresholds.get("flood_index", 0.75) or 1.0
    flood_score = min(max(_sf(flood) / float(flood_norm), 0.0), 1.0)
    trend_score = (
        0.0 if trend is None
        else {1: 1.0, 0: 0.5, -1: 0.0}.get(int(float(trend)), 0.0)
    )
    eo_index = 0.4 * soil_score + 0.3 * flood_score + 0.3 * trend_score

    flood_risk = min(0.5 * sensor_index + 0.5 * eo_index, 1.0)

    final_sensor = dict(sensor_data)
    final_sensor["timestamp"] = _to_canonical_ts(sensor_data.get("timestamp", ""))

    return {
        "flood_risk":    flood_risk,
        "alert_level":   compute_alert_level(flood_risk, thresholds),
        "eo_features":   {"soil_saturation": soil, "flood_extent": flood, "wetness_trend": trend},
        "sensor_record": final_sensor,
    }


# ---------------------------------------------------------------------------
# Event logging  — writes every sensor row to events.csv
# ---------------------------------------------------------------------------

def log_event(result=None, log_path=None, sensor_csv=None, eo_csv=None):
    """Merge ALL sensor rows with EO data and write to events.csv."""
    sensor_csv = Path(sensor_csv) if sensor_csv else _SENSOR_CSV
    eo_csv     = Path(eo_csv)     if eo_csv     else _SENTINEL1_CSV
    log_path   = Path(log_path)   if log_path   else _LOG_PATH

    # Ensure output directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_rows(path):
        """Return a LIST of normalized row dicts — never collapses duplicates."""
        if not path.exists():
            print(f"[WARN] CSV not found: {path}")
            return []
        with path.open("r", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))
        result = []
        for i, r in enumerate(raw_rows):
            raw_ts = (
                r.get("timestamp") or r.get("Timestamp") or
                r.get("time")      or r.get("ts") or ""
            ).strip()
            canonical = _to_canonical_ts(raw_ts) if raw_ts else ""
            parsed    = _parse_iso(canonical)     if canonical else None
            result.append({
                "raw":    {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()},
                "iso_ts": canonical,
                "parsed": parsed,
                "index":  i,
            })
        return result

    sensor_rows = _read_rows(sensor_csv)
    eo_rows     = _read_rows(eo_csv)

    print(f"[INFO] Sensor rows loaded : {len(sensor_rows)} from {sensor_csv}")
    print(f"[INFO] EO rows loaded     : {len(eo_rows)} from {eo_csv}")

    # EO lookup by canonical timestamp
    eo_map = {}
    for r in eo_rows:
        eo_map.setdefault(r["iso_ts"], r)

    fieldnames = [
        "timestamp", "warning_level",
        "water_level", "rainfall", "humidity",
        "soil_saturation", "flood_extent", "wetness_trend",
        "risk",
    ]

    # Sort sensor rows chronologically (list preserved in full — no dict collapse)
    sensor_rows_sorted = sorted(
        sensor_rows,
        key=lambda r: (0, r["parsed"]) if r["parsed"] is not None else (1, r["iso_ts"])
    )

    # EO-only rows
    sensor_ts_set  = {r["iso_ts"] for r in sensor_rows}
    eo_only_sorted = sorted(
        [r for r in eo_rows if r["iso_ts"] not in sensor_ts_set],
        key=lambda r: (0, r["parsed"]) if r["parsed"] is not None else (1, r["iso_ts"])
    )

    rows_written = 0
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for srow in sensor_rows_sorted:
            sraw = srow["raw"]
            k    = srow["iso_ts"]

            def _fv(key, fallback=""):
                try:
                    return float(sraw.get(key) or fallback)
                except Exception:
                    return None

            water    = _fv("water_level") or _fv("water")
            rainfall = _fv("rainfall")
            humidity = _fv("humidity")

            erow = eo_map.get(k)
            if erow:
                eraw = erow["raw"]
                eo_data = {
                    "soil_saturation": _try_float(eraw.get("soil_saturation")),
                    "flood_extent":    _try_float(eraw.get("flood_extent")),
                    "wetness_trend":   _try_int(eraw.get("wetness_trend")),
                }
            else:
                eo_data = None  # falls back to whole-CSV aggregate inside compute_flood_risk

            res = compute_flood_risk({
                "timestamp":   k,
                "water_level": water    if water    is not None else 0.0,
                "rainfall":    rainfall if rainfall is not None else 0.0,
                "humidity":    humidity if humidity is not None else 0.0,
            }, eo_features=eo_data, thresholds=thresholds)

            writer.writerow({
                "timestamp":       k,
                "warning_level":   res.get("alert_level", ""),
                "water_level":     "" if water    is None else water,
                "rainfall":        "" if rainfall is None else rainfall,
                "humidity":        "" if humidity is None else humidity,
                "soil_saturation": "" if (not eo_data or eo_data.get("soil_saturation") is None) else eo_data["soil_saturation"],
                "flood_extent":    "" if (not eo_data or eo_data.get("flood_extent")    is None) else eo_data["flood_extent"],
                "wetness_trend":   "" if (not eo_data or eo_data.get("wetness_trend")   is None) else eo_data["wetness_trend"],
                "risk":            res.get("flood_risk", ""),
            })
            rows_written += 1

        for erow in eo_only_sorted:
            eraw = erow["raw"]
            writer.writerow({
                "timestamp":       erow["iso_ts"],
                "warning_level":   "",
                "water_level":     "",
                "rainfall":        "",
                "humidity":        "",
                "soil_saturation": _try_float(eraw.get("soil_saturation")) or "",
                "flood_extent":    _try_float(eraw.get("flood_extent"))    or "",
                "wetness_trend":   _try_int(eraw.get("wetness_trend"))     or "",
                "risk":            "",
            })
            rows_written += 1

    print(f"[INFO] events.csv written : {rows_written} rows -> {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(sensor_csv: str | None = None, eo_csv: str | None = None, log_path: str | None = None):
    """Read sensor + EO CSVs and produce the merged events.csv."""
    sensor_csv = Path(sensor_csv) if sensor_csv else _SENSOR_CSV
    eo_csv     = Path(eo_csv)     if eo_csv     else _SENTINEL1_CSV
    log_path   = Path(log_path)   if log_path   else _LOG_PATH

    log_event(result=None, log_path=log_path, sensor_csv=sensor_csv, eo_csv=eo_csv)


if __name__ == "__main__":
    main()