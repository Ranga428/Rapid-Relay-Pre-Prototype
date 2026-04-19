"""
showcase_transmission_speed.py
================================
Showcase Add-on — Transmission Speed Recorder

PURPOSE
-------
Measures how long it takes for a sensor reading ingested from Supabase
to appear as a prediction in showcase_predict.csv.

For each date that exists in BOTH showcase_sensor.csv AND
showcase_predict.csv, it computes:

    transmission_latency_seconds = prediction_timestamp - ingest_timestamp

The "ingest timestamp" is the datetime the sensor row's daily aggregate
was written (the timestamp column in showcase_sensor.csv, which reflects
the original Supabase reading time via the Date + Time columns).

The "prediction timestamp" is taken as the file modification time of
showcase_predict.csv at the time the matching prediction row was written
— approximated here as the prediction row's timestamp field (daily
midnight) aligned to the same date.

Because both CSVs store daily aggregates at midnight (T00:00:00), a
direct timestamp subtraction gives zero. This script therefore also
records the WALL-CLOCK latency log written by showcase_start.py into
transmission_log.csv, and computes the true end-to-end delta from
those wall-clock records when available.

TWO MODES
---------
1. LOG MODE  (called by showcase_start.py during a live run)
   Records a single entry: ingest wall-clock time + prediction wall-clock
   time for the current run. Appends to transmission_log.csv.

2. REPORT MODE  (run standalone)
   Reads transmission_log.csv and prints a latency summary table.

OUTPUT
------
    data/sensor/transmission_log.csv
    Columns:
        run_id              — sequential run number
        ingest_wall_clock   — UTC datetime sensor ingest completed
        predict_wall_clock  — UTC datetime prediction completed
        latency_seconds     — predict_wall_clock - ingest_wall_clock
        latency_hms         — human-readable HH:MM:SS
        sensor_date         — the date of the ingested sensor row
        prediction_date     — the date of the prediction produced
        tier                — risk tier of that prediction
        probability         — flood probability

Usage
-----
    # Called by showcase_start.py (log one run)
    from showcase_transmission_speed import log_run
    log_run(ingest_ts, predict_ts, sensor_date, prediction_date, tier, prob)

    # Standalone report
    python showcase_transmission_speed.py
    python showcase_transmission_speed.py --last 10
"""

import os
import argparse
from datetime import datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "sensor")

TRANSMISSION_LOG = os.path.join(DATA_DIR, "transmission_log.csv")

SENSOR_CSV  = os.path.join(DATA_DIR, "showcase_sensor.csv")
PREDICT_CSV = os.path.join(PROJECT_ROOT, "predictions", "showcase_predict.csv")

LOG_COLUMNS = [
    "run_id",
    "ingest_wall_clock",
    "predict_wall_clock",
    "latency_seconds",
    "latency_hms",
    "sensor_date",
    "prediction_date",
    "tier",
    "probability",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _parse_utc(ts_str: str) -> datetime:
    """
    Parse a timestamp string to a UTC-aware datetime.

    Handles formats:
        2025-07-14T12:34:56           (assumed UTC, no tz info)
        2025-07-14T12:34:56+00:00     (explicit UTC)
        2025-07-14T12:34:56Z          (UTC suffix)
        2025-07-14 12:34:56           (space separator, assumed UTC)
        2025-07-14                    (date-only, midnight UTC)
    """
    if isinstance(ts_str, datetime):
        if ts_str.tzinfo is None:
            return ts_str.replace(tzinfo=timezone.utc)
        return ts_str.astimezone(timezone.utc)

    ts_str = str(ts_str).strip()

    # Replace trailing Z with +00:00 for fromisoformat compatibility
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"

    # Date-only → midnight UTC
    if len(ts_str) == 10:
        ts_str = ts_str + "T00:00:00"

    # Normalize space separator
    ts_str = ts_str.replace(" ", "T", 1)

    try:
        dt = datetime.fromisoformat(ts_str)
    except ValueError:
        dt = pd.Timestamp(ts_str).to_pydatetime()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def _seconds_to_hms(seconds: float) -> str:
    """Convert a float number of seconds to HH:MM:SS string."""
    seconds = max(0.0, float(seconds))
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _get_next_run_id() -> int:
    """Return the next sequential run_id."""
    if not os.path.exists(TRANSMISSION_LOG):
        return 1
    try:
        df = pd.read_csv(TRANSMISSION_LOG)
        if df.empty or "run_id" not in df.columns:
            return 1
        return int(df["run_id"].max()) + 1
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_run(
    ingest_wall_clock:  str,
    predict_wall_clock: str,
    sensor_date:        str,
    prediction_date:    str,
    tier:               str,
    probability:        float,
) -> dict:
    """
    Record one transmission event to transmission_log.csv.

    Parameters
    ----------
    ingest_wall_clock   : ISO UTC datetime when sensor ingest completed
    predict_wall_clock  : ISO UTC datetime when prediction completed
    sensor_date         : YYYY-MM-DD of the ingested sensor row
    prediction_date     : YYYY-MM-DD of the prediction produced
    tier                : risk tier string ("CLEAR", "WATCH", "WARNING", "DANGER")
    probability         : flood probability float 0.0–1.0

    Returns
    -------
    dict with all logged fields.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        t_ingest  = _parse_utc(ingest_wall_clock)
        t_predict = _parse_utc(predict_wall_clock)
    except Exception as e:
        print(f"  [transmission_speed] WARNING: Could not parse timestamps: {e}")
        return {}

    latency_s  = (t_predict - t_ingest).total_seconds()
    latency_hms = _seconds_to_hms(latency_s)
    run_id      = _get_next_run_id()

    record = {
        "run_id":             run_id,
        "ingest_wall_clock":  t_ingest.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predict_wall_clock": t_predict.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latency_seconds":    round(latency_s, 2),
        "latency_hms":        latency_hms,
        "sensor_date":        sensor_date,
        "prediction_date":    prediction_date,
        "tier":               tier,
        "probability":        round(float(probability), 4),
    }

    df_row = pd.DataFrame([record])

    if not os.path.exists(TRANSMISSION_LOG):
        df_row.to_csv(TRANSMISSION_LOG, index=False)
        print(f"  [transmission_speed] Created transmission_log.csv — Run #{run_id}")
    else:
        df_row.to_csv(TRANSMISSION_LOG, mode="a", header=False, index=False)
        print(f"  [transmission_speed] Logged Run #{run_id} — latency: {latency_hms} ({latency_s:.1f}s)")

    return record


def compute_from_csvs() -> pd.DataFrame:
    """
    Compute date-level latency by comparing timestamps across
    showcase_sensor.csv and showcase_predict.csv.

    Since both CSVs store daily-aggregate midnight timestamps (T00:00:00),
    the timestamp columns themselves don't carry wall-clock timing.
    This function instead aligns rows by date and reports zero-latency
    (same-day) or cross-day gaps where a sensor date has no matching
    prediction yet.

    Returns a DataFrame summarising alignment between the two CSVs.
    """
    results = []

    if not os.path.exists(SENSOR_CSV):
        print(f"  [transmission_speed] showcase_sensor.csv not found: {SENSOR_CSV}")
        return pd.DataFrame()

    if not os.path.exists(PREDICT_CSV):
        print(f"  [transmission_speed] showcase_predict.csv not found: {PREDICT_CSV}")
        return pd.DataFrame()

    sensor_df  = pd.read_csv(SENSOR_CSV)
    predict_df = pd.read_csv(PREDICT_CSV)

    # Normalise timestamp → date string
    sensor_df["date"]  = pd.to_datetime(sensor_df["timestamp"]).dt.strftime("%Y-%m-%d")
    predict_df["date"] = pd.to_datetime(predict_df["timestamp"]).dt.strftime("%Y-%m-%d")

    sensor_dates  = set(sensor_df["date"].tolist())
    predict_dates = set(predict_df["date"].tolist())
    all_dates     = sorted(sensor_dates | predict_dates)

    for d in all_dates:
        in_sensor  = d in sensor_dates
        in_predict = d in predict_dates

        if in_sensor and in_predict:
            pred_row = predict_df[predict_df["date"] == d].iloc[0]
            results.append({
                "date":         d,
                "has_sensor":   True,
                "has_predict":  True,
                "tier":         pred_row.get("risk_tier", "N/A"),
                "probability":  round(float(pred_row.get("flood_probability", 0)), 4),
                "status":       "BOTH — same-day (wall-clock latency in transmission_log.csv)",
            })
        elif in_sensor and not in_predict:
            results.append({
                "date":        d,
                "has_sensor":  True,
                "has_predict": False,
                "tier":        "N/A",
                "probability": None,
                "status":      "SENSOR ONLY — prediction not yet generated",
            })
        else:
            results.append({
                "date":        d,
                "has_sensor":  False,
                "has_predict": True,
                "tier":        predict_df[predict_df["date"] == d].iloc[0].get("risk_tier", "N/A"),
                "probability": None,
                "status":      "PREDICT ONLY — no matching sensor row",
            })

    return pd.DataFrame(results)


def print_report(last_n: int = 0) -> None:
    """
    Print a latency summary from transmission_log.csv.

    last_n : if > 0, show only the last N runs. 0 = show all.
    """
    print("\n" + "=" * 70)
    print("  TRANSMISSION SPEED REPORT — Showcase Flood EWS")
    print("=" * 70)
    print(f"  Log file  : {TRANSMISSION_LOG}")
    print(f"  Sensor CSV: {SENSOR_CSV}")
    print(f"  Predict CSV: {PREDICT_CSV}")

    # ── CSV alignment section ────────────────────────────────────────────
    print("\n  ── CSV Date Alignment ──────────────────────────────────────────")
    alignment = compute_from_csvs()
    if alignment.empty:
        print("  No data to align.")
    else:
        both      = (alignment["has_sensor"] & alignment["has_predict"]).sum()
        sensor_only = (alignment["has_sensor"] & ~alignment["has_predict"]).sum()
        pred_only  = (~alignment["has_sensor"] & alignment["has_predict"]).sum()
        print(f"  Total dates checked   : {len(alignment):,}")
        print(f"  Both sensor+predict   : {both:,}  ✅")
        print(f"  Sensor only (pending) : {sensor_only:,}  ⏳")
        print(f"  Predict only (orphan) : {pred_only:,}  ⚠️")

    # ── Wall-clock latency section ───────────────────────────────────────
    print("\n  ── Wall-Clock Latency (from transmission_log.csv) ──────────────")
    if not os.path.exists(TRANSMISSION_LOG):
        print("  transmission_log.csv not found.")
        print("  Run showcase_start.py at least once to populate it.")
    else:
        try:
            log_df = pd.read_csv(TRANSMISSION_LOG)
            if log_df.empty:
                print("  No entries in transmission_log.csv yet.")
            else:
                if last_n > 0:
                    log_df = log_df.tail(last_n)
                    print(f"  Showing last {last_n} run(s):")

                if "latency_seconds" in log_df.columns:
                    lats = log_df["latency_seconds"].dropna()
                    if len(lats) > 0:
                        print(f"\n  Summary over {len(lats)} run(s):")
                        print(f"    Min latency  : {_seconds_to_hms(lats.min())}  ({lats.min():.1f}s)")
                        print(f"    Max latency  : {_seconds_to_hms(lats.max())}  ({lats.max():.1f}s)")
                        print(f"    Mean latency : {_seconds_to_hms(lats.mean())}  ({lats.mean():.1f}s)")
                        print(f"    Median       : {_seconds_to_hms(lats.median())}  ({lats.median():.1f}s)")
                        print(f"    Std dev      : {lats.std():.1f}s")

                print(f"\n  {'Run':<6} {'Ingest (UTC)':<22} {'Predict (UTC)':<22} "
                      f"{'Latency':>10}  {'Tier':<8}  {'Prob':>6}  Sensor Date")
                print(f"  {'-'*6} {'-'*22} {'-'*22} {'-'*10}  {'-'*8}  {'-'*6}  {'-'*12}")

                for _, row in log_df.iterrows():
                    run_id      = int(row.get("run_id", 0))
                    ingest_ts   = str(row.get("ingest_wall_clock",  "N/A"))[:19]
                    predict_ts  = str(row.get("predict_wall_clock", "N/A"))[:19]
                    latency_hms = str(row.get("latency_hms",       "N/A"))
                    tier        = str(row.get("tier",               "N/A"))
                    prob        = row.get("probability", 0.0)
                    sensor_date = str(row.get("sensor_date",        "N/A"))
                    print(f"  {run_id:<6} {ingest_ts:<22} {predict_ts:<22} "
                          f"{latency_hms:>10}  {tier:<8}  {prob:>6.1%}  {sensor_date}")

        except Exception as e:
            print(f"  ERROR reading transmission_log.csv: {e}")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "showcase_transmission_speed — latency reporter.\n"
            "Reads transmission_log.csv and prints a summary."
        )
    )
    parser.add_argument(
        "--last", type=int, default=0,
        help="Show only the last N runs (0 = all runs, default: 0).",
    )
    args = parser.parse_args()
    print_report(last_n=args.last)
