"""
showcase_availability.py
========================
Showcase Add-on — System Availability Calculator

PURPOSE
-------
Calculates and reports system availability for the Showcase Flood EWS
pipeline. Availability is measured across three dimensions:

  1. DATA AVAILABILITY
     Percentage of expected calendar days that have a complete row in
     each showcase CSV (showcase_sensor.csv, showcase_proxy.csv,
     showcase_merge.csv, showcase_predict.csv).

  2. PIPELINE AVAILABILITY
     Based on transmission_log.csv — percentage of scheduled runs that
     completed successfully vs total expected runs.

  3. ALERT CHANNEL AVAILABILITY
     Based on transmission_log.csv and alert state files — percentage
     of alert dispatches that were successfully sent per channel
     (Facebook, Telegram, Supabase).

AVAILABILITY FORMULA
--------------------
    availability = (uptime_days / total_expected_days) × 100
    downtime_days = total_expected_days − uptime_days

For data availability:
    total_expected = calendar days from HISTORY_START to today
    uptime_days    = days with a complete (non-NaN) row in the CSV
    downtime_days  = days without a complete row (gap days)

For pipeline availability:
    total_expected = total rows in transmission_log.csv (each = one run attempt)
    uptime_days    = rows where latency_seconds > 0 and tier != "ERROR"
    downtime_days  = failed / incomplete run rows

For alert availability:
    Reads showcase_last_fb_posted.json and showcase_last_tg_sent.json
    and cross-references against transmission_log.csv run count.

OUTPUT
------
    data/sensor/availability_report.csv
    Columns:
        report_date       — date this report was generated
        metric            — name of the availability metric
        uptime_days       — count of successful / present events
        downtime_days     — count of missing / failed events
        total_expected    — uptime + downtime
        availability_pct  — (uptime / total) × 100, rounded to 2 dp
        notes             — human-readable explanation of the calculation

Usage
-----
    # Standalone report
    python showcase_availability.py

    # Save report to CSV
    python showcase_availability.py --save

    # Custom start date
    python showcase_availability.py --start 2025-07-01
"""

import os
import json
import argparse
from datetime import date, datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "sensor")
PREDICT_DIR  = os.path.join(PROJECT_ROOT, "predictions")

SENSOR_CSV   = os.path.join(DATA_DIR, "showcase_sensor.csv")
PROXY_CSV    = os.path.join(DATA_DIR, "showcase_proxy.csv")
MERGE_CSV    = os.path.join(DATA_DIR, "showcase_merge.csv")
PREDICT_CSV  = os.path.join(PREDICT_DIR, "showcase_predict.csv")

TRANSMISSION_LOG = os.path.join(DATA_DIR, "transmission_log.csv")

FB_STATE_FILE  = os.path.join(SCRIPT_DIR, "showcase_last_fb_posted.json")
TG_STATE_FILE  = os.path.join(SCRIPT_DIR, "showcase_last_tg_sent.json")

AVAILABILITY_REPORT = os.path.join(DATA_DIR, "availability_report.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_utc() -> date:
    return datetime.now(tz=timezone.utc).date()


def _parse_date_col(df: pd.DataFrame, col: str = "timestamp") -> pd.Series:
    """Return a Series of date strings (YYYY-MM-DD) from the timestamp column."""
    return pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")


def _load_csv_dates(csv_path: str) -> set[str]:
    """Return the set of date strings present in a CSV's timestamp column."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["timestamp"])
        return set(_parse_date_col(df).tolist())
    except Exception:
        return set()


def _load_csv_complete_dates(csv_path: str,
                              sensor_cols: list | None = None) -> set[str]:
    """
    Return the set of dates that have a COMPLETE row (no NaN in sensor cols).
    """
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        if sensor_cols:
            cols_present = [c for c in sensor_cols if c in df.columns]
            df = df.dropna(subset=cols_present, how="any")
        df["_date"] = _parse_date_col(df)
        return set(df["_date"].tolist())
    except Exception:
        return set()


def _expected_dates(start: date, end: date) -> list[str]:
    """Return all YYYY-MM-DD strings from start to end inclusive."""
    return [
        (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end - start).days + 1)
    ]


def _availability_pct(uptime: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(100.0 * uptime / total, 2)


def _build_metric(metric: str,
                  uptime: int,
                  total: int,
                  notes: str) -> dict:
    """
    Build a standardised metric dict with explicit uptime/downtime columns.

    CSV columns produced
    --------------------
    metric            — metric name
    uptime_days       — successful / present count
    downtime_days     — missing / failed count
    total_expected    — uptime + downtime
    availability_pct  — (uptime / total) × 100
    notes             — calculation explanation
    """
    downtime = max(total - uptime, 0)
    return {
        "metric":           metric,
        "uptime_days":      uptime,
        "downtime_days":    downtime,
        "total_expected":   total,
        "availability_pct": _availability_pct(uptime, total),
        "notes":            notes,
    }


# ---------------------------------------------------------------------------
# 1. Data availability
# ---------------------------------------------------------------------------

def compute_data_availability(start_date: date) -> list[dict]:
    today      = _today_utc()
    expected   = _expected_dates(start_date, today)
    n_expected = len(expected)

    sensor_cols = ["waterlevel", "soil_moisture", "humidity"]
    results = []

    for label, csv_path in [
        ("sensor (showcase_sensor.csv)",   SENSOR_CSV),
        ("proxy  (showcase_proxy.csv)",    PROXY_CSV),
        ("merge  (showcase_merge.csv)",    MERGE_CSV),
        ("predict(showcase_predict.csv)",  PREDICT_CSV),
    ]:
        file_exists = os.path.exists(csv_path)
        complete    = _load_csv_complete_dates(csv_path, sensor_cols) if file_exists else set()
        n_uptime    = sum(1 for d in expected if d in complete)

        if file_exists:
            notes = (
                f"Calendar days {start_date} → {today} with a complete non-null row. "
                f"uptime={n_uptime} days present, "
                f"downtime={n_expected - n_uptime} days missing."
            )
        else:
            notes = f"CSV file not found at {csv_path} — all days counted as downtime."

        results.append(_build_metric(
            metric   = f"Data availability — {label}",
            uptime   = n_uptime,
            total    = n_expected,
            notes    = notes,
        ))

    return results


# ---------------------------------------------------------------------------
# 2. Pipeline run availability
# ---------------------------------------------------------------------------

def compute_pipeline_availability() -> list[dict]:
    results = []

    if not os.path.exists(TRANSMISSION_LOG):
        results.append(_build_metric(
            metric = "Pipeline availability (runs completed)",
            uptime = 0,
            total  = 0,
            notes  = "transmission_log.csv not found — run showcase_start.py first.",
        ))
        return results

    try:
        log_df = pd.read_csv(TRANSMISSION_LOG)
    except Exception as e:
        results.append(_build_metric(
            metric = "Pipeline availability (runs completed)",
            uptime = 0,
            total  = 0,
            notes  = f"Could not read transmission_log.csv: {e}",
        ))
        return results

    total = len(log_df)

    # A run is successful if latency_seconds > 0 and tier is a known value
    if "latency_seconds" in log_df.columns and "tier" in log_df.columns:
        successful = int((
            (log_df["latency_seconds"] > 0) &
            (log_df["tier"].isin(["CLEAR", "WATCH", "WARNING", "DANGER"]))
        ).sum())
    else:
        successful = total  # no error info → assume all OK

    results.append(_build_metric(
        metric = "Pipeline availability (runs completed)",
        uptime = successful,
        total  = total,
        notes  = (
            f"uptime={successful} runs with latency>0 and known tier. "
            f"downtime={total - successful} runs failed or produced unrecognised tier."
        ),
    ))

    # Latency speed buckets — expressed as share of total runs
    if "latency_seconds" in log_df.columns and total > 0:
        lats   = log_df["latency_seconds"].dropna()
        fast   = int((lats <= 60).sum())
        medium = int(((lats > 60) & (lats <= 300)).sum())
        slow   = int((lats > 300).sum())

        results.append(_build_metric(
            metric = "Pipeline speed — ≤60s latency runs",
            uptime = fast,
            total  = total,
            notes  = (
                f"Runs completing within 60 seconds end-to-end. "
                f"downtime here = runs that took longer than 60s ({total - fast})."
            ),
        ))
        results.append(_build_metric(
            metric = "Pipeline speed — 60–300s latency runs",
            uptime = medium,
            total  = total,
            notes  = (
                f"Runs completing between 60–300 seconds. "
                f"downtime = runs outside this bracket ({total - medium})."
            ),
        ))
        results.append(_build_metric(
            metric = "Pipeline speed — >300s latency runs",
            uptime = slow,
            total  = total,
            notes  = (
                f"Runs taking more than 5 minutes. "
                f"downtime = runs that finished faster ({total - slow})."
            ),
        ))

    return results


# ---------------------------------------------------------------------------
# 3. Alert channel availability
# ---------------------------------------------------------------------------

def compute_alert_availability() -> list[dict]:
    results = []

    if not os.path.exists(TRANSMISSION_LOG):
        return results

    try:
        log_df = pd.read_csv(TRANSMISSION_LOG)
        total_runs = len(log_df)
    except Exception:
        return results

    if total_runs == 0:
        return results

    # Facebook
    fb_ok = False
    try:
        with open(FB_STATE_FILE, "r") as f:
            fb_data = json.load(f)
        fb_ok = bool(fb_data.get("last_posted_timestamp"))
    except Exception:
        fb_ok = False

    # Telegram
    tg_ok = False
    try:
        with open(TG_STATE_FILE, "r") as f:
            tg_data = json.load(f)
        tg_ok = bool(tg_data.get("last_telegram_timestamp"))
    except Exception:
        tg_ok = False

    # Count distinct logged prediction dates for per-date Supabase comparison
    if "prediction_date" in log_df.columns:
        n_dates_logged = log_df["prediction_date"].nunique()
    else:
        n_dates_logged = total_runs

    results.append(_build_metric(
        metric = "Alert channel — Facebook (at least one post confirmed)",
        uptime = 1 if fb_ok else 0,
        total  = 1,
        notes  = (
            "Last post timestamp found in showcase_last_fb_posted.json. "
            "uptime=1 means ≥1 successful post exists; downtime=1 means no post recorded."
            if fb_ok else
            "No successful post recorded in showcase_last_fb_posted.json state file. "
            "uptime=0, downtime=1."
        ),
    ))

    results.append(_build_metric(
        metric = "Alert channel — Telegram (at least one send confirmed)",
        uptime = 1 if tg_ok else 0,
        total  = 1,
        notes  = (
            "Last send timestamp found in showcase_last_tg_sent.json. "
            "uptime=1 means ≥1 successful send; downtime=1 means no send recorded."
            if tg_ok else
            "No successful send recorded in showcase_last_tg_sent.json state file. "
            "uptime=0, downtime=1."
        ),
    ))

    # Supabase: predict CSV rows vs logged run dates
    predict_rows = 0
    if os.path.exists(PREDICT_CSV):
        try:
            predict_rows = len(pd.read_csv(PREDICT_CSV))
        except Exception:
            predict_rows = 0

    results.append(_build_metric(
        metric = "Alert channel — Supabase (rows in showcase_predict.csv)",
        uptime = predict_rows,
        total  = n_dates_logged,
        notes  = (
            f"uptime={predict_rows} rows present in showcase_predict.csv. "
            f"total={n_dates_logged} distinct prediction dates logged. "
            f"downtime={max(n_dates_logged - predict_rows, 0)} dates without a prediction row."
        ),
    ))

    return results


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def generate_report(start_date: date, save: bool = False) -> pd.DataFrame:
    today  = _today_utc()
    n_days = (today - start_date).days + 1

    print("\n" + "=" * 72)
    print("  SYSTEM AVAILABILITY REPORT — Showcase Flood EWS")
    print("=" * 72)
    print(f"  Report date   : {today}")
    print(f"  History start : {start_date}")
    print(f"  Calendar days : {n_days}")
    print(f"  Sensor CSV    : {SENSOR_CSV}")
    print(f"  Predict CSV   : {PREDICT_CSV}")
    print(f"  Trans. log    : {TRANSMISSION_LOG}")
    print(f"\n  Columns: uptime_days | downtime_days | total_expected | availability_pct")

    all_metrics: list[dict] = []

    # ── Section 1: Data availability ─────────────────────────────────────
    print(f"\n  ── 1. DATA AVAILABILITY ──────────────────────────────────────────")
    _print_header()
    data_metrics = compute_data_availability(start_date)
    for m in data_metrics:
        _print_metric(m)
        all_metrics.append(m)

    # ── Section 2: Pipeline availability ──────────────────────────────────
    print(f"\n  ── 2. PIPELINE AVAILABILITY ──────────────────────────────────────")
    _print_header()
    pipeline_metrics = compute_pipeline_availability()
    if not pipeline_metrics:
        print("  No pipeline runs logged yet.")
    for m in pipeline_metrics:
        _print_metric(m)
        all_metrics.append(m)

    # ── Section 3: Alert channel availability ─────────────────────────────
    print(f"\n  ── 3. ALERT CHANNEL AVAILABILITY ─────────────────────────────────")
    _print_header()
    alert_metrics = compute_alert_availability()
    if not alert_metrics:
        print("  No alert data available yet.")
    for m in alert_metrics:
        _print_metric(m)
        all_metrics.append(m)

    print("\n" + "=" * 72 + "\n")

    if not all_metrics:
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics)
    # Enforce column order
    col_order = [
        "metric",
        "uptime_days",
        "downtime_days",
        "total_expected",
        "availability_pct",
        "notes",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df.insert(0, "report_date", str(today))

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(AVAILABILITY_REPORT, index=False)
        print(f"  Saved → {AVAILABILITY_REPORT}  ({len(df)} metrics)")
        print(f"  Columns: {', '.join(df.columns.tolist())}\n")

    return df


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_header() -> None:
    print(f"  {'metric':<52}  {'up':>6}  {'down':>6}  {'total':>6}  {'pct':>7}")
    print(f"  {'-'*52}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")


def _print_metric(m: dict) -> None:
    bar_len = int(m["availability_pct"] / 5)  # max 20 chars
    bar     = "█" * bar_len + "░" * (20 - bar_len)
    label   = m["metric"][:52]
    print(
        f"  {label:<52}  "
        f"{m['uptime_days']:>6}  "
        f"{m['downtime_days']:>6}  "
        f"{m['total_expected']:>6}  "
        f"{m['availability_pct']:>6.2f}%"
    )
    print(f"  [{bar}]  {m['notes'][:80]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "showcase_availability — System availability calculator.\n"
            "Measures data completeness, pipeline uptime, and alert channel health.\n"
            "CSV output columns: report_date, metric, uptime_days, downtime_days,\n"
            "                    total_expected, availability_pct, notes."
        )
    )
    parser.add_argument(
        "--start", type=str, default="2025-07-01",
        help="Start date for availability calculation (YYYY-MM-DD, default: 2025-07-01).",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the availability report to availability_report.csv.",
    )
    args = parser.parse_args()

    try:
        start = date.fromisoformat(args.start)
    except ValueError:
        print(f"  ERROR: Invalid date format '{args.start}'. Use YYYY-MM-DD.")
        exit(1)

    generate_report(start_date=start, save=args.save)