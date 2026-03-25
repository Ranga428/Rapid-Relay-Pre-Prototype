"""
main.py
=======
Rapid Relay — Flood Early Warning System
Pipeline Orchestrator — Obando, Bulacan

PIPELINE OVERVIEW
-----------------
This file stitches together two pipelines:

  DAILY PIPELINE  (runs every time new sensor data arrives)
  ┌─────────────────────────────────────────────────────────┐
  │  1. Sensor ingestion  (Sat_SensorData_proxy or real CSV) │
  │  2. Prediction        (LGBM_Predict / predict.py)        │
  │  3. Calibration       (calibrator — load thresholds)     │
  │  4. Alert             (notifier — SMS / log / dashboard) │
  └─────────────────────────────────────────────────────────┘

  EO PIPELINE  (runs ~every 12 days when new Sentinel-1 pass is available)
  ┌─────────────────────────────────────────────────────────┐
  │  1. GEE extraction    (sentinel1_GEE.py)                 │
  │  2. EO feature build  (eo_features.py)                   │
  │  NOTE: retraining is a separate manual step              │
  └─────────────────────────────────────────────────────────┘

USAGE
-----
  # Run full daily pipeline (sensor → predict → alert)
  python main.py

  # Run full daily pipeline using simulated sensor data
  python main.py --mode sim

  # Run EO ingestion only (GEE + eo_features)
  python main.py --eo-only

  # Run daily pipeline + EO ingestion together
  python main.py --with-eo

  # Run on a schedule (every N hours)
  python main.py --schedule --interval 12

  # Dry run — show what would run without executing
  python main.py --dry-run

FAILURE BEHAVIOUR
-----------------
  - If sensor ingestion fails    → pipeline stops (no data, no point predicting)
  - If prediction fails          → pipeline stops (nothing to alert on)
  - If calibration fails         → WARNING logged, prediction output still used
  - If alerting fails            → WARNING logged, predictions already saved to CSV
  - If GEE fails                 → WARNING logged, existing sentinel1_timeseries.csv kept
  - If eo_features fails         → WARNING logged, existing eo_features.csv kept

ADDING NEW STEPS
----------------
  Each step is a function that returns True (success) or False (failure).
  Add new steps to DAILY_STEPS or EO_STEPS lists in run_daily() / run_eo().
"""

import os
import sys
import time
import logging
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — all scripts live under flood_preprototype/scripts/
# ---------------------------------------------------------------------------

ROOT_DIR    = Path(__file__).parent.resolve()
SCRIPTS_DIR = ROOT_DIR / "flood_preprototype" / "scripts"
MODELS_DIR  = ROOT_DIR / "flood_preprototype" / "models"
ALERTS_DIR  = ROOT_DIR / "flood_preprototype" / "alerts"
LOGS_DIR    = ROOT_DIR / "flood_preprototype" / "logs"
DATA_DIR    = ROOT_DIR / "flood_preprototype" / "data"

# Add all relevant dirs to sys.path so imports resolve cleanly
for p in [str(ROOT_DIR), str(SCRIPTS_DIR), str(MODELS_DIR), str(ALERTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Logging — writes to logs/pipeline.log and stdout
# ---------------------------------------------------------------------------

LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("rapid_relay")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def separator(title: str = "") -> None:
    line = "=" * 60
    if title:
        log.info(line)
        log.info(f"  {title}")
        log.info(line)
    else:
        log.info(line)


def run_step(
    name: str,
    fn,
    *args,
    critical: bool = True,
    dry_run: bool = False,
    **kwargs,
) -> bool:
    """
    Execute a single pipeline step safely.

    Parameters
    ----------
    name     : Human-readable step name for logging.
    fn       : Callable to execute.
    critical : If True, a failure will halt the pipeline.
               If False, a failure logs a WARNING but pipeline continues.
    dry_run  : If True, log what would run without calling fn.

    Returns
    -------
    True if step succeeded (or dry_run), False if it failed.
    """
    log.info(f"  ▶  {name}")
    if dry_run:
        log.info(f"     [DRY RUN] would call: {fn.__module__}.{fn.__name__}")
        return True
    try:
        fn(*args, **kwargs)
        log.info(f"  ✅ {name} — OK")
        return True
    except Exception as exc:
        if critical:
            log.error(f"  ❌ {name} — FAILED (critical)")
            log.error(f"     {exc}")
            log.debug(traceback.format_exc())
        else:
            log.warning(f"  ⚠️  {name} — FAILED (non-critical, continuing)")
            log.warning(f"     {exc}")
            log.debug(traceback.format_exc())
        return False


# ---------------------------------------------------------------------------
# Lazy imports — only imported when the step actually runs.
# This way a missing/broken module doesn't crash the entire file on import.
# ---------------------------------------------------------------------------

def _import_sensor_proxy():
    import Sat_SensorData_proxy
    return Sat_SensorData_proxy

def _import_sentinel1_gee():
    import sentinel1_GEE
    return sentinel1_GEE

def _import_eo_features():
    import eo_features
    return eo_features

def _import_predictor():
    """
    Try LGBM predictor first, fall back to XGBoost predict.py.
    Both expose a main() entry point.
    """
    try:
        import LGBM_Predict as predictor
        log.info("     Using model: LGBM_Predict")
        return predictor
    except ImportError:
        import predict as predictor
        log.info("     Using model: XGBoost predict")
        return predictor

def _import_calibrator():
    from models import calibrator
    return calibrator

def _import_notifier():
    from alerts import notifier
    return notifier


# ---------------------------------------------------------------------------
# Pipeline step wrappers
# (Each wraps a module's main() in a consistent callable signature)
# ---------------------------------------------------------------------------

def step_sensor_proxy(mode: str) -> None:
    """Ingest sensor data. mode='sim' uses simulated CSV, mode='real' skips proxy."""
    if mode == "real":
        log.info("     Sensor mode: real — using existing sensor CSV, no proxy needed.")
        _check_sensor_csv_exists()
        return
    mod = _import_sensor_proxy()
    mod.main()


def step_predict() -> None:
    mod = _import_predictor()
    mod.main()


def step_calibrate() -> None:
    mod = _import_calibrator()
    mod.load_thresholds()


def step_notify() -> None:
    mod = _import_notifier()
    mod.notify()


def step_sentinel1_gee() -> None:
    mod = _import_sentinel1_gee()
    mod.main()


def step_eo_features() -> None:
    mod = _import_eo_features()
    mod.main()


# ---------------------------------------------------------------------------
# Guard: check sensor CSV exists before predicting
# ---------------------------------------------------------------------------

SENSOR_CSV = DATA_DIR / "sensor" / "obando_environmental_data.csv"
SIMULATED_CSV = DATA_DIR / "sensor" / "simulated.csv"

def _check_sensor_csv_exists() -> None:
    if not SENSOR_CSV.exists():
        raise FileNotFoundError(
            f"Sensor CSV not found: {SENSOR_CSV}\n"
            f"  If using simulated data, run with --mode sim.\n"
            f"  If using real data, ensure the sensor CSV is present."
        )
    log.info(f"     Sensor CSV found: {SENSOR_CSV}")


# ---------------------------------------------------------------------------
# Daily pipeline
# ---------------------------------------------------------------------------

def run_daily(mode: str = "real", dry_run: bool = False) -> bool:
    """
    Run the daily sensor → predict → calibrate → alert pipeline.

    Returns True if all critical steps succeeded.
    """
    separator("DAILY PIPELINE")
    log.info(f"  Sensor mode : {mode}")
    log.info(f"  Dry run     : {dry_run}")

    steps = [
        # (name, callable, args, critical)
        ("Sensor ingestion",  step_sensor_proxy, (mode,), True),
        ("Flood prediction",  step_predict,      (),       True),
        ("Threshold calibration", step_calibrate, (),      False),  # non-critical
        ("Alert / notification",  step_notify,    (),      False),  # non-critical
    ]

    for name, fn, args, critical in steps:
        ok = run_step(name, fn, *args, critical=critical, dry_run=dry_run)
        if not ok and critical:
            log.error(f"  Pipeline halted at critical step: {name}")
            return False

    separator("DAILY PIPELINE COMPLETE")
    return True


# ---------------------------------------------------------------------------
# EO pipeline (~12-day cadence, when new SAR pass is available)
# ---------------------------------------------------------------------------

def run_eo(dry_run: bool = False) -> bool:
    """
    Run the EO ingestion pipeline: GEE extraction → eo_features.

    This is typically run manually or on a ~12-day cron job
    aligned with the Sentinel-1 revisit cycle over Obando.
    On failure the existing sentinel1_timeseries.csv is preserved.
    """
    separator("EO PIPELINE  (Sentinel-1 / GEE)")

    steps = [
        # Both steps non-critical: if GEE is unreachable, keep existing CSV
        ("GEE extraction (sentinel1_GEE)",  step_sentinel1_gee, (), False),
        ("EO feature engineering",          step_eo_features,   (), False),
    ]

    for name, fn, args, critical in steps:
        ok = run_step(name, fn, *args, critical=critical, dry_run=dry_run)
        if not ok:
            log.warning(
                f"  EO step failed: {name}. "
                f"Existing EO data preserved — prediction will use last known values."
            )

    separator("EO PIPELINE COMPLETE")
    return True


# ---------------------------------------------------------------------------
# Scheduled mode
# ---------------------------------------------------------------------------

def run_scheduled(interval_hours: int, mode: str, with_eo: bool) -> None:
    log.info(f"  Scheduled mode — every {interval_hours}h")
    log.info(f"  Press Ctrl+C to stop.")

    eo_interval_hours = 12 * 24   # ~12 days in hours
    hours_since_eo    = eo_interval_hours  # trigger EO on first run if with_eo

    while True:
        try:
            run_daily(mode=mode)

            if with_eo:
                hours_since_eo += interval_hours
                if hours_since_eo >= eo_interval_hours:
                    run_eo()
                    hours_since_eo = 0
                else:
                    remaining = eo_interval_hours - hours_since_eo
                    log.info(f"  EO pipeline next in ~{remaining}h")

            next_run = datetime.now().strftime("%Y-%m-%d %H:%M")
            log.info(f"  Next run in {interval_hours}h  (scheduled at {next_run})")
            time.sleep(interval_hours * 3600)

        except KeyboardInterrupt:
            log.info("  Stopped by user.")
            break
        except Exception as exc:
            log.error(f"  Unexpected error in scheduled run: {exc}")
            log.debug(traceback.format_exc())
            log.info(f"  Retrying in {interval_hours}h...")
            time.sleep(interval_hours * 3600)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rapid Relay — Flood Early Warning Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # daily pipeline, real sensor data
  python main.py --mode sim             # daily pipeline, simulated sensor data
  python main.py --eo-only              # EO ingestion only (GEE + eo_features)
  python main.py --with-eo              # daily pipeline + EO ingestion
  python main.py --schedule --interval 12   # run every 12 hours
  python main.py --dry-run              # show what would run
        """,
    )
    parser.add_argument(
        "--mode", choices=["real", "sim"], default="real",
        help="Sensor data source. 'real' = obando_environmental_data.csv, "
             "'sim' = Sat_SensorData_proxy (simulated.csv). Default: real",
    )
    parser.add_argument(
        "--eo-only", action="store_true",
        help="Run EO ingestion pipeline only (GEE + eo_features). Skip daily pipeline.",
    )
    parser.add_argument(
        "--with-eo", action="store_true",
        help="Run daily pipeline AND EO ingestion pipeline.",
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a repeating schedule instead of once.",
    )
    parser.add_argument(
        "--interval", type=int, default=12,
        help="Hours between scheduled runs (default: 12).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without actually executing any steps.",
    )
    args = parser.parse_args()

    separator("Rapid Relay — Flood Early Warning System")
    log.info(f"  Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Root dir     : {ROOT_DIR}")
    log.info(f"  Log file     : {LOG_FILE}")
    log.info(f"  Mode         : {args.mode}")
    log.info(f"  EO only      : {args.eo_only}")
    log.info(f"  With EO      : {args.with_eo}")
    log.info(f"  Scheduled    : {args.schedule}  (interval={args.interval}h)")
    log.info(f"  Dry run      : {args.dry_run}")

    if args.schedule:
        run_scheduled(
            interval_hours=args.interval,
            mode=args.mode,
            with_eo=args.with_eo,
        )
        return

    if args.eo_only:
        run_eo(dry_run=args.dry_run)
        return

    run_daily(mode=args.mode, dry_run=args.dry_run)

    if args.with_eo:
        run_eo(dry_run=args.dry_run)

    separator("ALL DONE")
    log.info(f"  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()