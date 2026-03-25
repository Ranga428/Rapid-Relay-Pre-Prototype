"""
Start.py
========
Flood Prediction — Daily Inference Orchestrator

PURPOSE
-------
Single entry point for daily operational deployment.
Calls existing pipeline scripts in order:

    Step 0  Sat_SensorData_proxy.py  — fetch today's sensor reading (incremental append)
    Step 1  RF_Predict.py            — predict on updated sensor CSV (Design Option 2)
    Step 2  FB_Post.py               — Facebook alert post (if WATCH/WARNING/DANGER)
    Step 3  Seasonal notification    — remind to retrain every ~90 days (manual trigger)

COMPARISON MODELS (optional, --all-models flag)
    XGB_Predict.py   — Design Option 1
    LGBM_Predict.py  — Design Option 3

RETRAINING (separate script, manual trigger)
    python Retraining_Pipeline.py --retrain
    This is NOT run automatically by Start.py. Instead, Start.py checks
    how many days have elapsed since the last model training and prints a
    prominent terminal notification when retraining is due (~every season /
    90 days). The actual retraining is always a manual decision.

HOW DEPLOYMENT WORKS
---------------------
1. A cron job or Windows Task Scheduler runs this script once per day
   (midnight or shortly after the sensor station uploads its daily reading).

2. Step 0: Sat_SensorData_proxy.py fetches the latest missing sensor rows
   and appends them to obando_environmental_data.csv. On a daily schedule
   this is typically 1 new row per run (fast, incremental).

3. Step 1: RF_Predict.run_pipeline() loads the updated CSV, builds features,
   filters to rows after LAST_TRAINING_DATE, and saves predictions + plot.

4. Step 2: If the latest prediction is WATCH/WARNING/DANGER, FB_Post.py
   is called to post an alert to the configured Facebook Page.

5. Step 3: Days since last model training are checked. If >= RETRAIN_NOTIFY_DAYS
   (default 90), a prominent seasonal retrain reminder is printed to the
   terminal and optionally written to a log file.

NO LABELS NEEDED AT RUNTIME
-----------------------------
The model was trained with satellite-derived flood labels (Sentinel-1 every
~12 days). At inference time NO satellite data and NO labels are needed. The
model learned "what sensor patterns precede a flood" during training.

Sentinel-1 data is only needed when retraining. After new satellite passes
accumulate (every 6-12 months), run:
    python Retraining_Pipeline.py --retrain

FIX vs old Start.py
-------------------
- Added Step 0: Sat_SensorData_proxy.py runs before RF_Predict so the sensor
  CSV is always current before predictions are made.
- Added Step 3: Seasonal retrain notification based on last_training_date in
  the .pkl artifact — no manual constants needed.
- Threshold is always read from the saved .pkl artifact.
- Sensor extractor is imported and called via run_pipeline() — same pattern
  as RF_Predict, single code path.

Usage
-----
    # Run once manually
    python Start.py

    # Run on a schedule (every 24 hours)
    python Start.py --schedule --interval 24

    # Also run XGB and LGBM for comparison logging
    python Start.py --all-models

    # Skip Facebook posting (dry run / testing)
    python Start.py --no-post

    # Skip sensor update (use existing CSV as-is)
    python Start.py --skip-sensor

    # Full scheduled run
    python Start.py --schedule --interval 24 --all-models
"""

import os
import sys
import time
import argparse
import warnings
import traceback
from datetime import date

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


# ===========================================================================
# CONFIG
# ===========================================================================

# Alert tiers that trigger a Facebook post
ALERT_TIERS = {"WATCH", "WARNING", "DANGER"}

# Days since last model training before a seasonal retrain notification fires.
# 90 days ≈ one season (wet/dry transition). Change to 180 for semi-annual.
RETRAIN_NOTIFY_DAYS = 90

# Path for retrain notification log (append-only)
LOG_DIR      = os.path.join(SCRIPT_DIR, "..", "logs")
RETRAIN_LOG  = os.path.join(LOG_DIR, "retrain_notifications.log")

# ===========================================================================
# END CONFIG
# ===========================================================================


def separator(title=""):
    line = "=" * 55
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ---------------------------------------------------------------------------
# Import pipeline modules
# ---------------------------------------------------------------------------

def import_sensor_module():
    """Import Sat_SensorData_proxy for Step 0 sensor update."""
    try:
        import Sat_SensorData_proxy
        return Sat_SensorData_proxy
    except ImportError as e:
        print(f"\n  WARNING: Could not import Sat_SensorData_proxy.py — {e}")
        print("  Sensor CSV will NOT be updated before prediction.")
        return None


def import_predict_modules():
    """Import RF_Predict (primary model). Hard exit if missing."""
    try:
        import RF_Predict
        return RF_Predict
    except ImportError as e:
        sys.exit(
            f"\n  ERROR: Could not import RF_Predict.py.\n"
            f"  Make sure it is in the same folder as Start.py.\n"
            f"  Detail: {e}\n"
        )


def import_comparison_modules():
    """Import XGB and LGBM predict scripts for optional comparison logging."""
    modules = {}
    for name in ("XGB_Predict", "LGBM_Predict"):
        try:
            mod = __import__(name)
            modules[name] = mod
            print(f"  Loaded: {name}.py")
        except ImportError as e:
            print(f"  WARNING: Could not import {name}.py — skipping. ({e})")
    return modules


def import_fb_post():
    """Import FB_Post.py softly — pipeline continues if missing."""
    try:
        import FB_Post
        return FB_Post
    except ImportError:
        print("  NOTE: FB_Post.py not found — Facebook posting disabled.")
        return None
    except Exception as e:
        print(f"  NOTE: FB_Post.py failed to import ({e}) — posting disabled.")
        return None


# ---------------------------------------------------------------------------
# Retrain notification
# ---------------------------------------------------------------------------

def check_retrain_notification(RF_Predict) -> None:
    """
    Read last_training_date from the RF model artifact.
    If >= RETRAIN_NOTIFY_DAYS have elapsed since that date, print a
    prominent terminal notification and log it.

    This is notification ONLY — retraining is always a manual step:
        python Retraining_Pipeline.py --retrain
    """
    try:
        import joblib
        model_path = RF_Predict.MODEL_PATH
        artifact   = joblib.load(model_path)
        last_train = artifact.get("last_training_date")

        if last_train is None:
            return

        last_train_date = pd.Timestamp(last_train).date()
        days_elapsed    = (date.today() - last_train_date).days
        months_elapsed  = days_elapsed / 30.0

        if days_elapsed >= RETRAIN_NOTIFY_DAYS:
            separator("⚠️  SEASONAL RETRAIN REMINDER")
            print(f"  Last model training : {last_train_date}")
            print(f"  Days elapsed        : {days_elapsed}  (~{months_elapsed:.1f} months)")
            print(f"  Threshold           : {RETRAIN_NOTIFY_DAYS} days")
            print()
            print("  The model has not been retrained this season.")
            print("  New Sentinel-1 passes and sensor readings have accumulated.")
            print("  Retraining is recommended to incorporate recent flood patterns.")
            print()
            print("  To retrain (manual step):")
            print("    python Retraining_Pipeline.py --retrain")
            print()
            print("  To also update the satellite label archive first:")
            print("    python Retraining_Pipeline.py --retrain")
            print("    (sentinel1_GEE.py runs automatically as Step 1)")
            separator()

            # Log the notification
            os.makedirs(LOG_DIR, exist_ok=True)
            try:
                with open(RETRAIN_LOG, "a") as f:
                    f.write(
                        f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}] "
                        f"Retrain notification fired — last_training_date={last_train_date}, "
                        f"days_elapsed={days_elapsed}\n"
                    )
            except Exception:
                pass
        else:
            days_remaining = RETRAIN_NOTIFY_DAYS - days_elapsed
            print(f"  Retrain check: {days_elapsed}d since last training "
                  f"({days_remaining}d until seasonal reminder)")

    except Exception as e:
        print(f"  NOTE: Could not check retrain status — {e}")


# ---------------------------------------------------------------------------
# Read latest prediction from saved CSV
# ---------------------------------------------------------------------------

def read_latest_result(csv_path: str) -> dict | None:
    if not os.path.exists(csv_path):
        print(f"  WARNING: Predictions CSV not found at {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
        if len(df) == 0:
            return None
        latest = df.iloc[-1]
        return {
            "timestamp":   str(latest.name.date()),
            "probability": float(latest.get("flood_probability", 0.0)),
            "risk_tier":   str(latest.get("risk_tier", "CLEAR")),
        }
    except Exception as e:
        print(f"  WARNING: Could not read predictions CSV — {e}")
        return None


# ---------------------------------------------------------------------------
# Facebook post dispatch
# ---------------------------------------------------------------------------

def dispatch_fb_post(FB_Post, tier: str, probability: float, timestamp: str) -> None:
    try:
        FB_Post.post_alert(risk_tier=tier, probability=probability, timestamp=timestamp)
    except AttributeError:
        try:
            FB_Post.post(risk_tier=tier, probability=probability, timestamp=timestamp)
        except AttributeError:
            print(
                "  WARNING: FB_Post.py has neither post_alert() nor post().\n"
                "  Add a post_alert(risk_tier, probability, timestamp) function."
            )
        except Exception as e2:
            print(f"  Facebook post failed: {e2}")
    except Exception as e:
        print(f"  Facebook post failed: {e}")


# ---------------------------------------------------------------------------
# Main daily orchestration
# ---------------------------------------------------------------------------

def run_daily(
    run_all_models: bool = False,
    no_post:        bool = False,
    skip_sensor:    bool = False,
) -> None:
    separator("Flood Early Warning System — Daily Inference")
    print(f"  Date            : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Primary model   : RF sensor (Design Option 2 — 91.1% Watch recall)")
    print(f"  Training cutoff : read from .pkl artifact")
    print(f"  Inference mode  : LIVE — sensor CSV only, no satellite required")
    print(f"  Sensor update   : {'disabled (--skip-sensor)' if skip_sensor else 'enabled (incremental append)'}")
    print(f"  Facebook post   : {'disabled (--no-post)' if no_post else 'enabled if WATCH/WARNING/DANGER'}")
    print(f"  Comparison run  : {'RF + XGB + LGBM' if run_all_models else 'RF only'}")

    # ── Step 0: Update sensor CSV ─────────────────────────────────────────
    separator("Step 0 — Sat_SensorData_proxy.py (sensor update)")
    if skip_sensor:
        print("  Skipped — using existing sensor CSV.")
    else:
        Sensor = import_sensor_module()
        if Sensor is not None:
            try:
                updated = Sensor.run_pipeline()
                if updated:
                    print("  ✓ Sensor CSV updated with latest readings.")
                else:
                    print("  Sensor CSV already up to date — no new rows appended.")
            except Exception as e:
                print(f"\n  WARNING: Sensor update failed: {e}")
                print("  Continuing with existing sensor CSV.")
                traceback.print_exc()
        else:
            print("  Sensor update skipped — module not available.")

    # ── Step 1: Import RF_Predict ─────────────────────────────────────────
    RF_Predict = import_predict_modules()

    # ── Step 2: Run RF pipeline ───────────────────────────────────────────
    separator("Step 1 — RF_Predict.py (primary — Design Option 2)")
    rf_ok = False
    try:
        RF_Predict.run_pipeline()
        rf_ok = True
    except SystemExit as e:
        print(f"\n  RF_Predict exited early: {e}")
        print("  Most likely cause: no new sensor rows after the training cutoff.")
        print("  Check that the sensor CSV has rows dated after the training cutoff.")
    except Exception as e:
        print(f"\n  RF_Predict.py ERROR: {e}")
        traceback.print_exc()

    # ── Step 3: Read result and decide on Facebook post ───────────────────
    latest = read_latest_result(RF_Predict.PREDICTIONS_CSV) if rf_ok else None

    if latest:
        tier  = latest["risk_tier"]
        prob  = latest["probability"]
        ts    = latest["timestamp"]
        emoji = {"CLEAR": "🟢", "WATCH": "🟡", "WARNING": "🟠", "DANGER": "🔴"}.get(tier, "")
        print(f"\n  Latest prediction : {ts}  {emoji} {tier}  ({prob:.1%})")

        if tier in ALERT_TIERS and not no_post:
            separator("Step 2 — FB_Post.py (alert dispatch)")
            FB_Post = import_fb_post()
            if FB_Post is not None:
                dispatch_fb_post(FB_Post, tier, prob, ts)
            else:
                print("  Facebook post skipped — FB_Post.py not available.")
        elif tier not in ALERT_TIERS:
            print("  CLEAR today — no Facebook post.")
        else:
            print("  --no-post flag set — skipping Facebook post.")
    else:
        if rf_ok:
            print("\n  WARNING: RF ran but could not read back results.")
        print("  Skipping Facebook post.")

    # ── Step 4: Seasonal retrain notification ─────────────────────────────
    separator("Step 3 — Seasonal Retrain Check")
    check_retrain_notification(RF_Predict)

    # ── Step 5: Optional comparison run (XGB + LGBM) ─────────────────────
    if run_all_models:
        comparison_mods = import_comparison_modules()

        if "XGB_Predict" in comparison_mods:
            separator("Step 4a — XGB_Predict.py (Design Option 1 — comparison)")
            try:
                comparison_mods["XGB_Predict"].run_pipeline()
            except SystemExit:
                print("  XGB: no new rows or early exit.")
            except Exception as e:
                print(f"  XGB_Predict.py ERROR: {e}")

        if "LGBM_Predict" in comparison_mods:
            separator("Step 4b — LGBM_Predict.py (Design Option 3 — comparison)")
            try:
                comparison_mods["LGBM_Predict"].run_pipeline()
            except SystemExit:
                print("  LGBM: no new rows or early exit.")
            except Exception as e:
                print(f"  LGBM_Predict.py ERROR: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    separator("DONE")
    if rf_ok:
        print(f"  RF CSV  : {RF_Predict.PREDICTIONS_CSV}")
        print(f"  RF plot : {RF_Predict.PLOT_FILE}")
    separator()


# ---------------------------------------------------------------------------
# Scheduled mode
# ---------------------------------------------------------------------------

def run_scheduled(interval_hours: int, run_all_models: bool, no_post: bool, skip_sensor: bool) -> None:
    print(f"\n  Scheduled mode — running every {interval_hours}h")
    print(f"  Press Ctrl+C to stop.\n")
    while True:
        try:
            run_daily(run_all_models=run_all_models, no_post=no_post, skip_sensor=skip_sensor)
            next_run = pd.Timestamp.now() + pd.Timedelta(hours=interval_hours)
            print(f"\n  Next run : {next_run.strftime('%Y-%m-%d %H:%M')}")
            time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            print("\n  Stopped by user.")
            break
        except Exception as exc:
            print(f"\n  Unexpected error: {exc}")
            traceback.print_exc()
            print(f"  Retrying in {interval_hours}h...")
            time.sleep(interval_hours * 3600)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Flood Early Warning System — Daily Inference Orchestrator.\n"
            "Step 0: update sensor CSV. Step 1: RF prediction. "
            "Step 2: FB post if alert. Step 3: seasonal retrain notification."
        )
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a repeating schedule instead of once.",
    )
    parser.add_argument(
        "--interval", type=int, default=24,
        help="Hours between scheduled runs (default: 24).",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Also run XGB_Predict.py and LGBM_Predict.py for comparison logging.",
    )
    parser.add_argument(
        "--no-post", action="store_true",
        help="Skip Facebook posting. Useful for testing.",
    )
    parser.add_argument(
        "--skip-sensor", action="store_true",
        help="Skip Step 0 sensor CSV update. Use existing CSV as-is.",
    )
    args = parser.parse_args()

    if args.schedule:
        run_scheduled(
            interval_hours=args.interval,
            run_all_models=args.all_models,
            no_post=args.no_post,
            skip_sensor=args.skip_sensor,
        )
    else:
        run_daily(
            run_all_models=args.all_models,
            no_post=args.no_post,
            skip_sensor=args.skip_sensor,
        )