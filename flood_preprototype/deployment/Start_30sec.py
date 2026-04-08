"""
Start_30sec.py
==============
Flood Prediction — Inference Orchestrator (30-second interval for testing)

Same as Start.py but runs every 30 seconds instead of midnight-pinned daily.

Usage
-----
    # Run once manually
    python Start_30sec.py

    # Run on 30-second loop (Ctrl+C to stop)
    python Start_30sec.py --schedule

    # Skip hardware sensor ingest
    python Start_30sec.py --schedule --skip-sensor

    # Skip GEE proxy fetch
    python Start_30sec.py --schedule --skip-proxy

    # Skip Facebook posting (dry run)
    python Start_30sec.py --schedule --no-post

    # Also run XGB and LGBM for comparison logging
    python Start_30sec.py --schedule --all-models
"""

import os
import sys
import time
import argparse
import warnings
import traceback
from datetime import date, datetime, timedelta

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "ml_pipeline"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "alerts"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "deployment"))

import Alert


# ===========================================================================
# CONFIG
# ===========================================================================

RETRAIN_NOTIFY_DAYS = 90
INTERVAL_SECONDS    = 10          # ← change this to adjust the loop interval

LOG_DIR     = os.path.join(SCRIPT_DIR, "..", "logs")
RETRAIN_LOG = os.path.join(LOG_DIR, "retrain_notifications.log")

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

def import_sensor_ingest():
    try:
        import sensor_ingest
        return sensor_ingest
    except ImportError as e:
        print(f"\n  WARNING: Could not import sensor_ingest.py — {e}")
        print("  Hardware sensor CSV will NOT be updated.")
        return None


def import_proxy_module():
    try:
        import Sat_SensorData_proxy
        return Sat_SensorData_proxy
    except ImportError as e:
        print(f"\n  WARNING: Could not import Sat_SensorData_proxy.py — {e}")
        print("  Proxy (satellite) CSV will NOT be updated.")
        return None


def import_merge_module():
    try:
        import merge_sensor
        return merge_sensor
    except ImportError as e:
        print(f"\n  WARNING: Could not import merge_sensor.py — {e}")
        print("  combined_sensor_context.csv will NOT be updated.")
        return None


def import_predict_module():
    try:
        import RF_Predict
        return RF_Predict
    except ImportError as e:
        sys.exit(
            f"\n  ERROR: Could not import RF_Predict.py.\n"
            f"  Make sure it is in the same folder as Start_30sec.py.\n"
            f"  Detail: {e}\n"
        )


def import_comparison_modules():
    modules = {}
    for name in ("XGB_Predict", "LGBM_Predict"):
        try:
            mod = __import__(name)
            modules[name] = mod
            print(f"  Loaded: {name}.py")
        except ImportError as e:
            print(f"  WARNING: Could not import {name}.py — skipping. ({e})")
    return modules


# ---------------------------------------------------------------------------
# Retrain notification
# ---------------------------------------------------------------------------

def check_retrain_notification(RF_Predict) -> None:
    try:
        import joblib
        artifact   = joblib.load(RF_Predict.MODEL_FILE)
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
            separator()

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

def read_latest_result(RF_Predict) -> dict | None:
    csv_path = RF_Predict.PREDICTIONS_CSV
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
# Countdown sleep
# ---------------------------------------------------------------------------

def sleep_with_countdown(seconds: int) -> None:
    """Sleep for `seconds` with a live terminal countdown."""
    print(f"\n  Next run in {seconds} seconds...")
    for remaining in range(seconds, 0, -1):
        print(f"\r  Countdown: {remaining}s ", end="", flush=True)
        time.sleep(1)
    print()


# ---------------------------------------------------------------------------
# Main daily orchestration
# ---------------------------------------------------------------------------

def run_daily(
    skip_sensor:    bool = False,
    skip_proxy:     bool = False,
    no_post:        bool = False,
    run_all_models: bool = False,
) -> None:

    separator("Flood Early Warning System — Inference Run")
    print(f"  Run time        : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Primary model   : RF sensor (live mode — incremental)")
    print(f"  Hardware ingest : {'disabled (--skip-sensor)' if skip_sensor else 'enabled (Supabase incremental)'}")
    print(f"  Proxy fetch     : {'disabled (--skip-proxy)'  if skip_proxy  else 'enabled (GEE incremental)'}")
    print(f"  Merge           : enabled (combined_sensor_context.csv incremental)")
    print(f"  Facebook post   : {'disabled (--no-post)' if no_post else 'enabled if WATCH/WARNING/DANGER'}")
    print(f"  Comparison run  : {'RF + XGB + LGBM' if run_all_models else 'RF only'}")
    print(f"  Alert source    : flood_rf_sensor_predictions.csv (always RF)")
    print(f"  Interval        : every {INTERVAL_SECONDS} seconds")

    # ── Step 0a: Hardware sensor ingest ──────────────────────────────────
    separator("Step 0a — sensor_ingest.py (Supabase hardware pull)")
    if skip_sensor:
        print("  Skipped — --skip-sensor flag set.")
    else:
        SensorIngest = import_sensor_ingest()
        if SensorIngest is not None:
            try:
                updated = SensorIngest.ingest_latest()
                if updated:
                    print("  ✓ Hardware sensor CSV updated with latest Supabase rows.")
                else:
                    print("  Hardware sensor CSV already up to date — no new rows.")
            except Exception as e:
                print(f"\n  WARNING: Hardware ingest failed: {e}")
                print("  Continuing with existing obando_sensor_data.csv.")
                traceback.print_exc()
        else:
            print("  Skipped — sensor_ingest module not available.")

    # ── Step 0b: Proxy data fetch ─────────────────────────────────────────
    separator("Step 0b — Sat_SensorData_proxy.py (GEE proxy pull)")
    if skip_proxy:
        print("  Skipped — --skip-proxy flag set.")
    else:
        Proxy = import_proxy_module()
        if Proxy is not None:
            try:
                updated = Proxy.run_pipeline(force_full=False)
                if updated:
                    print("  ✓ Proxy CSV updated with latest GEE data.")
                else:
                    print("  Proxy CSV already up to date — no new rows.")
            except Exception as e:
                print(f"\n  WARNING: Proxy fetch failed: {e}")
                print("  Continuing with existing obando_environmental_data.csv.")
                traceback.print_exc()
        else:
            print("  Skipped — Sat_SensorData_proxy module not available.")

    # ── Step 0c: Merge ────────────────────────────────────────────────────
    separator("Step 0c — merge_sensor.py (merge proxy + hardware)")
    Merge = import_merge_module()
    if Merge is not None:
        try:
            new_rows = Merge.run_pipeline()
            if new_rows is not None and len(new_rows) > 0:
                print(f"  ✓ Merged {len(new_rows)} new row(s) into combined_sensor_context.csv.")
            else:
                print("  combined_sensor_context.csv already up to date — no new rows.")
        except Exception as e:
            print(f"\n  WARNING: Merge failed: {e}")
            print("  RF_Predict will use whatever combined_sensor_context.csv currently exists.")
            traceback.print_exc()
    else:
        print("  Skipped — merge_sensor module not available.")
        print("  RF_Predict will fall back to obando_environmental_data.csv (proxy only).")

    # ── Step 1: Import RF_Predict ─────────────────────────────────────────
    RF_Predict = import_predict_module()

    # ── Step 2: Run RF pipeline ───────────────────────────────────────────
    separator("Step 1 — RF_Predict.py (live mode — incremental)")
    rf_ok = False
    try:
        RF_Predict.run_pipeline()
        rf_ok = True
    except SystemExit as e:
        print(f"\n  RF_Predict exited early: {e}")
        print("  Most likely: no new sensor rows after the training cutoff.")
        print("  Check that combined_sensor_context.csv has rows after LAST_TRAINING_DATE.")
    except Exception as e:
        print(f"\n  RF_Predict.py ERROR: {e}")
        traceback.print_exc()

    # ── Step 3: Optional comparison run ──────────────────────────────────
    if run_all_models:
        comparison_mods = import_comparison_modules()

        if "XGB_Predict" in comparison_mods:
            separator("Step 2a — XGB_Predict.py (comparison)")
            try:
                comparison_mods["XGB_Predict"].run_pipeline()
            except SystemExit:
                print("  XGB: no new rows or early exit.")
            except Exception as e:
                print(f"  XGB_Predict.py ERROR: {e}")

        if "LGBM_Predict" in comparison_mods:
            separator("Step 2b — LGBM_Predict.py (comparison)")
            try:
                comparison_mods["LGBM_Predict"].run_pipeline()
            except SystemExit:
                print("  LGBM: no new rows or early exit.")
            except Exception as e:
                print(f"  LGBM_Predict.py ERROR: {e}")

    # ── Step 4: Alert dispatch ────────────────────────────────────────────
    separator("Step 3 — Alert.py (dispatch from flood_rf_sensor_predictions.csv)")
    latest = read_latest_result(RF_Predict) if rf_ok else None

    if latest:
        tier  = latest["risk_tier"]
        prob  = latest["probability"]
        ts    = latest["timestamp"]
        emoji = {"CLEAR": "🟢", "WATCH": "🟡", "WARNING": "🟠", "DANGER": "🔴"}.get(tier, "")
        print(f"\n  Latest RF prediction : {ts}  {emoji} {tier}  ({prob:.1%})")

        if no_post:
            print("  --no-post flag set — skipping alert dispatch.")
        elif tier in Alert.ALERT_TIERS:
            Alert.dispatch_alert(tier=tier, probability=prob, timestamp=ts)
        else:
            Alert.dispatch_alert(tier=tier, probability=prob, timestamp=ts)
    else:
        if rf_ok:
            print("\n  WARNING: RF ran but could not read back results.")
        print("  Skipping alert dispatch.")

    # ── Step 5: Seasonal retrain notification ─────────────────────────────
    separator("Step 4 — Seasonal Retrain Check")
    check_retrain_notification(RF_Predict)

    # ── Summary ───────────────────────────────────────────────────────────
    separator("DONE")
    if rf_ok:
        print(f"  RF CSV  : {RF_Predict.PREDICTIONS_CSV}")
        print(f"  RF plot : {RF_Predict.PLOT_FILE}")
    separator()


# ---------------------------------------------------------------------------
# Scheduled mode — 30-second interval with countdown
# ---------------------------------------------------------------------------

def run_scheduled(
    skip_sensor:    bool = False,
    skip_proxy:     bool = False,
    no_post:        bool = False,
    run_all_models: bool = False,
) -> None:
    print(f"\n  Scheduled mode — runs every {INTERVAL_SECONDS} seconds")
    print(f"  Press Ctrl+C to stop.\n")

    while True:
        try:
            run_daily(
                skip_sensor    = skip_sensor,
                skip_proxy     = skip_proxy,
                no_post        = no_post,
                run_all_models = run_all_models,
            )
            sleep_with_countdown(INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\n  Stopped by user.")
            break
        except Exception as exc:
            print(f"\n  Unexpected error in scheduled run: {exc}")
            traceback.print_exc()
            print(f"  Retrying in {INTERVAL_SECONDS} seconds...")
            sleep_with_countdown(INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Flood Early Warning System — Inference Orchestrator (30-second interval).\n"
            "Steps: hardware ingest → proxy fetch → merge → RF predict → "
            "XGB/LGBM (optional) → alert → retrain check.\n"
            "All steps are incremental — only new rows processed each run.\n"
            "Alert always dispatches from flood_rf_sensor_predictions.csv."
        )
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help=f"Run on a {INTERVAL_SECONDS}-second loop instead of once.",
    )
    parser.add_argument(
        "--skip-sensor", action="store_true",
        help="Skip Step 0a — Supabase hardware ingest.",
    )
    parser.add_argument(
        "--skip-proxy", action="store_true",
        help="Skip Step 0b — GEE proxy fetch.",
    )
    parser.add_argument(
        "--no-post", action="store_true",
        help="Skip all alert dispatch. Useful for testing.",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Also run XGB_Predict.py and LGBM_Predict.py for comparison logging.",
    )
    args = parser.parse_args()

    if args.schedule:
        run_scheduled(
            skip_sensor    = args.skip_sensor,
            skip_proxy     = args.skip_proxy,
            no_post        = args.no_post,
            run_all_models = args.all_models,
        )
    else:
        run_daily(
            skip_sensor    = args.skip_sensor,
            skip_proxy     = args.skip_proxy,
            no_post        = args.no_post,
            run_all_models = args.all_models,
        )