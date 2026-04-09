"""
Start.py
========
Flood Prediction — Daily Inference Orchestrator

PURPOSE
-------
Single entry point for daily operational deployment.
Runs the full pipeline in order, all incremental:

    Step 0a  sensor_ingest.py         — pull new hardware readings from Supabase
    Step 0b  Sat_SensorData_proxy.py  — pull new proxy data from GEE
    Step 0c  merge_sensor.py          — merge both into combined_sensor_context.csv
    Step 1   XGB_Predict.py           — live mode inference on combined CSV
    Step 2   RF_Predict.py            — comparison run (if --all-models)
    Step 3   LGBM_Predict.py          — comparison run (if --all-models)
    Step 4   Alert.py                 — dispatch alert based on flood_xgb_sensor_predictions.csv
    Step 5   Seasonal notification    — remind to retrain every ~90 days (manual trigger)

ALL STEPS ARE INCREMENTAL
--------------------------
Every script only processes rows newer than what is already on disk:
    sensor_ingest   — fetches Supabase rows newer than latest obando_sensor_data.csv timestamp
    Sat_SensorData  — fetches GEE rows newer than latest obando_environmental_data.csv timestamp
    merge_sensor    — merges only rows newer than latest combined_sensor_context.csv timestamp
    XGB_Predict     — predicts only rows after LAST_TRAINING_DATE (filter_new_rows)
    Alert           — FB check_duplicate=True; Supabase appends only new prediction rows

ALERT DISPATCH
--------------
Alert.py always runs last, after all model predictions are complete.
It reads the latest row from flood_xgb_sensor_predictions.csv — the XGB
model is the primary operational model and is always the alert source.
RF and LGBM are comparison-only and do not trigger alerts.

SCHEDULER — MIDNIGHT-PINNED
-----------------------------
    python Start.py --schedule

Runs once immediately on start, then sleeps until the next calendar midnight
(local system time) before each subsequent run. This means the pipeline always
fires at midnight regardless of when you first started the script — not every
24 hours from launch time.

If a run takes longer than expected and the next midnight has already passed,
the script runs immediately and re-aligns to the following midnight.

RETRAINING (manual, separate script)
--------------------------------------
    python Retraining_Pipeline.py --retrain

NOT run automatically. Start.py checks days since last training and prints
a prominent terminal notice when retraining is due (~every 90 days).

Usage
-----
    # Run once manually
    python Start.py

    # Run on midnight schedule (keeps running, Ctrl+C to stop)
    python Start.py --schedule

    # Skip hardware sensor ingest (no Supabase connection / testing)
    python Start.py --skip-sensor

    # Skip GEE proxy fetch (no GEE auth / testing)
    python Start.py --skip-proxy

    # Skip Facebook posting (dry run / testing)
    python Start.py --no-post

    # Full scheduled run
    python Start.py --schedule

    # Also run RF and LGBM for comparison logging
    python Start.py --all-models
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

# Days since last model training before a seasonal retrain notification fires.
# 90 days ≈ one season (wet/dry transition).
RETRAIN_NOTIFY_DAYS = 90

# Path for retrain notification log (append-only)
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
    """Import sensor_ingest for Step 0a — Supabase hardware pull."""
    try:
        import sensor_ingest
        return sensor_ingest
    except ImportError as e:
        print(f"\n  WARNING: Could not import sensor_ingest.py — {e}")
        print("  Hardware sensor CSV will NOT be updated.")
        return None


def import_proxy_module():
    """Import Sat_SensorData_proxy for Step 0b — GEE proxy pull."""
    try:
        import Sat_SensorData_proxy
        return Sat_SensorData_proxy
    except ImportError as e:
        print(f"\n  WARNING: Could not import Sat_SensorData_proxy.py — {e}")
        print("  Proxy (satellite) CSV will NOT be updated.")
        return None


def import_merge_module():
    """Import merge_sensor for Step 0c — merge proxy + hardware."""
    try:
        import merge_sensor
        return merge_sensor
    except ImportError as e:
        print(f"\n  WARNING: Could not import merge_sensor.py — {e}")
        print("  combined_sensor_context.csv will NOT be updated.")
        return None


def import_predict_module():
    """Import XGB_Predict (primary model). Hard exit if missing."""
    try:
        import XGB_Predict
        return XGB_Predict
    except ImportError as e:
        sys.exit(
            f"\n  ERROR: Could not import XGB_Predict.py.\n"
            f"  Make sure it is in the same folder as Start.py.\n"
            f"  Detail: {e}\n"
        )


def import_comparison_modules():
    """Import RF and LGBM predict scripts for optional comparison logging."""
    modules = {}
    for name in ("RF_Predict", "LGBM_Predict"):
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

def check_retrain_notification(XGB_Predict) -> None:
    """
    Read last_training_date from the XGB model artifact.
    If >= RETRAIN_NOTIFY_DAYS have elapsed, print a prominent terminal
    notification and log it. Retraining is always a manual step.
    """
    try:
        import joblib
        artifact   = joblib.load(XGB_Predict.MODEL_FILE)
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

def read_latest_result(XGB_Predict) -> dict | None:
    """
    Read the most recent row from flood_xgb_sensor_predictions.csv.
    This is always the alert source — XGB is the primary operational model.
    RF and LGBM comparison runs do not affect alert dispatch.
    """
    csv_path = XGB_Predict.PREDICTIONS_CSV
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
# Midnight-pinned sleep
# ---------------------------------------------------------------------------

def seconds_until_midnight() -> float:
    """
    Return the number of seconds from now until the next calendar midnight
    (local system time). Minimum 1 second to avoid tight loops.
    """
    now       = datetime.now()
    tomorrow  = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    delta     = (tomorrow - now).total_seconds()
    return max(delta, 1.0)


# ---------------------------------------------------------------------------
# Main daily orchestration
# ---------------------------------------------------------------------------

def run_daily(
    skip_sensor:    bool = False,
    skip_proxy:     bool = False,
    no_post:        bool = False,
    run_all_models: bool = False,
) -> None:

    separator("Flood Early Warning System — Daily Inference")
    print(f"  Run time        : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Primary model   : XGB sensor (live mode — incremental)")
    print(f"  Hardware ingest : {'disabled (--skip-sensor)' if skip_sensor else 'enabled (Supabase incremental)'}")
    print(f"  Proxy fetch     : {'disabled (--skip-proxy)'  if skip_proxy  else 'enabled (GEE incremental)'}")
    print(f"  Merge           : enabled (combined_sensor_context.csv incremental)")
    print(f"  Facebook post   : {'disabled (--no-post)' if no_post else 'enabled if WATCH/WARNING/DANGER'}")
    print(f"  Comparison run  : {'XGB + RF + LGBM' if run_all_models else 'XGB only'}")
    print(f"  Alert source    : flood_xgb_sensor_predictions.csv (always XGB)")

    # ── Step 0a: Hardware sensor ingest (Supabase → obando_sensor_data.csv) ──
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

    # ── Step 0b: Proxy data fetch (GEE → obando_environmental_data.csv) ────
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

    # ── Step 0c: Merge (proxy + hardware → combined_sensor_context.csv) ────
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
            print("  XGB_Predict will use whatever combined_sensor_context.csv currently exists.")
            traceback.print_exc()
    else:
        print("  Skipped — merge_sensor module not available.")
        print("  XGB_Predict will fall back to obando_environmental_data.csv (proxy only).")

    # ── Step 1: Import XGB_Predict ─────────────────────────────────────────
    XGB_Predict = import_predict_module()

    # ── Step 2: Run XGB pipeline (live mode — incremental) ─────────────────
    separator("Step 1 — XGB_Predict.py (live mode — incremental)")
    xgb_ok = False
    try:
        XGB_Predict.run_pipeline()
        xgb_ok = True
    except SystemExit as e:
        print(f"\n  XGB_Predict exited early: {e}")
        print("  Most likely: no new sensor rows after the training cutoff.")
        print("  Check that combined_sensor_context.csv has rows after LAST_TRAINING_DATE.")
    except Exception as e:
        print(f"\n  XGB_Predict.py ERROR: {e}")
        traceback.print_exc()

    # ── Step 3: Optional comparison run (RF + LGBM) ────────────────────────
    if run_all_models:
        comparison_mods = import_comparison_modules()

        if "RF_Predict" in comparison_mods:
            separator("Step 2a — RF_Predict.py (comparison)")
            try:
                comparison_mods["RF_Predict"].run_pipeline()
            except SystemExit:
                print("  RF: no new rows or early exit.")
            except Exception as e:
                print(f"  RF_Predict.py ERROR: {e}")

        if "LGBM_Predict" in comparison_mods:
            separator("Step 2b — LGBM_Predict.py (comparison)")
            try:
                comparison_mods["LGBM_Predict"].run_pipeline()
            except SystemExit:
                print("  LGBM: no new rows or early exit.")
            except Exception as e:
                print(f"  LGBM_Predict.py ERROR: {e}")

    # ── Step 4: Alert dispatch — always based on XGB predictions CSV ───────
    #
    # Reads flood_xgb_sensor_predictions.csv after ALL model runs are complete.
    # RF and LGBM are comparison-only — they never affect alert dispatch.
    #
    separator("Step 3 — Alert.py (dispatch from flood_xgb_sensor_predictions.csv)")
    latest = read_latest_result(XGB_Predict) if xgb_ok else None

    if latest:
        tier  = latest["risk_tier"]
        prob  = latest["probability"]
        ts    = latest["timestamp"]
        emoji = {"CLEAR": "🟢", "WATCH": "🟡", "WARNING": "🟠", "DANGER": "🔴"}.get(tier, "")
        print(f"\n  Latest XGB prediction : {ts}  {emoji} {tier}  ({prob:.1%})")

        if no_post:
            print("  --no-post flag set — skipping alert dispatch.")
        elif tier in Alert.ALERT_TIERS:
            Alert.dispatch_alert(tier=tier, probability=prob, timestamp=ts)
        else:
            # CLEAR — FB + Supabase still fire via ALWAYS_FIRE_CHANNELS
            Alert.dispatch_alert(tier=tier, probability=prob, timestamp=ts)
    else:
        if xgb_ok:
            print("\n  WARNING: XGB ran but could not read back results.")
        print("  Skipping alert dispatch.")

    # ── Step 5: Seasonal retrain notification ──────────────────────────────
    separator("Step 4 — Seasonal Retrain Check")
    check_retrain_notification(XGB_Predict)

    # ── Summary ────────────────────────────────────────────────────────────
    separator("DONE")
    if xgb_ok:
        print(f"  XGB CSV  : {XGB_Predict.PREDICTIONS_CSV}")
        print(f"  XGB plot : {XGB_Predict.PLOT_FILE}")
    separator()


# ---------------------------------------------------------------------------
# Scheduled mode — midnight-pinned
# ---------------------------------------------------------------------------

def run_scheduled(
    skip_sensor:    bool = False,
    skip_proxy:     bool = False,
    no_post:        bool = False,
    run_all_models: bool = False,
) -> None:
    print(f"\n  Scheduled mode — midnight-pinned (runs daily at 00:00 local time)")
    print(f"  Press Ctrl+C to stop.\n")

    while True:
        try:
            run_daily(
                skip_sensor    = skip_sensor,
                skip_proxy     = skip_proxy,
                no_post        = no_post,
                run_all_models = run_all_models,
            )

            total_secs = seconds_until_midnight()
            next_run   = datetime.now() + timedelta(seconds=total_secs)
            next_str   = next_run.strftime("%Y-%m-%d 00:00")
            print(f"\n  Next run : {next_str}")

            remaining = total_secs
            try:
                while remaining > 0:
                    h, r = divmod(int(remaining), 3600)
                    m, s = divmod(r, 60)
                    print(f"  Sleeping : {h:02d}:{m:02d}:{s:02d} until midnight", end="\r", flush=True)
                    time.sleep(1)
                    remaining -= 1
                print(" " * 55, end="\r")  # clear the line
            except KeyboardInterrupt:
                print("\n  Stopped by user.")
                return

        except KeyboardInterrupt:
            print("\n  Stopped by user.")
            break
        except Exception as exc:
            print(f"\n  Unexpected error in scheduled run: {exc}")
            traceback.print_exc()
            secs = seconds_until_midnight()
            print(f"  Retrying at next midnight...")
            time.sleep(secs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Flood Early Warning System — Daily Inference Orchestrator.\n"
            "Steps: hardware ingest → proxy fetch → merge → XGB predict → "
            "RF/LGBM (optional) → alert → retrain check.\n"
            "All steps are incremental — only new rows processed each run.\n"
            "Alert always dispatches from flood_xgb_sensor_predictions.csv."
        )
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a midnight-pinned daily schedule instead of once.",
    )
    parser.add_argument(
        "--skip-sensor", action="store_true",
        help="Skip Step 0a — Supabase hardware ingest. Use existing obando_sensor_data.csv.",
    )
    parser.add_argument(
        "--skip-proxy", action="store_true",
        help="Skip Step 0b — GEE proxy fetch. Use existing obando_environmental_data.csv.",
    )
    parser.add_argument(
        "--no-post", action="store_true",
        help="Skip all alert dispatch. Useful for testing.",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Also run RF_Predict.py and LGBM_Predict.py for comparison logging.",
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