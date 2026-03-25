"""
Start.py
========
Flood Prediction — Daily Inference Orchestrator

PURPOSE
-------
This is the single entry point for daily operational deployment.
It does NOT reimplement any logic — it calls the existing pipeline
scripts that are already in this folder:

    RF_Predict.py      — Design Option 2 (recommended deployment model)
    XGB_Predict.py     — Design Option 1 (comparison / reference)
    LGBM_Predict.py    — Design Option 3 (comparison / reference)
    FB_Post.py         — Facebook Graph API alert posting

HOW DEPLOYMENT WORKS
---------------------
1. A cron job (Windows Task Scheduler or Linux cron) runs this script
   once per day — typically at midnight or shortly after the sensor
   station uploads its daily reading.

2. This script calls RF_Predict.run_pipeline() in LIVE mode.
   Live mode inside RF_Predict.py:
     - Loads the full sensor CSV history (needed for rolling-window
       features like waterlevel_pct_rank_30d which needs 30 days context)
     - Builds sensor-only features via feature_engineering.py
     - Filters to rows after LAST_TRAINING_DATE (unseen rows only)
     - Runs flood_rf_sensor.pkl
     - Saves predictions CSV and plot to ../predictions/

3. If the resulting alert tier is WATCH, WARNING, or DANGER, it calls
   FB_Post.py to post an alert to the configured Facebook Page.

4. Optionally also runs XGB_Predict.py and LGBM_Predict.py for
   comparison logging (use --all-models flag).

NO LABELS NEEDED AT RUNTIME
-----------------------------
The model was trained with satellite-derived flood labels (Sentinel-1
every 12 days). At inference time NO satellite data and NO labels are
needed. The model learned "what sensor patterns precede a flood" during
training. In deployment it outputs a flood probability from sensor
features only.

Sentinel-1 data is only needed when retraining. After new satellite
passes accumulate (every 6-12 months), run:
    python prepare_dataset.py          (rebuild flood_dataset.csv)
    python RF_train_flood_model.py     (retrain RF)
Then update LAST_TRAINING_DATE below and in RF_Predict.py to match
the new VAL_END.

FIX vs old Start.py
-------------------
Old version reimplemented feature building and model loading from
scratch, duplicating logic already in RF_Predict.py. It also had a
threshold offset bug where WATCH_OFFSET = -0.07 could push the watch
threshold to ~0.0 when the artifact threshold is 0.05.

New version imports run_pipeline() directly from RF_Predict so there
is exactly one code path for both daily deployment and batch evaluation.
Threshold is always read from the saved .pkl artifact — no manual
constants, no offset arithmetic.

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

    # Full scheduled run with all models and FB posting
    python Start.py --schedule --interval 24 --all-models
"""

import os
import sys
import time
import argparse
import warnings
import traceback

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


# ===========================================================================
# CONFIG
# ===========================================================================

# LAST_TRAINING_DATE is no longer a manual constant here.
# RF_Predict.load_model() reads it automatically from the saved .pkl
# artifact (written by RF_train_flood_model.py as artifact["last_training_date"]).
# After retraining you only need to deploy the new .pkl — no config edits.

# Alert tiers that trigger a Facebook post
ALERT_TIERS = {"WATCH", "WARNING", "DANGER"}

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
# Import existing pipeline scripts
# ---------------------------------------------------------------------------

def import_predict_modules():
    """
    Import the three predict scripts from the same folder.
    Each already has run_pipeline() with full live-mode logic.
    Calling run_pipeline() with no arguments = live mode.
    """
    try:
        import RF_Predict
        return RF_Predict
    except ImportError as e:
        sys.exit(
            f"\n  ERROR: Could not import RF_Predict.py.\n"
            f"  Make sure it is in the same folder as Start.py.\n"
            f"  Path checked: {SCRIPT_DIR}\n"
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
    """
    Import FB_Post.py for Facebook alert posting.
    Returns None gracefully if not configured — pipeline still runs,
    just skips the social post.
    """
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
# Read latest prediction from the saved CSV
# ---------------------------------------------------------------------------

def read_latest_result(csv_path: str) -> dict | None:
    """
    Read the most recent row from the predictions CSV that RF_Predict
    just wrote. Returns the tier, probability, and timestamp so we can
    decide whether to post to Facebook.
    """
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
    """
    Call FB_Post.py to post the alert. Tries post_alert() first (the
    expected function name), falls back to post() if not found.
    """
    try:
        FB_Post.post_alert(
            risk_tier=tier,
            probability=probability,
            timestamp=timestamp,
        )
    except AttributeError:
        # FB_Post.py may use a different function name
        try:
            FB_Post.post(risk_tier=tier, probability=probability, timestamp=timestamp)
        except AttributeError:
            print(
                "  WARNING: FB_Post.py has neither post_alert() nor post().\n"
                "  Add a post_alert(risk_tier, probability, timestamp) function to FB_Post.py."
            )
        except Exception as e2:
            print(f"  Facebook post failed: {e2}")
    except Exception as e:
        print(f"  Facebook post failed: {e}")


# ---------------------------------------------------------------------------
# Main daily orchestration
# ---------------------------------------------------------------------------

def run_daily(run_all_models: bool = False, no_post: bool = False) -> None:
    separator("Flood Early Warning System — Daily Inference")
    print(f"  Date            : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Primary model   : RF sensor (Design Option 2 — 91.1% Watch recall)")
    print(f"  Training cutoff : read from .pkl artifact (set during last retrain)")
    print(f"  Inference mode  : LIVE — sensor CSV only, no satellite required")
    print(f"  Facebook post   : {'disabled (--no-post)' if no_post else 'enabled if WATCH/WARNING/DANGER'}")
    print(f"  Comparison run  : {'RF + XGB + LGBM' if run_all_models else 'RF only (use --all-models for others)'}")

    # ── Step 1: Import RF_Predict (primary model) ─────────────────────────
    RF_Predict = import_predict_modules()

    # ── Step 2: Run RF pipeline in live mode ──────────────────────────────
    # run_pipeline() with no arguments = live mode:
    #   - loads full sensor CSV (../data/sensor/obando_environmental_data.csv)
    #   - builds features via feature_engineering.py
    #   - filters to rows after LAST_TRAINING_DATE in RF_Predict.py's CONFIG
    #   - runs flood_rf_sensor.pkl
    #   - saves ../predictions/flood_rf_live_predictions.csv and .png
    separator("Step 1 — RF_Predict.py (primary — Design Option 2)")
    rf_ok = False
    try:
        RF_Predict.run_pipeline()
        rf_ok = True
    except SystemExit as e:
        print(f"\n  RF_Predict exited early: {e}")
        print("  Most likely cause: no new sensor rows after the training cutoff.")
        print(f"  Check that the sensor CSV has rows dated after the training cutoff.")
        print("  If you have retrained, deploy the new .pkl files — no config edits needed.")
        print("  Start.py and RF_Predict.py to the new VAL_END date.")
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

    # ── Step 4: Optional comparison run (XGB + LGBM) ─────────────────────
    if run_all_models:
        comparison_mods = import_comparison_modules()

        if "XGB_Predict" in comparison_mods:
            separator("Step 3a — XGB_Predict.py (Design Option 1 — comparison)")
            try:
                comparison_mods["XGB_Predict"].run_pipeline()
            except SystemExit:
                print("  XGB: no new rows or early exit.")
            except Exception as e:
                print(f"  XGB_Predict.py ERROR: {e}")

        if "LGBM_Predict" in comparison_mods:
            separator("Step 3b — LGBM_Predict.py (Design Option 3 — comparison)")
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

def run_scheduled(interval_hours: int, run_all_models: bool, no_post: bool) -> None:
    print(f"\n  Scheduled mode — running every {interval_hours}h")
    print(f"  Press Ctrl+C to stop.\n")
    while True:
        try:
            run_daily(run_all_models=run_all_models, no_post=no_post)
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
            "Runs RF_Predict.py in live mode then posts to Facebook if alert."
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
    args = parser.parse_args()

    if args.schedule:
        run_scheduled(
            interval_hours=args.interval,
            run_all_models=args.all_models,
            no_post=args.no_post,
        )
    else:
        run_daily(
            run_all_models=args.all_models,
            no_post=args.no_post,
        )