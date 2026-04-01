"""
Retraining_Pipeline.py
======================
Flood Early Warning System — Periodic Retraining Orchestrator

PURPOSE
-------
This script runs the FULL retraining cycle whenever new satellite data
is available. It is NOT part of the daily prediction loop — it runs
on a separate, longer schedule (every 12–14 days to match Sentinel-1
revisit, or manually when you decide to retrain).

WHEN TO RUN
-----------
    Trigger 1 — Sentinel-1 revisit (every ~12 days)
        New SAR passes are available from Copernicus. Run this script
        to pull them, extend the flood label archive, and optionally
        retrain if enough new labeled data has accumulated.

    Trigger 2 — Periodic retraining (every 6–12 months)
        After 6+ months of new data, full retraining is recommended
        to incorporate new flood patterns into the model weights.

    Trigger 3 — Manual
        Run at any time with: python Retraining_Pipeline.py --retrain

WHAT THIS SCRIPT DOES
---------------------
    Step 1  sentinel1_GEE.py — pull new Sentinel-1 SAR passes from
            Google Earth Engine, compute flood labels (triple condition),
            extend sentinel1_timeseries.csv.

    Step 2  prepare_dataset.py — rebuild flood_dataset.csv from the
            updated sensor + satellite data.

    Step 3  (optional, --retrain flag)
            RF_train_flood_model.py  — retrain Random Forest
            XGB_train_flood_model.py — retrain XGBoost
            LGBM_train_flood_model.py — retrain LightGBM

            After retraining, new .pkl files are written to ../model/
            with the updated last_training_date baked in. The daily
            Start.py reads this automatically — no manual config edit.

    Step 4  extract_test_set.py — regenerate test CSV if requested.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
    - It does NOT touch Start.py's constants — no manual edits needed.
    - It does NOT run predictions. That is Start.py's job.
    - It does NOT require the sensor CSV to be updated — sensor data
      collection is a separate process (sensor station or GEE extractor).

ON SENSOR DATA (the GEE extractor / real sensor question)
----------------------------------------------------------
    Currently: Sat_SensorData_proxy.py (GEE extractor) generates the
    sensor CSV (waterlevel, soil_moisture, humidity) from satellite
    products as a proxy for real hardware sensors.

    Future: When real hardware sensors are deployed, they will write
    to the same CSV format (timestamp, waterlevel, soil_moisture,
    humidity). The model does not care about the source — only the
    column names and value distribution matter.

    IMPORTANT — Distribution shift when switching to real sensors:
    The model was trained on z-scored GEE-derived water level values.
    Real sensors will produce different absolute values. To handle this:

        Phase 1 (first 6 months of real sensors):
            Run both GEE extractor AND real sensors in parallel.
            Validate that the real sensor readings correlate with
            GEE-derived values in the same conditions.

        Phase 2 (after first confirmed flood event on real sensors):
            Append real sensor rows to the existing CSV and retrain.
            Keep 2017–2022 GEE rows for historical label coverage.
            The model will learn the real sensor's distribution.

        Phase 3 (after 12+ months of real data):
            Retrain on real sensor data only. GEE extractor retired.

    The z-score normalisation in prepare_dataset.py is computed on the
    full input series — when real sensor data dominates, the z-scores
    will automatically recalibrate to the real sensor's range.

SCHEDULE RECOMMENDATION
-----------------------
    Daily (midnight):           python Start.py
    Every 12–14 days:           python Retraining_Pipeline.py
    Every 6–12 months:          python Retraining_Pipeline.py --retrain

    Windows Task Scheduler:
        Action: python Retraining_Pipeline.py
        Trigger: Weekly, every Sunday at 02:00

    Linux cron:
        0 2 */12 * * cd /path/to/scripts && python Retraining_Pipeline.py

Usage
-----
    # Pull new satellite data only (no retraining)
    python Retraining_Pipeline.py

    # Pull new data AND retrain all three models
    python Retraining_Pipeline.py --retrain

    # Pull new data, retrain, and regenerate test set
    python Retraining_Pipeline.py --retrain --update-test

    # Skip GEE pull (just retrain on existing data)
    python Retraining_Pipeline.py --retrain --skip-gee

    # Dry run — print what would happen without executing
    python Retraining_Pipeline.py --retrain --dry-run
"""

import os
import sys
import subprocess
import argparse
import traceback
from datetime import datetime

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


# ===========================================================================
# CONFIG — paths relative to this script's directory
# ===========================================================================

SENTINEL1_CSV = os.path.join(
    SCRIPT_DIR, r"..\data\sentinel1\GEE-Processing\sentinel1_timeseries.csv"
)
FLOOD_DATASET_CSV = os.path.join(SCRIPT_DIR, r"..\data\flood_dataset.csv")
MODEL_DIR         = os.path.join(SCRIPT_DIR, r"..\model")
LOG_FILE          = os.path.join(SCRIPT_DIR, r"..\logs\retraining_log.txt")

# Scripts to run (in order)
_PROJECT_ROOT        = os.path.dirname(SCRIPT_DIR)
_SCRIPTS_DIR         = os.path.join(_PROJECT_ROOT, "scripts")
_ML_PIPELINE_DIR     = os.path.join(_PROJECT_ROOT, "ml_pipeline")

SENTINEL1_SCRIPT     = os.path.join(_ML_PIPELINE_DIR, "sentinel1_GEE.py")
PREPARE_SCRIPT       = os.path.join(_ML_PIPELINE_DIR, "prepare_dataset.py")
RF_TRAIN_SCRIPT      = os.path.join(_SCRIPTS_DIR, "RF_train_flood_model.py")
XGB_TRAIN_SCRIPT     = os.path.join(_SCRIPTS_DIR, "XGB_train_flood_model.py")
LGBM_TRAIN_SCRIPT    = os.path.join(_SCRIPTS_DIR, "LGBM_train_flood_model.py")
EXTRACT_TEST_SCRIPT  = os.path.join(_SCRIPTS_DIR, "extract_test_set.py")

# ===========================================================================
# END CONFIG
# ===========================================================================


def separator(title=""):
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def log(message: str, log_path: str = LOG_FILE) -> None:
    """Append a timestamped line to the retraining log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(f"  LOG: {message}")
    try:
        with open(log_path, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run_script(script_path: str, label: str, dry_run: bool = False) -> bool:
    """
    Run a Python script as a subprocess.
    Returns True on success, False on failure.
    Uses the same Python interpreter that is running this script.
    """
    if not os.path.exists(script_path):
        print(f"\n  ERROR: {label} not found at:\n    {script_path}")
        log(f"FAIL — {label} not found")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would run: python {os.path.basename(script_path)}")
        return True

    print(f"\n  Running: python {os.path.basename(script_path)}")
    print(f"  {'─' * 50}")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            check=False,
        )
        if result.returncode == 0:
            log(f"OK — {label}")
            return True
        else:
            log(f"FAIL — {label} (exit code {result.returncode})")
            return False
    except Exception as e:
        print(f"\n  ERROR running {label}: {e}")
        traceback.print_exc()
        log(f"ERROR — {label}: {e}")
        return False


def check_new_satellite_data() -> dict:
    """
    Read sentinel1_timeseries.csv and report on latest pass.
    Returns summary dict for logging.
    """
    if not os.path.exists(SENTINEL1_CSV):
        return {"exists": False, "rows": 0, "latest": None}
    try:
        df = pd.read_csv(SENTINEL1_CSV, parse_dates=["timestamp"])
        df = df.sort_values("timestamp")
        return {
            "exists":  True,
            "rows":    len(df),
            "latest":  str(df["timestamp"].iloc[-1].date()),
            "floods":  int(df["flood_label"].sum()),
            "pct":     round(100 * df["flood_label"].mean(), 1),
        }
    except Exception as e:
        return {"exists": True, "rows": 0, "latest": None, "error": str(e)}


def check_model_artifacts() -> dict:
    """Check which model pkls exist and what their last_training_date is."""
    import joblib
    results = {}
    for name, filename in [
        ("RF",   "flood_rf_sensor.pkl"),
        ("XGB",  "flood_xgb_sensor.pkl"),
        ("LGBM", "flood_lgbm_sensor.pkl"),
    ]:
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                artifact = joblib.load(path)
                results[name] = {
                    "exists": True,
                    "last_training_date": artifact.get("last_training_date", "unknown"),
                    "watch_threshold":    artifact.get("watch_threshold", "?"),
                }
            except Exception as e:
                results[name] = {"exists": True, "error": str(e)}
        else:
            results[name] = {"exists": False}
    return results


# ---------------------------------------------------------------------------
# Main retraining pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    retrain:     bool = False,
    skip_gee:    bool = False,
    update_test: bool = False,
    dry_run:     bool = False,
) -> None:

    separator("Flood EWS — Retraining Pipeline")
    print(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Skip GEE    : {skip_gee}")
    print(f"  Retrain     : {retrain}")
    print(f"  Update test : {update_test}")
    print(f"  Dry run     : {dry_run}")

    log("=== Retraining pipeline started ===")

    # ── Status check before running ────────────────────────────────────────
    separator("Current State")
    sat_info = check_new_satellite_data()
    print(f"  Satellite CSV : {'EXISTS' if sat_info['exists'] else 'NOT FOUND'}")
    if sat_info.get("rows"):
        print(f"    Rows   : {sat_info['rows']}")
        print(f"    Latest : {sat_info['latest']}")
        print(f"    Floods : {sat_info['floods']} ({sat_info['pct']}%)")

    model_info = check_model_artifacts()
    for name, info in model_info.items():
        if info.get("exists"):
            ldt = info.get("last_training_date", "unknown")
            wt  = info.get("watch_threshold", "?")
            print(f"  {name} model  : trained through {ldt}  (watch={wt})")
        else:
            print(f"  {name} model  : NOT FOUND — run training scripts first")

    # ── Step 1: Pull new Sentinel-1 data from GEE ─────────────────────────
    if not skip_gee:
        separator("Step 1 — sentinel1_GEE.py  (pull new SAR passes + labels)")
        print("  This calls Google Earth Engine to pull new Sentinel-1 passes,")
        print("  compute GPM rainfall and ERA5 runoff, and assign flood labels.")
        print("  Requires GEE authentication (ee.Initialize).")
        print("  Expected runtime: 10–40 minutes depending on date range.")

        ok = run_script(SENTINEL1_SCRIPT, "sentinel1_GEE.py", dry_run)
        if not ok:
            print("\n  WARNING: GEE pull failed. Check GEE credentials and quota.")
            print("  Continuing with existing satellite data.")
            log("WARN — GEE pull failed, continuing with existing data")
        else:
            sat_after = check_new_satellite_data()
            if sat_after.get("rows", 0) > sat_info.get("rows", 0):
                new_rows = sat_after["rows"] - sat_info.get("rows", 0)
                print(f"\n  New passes added: {new_rows}")
                log(f"GEE pull OK — {new_rows} new passes, latest={sat_after['latest']}")
            else:
                print("\n  No new passes found (up to date).")
                log("GEE pull OK — no new passes")
    else:
        print("\n  Step 1 skipped (--skip-gee).")
        log("Step 1 skipped (--skip-gee)")

    # ── Step 2: Rebuild flood_dataset.csv ─────────────────────────────────
    separator("Step 2 — prepare_dataset.py  (rebuild flood_dataset.csv)")
    print("  Merges updated satellite data with sensor CSV.")
    print("  Rebuilds all features and aligns labels.")
    print("  Expected runtime: 1–3 minutes.")

    ok = run_script(PREPARE_SCRIPT, "prepare_dataset.py", dry_run)
    if not ok:
        print("\n  ERROR: prepare_dataset.py failed. Cannot retrain without dataset.")
        log("FAIL — prepare_dataset.py")
        if retrain:
            print("  Aborting retraining.")
            return
    else:
        log("prepare_dataset.py OK")

    # ── Step 3: Retrain models (optional) ─────────────────────────────────
    if retrain:
        separator("Step 3 — Retraining all three models")
        print("  Each train script:")
        print("    - Splits data chronologically (train/val/test)")
        print("    - Trains model with flood_weight=12.0")
        print("    - Tunes thresholds on val set")
        print("    - Saves .pkl with last_training_date baked in")
        print("    - Start.py reads last_training_date automatically")
        print("  Expected runtime: 2–5 minutes per model.")

        for script, label in [
            (RF_TRAIN_SCRIPT,   "RF_train_flood_model.py"),
            (XGB_TRAIN_SCRIPT,  "XGB_train_flood_model.py"),
            (LGBM_TRAIN_SCRIPT, "LGBM_train_flood_model.py"),
        ]:
            sep = f"  {label}"
            print(f"\n{sep}")
            ok = run_script(script, label, dry_run)
            if ok:
                log(f"{label} OK")
            else:
                print(f"  WARNING: {label} failed. Previous model retained.")
                log(f"FAIL — {label}")

        # Confirm new artifacts
        if not dry_run:
            print("\n  Updated model artifacts:")
            new_info = check_model_artifacts()
            for name, info in new_info.items():
                if info.get("exists"):
                    ldt = info.get("last_training_date", "unknown")
                    wt  = info.get("watch_threshold", "?")
                    print(f"    {name}: last_training_date={ldt}  watch_threshold={wt}")

        log("Retraining complete")
    else:
        print("\n  Step 3 skipped (use --retrain to also retrain models).")
        log("Step 3 skipped (no --retrain)")

    # ── Step 4: Regenerate test set (optional) ─────────────────────────────
    if update_test:
        separator("Step 4 — extract_test_set.py  (regenerate test CSV)")
        ok = run_script(EXTRACT_TEST_SCRIPT, "extract_test_set.py", dry_run)
        if ok:
            log("extract_test_set.py OK")
        else:
            log("FAIL — extract_test_set.py")
    else:
        print("\n  Step 4 skipped (use --update-test to regenerate test CSV).")

    # ── Done ───────────────────────────────────────────────────────────────
    separator("DONE")
    print(f"  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if retrain and not dry_run:
        print()
        print("  Next steps:")
        print("    1. The new .pkl files are live in ../model/")
        print("    2. Start.py reads last_training_date from the .pkl automatically")
        print("    3. Run: python Start.py --no-post  to verify predictions")
        print("    4. If results look good, Start.py daily schedule continues unchanged")
    log("=== Retraining pipeline complete ===")
    separator()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Flood EWS — Periodic Retraining Pipeline.\n"
            "Pulls new Sentinel-1 data via GEE and optionally retrains models."
        )
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Also retrain RF, XGBoost, and LightGBM after rebuilding the dataset.",
    )
    parser.add_argument(
        "--skip-gee", action="store_true",
        help="Skip the GEE satellite data pull. Use existing sentinel1_timeseries.csv.",
    )
    parser.add_argument(
        "--update-test", action="store_true",
        help="Regenerate flood_dataset_test.csv via extract_test_set.py.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without actually executing anything.",
    )
    args = parser.parse_args()

    run_pipeline(
        retrain     = args.retrain,
        skip_gee    = args.skip_gee,
        update_test = args.update_test,
        dry_run     = args.dry_run,
    )