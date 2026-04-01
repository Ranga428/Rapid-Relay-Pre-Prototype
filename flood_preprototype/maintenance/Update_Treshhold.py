"""
update_thresholds.py
====================
Standalone utility to update Watch and Warning thresholds in a deployed
model artifact (.pkl) without retraining.

WHEN TO USE
-----------
After running --threshold-search on any predict_*.py script, it prints
optimal Watch and Warning thresholds calibrated to the data you passed.
This script applies those values into the .pkl artifact so that all
subsequent live and batch runs pick them up automatically.

Usage
-----
    # Update RF sensor model thresholds
    python update_thresholds.py --model ..\\model\\flood_rf_sensor.pkl
        --watch 0.11 --warning 0.38

    # Update XGBoost sensor model thresholds
    python update_thresholds.py --model ..\\model\\flood_xgb_sensor.pkl
        --watch 0.13 --warning 0.45

    # Update LightGBM sensor model thresholds
    python update_thresholds.py --model ..\\model\\flood_lgbm_sensor.pkl
        --watch 0.10 --warning 0.40

    # Preview current thresholds without changing anything
    python update_thresholds.py --model ..\\model\\flood_rf_sensor.pkl

WHAT IT CHANGES
---------------
    artifact["threshold"]         = watch_threshold  (primary deployment)
    artifact["watch_threshold"]   = watch_threshold
    artifact["warning_threshold"] = warning_threshold

    The artifact["model"] and artifact["feature_columns"] are never touched.

CAUTION
-------
If you derived the new thresholds from the test set (flood_dataset_test.csv),
be aware that this slightly optimizes for the data you evaluated on. This
is acceptable when no separate calibration set exists, but note that
reported metrics after this update will be slightly optimistic.

The previous thresholds are printed before updating so you can revert
manually if needed.
"""

import argparse
import joblib
import os
import sys


def load_and_show(model_path: str) -> dict:
    if not os.path.exists(model_path):
        sys.exit(f"\n  ERROR: Model file not found.\n  Expected : {model_path}")

    artifact = joblib.load(model_path)
    print(f"\n  Model            : {model_path}")
    print(f"  Model type       : {type(artifact['model']).__name__}")
    print(f"  Feature count    : {len(artifact['feature_columns'])}")
    print(f"\n  Current thresholds:")
    print(f"    watch_threshold   : {artifact.get('watch_threshold',   'NOT SET')}")
    print(f"    warning_threshold : {artifact.get('warning_threshold', 'NOT SET')}")
    print(f"    threshold (alias) : {artifact.get('threshold',         'NOT SET')}")
    return artifact


def update(model_path: str, watch: float, warning: float) -> None:
    artifact = load_and_show(model_path)

    print(f"\n  Updating to:")
    print(f"    watch_threshold   : {watch:.4f}")
    print(f"    warning_threshold : {warning:.4f}")

    if warning <= watch:
        sys.exit(
            f"\n  ERROR: warning_threshold ({warning}) must be > "
            f"watch_threshold ({watch}).\n"
            f"  The WARNING tier requires higher confidence than WATCH."
        )

    artifact["threshold"]         = watch
    artifact["watch_threshold"]   = watch
    artifact["warning_threshold"] = warning

    joblib.dump(artifact, model_path)

    print(f"\n  ✅  Saved → {model_path}")
    print(f"  Subsequent runs of predict_*.py will use these thresholds.")
    print(f"  DANGER threshold at inference time = warning + 0.10 = {warning + 0.10:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update Watch / Warning thresholds in a deployed model artifact."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the .pkl model artifact to update.",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        help="New Watch threshold (e.g. 0.11). If omitted, only prints current values.",
    )
    parser.add_argument(
        "--warning",
        type=float,
        default=None,
        help="New Warning threshold (e.g. 0.38). Must be greater than --watch.",
    )
    args = parser.parse_args()

    if args.watch is None and args.warning is None:
        # Preview mode — just print current values
        load_and_show(args.model)
        print(f"\n  (No changes made — pass --watch and --warning to update.)")
    elif args.watch is None or args.warning is None:
        sys.exit(
            "\n  ERROR: Both --watch and --warning must be provided together.\n"
            "  Run with neither to preview current values."
        )
    else:
        update(args.model, args.watch, args.warning)