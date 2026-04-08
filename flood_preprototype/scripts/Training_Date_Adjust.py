import joblib
import os

MODEL_DIR = r"D:\Rapid-Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\model"

MODEL_FILES = {
    "RF":   os.path.join(MODEL_DIR, "flood_rf_sensor.pkl"),
    "XGB":  os.path.join(MODEL_DIR, "flood_xgb_sensor.pkl"),
    "LGBM": os.path.join(MODEL_DIR, "flood_lgbm_sensor.pkl"),
    "RF1":   os.path.join(MODEL_DIR, "flood_rf_full.pkl"),
    "XGB1":  os.path.join(MODEL_DIR, "flood_xgb_full.pkl"),
    "LGBM1": os.path.join(MODEL_DIR, "flood_lgbm_full.pkl"),
}

NEW_DATE = "2025-06-30"

for name, path in MODEL_FILES.items():
    if not os.path.exists(path):
        print(f"  [{name}] NOT FOUND — skipping: {path}")
        continue

    artifact = joblib.load(path)
    old_date = artifact.get("last_training_date", "NOT SET")
    print(f"  [{name}] Current last_training_date: {old_date}")

    artifact["last_training_date"] = NEW_DATE
    joblib.dump(artifact, path)
    print(f"  [{name}] Updated to: {NEW_DATE}")
    print()

print("Done.")