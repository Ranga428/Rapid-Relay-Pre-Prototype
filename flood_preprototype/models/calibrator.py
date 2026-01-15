import yaml

def load_thresholds(path="config/thresholds.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def calibrate_thresholds(thresholds):
    calibrated = thresholds.copy()
    calibrated["calibrated_water_level"] = (
        thresholds["base_water_level"] * thresholds["flood_risk_factor"]
    )
    return calibrated
