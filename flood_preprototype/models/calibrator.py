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

def calibrate_with_eo(eo_file):
    """
    Converts EO presence into calibration weight.
    Placeholder for backscatter / soil moisture later.
    """
    if eo_file:
        return 1.1  # EO-informed boost
    return 1.0
