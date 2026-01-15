def predict(sensor_data, thresholds):
    wl = sensor_data["water_level"]
    rain = sensor_data["rainfall"]
    hum = sensor_data["humidity"]

    if (
        wl >= thresholds["calibrated_water_level"]
        and rain >= thresholds["rainfall_limit"]
        and hum >= thresholds["humidity_limit"]
    ):
        return "CRITICAL"

    if wl >= thresholds["base_water_level"] and rain >= thresholds["rainfall_limit"]:
        return "WARNING"

    if wl >= thresholds["base_water_level"] * 0.9:
        return "ADVISORY"

    return "NORMAL"
