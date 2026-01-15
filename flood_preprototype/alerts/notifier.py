def notify(alert_level, sensor_data):
    message = (
        f"[{alert_level}] "
        f"Water={sensor_data['water_level']}cm | "
        f"Rain={sensor_data['rainfall']}mm/hr | "
        f"Humidity={sensor_data['humidity']}%"
    )
    print(message)
