import random

def generate_sensor_data():
    water_level = random.uniform(20, 80)
    rainfall = random.uniform(0, 30)
    return {
        "water_level": water_level,
        "rainfall": rainfall
    }
