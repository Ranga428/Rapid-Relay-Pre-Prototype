import random

def extract_eo_features(eo_files):
    """
    Placeholder EO-derived indicators
    (replace later with real EO analytics)
    """

    soil_saturation_index = round(random.uniform(0.6, 0.9), 2)
    flood_extent_ratio = round(random.uniform(0.0, 0.4), 2)
    wetness_trend = random.choice([-1, 0, 1])

    return {
        "soil_saturation": soil_saturation_index,
        "flood_extent": flood_extent_ratio,
        "wetness_trend": wetness_trend
    }
