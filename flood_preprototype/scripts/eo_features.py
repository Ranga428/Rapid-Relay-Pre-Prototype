import random
import pandas as pd
from datetime import datetime
import os

EO_CSV_PATH = "data/sentinel1/eo_features.csv"

def extract_eo_features(save_csv=True):
    """
    Placeholder EO-derived indicators
    (replace later with real EO analytics)
    """

    # Generate random proxies
    soil_saturation_index = round(random.uniform(0.6, 0.9), 2)
    flood_extent_ratio = round(random.uniform(0.0, 0.4), 2)
    wetness_trend = random.choice([-1, 0, 1])

    features = {
        "timestamp": datetime.utcnow().isoformat(),
        "soil_saturation": soil_saturation_index,
        "flood_extent": flood_extent_ratio,
        "wetness_trend": wetness_trend
    }

    # Save to CSV if requested
    if save_csv:
        os.makedirs(os.path.dirname(EO_CSV_PATH), exist_ok=True)

        # Create DataFrame for the new row
        df_new = pd.DataFrame([features])

        if os.path.exists(EO_CSV_PATH):
            # Read existing CSV
            df_existing = pd.read_csv(EO_CSV_PATH)
            # Concatenate safely
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new

        # Save
        df_all.to_csv(EO_CSV_PATH, index=False)

    return features

