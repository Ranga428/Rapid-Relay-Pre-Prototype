# Simple linear calibration
sar_img_calibrated = sar_img.astype(float)  # Convert to float
sar_img_calibrated = sar_img_calibrated ** 2  # Approximate backscatter (simplified)
# For real calibration, SNAP or formula from ESA documentation is needed

# Soild Saturation
soil_saturation = np.mean(sar_img_calibrated[mask])
# Normalize to 0–1
soil_saturation = (soil_saturation - np.min(sar_img_calibrated)) / (np.max(sar_img_calibrated) - np.min(sar_img_calibrated))

# Flood Extent
water_mask = sar_img_calibrated < 0.2  # Adjust threshold
flood_extent = np.sum(water_mask[mask]) / np.sum(mask)  # 0–1 fraction

# Wetness Trend
prev_sar = rasterio.open('data/sentinel1/prev_scene.tiff').read(1)
trend_change = np.mean(sar_img_calibrated[mask] - prev_sar[mask])

if trend_change > 0.05:
    wetness_trend = 1   # wetter
elif trend_change < -0.05:
    wetness_trend = -1  # drying
else:
    wetness_trend = 0   # stable

# Save features to CSV
import pandas as pd
from datetime import datetime

features = {
    "timestamp": datetime.utcnow().isoformat(),
    "soil_saturation": soil_saturation,
    "flood_extent": flood_extent,
    "wetness_trend": wetness_trend
}

df = pd.DataFrame([features])
csv_path = "data/sentinel1/eo_features.csv"
df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
