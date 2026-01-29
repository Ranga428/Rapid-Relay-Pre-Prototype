import rasterio
import os
import numpy as np

scene_folder = "data/sentinel1/S1A_IW_GRDH_1SDV_20260115T230423_20260115T230440_005920_00BDFA.SAFE/measurement/"
tiff_file = [f for f in os.listdir(scene_folder) if f.endswith('.tiff')][0]

with rasterio.open(os.path.join(scene_folder, tiff_file)) as src:
    sar_img = src.read(1)  # Single band
    profile = src.profile
