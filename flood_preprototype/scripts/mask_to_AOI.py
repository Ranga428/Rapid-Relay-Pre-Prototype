import rasterio.features
from shapely.geometry import shape

# Assume gdf has AOI polygon
mask = rasterio.features.geometry_mask(
    [shape(gdf.geometry[0])],
    out_shape=sar_img.shape,
    transform=src.transform,
    invert=True
)
