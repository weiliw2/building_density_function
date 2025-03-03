import rasterio
import geopandas as gpd
import rasterstats
import numpy as np
import pandas as pd
from pyproj import Geod

geojson_path = "/Users/weilynnw/Desktop/GHSL:overtrue/proceesed_data/split_buildings.geojson"
buildings = gpd.read_file(geojson_path)
print(f"Processing {len(buildings)} buildings in the input file")

raster_path = "/Users/weilynnw/Desktop/GHSL:overtrue/proceesed_data/cuttingGHSL/result.tif"

geod = Geod(ellps="WGS84")
buildings["geodesic_area_m2"] = buildings.geometry.apply(lambda geom: abs(geod.geometry_area_perimeter(geom)[0]))
#buildings = buildings[['id', 'geodesic_area_m2', 'geometry']]
print(buildings.head())

with rasterio.open(raster_path) as src:
    raster_transform = src.transform  # Get affine transform
    raster_meta = src.meta.copy()  # Copy metadata
    raster_shape = (src.height, src.width)  # Get raster dimensions (rows, cols)
    raster_crs = src.crs  # Get raster coordinate system
    cell_size_x, cell_size_y = src.res  # Get raster cell size (resolution)

print(f"Raster Grid Size: {raster_shape} (Rows x Cols)")
print(f"Cell Size: {cell_size_x} x {cell_size_y} meters")
building_raster = np.zeros(raster_shape, dtype=np.float32)  # Use float for summed areas
shapes = [(geom, area) for geom, area in zip(buildings.geometry, buildings["geodesic_area_m2"])]

rasterio.features.rasterize(
    shapes,
    out=building_raster,  # Write rasterized buildings into this array
    transform=raster_transform,  # Ensures alignment with the GHSL raster grid
    fill=0,  # Background (non-building areas) will be 0
    all_touched=True,
    merge_alg=rasterio.enums.MergeAlg.add   # Ensures partial coverage of cells is included
)

print("Rasterization Complete! Each grid cell contains the summed building area.")

with rasterio.open(raster_path) as src:
    ghsl_array = src.read(1)  # Read GHSL population density values
    #print(ghsl_array)

df_building = pd.DataFrame(building_raster)
print(df_building)
#
# Compute error rate per grid cell
error_rate = np.where(building_raster != 0, (ghsl_array - building_raster) / building_raster, np.nan)

print("Computed GHSL Error Rates per Grid Cell!")

print(error_rate)
valid_cells = ~np.isnan(error_rate)  # Mask out NaN values
mean_absolute_error = np.nanmean(np.abs(error_rate[valid_cells]))  # Mean absolute error
mean_error = np.nanmean(error_rate[valid_cells])  # Mean bias

print(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f}")
print(f"Mean Error (Bias): {mean_error:.4f}")