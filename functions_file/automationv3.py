import rasterio
import geopandas as gpd
import rasterstats
import numpy as np
import pandas as pd
from pyproj import Geod

geojson_path = "/Users/weilynnw/Desktop/building_density_error/split_building/Berilin.geojson"
buildings = gpd.read_file(geojson_path)
print(f"Processing {len(buildings)} buildings in the input file")

raster_path = "/Users/weilynnw/Desktop/building_density_error/proceesed_data/cuttingGHSL/result.tif"

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

# Modified error calculation to handle all cases
error_rate = np.where(
    (ghsl_array == 0) & (building_raster == 0),  # Case 1: Both zero
    0,  # No error
    np.where(
        (ghsl_array > 0) & (building_raster == 0),  # Case 2: GHSL > 0, buildings = 0
        1,  # 100% error
        np.where(
            (ghsl_array == 0) & (building_raster > 0),  # Case 3: GHSL = 0, buildings > 0
            -1,  # -100% error (complete underestimation)
            (ghsl_array - building_raster) / building_raster  # Normal case
        )
    )
)

print("Computed GHSL Error Rates per Grid Cell!")

print(error_rate)
valid_cells = ~np.isnan(error_rate)
mean_absolute_error = np.nanmean(np.abs(error_rate[valid_cells]))  # Mean absolute error

print(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f}")

rmse = np.sqrt(np.nanmean(error_rate[valid_cells] ** 2))

print(f"Root Mean Square Error (RMSE): {rmse:.4f}")

rows, cols = error_rate.shape

for row in range(rows):
    for col in range(cols):
        print(f"Cell ({row}, {col}): GHSL={ghsl_array[row, col]}, Buildings={building_raster[row, col]}, Error Rate={error_rate[row, col]:.4f}")

import pickle

results = {
    'error_rate': error_rate,
    'building_raster': building_raster,
    'ghsl_array': ghsl_array,
    'mean_absolute_error': mean_absolute_error,
    'rmse': rmse,
    'transform': raster_transform,
    'metadata': raster_meta
}

# Save the results
with open('building_error_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved to building_error_results.pkl")