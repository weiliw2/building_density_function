import geopandas as gpd
import rasterio
from shapely.geometry import box
from rasterio.transform import from_bounds


geojson_path = "/Users/weilynnw/Desktop/building_density_error/intrested_area/Berlin.geojson"
buildings = gpd.read_file(geojson_path)

# Load raster metadata
raster_path = "/Users/weilynnw/Desktop/building_density_error/proceesed_data/cuttingGHSL/result.tif"
with rasterio.open(raster_path) as src:
    raster_transform = src.transform
    raster_crs = src.crs
    raster_width = src.width
    raster_height = src.height
    raster_bounds = src.bounds

if buildings.crs != raster_crs:
    buildings = buildings.to_crs(raster_crs)

# Step 1: Generate Grid Cells from Raster
pixel_size = raster_transform.a  # Assuming square pixels
grid_cells = []
grid_ids = []

for row in range(raster_height):
    for col in range(raster_width):
        minx, miny = raster_transform * (col, row)
        maxx, maxy = raster_transform * (col + 1, row + 1)
        grid_cells.append(box(minx, miny, maxx, maxy))
        grid_ids.append(f"cell_{row}_{col}")  # Unique ID per grid cell

# Convert grid cells to GeoDataFrame
grid_gdf = gpd.GeoDataFrame({"grid_id": grid_ids, "geometry": grid_cells}, crs=raster_crs)

# Step 2: Overlay buildings onto the grid to split polygons at grid boundaries
split_buildings = gpd.overlay(buildings, grid_gdf, how="intersection")

# Step 3: Use Spatial Join to Assign Grid IDs to Split Polygons
split_buildings = split_buildings.sjoin(grid_gdf[["grid_id", "geometry"]], how="left", predicate="intersects")

# Step 4: Assign Original Building IDs
split_buildings["original_building_id"] = split_buildings.index  # Retain original ID

# Step 5: Save and Verify
split_buildings.to_file("/Users/weilynnw/Desktop/building_density_error/split_building/Berilin.geojson", driver="GeoJSON")
print(f"Saved split polygons to 'split_buildings.geojson'")

# Step 6: Visualize
"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
grid_gdf.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.5)
split_buildings.plot(ax=ax, color="red", alpha=0.5)

plt.title("Split Buildings Overlay on Grid")
plt.show()
"""