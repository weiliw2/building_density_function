#!/usr/bin/env python3

import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import duckdb
from geopy.distance import geodesic
from datetime import datetime
import os
import tempfile
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

def calculate_bounding_box(city_center, x_km, y_km):
    """
    Calculate a bounding box around a city center point.
    
    Parameters:
    city_center (tuple): Latitude and longitude of the city center.
    x_km (float): East-west distance in km.
    y_km (float): North-south distance in km.
    
    Returns:
    dict: Coordinates of the bounding box (north, south, east, west).
    """
    lat, lon = city_center
    return {
        "north": (geodesic(kilometers=y_km).destination((lat, lon), 0).latitude, lon),
        "south": (geodesic(kilometers=y_km).destination((lat, lon), 180).latitude, lon),
        "east": (lat, geodesic(kilometers=x_km).destination((lat, lon), 90).longitude),
        "west": (lat, geodesic(kilometers=x_km).destination((lat, lon), 270).longitude)
    }

def geographic_to_pixel(lat, lon, transform):
    """Convert geographic coordinates to pixel coordinates"""
    col, row = ~transform * (lon, lat)
    return int(col), int(row)

def process_city(city_name, city_center, x_km, y_km, input_raster_path, output_dir, azure_connection=None):
    """
    Process a city area in one go:
    1. Calculate bounding box
    2. Extract raster data for the area
    3. Download and process buildings data
    4. Split buildings based on raster cells
    
    Parameters:
    city_name (str): Name of the city
    city_center (tuple): (latitude, longitude) of city center
    x_km, y_km (float): Area dimensions in km
    input_raster_path (str): Path to input raster file
    output_dir (str): Directory to save final outputs
    azure_connection (str, optional): Azure connection string for Overture maps
    """
    print(f"Processing {city_name}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Calculate bounding box
    bounding_box = calculate_bounding_box(city_center, x_km, y_km)
    print("Bounding Box Coordinates:", bounding_box)
    
    # Transform for coordinate conversion
    transform = rasterio.Affine(
        0.008333333300326820764, 0.0, -180.0012492646600606,
        0.0, -0.008333333299795072507, 89.0995831776455987
    )
    
    pixel_coords = {key: geographic_to_pixel(*bounding_box[key], transform) for key in bounding_box}
    
    # Calculate min/max pixel values
    min_col = min(p[0] for p in pixel_coords.values())
    max_col = max(p[0] for p in pixel_coords.values())
    min_row = min(p[1] for p in pixel_coords.values())
    max_row = max(p[1] for p in pixel_coords.values())
    bounding_box_pixels = (min_col, min_row, max_col, max_row)
    
    # Convert pixel coordinates back to geographic coordinates
    min_lon, max_lat = transform * (min_col, min_row)
    max_lon, min_lat = transform * (max_col, max_row)
    
    print(f"Bounding Box Pixels: ({min_col}, {min_row}) to ({max_col}, {max_row})")
    print(f"Geographic bounds: {min_lon}, {min_lat}, {max_lon}, {max_lat}")
    
    # Step 2: Extract raster data using a temporary file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_raster:
        temp_raster_path = tmp_raster.name
        
    try:
        print(f"Extracting raster data for {city_name}...")
        with rasterio.open(input_raster_path) as src:
            # Define the window based on pixel coordinates
            window = Window.from_slices((min_row, max_row), (min_col, max_col))
            
            # Read the data within the window
            clipped_data = src.read(window=window)
            
            # Adjust the transform for the clipped region
            window_transform = src.window_transform(window)
            
            # Write the clipped raster to the temporary file
            profile = src.profile
            profile.update({
                "height": window.height,
                "width": window.width,
                "transform": window_transform
            })
            
            with rasterio.open(temp_raster_path, "w", **profile) as dst:
                dst.write(clipped_data)
                
            print(f"Temporary raster created: {temp_raster_path}")
            
            # Step 3: Download buildings using DuckDB (if connection string provided)
            buildings_gdf = None
            if azure_connection:
                with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp_buildings:
                    temp_buildings_path = tmp_buildings.name
                
                try:
                    print(f"Downloading buildings data for {city_name}...")
                    con = duckdb.connect(database=':memory:')
                    con.execute("LOAD spatial;")
                    con.execute(f"SET azure_storage_connection_string = '{azure_connection}';")
                    
                    query = f"""
                    COPY(
                      SELECT id, geometry
                      FROM read_parquet('azure://release/2025-02-19.0/theme=buildings/type=building/*', filename=true, hive_partitioning=1)
                      WHERE bbox.xmin BETWEEN {min_lon} AND {max_lon}
                        AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
                    ) TO '{temp_buildings_path}' WITH (FORMAT GDAL, DRIVER 'GeoJSON');
                    """
                    
                    con.execute(query)
                    print(f"Downloaded buildings saved to temporary file")
                    
                    # Load the buildings data
                    buildings_gdf = gpd.read_file(temp_buildings_path)
                except Exception as e:
                    print(f"Error downloading buildings: {e}")
                    # If you have a local buildings file as fallback, you could load it here
                finally:
                    if os.path.exists(temp_buildings_path):
                        os.unlink(temp_buildings_path)
            else:
                print("No Azure connection string provided, skipping buildings download")
                # Option to load buildings from an existing file if needed
                # buildings_gdf = gpd.read_file("path/to/existing/buildings.geojson")
            
            # Step 4: Split buildings based on raster cells (if buildings data available)
            if buildings_gdf is not None and not buildings_gdf.empty:
                print(f"Splitting buildings based on raster cells...")
                
                with rasterio.open(temp_raster_path) as raster_src:
                    raster_transform = raster_src.transform
                    raster_crs = raster_src.crs
                    raster_width = raster_src.width
                    raster_height = raster_src.height
                    raster_bounds = raster_src.bounds
                
                # Ensure consistent CRS
                if buildings_gdf.crs != raster_crs:
                    buildings_gdf = buildings_gdf.to_crs(raster_crs)
                
                # Generate Grid Cells from Raster
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
                
                # Overlay buildings onto the grid to split polygons at grid boundaries
                split_buildings = gpd.overlay(buildings_gdf, grid_gdf, how="intersection")
                
                # Use Spatial Join to Assign Grid IDs to Split Polygons
                split_buildings = split_buildings.sjoin(grid_gdf[["grid_id", "geometry"]], how="left", predicate="intersects")
                
                # Assign Original Building IDs
                split_buildings["original_building_id"] = split_buildings.index  # Retain original ID
                
                # Save the final output
                final_buildings_path = os.path.join(output_dir, f"{city_name}_split_buildings.geojson")
                split_buildings.to_file(final_buildings_path, driver="GeoJSON")
                print(f"Saved split buildings to: {final_buildings_path}")
                
                # Save the clipped raster
                final_raster_path = os.path.join(output_dir, f"{city_name}_raster.tif")
                with rasterio.open(temp_raster_path) as src:
                    with rasterio.open(final_raster_path, 'w', **src.profile) as dst:
                        dst.write(src.read())
                print(f"Saved raster to: {final_raster_path}")
            else:
                print("No buildings data available, skipping building split step")
                
                # Still save the raster
                final_raster_path = os.path.join(output_dir, f"{city_name}_raster.tif")
                with rasterio.open(temp_raster_path) as src:
                    with rasterio.open(final_raster_path, 'w', **src.profile) as dst:
                        dst.write(src.read())
                print(f"Saved raster to: {final_raster_path}")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_raster_path):
            os.unlink(temp_raster_path)
    
    print(f"Processing completed for {city_name}")

# Example usage
if __name__ == "__main__":
    # Define your parameters
    city_name = "Auckland"
    city_center = (-36.8509, 174.7645)#(latitude, longitude)
    x_km, y_km = 10, 10  # 10km x 10km area
    
    # Path to input raster file
    input_raster = "/Users/weilynnw/Desktop/building_density_error/proceesed_data/cuttingGHSL/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
    
    # Output directory
    output_dir = "/Users/weilynnw/Desktop/building_density_error/combined_output"
    
    # Optional: Azure connection string for DuckDB to access Overture maps
    # If you don't have the connection string, pass None and the buildings download step will be skipped
    azure_connection = "DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net"
    
    # Process the city
    process_city(city_name, city_center, x_km, y_km, input_raster, output_dir, azure_connection)
    
    # You can process multiple cities in one run:
    """
    cities = [
        {"name": "Berlin", "center": (52.5200, 13.4050), "x_km": 10, "y_km": 10},
        {"name": "Paris", "center": (48.8566, 2.3522), "x_km": 10, "y_km": 10},
        {"name": "London", "center": (51.5074, -0.1278), "x_km": 10, "y_km": 10}
    ]
    
    for city in cities:
        process_city(
            city["name"], 
            city["center"], 
            city["x_km"], 
            city["y_km"], 
            input_raster, 
            output_dir, 
            azure_connection
        )
    """