import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from Get_city_area import bounding_box_pixels

min_col, min_row, max_col, max_row = bounding_box_pixels
print(f"Imported Bounding Box Pixels: ({min_col}, {min_row}) to ({max_col}, {max_row})")

input_raster = "/Users/weilynnw/Desktop/GHSL:overtrue/proceesed_data/cuttingGHSL/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"  # Replace with the local whole_global_30ssGHSL.tiff file path
output_raster = "/Users/weilynnw/Desktop/GHSL:overtrue/proceesed_data/cuttingGHSL/result.tif"  # Replace with the desired output file path

with rasterio.open(input_raster) as src:
    # Define the window based on pixel coordinates
    window = Window.from_slices((min_row, max_row), (min_col, max_col))
    
    # Read the data within the window
    clipped_data = src.read(window=window)
    
    # Adjust the transform for the clipped region
    transform = src.window_transform(window)
    
    # Write the clipped raster to a new file
    profile = src.profile
    profile.update({
        "height": window.height,
        "width": window.width,
        "transform": transform
    })
    
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(clipped_data)

print("Clipped_size raster saved to:", output_raster)
