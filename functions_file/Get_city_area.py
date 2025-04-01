from geopy.distance import geodesic
import rasterio
import duckdb
from datetime import datetime

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

# Convert geographic coordinates to pixel coordinates
def geographic_to_pixel(lat, lon, transform):
    col, row = ~transform * (lon, lat)
    return int(col), int(row)

# Main parameters
city_center = (52.5200, 13.4050)
x_km, y_km = 10, 10
bounding_box = calculate_bounding_box(city_center, x_km, y_km)

print("Bounding Box Coordinates:", bounding_box)

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

bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
#print("Formatted BBox:", bbox)

con = duckdb.connect(database=':memory:')
con.execute("LOAD spatial;")
con.execute("""
SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';
""")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"/Users/weilynnw/Desktop/building_density_error/intrested_area/Berlin.geojson"

query = f"""
COPY(
  SELECT id, geometry
  FROM read_parquet('azure://release/2025-02-19.0/theme=buildings/type=building/*', filename=true, hive_partitioning=1)
  WHERE bbox.xmin BETWEEN {min_lon} AND {max_lon}
    AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
) TO '{output_path}' WITH (FORMAT GDAL, DRIVER 'GeoJSON');
"""

try:
    con.execute(query)
    print("Downloaded buildings saved to:" + output_path)
except Exception as e:
    print("Error downloading buildings:", e)
