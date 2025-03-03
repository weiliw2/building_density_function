from geopy.distance import geodesic
import duckdb
import os

# Connect to DuckDB
conn = duckdb.connect("overture_data.db")

conn.execute("INSTALL spatial;")
conn.execute("LOAD spatial;")
conn.execute("INSTALL azure;")
conn.execute("LOAD azure;")

# Set Azure storage
conn.execute("""
SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';
""")

# Function to calculate a bounding box around a city center point
def calculate_bounding_box(city_center, width_km, height_km):
    latitude, longitude = city_center
    north = geodesic(kilometers=height_km).destination((latitude, longitude), 0).latitude
    south = geodesic(kilometers=height_km).destination((latitude, longitude), 180).latitude
    east = geodesic(kilometers=width_km).destination((latitude, longitude), 90).longitude
    west = geodesic(kilometers=width_km).destination((latitude, longitude), 270).longitude
    return [west, east, south, north]

# Define cities and coordinates
cities = {
    "New York City": (40.7128, -74.0060, 10, 10),
    "Los Angeles": (34.0522, -118.2437, 10, 10),
    "Chicago": (41.8781, -87.6298, 10, 10),
    "Houston": (29.7604, -95.3698, 10, 10),
    "San Francisco": (37.7749, -122.4194, 10, 10),
    "São Paulo": (-23.5505, -46.6333, 10, 10),
    "Buenos Aires": (-34.6037, -58.3816, 10, 10),
    "London": (51.5074, -0.1278, 10, 10),
    "Paris": (48.8566, 2.3522, 10, 10),
    "Berlin": (52.5200, 13.4050, 10, 10),
    "Cairo": (30.0444, 31.2357, 10, 10),
    "Johannesburg": (-26.2041, 28.0473, 10, 10),
    "Tokyo": (35.6895, 139.6917, 10, 10),
    "Beijing": (39.9042, 116.4074, 10, 10),
    "Shanghai": (31.2304, 121.4737, 10, 10),
    "Sydney": (-33.8688, 151.2093, 10, 10),
    "Melbourne": (-37.8136, 144.9631, 10, 10),
}

# Prompt user to select a city
print("Available cities:", ", ".join(cities.keys()))
selected_city = input("Enter the name of the city you want to download: ")

if selected_city not in cities:
    print("❌ Invalid city name. Please restart and enter a valid city.")
    exit()

# Extract city details
lat, lon, width_km, height_km = cities[selected_city]
bbox = calculate_bounding_box((lat, lon), width_km, height_km)
xmin, xmax, ymin, ymax = bbox

# Drop existing table if needed
conn.execute("DROP TABLE IF EXISTS temp_buildings;")

# Create a new table
conn.execute("""
CREATE TABLE temp_buildings (
    city TEXT,
    id TEXT,
    primary_name TEXT,
    height DOUBLE,
    geometry GEOMETRY
);
""")

# Query data for the selected city
query = f"""
INSERT INTO temp_buildings
SELECT
  '{selected_city}' AS city,  
  id,
  COALESCE(names.primary, 'Unnamed') AS primary_name, 
  height,
  geometry
FROM read_parquet('azure://release/2025-01-22.0/theme=buildings/type=building/*', filename=true, hive_partitioning=1)
WHERE bbox.xmin BETWEEN {xmin} AND {xmax}
AND bbox.ymin BETWEEN {ymin} AND {ymax}
"""
conn.execute(query)
print(f"✅ Data for {selected_city} added to the temp table")

# Export as GeoJSON first
output_dir = "/Users/weilynnw/Desktop/GHSL:overtrue/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

geojson_file = f"{output_dir}{selected_city.replace(' ', '_')}_buildings.geojson"

conn.execute(f"""
COPY (SELECT * FROM temp_buildings)
TO '{geojson_file}' (FORMAT GDAL, DRIVER 'GeoJSON');
""")

file_size = os.path.getsize(geojson_file) / (1024 * 1024)
print(f"✅ Saved {selected_city} data in {geojson_file} ({file_size:.2f} MB)")

print("🎉 GeoJSON download completed successfully!")