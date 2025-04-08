# Building Density Error Analysis

## Overview
This project compares the GHSL (Global Human Settlement Layer) dataset with building footprints from Overture Maps to analyze building density error rates. The analysis helps identify discrepancies between these datasets across different urban areas.

## Data Sources
1. **GHSL Data**: Download from [Copernicus Emergency Management Service](https://human-settlement.emergency.copernicus.eu/download.php).
2. **Overture Maps Data**: Retrieved using DuckDB with Azure connectivity or local files.

## Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install geopandas rasterio rasterstats numpy pandas matplotlib pyproj duckdb geopy shapely
```

## Workflow

### 1. Process City Data (`Proceesing_1.py`)
This script handles the complete workflow for a city:
- Calculates a bounding box around a city center
- Extracts raster data for the area
- Downloads and processes buildings data from Overture Maps
- Splits buildings based on raster cells

Usage:
```python
# Example for a single city
city_name = "Moscow"
city_center = (55.7558, 37.6173)  # (latitude, longitude)
x_km, y_km = 10, 10  # 10km x 10km area
input_raster = "path/to/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
output_dir = "path/to/output_directory"
azure_connection = "Your_Azure_Connection_String"  # Optional

process_city(city_name, city_center, x_km, y_km, input_raster, output_dir, azure_connection)

# For processing multiple cities
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
```

### 2. Statistical Analysis and Visualization (`stat_visual_2.py`)
This script performs statistical analysis on the processed data:
- Calculates error rates between GHSL and actual building areas
- Generates visualizations of error distributions
- Saves detailed cell-by-cell data for further analysis

Usage:
```bash
python stat_visual_2.py
# You will be prompted to enter:
# - Path to the GeoJSON file for the city
# - Path to the Raster file for the city
# - Whether to load existing results
```

## Output Files
The analysis generates several output files:
- `{city_name}_split_buildings.geojson`: Building footprints split by raster cells
- `{city_name}_raster.tif`: Clipped raster for the city area
- `{city_name}_cell_by_cell_data.csv`: Detailed error analysis for each raster cell
- `{city_name}_building_error_results.pkl`: Pickled results for later visualization
- `{city_name}_error_area_analysis.png`: Visualizations of error distribution

## Analysis Metrics
The script calculates several error metrics:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Error distribution statistics
- Correlation between building area and error rate

## Notes
- The Azure connection string is required to download buildings from Overture Maps. If not provided, you'll need to use locally available building data.
- For large datasets, the script can take significant processing time. Consider using the option to save and reload results.
- The visualization includes histograms of percentage error and scatter plots showing the relationship between building area and error rates.