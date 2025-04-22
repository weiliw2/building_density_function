# Building Density Analysis 

This repo contains a collection of Python scripts for analyzing and visualizing the relationship between GHSL (Global Human Settlement Layer) building density data and actual building footprints. It helps identify where GHSL data may overestimate or underestimate building densities in various cities around the world.

## Scripts Overview

The toolkit workflow consists of four main scripts that should be run in sequence:

1. **BuildingExtractor.py**: Extracts building data and raster data for specific cities
2. **ErrorAnalyze.py**: Calculates and visualizes error rates between GHSL and actual building footprints
3. **GlobalAggregate.py**: Aggregates data from multiple cities into a single dataset
4. **ContinentVisualize.py** : Creates continent-specific visualizations of error distributions

## Prerequisites

The following Python packages are required:

```
geopandas
rasterio
duckdb
geopy
numpy
pandas
matplotlib
seaborn
rasterstats
pyproj
pickle
```

## Data Sources
1. **GHSL Data**: Download from [Copernicus Emergency Management Service](https://human-settlement.emergency.copernicus.eu/download.php).
2. **Overture Maps Data**: Retrieved using DuckDB with Azure connectivity or local files.

## Workflow Instructions

### Step 1: Extract Building and Raster Data

Use `BuildingExtractor.py` to clip and process GHSL raster data and building footprints for specific cities.

```bash
python BuildingExtractor.py
```

#### Key parameters to configure:
- `city_name`: Name of the city
- `city_center`: Latitude and longitude coordinates of city center
- `x_km, y_km`: Area dimensions in kilometers (e.g., 10km x 10km)
- `input_raster`: Path to GHSL raster file
- `output_dir`: Directory to save outputs
- `azure_connection`: Azure connection string for Overture maps (optional)

#### Example:
```python
city_name = "Auckland"
city_center = (-36.8509, 174.7645)  # (latitude, longitude)
x_km, y_km = 10, 10  # 10km x 10km area
input_raster = "/path/to/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
output_dir = "/path/to/output"
```

#### Outputs:
- City-specific clipped raster (`{city_name}_raster.tif`)
- Split building footprints (`{city_name}_split_buildings.geojson`)

### Step 2: Calculate and Visualize Error Rates

Use `ErrorAnalyzer.py` to process the output from Step 1 and calculate error metrics between GHSL data and actual building footprints.

```bash
python ErrorAnalyzer.py
```

The script will prompt you for:
- GeoJSON file path (from Step 1)
- Raster file path (from Step 1)
- City coordinates
- Path for combined results CSV (optional)
- Whether to load existing results (if available)

#### Outputs:
- Cell-by-cell data CSV (`{city_name}_cell_data.csv`)
- Error analysis plots (`{city_name}_error_area_analysis.png`)
- Pickled results file (`{city_name}_building_error_results.pkl`)

### Step 3: Aggregate City Data

Use `GlobalAggregate.py` to combine data from multiple cities into a single dataset for global analysis.

```bash
python GlobalAggregate.py
```

#### Configuration:
- Modify `data_dir` to point to the directory containing your cell data files

#### Output:
- `aggregated_cities.csv`: Contains summarized metrics for all processed cities

### Step 4: Create Continent-Specific Visualizations

Use `ContinentVisualize.py` to generate visualizations showing error distribution patterns by continent.

```bash
python ContinentVisualize.py
```

#### Configuration:
- Ensure `directory` variable points to your cell data directory

#### Outputs:
- Continent-based error distribution plots (`error_rate_percentage_{continent}.png`)

## Data Structure

Each city's cell data CSV includes the following columns:
- City: City name
- Continent: Continent name
- Latitude/Longitude: City coordinates
- Row/Column: Grid cell position
- GHSL: GHSL building density value
- Buildings: Actual building area
- Error_Rate: (GHSL - Buildings) / Buildings
- MAE: Mean Absolute Error
- RMSE: Root Mean Square Error

## Error Interpretation

The error rate is calculated as:
- 0 = Perfect match (GHSL = Buildings)
- Positive values = GHSL overestimation
- Negative values = GHSL underestimation
- 1 (100%) = Complete overestimation (GHSL > 0, Buildings = 0)
- -1 (-100%) = Complete underestimation (GHSL = 0, Buildings > 0)

## Notes and Tips

1. For `BuildingExtractor.py`:
   - The script supports batch processing of multiple cities
   - If Overture maps access is unavailable, you can use locally available building data

2. For `ErrorAnalyzer.py`:
   - It's recommended to use the saved pickle files when available to avoid reprocessing
   - The visualization includes both histogram and scatter plot analysis

3. For `ContinentVisualizer.py`:
   - The script limits to 8 cities per continent for readability
   - Error rates > 100% are filtered out to focus on the most meaningful patterns

## Example Use Case

1. Extract Auckland building data and GHSL raster:
   ```python
   process_city("Auckland", (-36.8509, 174.7645), 10, 10, input_raster, output_dir)
   ```

2. Run error analysis on the extracted data:
   ```
   Enter GeoJSON path: /path/to/output/Auckland_split_buildings.geojson
   Enter Raster path: /path/to/output/Auckland_raster.tif
   Enter latitude: -36.8509
   Enter longitude: 174.7645
   ```

3. Aggregate multiple city results into a global dataset
4. Generate continent-specific visualizations to compare patterns

## Troubleshooting

- If you encounter memory issues with large cities, try reducing the area dimensions (x_km, y_km)
- For cities with sparse building data, the error metrics may be skewed; consider this in your interpretation
- If DuckDB connection fails, ensure you have the correct connection string or switch to local building data files