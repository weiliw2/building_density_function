# README

## Overview
This project compares the GHSL (Global Human Settlement Layer) dataset with Overture Maps to analyze building density.

## Data Sources
1. **GHSL Data**: Download from [Copernicus Emergency Management Service](https://human-settlement.emergency.copernicus.eu/download.php).
2. **Overture Maps Data**: Retrieved using the `overturemaps` CLI tool.

## Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- Required Python packages:
  ```bash
  pip install geopandas rasterio rasterstats numpy pandas pyproj overturemaps
  ```

## Workflow
Run the scripts in the following order:

1. **Get_city_area.py**: Modify this script to input the city center coordinates.
2. **resulttiff.py**: Processes the GHSL raster data.
3. **split_buildings.py**: Segments building footprints from Overture Maps.
4. **automationv3.py**: Automates data processing and analysis.
5. **visualization-script.py**: Generates visualizations for analysis.

## Usage
1. Edit `Get_city_area.py` to specify the city center coordinates.
2. Run each script sequentially.
3. Analyze the outputs and visualizations.

## Notes
- Ensure that your input data files are correctly formatted and placed in the expected directories.
- Modify script parameters as needed for your specific dataset.

