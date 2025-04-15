import rasterio
import geopandas as gpd
import rasterstats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Geod
import pickle
import os
import csv

# Dictionary of cities organized by continent
# This can be expanded with more cities as needed
CITY_METADATA = {
    "Europe": ["Moscow", "Berlin", "Paris", "London", "Rome", "Madrid", "Prague", "Vienna", "Amsterdam", "Athens"],
    "Asia": ["Tokyo", "Beijing", "Shanghai", "Singapore", "Seoul", "Bangkok", "Mumbai", "Delhi", "Hong Kong", "Taipei"],
    "North America": ["New York", "Los Angeles", "Chicago", "Toronto", "Mexico City", "Seattle", "Boston", "San Francisco", "Montreal", "Vancouver"],
    "South America": ["Sao Paulo", "Buenos Aires", "Rio de Janeiro", "Lima", "Bogota", "Santiago", "Caracas", "Quito", "Montevideo", "La Paz"],
    "Africa": ["Cairo", "Lagos", "Johannesburg", "Nairobi", "Casablanca", "Cape Town", "Accra", "Tunis", "Addis Ababa", "Dakar"],
    "Oceania": ["Sydney", "Melbourne", "Auckland", "Wellington", "Brisbane", "Perth", "Adelaide", "Christchurch", "Canberra", "Honolulu"]
}

# Function to get city continent
def get_city_metadata(city_name):
    """Get continent for a city"""
    # Clean up city name to handle variations (remove underscores, lowercase, etc.)
    clean_name = city_name.replace("_", " ").strip().lower()
    
    # Try matching with our dictionary
    for continent, cities in CITY_METADATA.items():
        for city in cities:
            if (clean_name == city.lower() or 
                clean_name in city.lower() or 
                city.lower() in clean_name):
                return {"continent": continent}
    
    # Default values if city is not found
    print(f"Warning: City '{city_name}' not found in metadata. Using default values.")
    return {"continent": "Unknown"}

def process_data(geojson_path, raster_path, results_csv_path=None, city_center=None):
    """Process geojson and raster data to calculate error rates"""
    # Extract city name from the geojson path
    city_name = os.path.basename(geojson_path).split('_')[0]  # Get first part of filename as city name
    print(f"City: {city_name}")
    print(f"Loading buildings from {geojson_path}")
    buildings = gpd.read_file(geojson_path)
    print(f"Processing {len(buildings)} buildings in the input file")
    
    # Get city metadata
    metadata = get_city_metadata(city_name)
    continent = metadata["continent"]
    
    # Use provided coordinates
    if city_center:
        coordinates = city_center
        print(f"Using provided coordinates: {coordinates}")
    else:
        coordinates = (0, 0)
        print("No coordinates provided, using (0, 0)")
    
    # Calculate geodesic area for each building
    geod = Geod(ellps="WGS84")
    buildings["geodesic_area_m2"] = buildings.geometry.apply(lambda geom: abs(geod.geometry_area_perimeter(geom)[0]))
    print(buildings.head())
    
    # Open raster file and get metadata
    with rasterio.open(raster_path) as src:
        raster_transform = src.transform
        raster_meta = src.meta.copy()
        raster_shape = (src.height, src.width)
        raster_crs = src.crs
        cell_size_x, cell_size_y = src.res
    
    print(f"Raster Grid Size: {raster_shape} (Rows x Cols)")
    print(f"Cell Size: {cell_size_x} x {cell_size_y} meters")
    
    # Create raster representation of buildings
    building_raster = np.zeros(raster_shape, dtype=np.float32)
    shapes = [(geom, area) for geom, area in zip(buildings.geometry, buildings["geodesic_area_m2"])]
    
    rasterio.features.rasterize(
        shapes,
        out=building_raster,
        transform=raster_transform,
        fill=0,
        all_touched=True,
        merge_alg=rasterio.enums.MergeAlg.add
    )
    
    print("Rasterization Complete! Each grid cell contains the summed building area.")
    
    # Read GHSL population density values
    with rasterio.open(raster_path) as src:
        ghsl_array = src.read(1)
    
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
    
    # Calculate error metrics
    valid_cells = ~np.isnan(error_rate)
    mean_absolute_error = np.nanmean(np.abs(error_rate[valid_cells]))
    rmse = np.sqrt(np.nanmean(error_rate[valid_cells] ** 2))
    
    print(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    
    # Create and save cell-by-cell data
    cell_data = []
    rows, cols = error_rate.shape
    for row in range(rows):
        for col in range(cols):
            cell_data.append({
                'City': city_name,
                'Continent': continent,
                'Latitude': coordinates[0],
                'Longitude': coordinates[1],
                'Row': row,
                'Column': col,
                'GHSL': ghsl_array[row, col],
                'Buildings': building_raster[row, col],
                'Error_Rate': error_rate[row, col],
                'MAE': mean_absolute_error,
                'RMSE': rmse
            })
    
    # Save cell data to CSV with city name
    cell_df = pd.DataFrame(cell_data)
    
    # Create individual city results filename
    cell_csv_file = f'{city_name}_cell_data.csv'
    cell_df.to_csv(cell_csv_file, index=False)
    print(f"Cell-by-cell data saved to {cell_csv_file}")
    
    # Save to the combined results CSV if provided
    if results_csv_path:
        # Create a summary row with city statistics
        summary_data = {
            'City': city_name,
            'Continent': continent,
            'Latitude': coordinates[0],
            'Longitude': coordinates[1],
            'MAE': mean_absolute_error,
            'RMSE': rmse,
            'Total_Cells': len(cell_data),
            'Valid_Cells': np.sum(valid_cells)
        }
        
        # Check if the file exists to determine if we need to write headers
        file_exists = os.path.isfile(results_csv_path)
        
        with open(results_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary_data)
        
        print(f"Summary data appended to {results_csv_path}")
    
    # Prepare results dictionary
    results = {
        'city_name': city_name,
        'continent': continent,
        'coordinates': coordinates,
        'error_rate': error_rate,
        'building_raster': building_raster,
        'ghsl_array': ghsl_array,
        'mean_absolute_error': mean_absolute_error,
        'rmse': rmse,
        'transform': raster_transform,
        'metadata': raster_meta
    }
    
    # Save the results with city name
    output_file = f'{city_name}_building_error_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_file}")
    
    return results

def visualize_results(results, city_name):
    """Create visualizations from analysis results for specific city"""
    error_rate = results['error_rate']
    building_raster = results['building_raster']
    ghsl_array = results['ghsl_array']
    mean_absolute_error = results['mean_absolute_error']
    rmse = results['rmse']
    continent = results.get('continent', 'Unknown')
    coordinates = results.get('coordinates', (0, 0))
    
    # Add metadata to the plot title
    plot_title = f'Building Error Analysis for {city_name}\n'
    plot_title += f'Continent: {continent}, Coordinates: {coordinates}'
    
    error_values = error_rate[~np.isnan(error_rate)].flatten()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Histogram of percentage error
    ax1.hist(error_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Percentage Error ((GHSL - Building)/Building)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Percentage Error')
    ax1.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    stats_text = f'Mean Absolute Error: {mean_absolute_error:.4f}\nRMSE: {rmse:.4f}'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # 2. Scatter plot of building area vs. error rate
    mask = ~np.isnan(error_rate)
    building_values = building_raster[mask].flatten()
    error_values_masked = error_rate[mask].flatten()
    
    max_points = 5000
    if len(building_values) > max_points:
        sample_indices = np.random.choice(len(building_values), max_points, replace=False)
        building_values_sample = building_values[sample_indices]
        error_values_sample = error_values_masked[sample_indices]
    else:
        building_values_sample = building_values
        error_values_sample = error_values_masked
    
    # Create scatter plot
    scatter = ax2.scatter(building_values_sample, error_values_sample, 
                         alpha=0.5, s=5, c='blue', edgecolor=None)
    ax2.set_xlabel('Building Area')
    ax2.set_ylabel('Percentage Error')
    ax2.set_title('Relationship Between Building Area and Error Rate')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--')
    
    # Calculate correlation
    correlation = np.corrcoef(building_values, error_values_masked)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    fig.suptitle(plot_title, fontsize=16, y=1.05)
    plt.savefig(f'{city_name}_error_area_analysis.png', dpi=300)
    plt.show()
    
    # Print summary statistics
    print("\nError Distribution Statistics:")
    print(f"City: {city_name}")
    print(f"Continent: {continent}")
    print(f"Coordinates: {coordinates}")
    print(f"Mean Error: {np.mean(error_values):.4f}")
    print(f"Median Error: {np.median(error_values):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Standard Deviation: {np.std(error_values):.4f}")
    print(f"Min: {np.min(error_values):.4f}")
    print(f"Max: {np.max(error_values):.4f}")
    print(f"25th Percentile: {np.percentile(error_values, 25):.4f}")
    print(f"75th Percentile: {np.percentile(error_values, 75):.4f}")
    print(f"\nCorrelation between Building Area and Error Rate: {correlation:.4f}")
    
    # Additional analysis for display only (no CSV export)
    area_bins = np.linspace(np.min(building_values), np.max(building_values), 10)
    area_labels = [f"{area_bins[i]:.1f}-{area_bins[i+1]:.1f}" for i in range(len(area_bins)-1)]
    
    # Assign each building to a bin
    bin_indices = np.digitize(building_values, area_bins) - 1
    bin_indices[bin_indices == len(area_bins)-1] = len(area_bins)-2  # Handle edge case
    
    # Calculate error statistics for each bin
    bin_stats = []
    for i in range(len(area_bins)-1):
        mask = bin_indices == i
        if np.any(mask):
            errors_in_bin = error_values_masked[mask]
            bin_stats.append({
                'City': city_name,
                'Continent': continent,
                'Latitude': coordinates[0],
                'Longitude': coordinates[1],
                'Area Range': area_labels[i],
                'Count': len(errors_in_bin),
                'Mean Error': np.mean(errors_in_bin),
                'Median Error': np.median(errors_in_bin),
                'MAE': np.mean(np.abs(errors_in_bin)),
                'Std Dev': np.std(errors_in_bin)
            })
    
    bin_df = pd.DataFrame(bin_stats)
    print("\nError Statistics by Building Area Range:")
    print(bin_df)
    
    # No longer saving bin statistics to CSV
    print("Bin statistics calculated but not saved to CSV")

def main():
    """Main function to run processing and visualization for a single city"""
    # Define file paths
    geojson_path = input("Enter the path to the GeoJSON file for the city: ")
    if not geojson_path:
        print("Error: You must provide a path to the GeoJSON file.")
        return
        
    raster_path = input("Enter the path to the Raster file for the city: ")
    if not raster_path:
        print("Error: You must provide a path to the Raster file.")
        return
    
    # Extract city name from the geojson path
    city_name = os.path.basename(geojson_path).split('_')[0]
    
    # Get coordinates from combined_script.py input
    try:
        lat = float(input(f"Enter latitude for {city_name} from combined_script.py: "))
        lon = float(input(f"Enter longitude for {city_name} from combined_script.py: "))
        city_center = (lat, lon)
    except ValueError:
        print("Invalid coordinates. Using (0,0) as default.")
        city_center = (0, 0)
    
    # Define the path for the combined results CSV (optional)
    combined_results_csv = input("Enter the path for the combined results CSV (or press Enter to skip): ")
    if not combined_results_csv:
        combined_results_csv = None
    
    # Option to load existing results instead of reprocessing data
    load_existing = input(f"Load existing results from {city_name}_building_error_results.pkl if available? (y/n): ").lower() == 'y'
    
    if load_existing:
        pkl_file = f'{city_name}_building_error_results.pkl'
        try:
            with open(pkl_file, 'rb') as f:
                results = pickle.load(f)
            print(f"Loaded existing results from {pkl_file}")
            
            # If we loaded from pickle but need to save to combined results
            if combined_results_csv:
                # Create a summary row with city statistics
                metadata = get_city_metadata(results.get('city_name', city_name))
                continent = results.get('continent', metadata.get('continent', 'Unknown'))
                coordinates = results.get('coordinates', city_center)
                
                summary_data = {
                    'City': results.get('city_name', city_name),
                    'Continent': continent,
                    'Latitude': coordinates[0],
                    'Longitude': coordinates[1],
                    'MAE': results.get('mean_absolute_error', 0),
                    'RMSE': results.get('rmse', 0),
                    'Total_Cells': np.size(results.get('error_rate', np.array([]))),
                    'Valid_Cells': np.sum(~np.isnan(results.get('error_rate', np.array([]))))
                }
                
                file_exists = os.path.isfile(combined_results_csv)
                
                with open(combined_results_csv, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=summary_data.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(summary_data)
                
                print(f"Summary data appended to {combined_results_csv}")
                
        except FileNotFoundError:
            print(f"No existing results found for {city_name}. Processing data...")
            results = process_data(geojson_path, raster_path, combined_results_csv, city_center)
    else:
        results = process_data(geojson_path, raster_path, combined_results_csv, city_center)
    
    city_name = results.get('city_name', city_name)
    
    # Visualize results with city name
    visualize_results(results, city_name)

if __name__ == "__main__":
    main()
