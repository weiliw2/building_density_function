import pandas as pd
import glob
import os
import numpy as np

def aggregate_city_data(data_dir='/Users/weilynnw/Desktop/building_density_error/Cell_data', file_pattern='*_cell_data.csv'):
    """
    Aggregates multiple city cell data files into a single summary file
    with one row per city.
    
    Args:
        data_dir: Directory containing the CSV files
        file_pattern: Pattern matching the CSV files to process
        
    Returns:
        DataFrame with aggregated city data including MAE, RMSE, and MAPE
    """
    full_pattern = os.path.join(data_dir, file_pattern)
    
    # List all matching files
    files = glob.glob(full_pattern)
    print(f"Found {len(files)} files to process")
    
    # List to store city summaries
    city_summaries = []

    for file_path in files:
        print(f"Processing {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Skip empty files
            if df.empty:
                print(f"File {file_path} is empty, skipping")
                continue
                
            # Get city name and continent from first row
            city = df['City'].iloc[0]
            continent = df['Continent'].iloc[0]
            
            # Get lat/long (should be the same for all rows in a city)
            latitude = df['Latitude'].iloc[0]
            longitude = df['Longitude'].iloc[0]
            
            # Sum the total GHSL and Buildings values
            ghsl_sum = round(df['GHSL'].sum())
            buildings_sum = df['Buildings'].sum()
            
            # Calculate MAE (Mean Absolute Error) from GHSL and Buildings columns
            mae = np.mean(np.abs(df['GHSL'] - df['Buildings']))
            
            # Calculate RMSE (Root Mean Square Error) from GHSL and Buildings columns
            rmse = np.sqrt(np.mean((df['GHSL'] - df['Buildings'])**2))
            
            # Calculate MAPE (Mean Absolute Percentage Error) from GHSL and Buildings columns
            # mean(|GHSL - Buildings| / |Buildings|) * 100
            # Avoid division by zero by filtering out zero values
            non_zero_mask = df['Buildings'] != 0
            if non_zero_mask.any():
                mape = np.mean(np.abs((df['GHSL'][non_zero_mask] - df['Buildings'][non_zero_mask]) / df['Buildings'][non_zero_mask])) * 100
            else:
                mape = np.nan
                print(f"Warning: No non-zero Buildings values in {city}, MAPE set to NaN")
            
            # Create summary row for this city
            city_summary = {
                'City': city,
                'Continent': continent,
                'Latitude': latitude,
                'Longitude': longitude,
                'GHSL_Sum': ghsl_sum,
                'Buildings_Sum': buildings_sum,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
            
            city_summaries.append(city_summary)
            print(f"Added summary for {city}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create DataFrame from summaries
    result_df = pd.DataFrame(city_summaries)
    
    return result_df

def main():
    data_dir = '/Users/weilynnw/Desktop/building_density_error/Cell_data'
    
    aggregated_data = aggregate_city_data(data_dir=data_dir)
    
    output_dir = '/Users/weilynnw/Desktop/building_density_error'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'aggregated_cities.csv')
    aggregated_data.to_csv(output_file, index=False)
    print(f"Saved aggregated data to {output_file}")
    
    print("\nAggregated Data:")
    print(aggregated_data)

if __name__ == "__main__":
    main()