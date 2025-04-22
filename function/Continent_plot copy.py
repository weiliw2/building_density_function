import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import numpy as np
directory = '/Users/weilynnw/Desktop/building_density_error/Cell_data'

csv_files = glob(os.path.join(directory, '*_cell_data.csv'))

if not csv_files:
    print(f"No CSV files found in {directory}")
    exit()

# Read all files to get cities and their continents
cities_data = []
for file_path in csv_files:
    # Extract city name from filename
    city_name = os.path.basename(file_path).split('_cell_data.csv')[0]
    
    try:
        df = pd.read_csv(file_path, nrows=1)
        
        if 'Continent' not in df.columns:
            print(f"Continent column not found in {file_path}")
            continue
            
        continent = df['Continent'].iloc[0]
        
        cities_data.append({
            'city': city_name,
            'continent': continent,
            'file_path': file_path
        })
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

cities_df = pd.DataFrame(cities_data)

# Group by continent
continent_groups = cities_df.groupby('continent')

for continent, group in continent_groups:
    files_in_continent = group['file_path'].tolist()
    cities_in_continent = group['city'].tolist()
    
    if len(files_in_continent) > 8:
        files_in_continent = files_in_continent[:8]
        cities_in_continent = cities_in_continent[:8]
    
    # Read error rates from all cities in this continent
    all_filtered_error_rates = []
    
    for file_path in files_in_continent:
        try:
            df = pd.read_csv(file_path)
            
            # Convert Error_Rate to percentage
            df['Error_Rate_Pct'] = df['Error_Rate'] * 100
            
            # Filter out extreme percentage values
            filtered_df = df[df['Error_Rate_Pct'] <= 100]
            all_filtered_error_rates.extend(filtered_df['Error_Rate_Pct'].tolist())
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Determine global min/max for this continent
    global_min = min(min(all_filtered_error_rates), -20)  # Extend to negative values
    global_max = max(max(all_filtered_error_rates), 100)  # Cap at 100% as per requirement
    global_bins = 30
    
    # Create a grid of subplots for this continent
    n_cities = len(files_in_continent)
    n_cols = min(3, n_cities)  # Maximum 3 columns
    n_rows = (n_cities + n_cols - 1) // n_cols  # Ceiling division

    plt.figure(figsize=(15, 4 * n_rows))
    
    # Process each file in this continent
    for i, (file_path, city_name) in enumerate(zip(files_in_continent, cities_in_continent)):
        try:
            df = pd.read_csv(file_path)
            
            df['Error_Rate_Pct'] = df['Error_Rate'] * 100
            
            # Filter error rates > 100%
            filtered_df = df[df['Error_Rate_Pct'] <= 100]
            
            ax = plt.subplot(n_rows, n_cols, i+1)
            
            # Calculate the overall tendency
            total_cells = len(filtered_df)
            over_count = sum(filtered_df['Error_Rate_Pct'] > 0)
            under_count = sum(filtered_df['Error_Rate_Pct'] <= 0)
            over_pct = (over_count / total_cells) * 100 if total_cells > 0 else 0
            under_pct = (under_count / total_cells) * 100 if total_cells > 0 else 0
            
            # Determine the dominant trend
            if over_pct > under_pct:
                dominant_trend = "Overestimation"
                dominant_color = 'indianred'
            else:
                dominant_trend = "Underestimation"
                dominant_color = 'steelblue'
            
            # Create a single histogram with color based on dominant trend
            sns.histplot(filtered_df['Error_Rate_Pct'], kde=True, bins=global_bins, 
                       color=dominant_color, label=dominant_trend, alpha=0.7)
            
            # Set title and labels
            plt.title(f"{city_name}")
            plt.xlabel("Error Rate (%)")
            plt.ylabel("Frequency")
            
            # Set common x limits for all plots in this continent
            plt.xlim(global_min, global_max)
            
            # Add a vertical line at 0 to highlight underestimation vs overestimation
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.9, linewidth=1.5)
            
            # Add mean and median lines
            mean_error = filtered_df['Error_Rate_Pct'].mean()
            median_error = filtered_df['Error_Rate_Pct'].median()
            plt.axvline(x=mean_error, color='green', linestyle='-', alpha=0.7)
            plt.axvline(x=median_error, color='purple', linestyle='-', alpha=0.7)
            
            # Add statistics as text annotation with dominant trend highlighted
            stats_text = (f"Mean: {mean_error:.1f}%\nMedian: {median_error:.1f}%\n"
                          f"Dominant Trend: {dominant_trend} ({max(over_pct, under_pct):.1f}%)\n"
                          f"Over: {over_pct:.1f}%, Under: {under_pct:.1f}%")
            
            plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        va='top', ha='left', fontsize=9, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add legend
            plt.legend(loc='best', fontsize=8)
            
            # Get KDE for the entire distribution
            kde_x = np.linspace(global_min, global_max, 1000)
            kde = sns.kdeplot(filtered_df['Error_Rate_Pct'], x=kde_x, bw_adjust=1).get_lines()[-1].get_data()
            
            # Fill area with light color based on dominant trend
            ax.fill_between(kde[0], kde[1], alpha=0.2, color=dominant_color)
            
            print(f"{continent} - {city_name}: Mean={mean_error:.1f}%, Median={median_error:.1f}%, Overall Trend={dominant_trend} ({max(over_pct, under_pct):.1f}%)")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plt.suptitle(f"Error Rate Distribution: {continent} Cities (Values ≤ 100%)", fontsize=16, y=0.98)

    legend_text = "Black: Zero Error | Green: Mean | Purple: Median | Red: Cities that trend toward Overestimation | Blue: Cities that trend toward Underestimation"
    plt.figtext(0.5, 0.01, legend_text, 
               ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.07)
    plt.savefig(os.path.join(directory, f'error_rate_percentage_{continent}.png'), dpi=300)
    print(f"Generated plot for {continent} with {n_cities} cities")

print("All continent-based plots generated!")