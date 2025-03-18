import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open('building_error_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract variables
error_rate = results['error_rate']
building_raster = results['building_raster']
ghsl_array = results['ghsl_array']
mean_absolute_error = results['mean_absolute_error']
rmse = results['rmse']

# Filter out NaN values for visualization
error_values = error_rate[~np.isnan(error_rate)].flatten()

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Histogram of percentage error
ax1.hist(error_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Percentage Error ((GHSL - Building)/Building)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Percentage Error')
ax1.axvline(x=0, color='r', linestyle='--', label='Zero Error')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add stats to histogram
stats_text = f'Mean Absolute Error: {mean_absolute_error:.4f}\nRMSE: {rmse:.4f}'
ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# 2. Scatter plot of building area vs. error rate
# Create mask for valid data points
mask = ~np.isnan(error_rate)
building_values = building_raster[mask].flatten()
error_values_masked = error_rate[mask].flatten()

# If there are too many points, sample them
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
plt.savefig('error_area_analysis.png', dpi=300)
plt.show()

# Print summary statistics
print("\nError Distribution Statistics:")
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

# Additional analysis: Group by building area ranges and analyze error rates
# Create bins for building areas
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

# Save this dataframe for further analysis
bin_df.to_csv('error_by_area_group.csv', index=False)
print("\nArea group statistics saved to 'error_by_area_group.csv'")