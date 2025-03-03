import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = '/Users/weilynnw/Desktop/RA_new/Chicago_result.csv'
data = pd.read_csv(file_path, low_memory=False)


useful_columns = ['fid', 'GHSL_val', 'id', 'each_building']
filtered_data = data[useful_columns].copy()
filtered_data.rename(columns={'fid': 'GHSL_fid'}, inplace=True)

# Summarize total building area within each unique GHSL_fid
summary_data = filtered_data.groupby(['GHSL_fid', 'GHSL_val']).agg(
    total_building_area=('each_building', 'sum')
).reset_index()

summary_data['error_percentage'] = (
    np.abs(summary_data['GHSL_val'] - summary_data['total_building_area']) / summary_data['total_building_area']
) * 100
print(summary_data.head())
# Calculate MAE
mae = np.mean(np.abs(summary_data['GHSL_val'] - summary_data['total_building_area']))

# Calculate RMSE
rmse = np.sqrt(np.mean((summary_data['GHSL_val'] - summary_data['total_building_area'])**2))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

output_path = '/Users/weilynnw/Desktop/GHSL:overtrue/chicago_result.csv'
summary_data.to_csv(output_path, index=False)

output_path
# Calculate Q1, Q3, and IQR
Q1 = summary_data['error_percentage'].quantile(0.25)
Q3 = summary_data['error_percentage'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

filtered_data = summary_data[
    (summary_data['error_percentage'] >= lower_bound) & 
    (summary_data['error_percentage'] <= upper_bound)
]
plt.figure(figsize=(8, 6))
plt.hist(filtered_data['error_percentage'], bins=20, edgecolor='black')
plt.title('Histogram of Error Percentage (Outliers Removed)')
plt.xlabel('Error Percentage')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
