import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import os

def visualize_cities_on_map(csv_path, output_path=None, marker_size_column='Buildings_Sum', 
                           color_column='Error_Rate', title='Global City Distribution'):
    """
    Visualize cities on a world map with customizable markers based on data attributes.
    
    Args:
        csv_path: Path to the aggregated cities CSV file
        output_path: Path to save the output image (optional)
        marker_size_column: Column to use for determining marker size
        color_column: Column to use for determining marker color
        title: Title for the map
    """
    # Read the aggregated city data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} cities from {csv_path}")
    
    # Create figure and set size
    plt.figure(figsize=(14, 8))
    
    # Create a Basemap instance - using Robinson projection which is good for world maps
    m = Basemap(projection='robin', resolution='l', lon_0=0)
    
    # Draw map features
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.3)
    m.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=8, linewidth=0.5)
    m.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1], fontsize=8, linewidth=0.5)
    m.fillcontinents(color='lightgray', lake_color='white', alpha=0.5)
    m.drawmapboundary(fill_color='white')
    
    # Set up colors based on continent
    continent_colors = {
        'Africa': 'red',
        'Asia': 'orange',
        'Europe': 'green',
        'North America': 'blue',
        'South America': 'purple',
        'Oceania': 'brown'
    }
    
    # Convert lat/lon to map coordinates
    x, y = m(df['Longitude'].values, df['Latitude'].values)
    
    # Normalize marker sizes - adjust the multiplier as needed
    sizes = df[marker_size_column].values
    size_multiplier = 100 / sizes.max() if sizes.max() > 0 else 1
    marker_sizes = sizes * size_multiplier
    
    # Normalize colors if using a data column for colors
    if color_column:
        colors = df[color_column].values
        norm = plt.Normalize(colors.min(), colors.max())
        
        # Create scatter plot
        scatter = m.scatter(x, y, s=marker_sizes, c=colors, cmap='viridis', 
                         alpha=0.7, edgecolors='black', linewidth=0.5, zorder=5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, shrink=0.7, label=color_column)
        cbar.ax.tick_params(labelsize=8)
    else:
        # Use continent-based colors
        colors = [continent_colors.get(continent, 'gray') for continent in df['Continent']]
        scatter = m.scatter(x, y, s=marker_sizes, c=colors, 
                         alpha=0.7, edgecolors='black', linewidth=0.5, zorder=5)
        
        # Add legend for continents
        for continent, color in continent_colors.items():
            if continent in df['Continent'].values:
                plt.scatter([], [], c=color, alpha=0.7, s=50, label=continent)
        plt.legend(loc='lower left', fontsize=8)
    
    # Add city labels for larger cities or important ones
    # Sort by the size column to label the most significant cities
    top_cities = df.sort_values(by=marker_size_column, ascending=False).head(10)
    for _, city in top_cities.iterrows():
        x_pos, y_pos = m(city['Longitude'], city['Latitude'])
        plt.annotate(city['City'], xy=(x_pos, y_pos), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    # Set title
    plt.title(title, fontsize=14)
    
    # Add note about marker size
    plt.figtext(0.02, 0.02, f"Marker size based on {marker_size_column}", fontsize=8)
    
    # Tight layout to use space efficiently
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to {output_path}")
    
    # Show the map
    plt.show()

def visualize_cities_folium(csv_path, output_path=None, marker_size_column='Buildings_Sum', 
                          color_column='Error_Rate'):
    """
    Alternative visualization using Folium (interactive web map)
    
    Args:
        csv_path: Path to the aggregated cities CSV file
        output_path: Path to save the output HTML file
        marker_size_column: Column to use for determining marker size
        color_column: Column to use for determining marker color
    """
    try:
        import folium
        from folium.plugins import MarkerCluster
        import branca.colormap as cm
    except ImportError:
        print("Please install folium: pip install folium")
        return
    
    # Read the aggregated city data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} cities from {csv_path}")
    
    # Create a map centered at a middle point
    mid_lat = df['Latitude'].mean()
    mid_lon = df['Longitude'].mean()
    city_map = folium.Map(location=[mid_lat, mid_lon], zoom_start=2, 
                         tiles='CartoDB positron')
    
    # Create a marker cluster
    marker_cluster = MarkerCluster().add_to(city_map)
    
    # Normalize sizes
    min_size = 5  # Minimum marker radius in pixels
    max_size = 25  # Maximum marker radius in pixels
    sizes = df[marker_size_column].values
    size_range = sizes.max() - sizes.min() if sizes.max() != sizes.min() else 1
    
    # Create color map if using a color column
    if color_column:
        values = df[color_column].values
        colormap = cm.linear.viridis.scale(values.min(), values.max())
        
        # Add color legend
        colormap.caption = color_column
        city_map.add_child(colormap)
    
    # Add markers for each city
    for _, city in df.iterrows():
        # Calculate marker size
        if sizes.max() != sizes.min():
            size = min_size + (city[marker_size_column] - sizes.min()) / size_range * (max_size - min_size)
        else:
            size = (min_size + max_size) / 2
        
        # Get color based on the value
        if color_column:
            color = colormap(city[color_column])
        else:
            color = 'blue'
        
        # Create popup content
        popup_content = f"""
        <b>{city['City']}, {city['Continent']}</b><br>
        Buildings: {city['Buildings_Sum']:.2f}<br>
        GHSL: {city['GHSL_Sum']}<br>
        Error Rate: {city['Error_Rate']:.4f}<br>
        MAE: {city['MAE']:.4f}<br>
        RMSE: {city['RMSE']:.4f}
        """
        
        # Add marker
        folium.CircleMarker(
            location=[city['Latitude'], city['Longitude']],
            radius=size,
            popup=folium.Popup(popup_content, max_width=200),
            tooltip=city['City'],
            color='black',
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1
        ).add_to(marker_cluster)
    
    # Save the map if output path provided
    if output_path:
        city_map.save(output_path)
        print(f"Interactive map saved to {output_path}")
    
    return city_map

def main():
    # Path to your aggregated data
    csv_path = '/Users/weilynnw/Desktop/building_density_error/aggregated_cities.csv'
    
    # Make sure the file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return
    
    # Create output directory for visualizations if it doesn't exist
    output_dir = '/Users/weilynnw/Desktop/building_density_error/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate static map using matplotlib and basemap
    try:
        # Static map with building count for size and error rate for color
        static_map_path = os.path.join(output_dir, 'cities_world_map.png')
        visualize_cities_on_map(
            csv_path=csv_path,
            output_path=static_map_path,
            marker_size_column='Buildings_Sum',
            color_column='Error_Rate',
            title='Global Distribution of Cities - Building Density and Error Rates'
        )
        print(f"Static map created at {static_map_path}")
    except Exception as e:
        print(f"Error creating static map: {e}")
        print("If Basemap is not installed, try: pip install basemap")
    
    # Generate interactive map using folium
    try:
        interactive_map_path = os.path.join(output_dir, 'cities_interactive_map.html')
        visualize_cities_folium(
            csv_path=csv_path,
            output_path=interactive_map_path,
            marker_size_column='Buildings_Sum',
            color_column='Error_Rate'
        )
        print(f"Interactive map created at {interactive_map_path}")
    except Exception as e:
        print(f"Error creating interactive map: {e}")
        print("Try installing folium: pip install folium")

if __name__ == "__main__":
    main()
