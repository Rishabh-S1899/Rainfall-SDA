import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import geopandas as gpd
import osmnx as ox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar



def filter_hp_uk_data(df):
    """
    Filter dataset to include only regions within Himachal Pradesh and Uttarakhand
    
    Parameters:
    df (pandas.DataFrame): Input dataframe with 'latitude' and 'longitude' columns
    
    Returns:
    pandas.DataFrame: Filtered dataframe
    """
    
    # Approximate bounding boxes for the states
    # Himachal Pradesh bounds
    hp_lat_min, hp_lat_max = 30.22, 33.25
    hp_lon_min, hp_lon_max = 75.47, 79.04
    
    # Uttarakhand bounds  
    uk_lat_min, uk_lat_max = 28.43, 31.45
    uk_lon_min, uk_lon_max = 77.34, 81.03
    
    # Create boolean masks for each state
    hp_mask = ((df['latitude'] >= hp_lat_min) & (df['latitude'] <= hp_lat_max) & 
               (df['longitude'] >= hp_lon_min) & (df['longitude'] <= hp_lon_max))
    
    uk_mask = ((df['latitude'] >= uk_lat_min) & (df['latitude'] <= uk_lat_max) & 
               (df['longitude'] >= uk_lon_min) & (df['longitude'] <= uk_lon_max))
    
    # Combine masks using OR operation
    combined_mask = hp_mask | uk_mask
    
    # Filter the dataframe
    filtered_df = df[combined_mask].copy()
    
    # Add a state column for identification
    filtered_df['state'] = 'Unknown'
    filtered_df.loc[hp_mask, 'state'] = 'Himachal Pradesh'
    filtered_df.loc[uk_mask, 'state'] = 'Uttarakhand'
    
    return filtered_df

def download_osm_boundaries():
    """
    Download state boundaries from OpenStreetMap using OSMnx.
    """
    try:
        # Get boundaries for Himachal Pradesh and Uttarakhand
        # We can pass a list of place names to geocode_to_gdf
        places = ['Himachal Pradesh, India', 'Uttarakhand, India']
        gdf = ox.geocode_to_gdf(places)
        print("Downloaded OSM boundaries")
        
        # Filter for the specific states
        hp_boundary = gdf[gdf['name'] == 'Himachal Pradesh'].geometry
        uk_boundary = gdf[gdf['name'] == 'Uttarakhand'].geometry
        
        if not hp_boundary.empty and not uk_boundary.empty:
            return hp_boundary.iloc[0], uk_boundary.iloc[0]
        else:
            print("States not found in OpenStreetMap data")
            return None, None
            
    except Exception as e:
        print(f"Error downloading OSM data: {e}")
        return None, None

def precise_boundary_filter(df, method='polygon'):
    """
    Main function to filter data using precise boundaries
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    method (str): 'polygon', 'osm', or 'natural_earth'
    
    Returns:
    pandas.DataFrame: Precisely filtered dataframe
    """
    
    print(f"Using {method} method for boundary filtering...")
    print(f"Original data points: {len(df)}")
    
    if method == 'polygon':
        # Use predefined polygons
        filtered_df = filter_hp_uk_data(df)
        
    elif method == 'osm':
        # Use OpenStreetMap boundaries
        hp_boundary, uk_boundary = download_osm_boundaries()
        print("Extracted OSM Boundaries")
        if hp_boundary and uk_boundary:
    # Create a GeoDataFrame from your DataFrame
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df['longitude'], df['latitude'])
            )
            gdf = gdf.set_crs('EPSG:4326')
            # Convert the boundaries to GeoDataFrames for the join
            hp_gdf = gpd.GeoDataFrame(geometry=[hp_boundary], crs='EPSG:4326')
            uk_gdf = gpd.GeoDataFrame(geometry=[uk_boundary], crs='EPSG:4326')

            # Perform geospatial joins to find points within the boundaries
            hp_points = gpd.sjoin(gdf, hp_gdf, how="inner", predicate="within")
            uk_points = gpd.sjoin(gdf, uk_gdf, how="inner", predicate="within")

            # Combine the results and add the state column
            filtered_df = pd.concat([hp_points, uk_points]).drop_duplicates(subset=['id']).copy()
            filtered_df['state'] = filtered_df.index.map(lambda idx: 'Himachal Pradesh' if idx in hp_points.index else 'Uttarakhand')

            # Optional: If you need a pandas DataFrame, convert it back
            filtered_df = pd.DataFrame(filtered_df.drop(columns=['geometry']))

        else:
            print("Failed to download OSM boundaries, falling back to polygon method")
            filtered_df = filter_hp_uk_data(df)

    print(f"Filtered data points: {len(filtered_df)}")
    print(f"Data retained: {len(filtered_df)/len(df)*100:.2f}%")

    if 'state' in filtered_df.columns:
        print("\nState-wise distribution:")
        print(filtered_df['state'].value_counts())
        
    return filtered_df

def plot_state_rainfall(data, state_name, years):
    """
    Generates and saves a grid of monthly rainfall plots for a given state.
    The grid size adapts to the number of months available in the data.

    Args:
        data (pd.DataFrame): The dataframe containing aggregated rainfall data for the state.
        state_name (str): The name of the state to plot.
        years (list): A list of years to set as ticks on the x-axis.
    """
    # Determine the unique months to plot
    months_to_plot = sorted(data['month'].unique())
    num_months = len(months_to_plot)

    if num_months == 0:
        print(f"No data available for {state_name}. Skipping plot generation.")
        return

    # Create a 2x2 subplot grid, suitable for up to 4 months.
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    fig.suptitle(f'Average Monthly Rainfall Trends ({years[0]}-{years[-1]}) in {state_name}', fontsize=24, y=1.0)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, month_num in enumerate(months_to_plot):
        ax = axes[i]
        month_name = calendar.month_name[month_num]
        
        # Filter data for the specific month
        month_data = data[data['month'] == month_num]
        
        if not month_data.empty:
            # Create a bar plot for the month
            ax.bar(month_data['year'], month_data['rainfall'], color='skyblue', edgecolor='black')
        
        # Formatting the subplot
        ax.set_title(month_name, fontsize=16)
        ax.set_xticks(years)
        ax.tick_params(axis='x', rotation=90, labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hide any unused subplots
    for i in range(num_months, len(axes)):
        axes[i].set_visible(False)

    # Set common labels
    fig.text(0.5, 0.01, 'Year', ha='center', va='center', fontsize=18)
    fig.text(0.01, 0.5, 'Average Rainfall (mm)', ha='center', va='center', rotation='vertical', fontsize=18)

    plt.tight_layout(rect=[0.02, 0.03, 1, 0.98])
    
    # Save the plot to a file
    filename = f"{state_name.replace(' ', '_')}_rainfall_plot.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    plt.show()

def generate_plots_from_dataframe(df):
    """
    Takes a DataFrame with rainfall data and generates monthly plots for each state.

    The input DataFrame should have 'year', 'month', 'state', 
    and 'monthly_rainfall' columns.
    """
    # --- 1. Data Preparation ---
    # Work with a copy to avoid modifying the original DataFrame.
    df_processed = df.copy()
    # Rename 'monthly_rainfall' to 'rainfall' for consistency with the plotting function.
    df_processed.rename(columns={'monthly_rainfall': 'rainfall'}, inplace=True)

    # --- 2. Data Aggregation ---
    # The data is monthly, but there might be multiple locations (lat/lon) per state.
    # We calculate the average rainfall for each month across all locations in a state.
    monthly_avg_rainfall = df_processed.groupby(['state', 'year', 'month'])['rainfall'].mean().reset_index()
    
    # --- 3. Generate Plots for Each State ---
    states = monthly_avg_rainfall['state'].unique()
    years = sorted(monthly_avg_rainfall['year'].unique())
    
    for state in states:
        state_data = monthly_avg_rainfall[monthly_avg_rainfall['state'] == state].copy()
        plot_state_rainfall(state_data, state, years)


# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_monthly_rainfall(df):
#     """
#     Aggregates hourly rainfall data to daily and plots monthly rainfall for each year.

#     Args:
#         df (pd.DataFrame): DataFrame with columns 'datetime', 'rainfall', and 'state'.
#     """
#     # Drop the redundant 'index_right' column if it exists
#     if 'index_right' in df.columns:
#         df = df.drop(columns=['index_right'])

#     # Convert 'datetime' column to datetime objects
#     df['datetime'] = pd.to_datetime(df['datetime'])

#     # Extract year, month, and day
#     df['year'] = df['datetime'].dt.year
#     df['month'] = df['datetime'].dt.month
#     df['day'] = df['datetime'].dt.day

#     # Filter for months June to September
#     df = df[df['month'].isin([6, 7, 8, 9])]

#     # Aggregate hourly data to daily rainfall for each state
#     daily_rainfall = df.groupby(['year', 'month', 'day', 'state'])['rainfall'].sum().reset_index()

#     # Get unique years from the data
#     years = sorted(daily_rainfall['year'].unique())
#     months = {6: 'June', 7: 'July', 8: 'August', 9: 'September'}

#     # Create a plot for each year
#     for year in years:
#         fig, axes = plt.subplots(2, 2, figsize=(18, 12))
#         fig.suptitle(f'Daily Rainfall for {year}', fontsize=20, y=1.02)

#         # Filter data for the current year
#         year_data = daily_rainfall[daily_rainfall['year'] == year]

#         for i, (month_num, month_name) in enumerate(months.items()):
#             ax = axes[i // 2, i % 2]
#             month_data = year_data[year_data['month'] == month_num]

#             if not month_data.empty:
#                 # Pivot data to have days as index and states as columns
#                 pivot_df = month_data.pivot_table(index='day', columns='state', values='rainfall', aggfunc='sum').fillna(0)
                
#                 # Plotting the stacked bar chart
#                 pivot_df.plot(kind='bar', stacked=True, ax=ax, width=0.8, 
#                               color={'Himachal Pradesh': 'skyblue', 'Uttarakhand': 'lightgreen'})
            
#             ax.set_title(month_name, fontsize=14)
#             ax.set_xlabel('Day of the Month', fontsize=12)
#             ax.set_ylabel('Total Rainfall (mm)', fontsize=12)
#             ax.tick_params(axis='x', rotation=45)
#             ax.legend(title='State')
#             ax.grid(axis='y', linestyle='--', alpha=0.7)

#         plt.tight_layout()
#         # Save the figure
#         plt.savefig(f'monthly_rainfall_{year}.png')
#         plt.close()

import os

def plot_monthly_rainfall_v2(df):
    """
    Aggregates hourly rainfall data to daily and plots monthly rainfall for each year and state.

    Args:
        df (pd.DataFrame): DataFrame with columns 'datetime', 'rainfall', and 'state'.
    """
    # Create the directory structure if it doesn't exist
    output_dir = 'plots/month_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Drop the redundant 'index_right' column if it exists
    if 'index_right' in df.columns:
        df = df.drop(columns=['index_right'])

    # Convert 'datetime' column to datetime objects
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract year, month, and day
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day

    # Filter for months June to September
    df = df[df['month'].isin([6, 7, 8, 9])]

    # Aggregate hourly data to daily rainfall for each state
    daily_rainfall = df.groupby(['year', 'month', 'day', 'state'])['rainfall'].sum().reset_index()

    # Get unique years and states from the data
    years = sorted(daily_rainfall['year'].unique())
    states = daily_rainfall['state'].unique()
    months = {6: 'June', 7: 'July', 8: 'August', 9: 'September'}

    # Create a plot for each year and each state
    for year in years:
        for state in states:
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            fig.suptitle(f'Daily Rainfall for {state} in {year}', fontsize=20, y=1.02)

            # Filter data for the current year and state
            year_state_data = daily_rainfall[(daily_rainfall['year'] == year) & (daily_rainfall['state'] == state)]

            for i, (month_num, month_name) in enumerate(months.items()):
                ax = axes[i // 2, i % 2]
                month_data = year_state_data[year_state_data['month'] == month_num]

                if not month_data.empty:
                    # Plotting the line plot with markers
                    ax.plot(month_data['day'], month_data['rainfall'], marker='o', linestyle='-', color='royalblue')
                
                ax.set_title(month_name, fontsize=14)
                ax.set_xlabel('Day of the Month', fontsize=12)
                ax.set_ylabel('Total Rainfall (mm)', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                # Set x-ticks to be integers representing days
                ax.set_xticks(range(1, 32)) 
                ax.set_xticklabels([str(i) if i % 2 != 0 else '' for i in range(1, 32)])
            plt.title(f'Daily Rainfall for {state} in {year}', fontsize=16)
            plt.tight_layout()
            # Sanitize state name for filename
            state_filename = state.replace(" ", "_")
            # Save the figure
            plt.savefig(f'{output_dir}/{year}_{state_filename}.png')
            print(f"Plot saved as {output_dir}/{year}_{state_filename}.png")
            plt.close()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
def plot_average_daily_rainfall_v2(df):
    """
    Calculates the average of daily total rainfall over the entire period and 
    creates both stacked bar and line plots with visible titles.

    Args:
        df (pd.DataFrame): DataFrame with columns 'datetime', 'rainfall', and 'state'.
    """
    # --- 1. Setup and Data Preparation ---
    bar_plot_dir = 'plots/average_daily_total_bar'
    line_plot_dir = 'plots/average_daily_total_line'
    os.makedirs(bar_plot_dir, exist_ok=True)
    os.makedirs(line_plot_dir, exist_ok=True)

    if 'index_right' in df.columns:
        df = df.drop(columns=['index_right'])

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day

    df = df[df['month'].isin([6, 7, 8, 9])]

    # --- 2. Corrected Aggregation ---
    daily_totals = df.groupby(['state', 'year', 'month', 'day'])['rainfall'].sum().reset_index()
    avg_daily_rainfall = daily_totals.groupby(['state', 'month', 'day'])['rainfall'].mean().reset_index()

    states = avg_daily_rainfall['state'].unique()
    months = {6: 'June', 7: 'July', 8: 'August', 9: 'September'}

    # --- 3. Generate Stacked Bar Plots ---
    fig_bar, axes_bar = plt.subplots(2, 2, figsize=(18, 12))
    fig_bar.suptitle('Average of Daily Total Rainfall (All Years) - Combined', fontsize=20)

    for i, (month_num, month_name) in enumerate(months.items()):
        ax = axes_bar[i // 2, i % 2]
        month_data = avg_daily_rainfall[avg_daily_rainfall['month'] == month_num]

        if not month_data.empty:
            pivot_df = month_data.pivot_table(index='day', columns='state', values='rainfall').fillna(0)
            pivot_df.plot(kind='bar', stacked=True, ax=ax, width=0.8,
                          color={'Himachal Pradesh': 'skyblue', 'Uttarakhand': 'lightgreen'})

        ax.set_title(month_name, fontsize=14)
        ax.set_xlabel('Day of the Month', fontsize=12)
        ax.set_ylabel('Average Daily Rainfall (mm)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='State')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # **FIX: Add rect parameter to make space for the suptitle**
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{bar_plot_dir}/average_daily_total_rainfall_bar_plot.png')
    plt.close(fig_bar)

    # --- 4. Generate Separate Line Plots ---
    for state in states:
        fig_line, axes_line = plt.subplots(2, 2, figsize=(18, 12))
        fig_line.suptitle(f'Average of Daily Total Rainfall (All Years) - {state}', fontsize=20)
        
        state_data = avg_daily_rainfall[avg_daily_rainfall['state'] == state]

        for i, (month_num, month_name) in enumerate(months.items()):
            ax = axes_line[i // 2, i % 2]
            month_data = state_data[state_data['month'] == month_num]

            if not month_data.empty:
                ax.plot(month_data['day'], month_data['rainfall'], marker='o', linestyle='-', color='royalblue')

            ax.set_title(month_name, fontsize=14)
            ax.set_xlabel('Day of the Month', fontsize=12)
            ax.set_ylabel('Average Daily Rainfall (mm)', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks(range(1, 32))
            ax.set_xticklabels([str(j) if j % 2 != 0 else '' for j in range(1, 32)])

        # **FIX: Add rect parameter to make space for the suptitle**
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        state_filename = state.replace(" ", "_")
        plt.savefig(f'{line_plot_dir}/average_daily_total_rainfall_line_plot_{state_filename}.png')
        plt.close(fig_line)


def plot_yearly_total_rainfall_comparison(df):
    """
    Calculates and plots the total monsoon rainfall for each year for both
    states on a single line plot for comparison.

    Args:
        df (pd.DataFrame): DataFrame with columns 'datetime', 'rainfall', and 'state'.
    """
    # --- 1. Setup and Data Preparation ---
    output_dir = 'plots/yearly_rainfall_comparison'
    os.makedirs(output_dir, exist_ok=True)

    if 'index_right' in df.columns:
        df = df.drop(columns=['index_right'])

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df = df[df['datetime'].dt.month.isin([6, 7, 8, 9])]

    # --- 2. Data Aggregation and Pivoting ---
    # Sum all rainfall records to get the total for each state and year
    yearly_total_rainfall = df.groupby(['state', 'year'])['rainfall'].sum().reset_index()

    # Pivot the table to have years as index and states as columns
    pivot_df = yearly_total_rainfall.pivot(index='year', columns='state', values='rainfall')

    # --- 3. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot both lines on the same axes
    pivot_df.plot(kind='line', marker='o', ax=ax)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Total Monsoon Rainfall (mm)', fontsize=12)
    ax.set_title('Total Monsoon Rainfall Comparison (2010-2024)', fontsize=16)
    ax.set_xticks(pivot_df.index)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='State')
    
    # Ensure y-axis starts at 0
    ax.set_ylim(0)

    fig.tight_layout()
    plt.savefig(f'{output_dir}/yearly_rainfall_comparison.png')
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from statsmodels.graphics.tsaplots import plot_acf

# def plot_rainfall_autocorrelation(df):
#     """
#     Creates yearly and monthly autocorrelation plots for the rainfall data.

#     Args:
#         df (pd.DataFrame): DataFrame with columns 'datetime', 'rainfall', and 'state'.
#     """
#     # --- 1. Setup and Data Preparation ---
#     output_dir = 'plots/autocorrelation'
#     os.makedirs(output_dir, exist_ok=True)

#     if 'index_right' in df.columns:
#         df = df.drop(columns=['index_right'])

#     df['datetime'] = pd.to_datetime(df['datetime'])
#     df['year'] = df['datetime'].dt.year
#     df['month'] = df['datetime'].dt.month

#     states = df['state'].unique()

#     # --- 2. Yearly Autocorrelation Plot ---
#     yearly_total = df.groupby(['state', 'year'])['rainfall'].sum()

#     for state in states:
#         fig, ax = plt.subplots(figsize=(10, 6))
#         plot_acf(yearly_total.loc[state], ax=ax, lags=10) # Show lags up to 10 years
#         ax.set_title(f'Yearly Rainfall Autocorrelation for {state}', fontsize=16)
#         ax.set_xlabel('Lag (Years)', fontsize=12)
#         ax.set_ylabel('Autocorrelation', fontsize=12)
        
#         state_filename = state.replace(" ", "_")
#         plt.savefig(f'{output_dir}/yearly_autocorrelation_{state_filename}.png')
#         plt.close(fig)

#     # --- 3. Monthly Autocorrelation Plots (Subplots) ---
#     monthly_total = df.groupby(['state', 'year', 'month'])['rainfall'].sum().reset_index()
#     months = {6: 'June', 7: 'July', 8: 'August', 9: 'September'}

#     for state in states:
#         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#         fig.suptitle(f'Monthly Rainfall Autocorrelation for {state} (2010-2024)', fontsize=18)
        
#         for i, (month_num, month_name) in enumerate(months.items()):
#             ax = axes[i // 2, i % 2]
            
#             # Get the time series for the specific month across all years
#             month_series = monthly_total[
#                 (monthly_total['state'] == state) & 
#                 (monthly_total['month'] == month_num)
#             ].set_index('year')['rainfall']
            
#             plot_acf(month_series, ax=ax, lags=6) # Lags up to 6 years is sufficient
#             ax.set_title(f'{month_name}', fontsize=14)
#             ax.set_xlabel('Lag (Years)')
#             ax.set_ylabel('Autocorrelation')

#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         state_filename = state.replace(" ", "_")
#         plt.savefig(f'{output_dir}/monthly_autocorrelation_{state_filename}.png')
#         plt.close(fig)


def plot_standard_acf(df):
    """
    Generates standard yearly, monthly, and daily ACF plots.

    Args:
        df (pd.DataFrame): DataFrame with 'datetime', 'rainfall', and 'state'.
    """
    # --- 1. Setup and Data Preparation ---
    output_dir = 'plots/standard_acf'
    os.makedirs(output_dir, exist_ok=True)

    if 'index_right' in df.columns:
        df = df.drop(columns=['index_right'])

    df['datetime'] = pd.to_datetime(df['datetime'])
    states = df['state'].unique()

    # --- 2. Yearly ACF Plot ---
    df['year'] = df['datetime'].dt.year
    yearly_total = df.groupby(['state', 'year'])['rainfall'].sum()

    for state in states:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_acf(yearly_total.loc[state], ax=ax, lags=10, title=f'Yearly Rainfall ACF for {state}')
        ax.set_xlabel('Lag (Years)')
        ax.set_ylabel('Autocorrelation')
        state_filename = state.replace(" ", "_")
        plt.savefig(f'{output_dir}/yearly_acf_{state_filename}.png')
        plt.close(fig)

    # --- 3. Monthly ACF Plot ---
    df['month_period'] = df['datetime'].dt.to_period('M')
    monthly_total = df.groupby(['state', 'month_period'])['rainfall'].sum()
    
    for state in states:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plotting up to 36 months to see potential annual cycles
        plot_acf(monthly_total.loc[state], ax=ax, lags=36, title=f'Monthly Rainfall ACF for {state}')
        ax.set_xlabel('Lag (Months)')
        ax.set_ylabel('Autocorrelation')
        state_filename = state.replace(" ", "_")
        plt.savefig(f'{output_dir}/monthly_acf_{state_filename}.png')
        plt.close(fig)

    # --- 4. Daily ACF Plot ---
    daily_total = df.groupby(['state', df['datetime'].dt.date])['rainfall'].sum()

    for state in states:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plotting up to 60 days to see weather persistence
        plot_acf(daily_total.loc[state], ax=ax, lags=60, title=f'Daily Rainfall ACF for {state}')
        ax.set_xlabel('Lag (Days)')
        ax.set_ylabel('Autocorrelation')
        state_filename = state.replace(" ", "_")
        plt.savefig(f'{output_dir}/daily_acf_{state_filename}.png')
        plt.close(fig)

def plot_monthly_acf_for_year(df, year, output_dir, lags=48):
    """
    For a given year, generates a subplot of ACF for each monsoon month.
    Each month will have a separate plot for each state.
    """
    print(f"\nGenerating monthly ACF plots for the year {year}...")
    monsoon_months = [6, 7, 8, 9]
    
    for state in df['state'].unique():
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Monthly Rainfall ACF for {state} - {year}', fontsize=16)
        axes = axes.flatten()

        for i, month in enumerate(monsoon_months):
            # Filter data for the specific state, year, and month
            monthly_data = df[(df['state'] == state) & (df.index.year == year) & (df.index.month == month)]
            
            if not monthly_data.empty:
                plot_acf(monthly_data['rainfall'], ax=axes[i], lags=lags, title=f'Month: {month}')
            else:
                axes[i].set_title(f'Month: {month} (No Data)')
                axes[i].text(0.5, 0.5, 'No Data Available', ha='center', va='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f'monthly_acf_{state}_{year}.png')
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.close(fig) # Close the figure to free memory

def plot_yearly_monsoon_acf(df, output_dir, lags=168):
    """
    Generates ACF plots for the entire monsoon season for each year.
    """
    print("\nGenerating ACF plots for each monsoon season by year...")
    years = df.index.year.unique()
    states = df['state'].unique()

    for year in years:
        fig, axes = plt.subplots(1, len(states), figsize=(14, 6))
        fig.suptitle(f'Monsoon Season Rainfall ACF - {year}', fontsize=16)
        if len(states) == 1:
            axes = [axes]

        for i, state in enumerate(states):
            # Filter data for the state and year
            yearly_data = df[(df['state'] == state) & (df.index.year == year)]
            
            if not yearly_data.empty:
                plot_acf(yearly_data['rainfall'], ax=axes[i], lags=lags, title=state)
            else:
                axes[i].set_title(f'{state} (No Data)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f'yearly_monsoon_acf_{year}.png')
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.close(fig) # Close the figure to free memory


def plot_overall_acf(df, output_dir, lags=336):
    """
    Generates a single ACF plot aggregating all years.
    """
    print("\nGenerating overall ACF plot for all years combined...")
    states = df['state'].unique()
    fig, axes = plt.subplots(1, len(states), figsize=(14, 6))
    fig.suptitle('Overall Rainfall ACF (2010-2024)', fontsize=16)
    if len(states) == 1:
            axes = [axes]

    for i, state in enumerate(states):
        state_data = df[df['state'] == state]
        if not state_data.empty:
            plot_acf(state_data['rainfall'], ax=axes[i], lags=lags, title=state)
        else:
            axes[i].set_title(f'{state} (No Data)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, 'overall_acf_all_years.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close(fig) # Close the figure to free memory


def generate_all_acf_plots(df):
    """
    Generates and saves all ACF plots into a 'plots' directory.
    """
    output_dir = 'plots/ACF_plots'
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved in '{output_dir}/' directory.")

    # Plot 1: Monthly ACF for a specific year (e.g., 2022)
    # Looping through a few recent years for demonstration
    for year in range(2010, 2025):
        plot_monthly_acf_for_year(df, year=year, output_dir=output_dir, lags=72)

    # Plot 2: ACF over the entire monsoon season for each year
    plot_yearly_monsoon_acf(df, output_dir=output_dir, lags=24*7)

    # Plot 3: ACF over all years combined
    plot_overall_acf(df, output_dir=output_dir, lags=24*14)