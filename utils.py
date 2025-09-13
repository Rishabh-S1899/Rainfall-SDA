import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import geopandas as gpd
import osmnx as ox


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