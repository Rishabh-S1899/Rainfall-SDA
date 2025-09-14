import pandas as pd
import numpy as np
import requests
import json
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import geopandas as gpd
import osmnx as ox
from utils import  precise_boundary_filter


df = pd.read_csv('datasets/rainfall_data_transformed_final.csv', encoding='utf-8')

df_long = df.melt(id_vars=['latitude','longitude'], var_name='datetime', value_name='rainfall')
df_long['datetime'] = pd.to_datetime(df_long['datetime'])
df_long['rainfall']=df_long['rainfall']*1000  # Convert from meters to 
df_long['id'] = df_long.index

df_long=precise_boundary_filter(df_long, method='osm')

df_long.to_csv('datasets/rainfall_data_final.csv', index=False)