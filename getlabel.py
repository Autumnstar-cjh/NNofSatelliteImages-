import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysal.lib import weights
from pysal.lib import cg as geometry
from shapely import Point

data = pd.read_excel('FL_DataSet/FL_coordinates_newbig_train.xlsx')
df = pd.read_csv('Florida_ct.csv', index_col=0)
FL_shapefile = gpd.read_file('tl_2020_12_tract/tl_2020_12_tract.shp')

# Convert the column types to match for merging
df['full_ct_fips'] = df['full_ct_fips'].astype('int64')
FL_shapefile['GEOID'] = FL_shapefile['GEOID'].astype('int64')

df_shp = pd.merge(FL_shapefile, df, how='left', left_on='GEOID', right_on='full_ct_fips')

new_df = pd.DataFrame(columns=['travel_driving_ratio'])

middle_points = []

for idx, row in FL_shapefile.iterrows():
    centroid = row['geometry'].centroid
    middle_points.append((centroid.x, centroid.y))

i = 1

for index, row in data.iterrows():

    y = row['latitude']
    x = row['longitude']
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.axis('off')
    df_shp.plot(column='travel_driving_ratio', cmap='magma', legend=True, alpha=1.0,
                vmin=0, vmax=1,
                legend_kwds={'label': "travel_driving_ratio", 'orientation': "vertical", 'shrink': 0.3},
                ax=ax)
    ax.set_title('travel_driving_ratio')

    ax.set_xlim(x, x + 0.00001)
    ax.set_ylim(y, y + 0.00001)
    plt.tight_layout()

    travel_driving_ratio = df_shp.loc[df_shp.geometry.contains(Point(x, y)), 'travel_driving_ratio'].values[0]

    new_df = new_df.append({'Number': i, 'target': travel_driving_ratio}, ignore_index=True)
    i = i + 1
    plt.close()

new_df.to_excel('FLlabel_travel_driving_ratio.xlsx', index=False)