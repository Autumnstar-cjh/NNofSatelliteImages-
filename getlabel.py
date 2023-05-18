import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysal.lib import weights
from pysal.lib import cg as geometry

data = pd.read_excel('newsatel.xlsx')

df = pd.read_csv('Florida_ct.csv', index_col = 0)
florida_shapefile = gpd.read_file('tl_2020_12_tract/tl_2020_12_tract.shp') # read the shapefile
print(df.shape)
print(florida_shapefile.shape)
florida_shapefile.plot(figsize = (10,10))


# adjust the object types to facilitate the merge
florida_shapefile['GEOID'] = florida_shapefile.GEOID.astype('int64')

# combine the dataframe with the shapefile.
# Note that it is important to choose how - e.g., inner, right, left, etc. Here I choose 'left' for teaching purposes.
df_shp = florida_shapefile.merge(df,
                                 how = 'left',
                                 left_on = 'GEOID',
                                 right_on = 'full_ct_fips')

print(df_shp.shape)
df_shp = df_shp.fillna(0.0)
fig, ax = plt.subplots(figsize=(8, 8))

ax.axis('off') # remove the axis
df_shp.plot(column = 'travel_driving_ratio', cmap = 'plasma', legend=True,
            legend_kwds={'label': "travel by driving ratio", 'orientation': "vertical", 'shrink': 0.3},
            ax = ax)
ax.set_title('travel by driving ratio')

plt.tight_layout()
plt.show()

# x is the longitude.
# y is the latitude.
i = 0
for index, row in data.iterrows():
    y = row['latitude']
    x = row['longitude']
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.axis('off') # remove the axies
    # df_shp.plot(facecolor="None", edgecolor='black', linewidth=0.1, ax = ax)
    df_shp.plot(column = 'travel_driving_ratio', cmap = 'magma', legend=True, alpha = 1.0,
            vmin = 0, vmax = 1,
            legend_kwds={'label': "travel by driving ratio", 'orientation': "vertical", 'shrink': 0.3},
            ax = ax)
    ax.set_title('travel by driving ratio')

    ax.set_xlim(x, x + 0.00001)
    ax.set_ylim(y, y + 0.00001)
    i = i + 1
    plt.tight_layout()
    plt.savefig("plot{}.png".format(i))

