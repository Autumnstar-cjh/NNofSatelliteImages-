import googlemaps
import urllib.request
import random
import pandas as pd

data = pd.read_excel('newsatel.xlsx')
gmaps = googlemaps.Client(key='YourKey')
i = 206
for index, row in data.iterrows():
    i = i + 1
    longitude = row['longitude']
    latitude = row['latitude']
    zoom = 20
    size = '1024x1024'
    url = "http://maps.googleapis.com/maps/api/staticmap?center=" + str(latitude) + "," + str(longitude) + "&zoom=" + str(zoom) + "&size=" + size + "&maptype=satellite&key=" + 'AIzaSyB7d-wrNrXQfEX51oEVmc-bO6E9kaClBWI'
    urllib.request.urlretrieve(url, str(i) + ".jpg")
