import googlemaps
import urllib.request
import pandas as pd

data = pd.read_excel('CA_testset_coord.xlsx')
gmaps = googlemaps.Client(key='AIzaSyB7d-wrNrXQfEX51oEVmc-bO6E9kaClBWI')
i = 0
zoom = 13
for index, row in data.iterrows():
    i = i + 1
    longitude = row['longitude']
    latitude = row['latitude']
    size = '1024x1024'
    url = "http://maps.googleapis.com/maps/api/staticmap?center=" + str(latitude) + "," + str(longitude) + "&zoom=" + str(zoom) + "&size=" + size + "&maptype=satellite&key=" + 'AIzaSyB7d-wrNrXQfEX51oEVmc-bO6E9kaClBWI'
    urllib.request.urlretrieve(url, str(i) + ".jpg")
