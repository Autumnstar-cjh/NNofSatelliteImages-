import googlemaps
import urllib.request
import pandas as pd
import os

data = pd.read_excel('FL_DataSet/FL_coordinates_train.xlsx')
gmaps = googlemaps.Client(key='AIzaSyB7d-wrNrXQfEX51oEVmc-bO6E9kaClBWI')
i = 0
zoom = 13

# 图片保存的文件夹路径
folder_path = 'FL_DataSet/trainset_zoom=13ST'

# 确保文件夹存在
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

i = 0
zoom = 13

for index, row in data.iterrows():
    i += 1
    longitude = row['longitude']
    latitude = row['latitude']
    size = '1024x1024'
    # 构建URL
    url = f"http://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size}&maptype=satellite&key=AIzaSyB7d-wrNrXQfEX51oEVmc-bO6E9kaClBWI"
    # 文件保存路径
    file_path = os.path.join(folder_path, f"{i}.jpg")
    # 下载图片并保存
    urllib.request.urlretrieve(url, file_path)