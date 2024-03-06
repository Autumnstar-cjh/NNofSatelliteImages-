import pandas as pd
import googlemaps
import urllib.request
import os

data = pd.read_excel('FL_Dataset/All_Coordinates.xlsx')
gmaps = googlemaps.Client(key='*')
i = 0

# 图片保存的文件夹路径
folder_path = 'FL_DataSet/all_SV'

for index, row in data.iterrows():
    i += 1
    longitude = row['longitude']
    latitude = row['latitude']
    size = '1024x1024'
    fov = 90
    heading = 0  # 你可以根据实际情况调整此值
    pitch = 0  # 你可以根据实际情况调整此值
    radius = 3000
    source = 'outdoor'
    url = f"https://maps.googleapis.com/maps/api/streetview?location={latitude},{longitude}&size={size}&fov={fov}&heading={heading}&pitch={pitch}&radius={radius}&source={source}&key=*"
    # 文件保存路径
    file_path = os.path.join(folder_path, f"{i}.jpg")
    # 下载图片并保存
    urllib.request.urlretrieve(url, file_path)


