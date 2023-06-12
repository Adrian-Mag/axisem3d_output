import numpy as np
from mayavi import mlab
import pandas as pd


def spherical_to_cart(rad, lat, lon):
    x = rad * np.cos(lat) * np.cos(lon)
    y = rad * np.cos(lat) * np.sin(lon)
    z = rad * np.sin(lat)
    return [x, y, z]


df = pd.read_csv('/home/adrian/PhD/sensitivity_rho.txt', sep=' ')

max_sensitivity = df['sensitivity'].abs().max()

mlab.figure()

#source 
source_rad = 6371000
source_lat = 0
source_lon = 0
[source_x, source_y, source_z] = spherical_to_cart(source_rad, source_lat, source_lon)
mlab.points3d(source_x, source_y, source_z, scale_factor=300000, color=(1,1,1))

# receiver
receiver_rad = 6371000
receiver_lat = 0
receiver_lon = 20
[receiver_x, receiver_y, receiver_z] = spherical_to_cart(receiver_rad, receiver_lat, receiver_lon)
mlab.points3d(receiver_x, receiver_y, receiver_z, scale_factor=300000, color=(1,1,1))


for index, row in df.iterrows():

    rad = row['radius']
    lat = np.deg2rad(row['latitude'])
    lon = np.deg2rad(row['longitude'])
    sensitivity = row['sensitivity']
    if abs(sensitivity) > 0: 
        [x, y, z] = spherical_to_cart(rad, lat, lon)
        
        # Define color table (including alpha), which must be uint8 and [0,255]
        color = np.asarray([0,0,0])
        if sensitivity > 0:
            color[0] = (1 - sensitivity / max_sensitivity) # red
            color[2] = 1 # blue
        else:
            color[0] = 1 # red
            color[2] = (1 + sensitivity / max_sensitivity) # blue
        color[1] = (1 - abs(sensitivity) / max_sensitivity) # green
        
        color = color.astype(np.uint8)
        
        
        mlab.points3d(x, y, z, scale_factor=100000, color=tuple(color))
    print(index)
    
mlab.show()