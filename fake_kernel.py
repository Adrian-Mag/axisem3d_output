import numpy as np
import pandas as pd


def func(rad, lat, lon):
    return rad

R = 6371000
R_min = 3400000
N = 20
X = np.linspace(-R,R,N)
Y = np.linspace(-R,R,N)
Z = np.linspace(-R,R,N)

sensitivity = {'radius': [], 'latitude': [], 'longitude': [], 'sensitivity': []}

for x in X:
    for y in Y:
        for z in Z:
            # integrate over time the dot product
            rad = np.sqrt(x**2 + y**2 + z**2)
            lat = np.rad2deg(np.arctan( z / np.sqrt( x**2 + y**2 ) ))
            lon = np.rad2deg(np.arctan2(y, x))
            if rad > R_min and rad < R:
                sensitivity['radius'].append(rad)
                sensitivity['latitude'].append(lat)
                sensitivity['longitude'].append(lon)
                sensitivity['sensitivity'].append(func(rad, lat, lon))

sensitivity_df = pd.DataFrame(sensitivity)
sensitivity_df.to_csv('data.txt', sep=' ', index=False)
