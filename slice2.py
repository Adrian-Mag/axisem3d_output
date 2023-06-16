import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Read the data from the file
data_file = "temperature_data.txt"  # Replace with the actual file name
data = np.loadtxt(data_file, skiprows=1)

# Extract the relevant columns from the data
radius = data[:, 0]
latitude = data[:, 1]
longitude = data[:, 2]
temperature = data[:, 3]

# Define the latitude and longitude values
lat_values = np.unique(latitude)
lon_values = np.unique(longitude)

# Define the dimensions of the grid
num_lat = len(lat_values)
num_lon = len(lon_values)

# Reshape the data into a 2D grid
temperature_grid = temperature.reshape((num_lat, num_lon))

# Create a meshgrid for plotting
lon_mesh, lat_mesh = np.meshgrid(lon_values, lat_values)

# Create a Basemap instance for the desired map projection
m = Basemap(projection='ortho', lat_0=0, lon_0=0)

# Convert lat/lon to x/y coordinates on the map
x, y = m(lon_mesh, lat_mesh)

# Plot the temperature slice
plt.figure(figsize=(8, 8))
m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='white', lake_color='lightblue')
m.contourf(x, y, temperature_grid, cmap='hot_r', alpha=0.7)
m.colorbar(location='bottom', label='Temperature')

# Draw parallels and meridians
m.drawparallels(np.arange(-90, 91, 30), labels=[True, False, False, False])
m.drawmeridians(np.arange(-180, 181, 60), labels=[False, False, False, True])

# Set plot title
plt.title('Temperature Distribution')

# Show the plot
plt.show()
