import numpy as np
import matplotlib 
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy import integrate
import sys
sys.path.append('/home/adrian/PhD/AxiSEM3D/Output_Handlers')
from element_output import element_output
import pandas as pd


def sph2cart(rad, lat, lon):
    x = rad * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))    
    y = rad * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = rad * np.sin(np.deg2rad(lat))
    
    return np.asarray([x, y, z])


def cart2sph(x, y, z):
    rad = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arctan( z / np.sqrt( x**2 + y**2 ) )
    lon = np.arctan2(y, x)
    
    return rad, lat, lon


# Parameters of the slice (by two points)
lat1 = 0        # deg 
lon1 = 0        # deg 
rad1 = 6371000  # m

lat2 = 0        # deg 
lon2 = 20        # deg 
rad2 = 6371000  # m

R_min = 4000000 # m
R_max = 6371000 # m
N_R = 10
N_theta_max = 100

# Get the radii where we will form the mesh
R = np.linspace(R_min, R_max, N_R)

# Transform to cartesian
point1 = sph2cart(rad1, lat1, lon1)
point2 = sph2cart(rad2, lat2, lon2)

# Do Gram-Schmidt orthogonalization to form slice basis
base1 = point1 / np.linalg.norm(point1)
base2 = point2 - np.dot(point2, base1) * base1
base2 /= np.linalg.norm(base2)

# Generate the mesh 
mesh = []
mesh_2d = []
for r in R:
    N_theta = int(N_theta_max * r/R_max)
    THETA = np.linspace(-np.pi / 2, np.pi / 2, N_theta)
    
    for theta in THETA:
        mesh.append(
            r * ( np.cos(theta) * base1 + np.sin(theta) * base2 )
        )

        mesh_2d.append([r, theta])


#############################
# COMPUTE SENSITIVITY ON MESH
#############################
# Get the forward and backward data 
path_to_forward = '/home/adrian/PhD/AxiSEM3D/CMB/simu1D_element/FORWARD'
path_to_backward = '/home/adrian/PhD/AxiSEM3D/CMB/simu1D_element/ADJOINT'

# Create element objects
forward_data = element_output(path_to_forward, [0,2,4])
backward_data = element_output(path_to_backward, [0,2,4])

# Earth's radius in m
R = 6371000

# get the forward time 
fw_time = forward_data.data_time
fw_dt = fw_time[1] - fw_time[0]
bw_time = backward_data.data_time
bw_dt = bw_time[1] - bw_time[0]

# initialize sensitivity 
sensitivity = {'radius': [], 'latitude': [], 'longitude': [], 'sensitivity': []}

for point in mesh:
    rad, lat, lon = cart2sph(point[0], point[1], point[2])
    lat *= 180 / np.pi
    lon *= 180 / np.pi
    
    point = [rad, lat, lon]

    # get forwards and backward waveforms at this point
    forward_waveform = np.nan_to_num(forward_data.load_data_at_point(point))
    backward_waveform = np.nan_to_num(backward_data.load_data_at_point(point))

    # Compute time derivatives wrt to time
    dfwdt = np.diff(forward_waveform)/fw_dt
    dbwdt = np.diff(backward_waveform)/bw_dt
    
    # flip backward waveform in the time axis
    reversed_dbwdt = np.flip(dbwdt, axis=1)

    # make dot product 
    fw_bw = (dfwdt * reversed_dbwdt).sum(axis=0)
    
    # integrate over time the dot product
    sensitivity['radius'].append(rad)
    sensitivity['latitude'].append(lat)
    sensitivity['longitude'].append(lon)
    sensitivity['sensitivity'].append(integrate.simpson(fw_bw, dx=fw_dt))

#######################
# VISUALIZE SENSITIVITY 
#######################

plt.figure()
for index, point in enumerate(mesh_2d):
    plt.scatter(point[0] * np.cos(point[1]),  point[0] * np.sin(point[1]), c = np.log10(abs(sensitivity['sensitivity'][index])), vmin=-40, vmax=-25)
plt.colorbar()
plt.show()
