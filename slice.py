import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def kernel_interpolator(slice_point, kernel_linear_mesh, sensitivities, R_min, R_max):
    if np.linalg.norm(slice_point) > R_min and np.linalg.norm(slice_point) < R_max: 
        # find closest slice_point in kernel mesh to our mesh slice_point
        index0 = np.argmin(((kernel_linear_mesh - slice_point)**2).sum(axis=1))
        sensitivity_interp = sensitivities[index0]
    else: 
        sensitivity_interp = np.nan
    
    return sensitivity_interp


############
# PARAMETERS
############

# Path to sensitivity kernel
path_to_kernel = '/home/adrian/PhD/AxiSEM3D/AxiSEM3D_Kernels/sensitivity_rho.txt'

# Parameters of the slice (by two points)
lat1 = 0        # deg 
lon1 = 0        # deg 
rad1 = 6371000  # m

lat2 = 0        # deg 
lon2 = 30       # deg 
rad2 = 6371000  # m

R_min = 3480000 # m
R_max = 6371000 # m
N_dim1 = 30
N_dim2 = 30

################
# IMPLEMENTATION
################

# Form vectors for the two points (Earth frame)
point1 = sph2cart(rad1, lat1, lon1)
point2 = sph2cart(rad2, lat2, lon2)

# Do Gram-Schmidt orthogonalization to form slice basis (Earth frame)
base1 = point1 / np.linalg.norm(point1)
base2 = point2 - np.dot(point2, base1) * base1
base2 /= np.linalg.norm(base2)

# Generate inplane slice mesh (Slice frame)
inplane_dim1 = np.linspace(-R_max, R_max, N_dim1)
inplane_dim2 = np.linspace(-R_max, R_max, N_dim2)
inplane_DIM1, inplane_DIM2 = np.meshgrid(inplane_dim1, inplane_dim2)

# Initialize sensitivity values on the slice (Slice frame)
inplane_sensitivity = np.zeros((N_dim1, N_dim2))

# Get the sensitivity data (Earth frame)
data = np.loadtxt(path_to_kernel, skiprows=1)

# Extract the relevant columns from the data (Earth frame)
radius = data[:, 0]
latitude = data[:, 1]
longitude = data[:, 2]
x = radius * np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(longitude))
y = radius * np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(longitude))
z = radius * np.sin(np.deg2rad(latitude))
sensitivities = data[:, 3]

# Transform into a list of NumPy arrays representing vectors (Earth frame)
kernel_linear_mesh = [np.array([x, y, z]) for x, y, z in zip(x, y, z)]

# Compute sensitivity for each inplane mesh point
with tqdm(total=N_dim1*N_dim2) as pbar:
    for index1 in range(N_dim1):
        for index2 in range(N_dim2):
            slice_point = inplane_dim1[index1] * base1 + inplane_dim2[index2] * base2  # Slice frame -> Earth frame
            inplane_sensitivity[index1, index2] = kernel_interpolator(slice_point, kernel_linear_mesh.copy(), sensitivities, R_min, R_max)
            pbar.update(1)

#######################
# VISUALIZE SENSITIVITY 
#######################

plt.figure()
plt.contourf(inplane_DIM1, inplane_DIM2, inplane_sensitivity)
plt.scatter()
plt.colorbar()
plt.show()
