import numpy as np
import matplotlib 
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


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
lon2 = 30        # deg 
rad2 = 6371000  # m

R_min = 3480000 # m
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
path_to_kernel = '/disks/data/PhD/AxiSEM3D-Kernels/KERNELS/sensitivity_rho.txt'
data = np.loadtxt(path_to_kernel, skiprows=1)

# Extract the relevant columns from the data
radius = data[:, 0]
latitude = data[:, 1]
longitude = data[:, 2]
x = radius * np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(longitude))
y = radius * np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(longitude))
z = radius * np.sin(np.deg2rad(latitude))
# Transform into a list of NumPy arrays representing vectors
kernel_mesh = [np.array([x, y, z]) for x, y, z in zip(x, y, z)]

sensitivities = data[:, 3]
sensitivity_interp = []
for point in mesh:
    kernel_mesh_copy = kernel_mesh.copy()
    point = np.array([point[0], point[1], point[2]])
    # find closest point in kernel mesh to our mesh point
    index0 = np.argmin(((kernel_mesh_copy - point)**2).sum(axis=1))
    kernel_mesh_copy[index0] += 1e10
    index1 = np.argmin(((kernel_mesh_copy - point)**2).sum(axis=1))
    kernel_mesh_copy[index1] += 1e10
    index2 = np.argmin(((kernel_mesh_copy - point)**2).sum(axis=1))

    sensitivity_interp.append((sensitivities[index0] + sensitivities[index1] + sensitivities[index2])/3)
#######################
# VISUALIZE SENSITIVITY 
#######################

plt.figure()
for index, point in enumerate(mesh_2d):
    if sensitivity_interp[index] is None:
        print('a')
    plt.scatter(point[0] * np.cos(point[1]),  point[0] * np.sin(point[1]), c = sensitivity_interp[index], vmin=-1e-28, vmax=1e-28)
plt.colorbar()
plt.show()
