from AxiSEM3D_Data_Handler.element_output import element_output
from obspy import read
import matplotlib.pyplot as plt
import numpy as np
import os

################
# LOAD REAL DATA
################

# Path to the mseed file
mseed_file = '/disks/data/PhD/CMB/simu3D_CMB/REAL_DATA/output/obspyfied/REAL_DATA.mseed'
data_name = mseed_file.split('/')[-1]
network = 'A'
station = '22'

# Load the mseed file
stream_real_data = read(mseed_file)
real_data_time = stream_real_data[0].times(type="relative")
dt_real_data = real_data_time[1] - real_data_time[0]
stream_real_data = stream_real_data.select(station="22")
print(stream_real_data)
print(real_data_time)

#####################
# LOAD SYNTHETIC DATA
#####################
depth = 0
lat = 0
lon = 30

element_path = '/disks/data/PhD/CMB/simu1D_element/FORWARD_DATA'
name_element = element_path.split('/')[-1]
element_obj = element_output(element_path, [0, 2, 4])
stream_forward = element_obj.stream(depth, lat, lon)
forward_time = element_obj.data_time
dt_forward = forward_time[1] - forward_time[0]

print(stream_forward)
print(forward_time)

#################
# COMPUTE RESIDUE 
#################

# Find the master time (minmax/maxmin)
t_max = min(real_data_time[-1], forward_time[-1])
t_min = max(real_data_time[0], forward_time[0])
dt = max(dt_real_data, dt_forward)
master_time = np.arange(t_min, t_max, dt)

# Project both arrays on the master time
interpolated_real_data = []
interpolated_forward_data = []

for i in range(3):
    interpolated_real_data.append(np.interp(master_time, real_data_time, stream_real_data[i].data))
    interpolated_forward_data.append(np.interp(master_time, forward_time, stream_forward[i].data))

# Subtract
residue = np.array(interpolated_forward_data) - np.array(interpolated_real_data)

# Save residue as STF.txt file ready to be given to AxiSEM3D
directory = 'CMB/STFS/' + name_element + '_' + data_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)
    print("Directory created:", directory)
    for index, channel in enumerate(['R', 'T', 'Z']):
        # Save results to a text file
        filename = directory + channel + '.txt'
        # Combine time and data arrays column-wise
        combined_data = np.column_stack((master_time, residue[index, :]))
        print(residue[index, :])
        # Save the combined data to a text file
        np.savetxt(filename, combined_data, fmt='%.16f', delimiter='\t')
else:
    print("Directory already exists:", directory)

