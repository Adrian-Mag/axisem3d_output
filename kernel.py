import sys
sys.path.append('/home/adrian/PhD/AxiSEM3D/Output_Handlers')
from AxiSEM3D_Data_Handler.element_output import element_output
import numpy as np
import pandas as pd
from scipy import integrate 
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import os 


print(os.getpid())

# Get the forward and backward data 
path_to_forward = '/disks/data/PhD/CMB/simu1D_element/FORWARD_DATA'
path_to_backward = '/disks/data/PhD/CMB/simu1D_element/BACKWARD_DATA'

# Create element objects
forward_data = element_output(path_to_forward, [0,2,4])
backward_data = element_output(path_to_backward, [0,2,4])

# import inversion mesh
points = pd.read_csv('/disks/data/PhD/CMB/stations/STA_3D_UNIFORM_STA.txt', sep=" ")

# Earth's radius in m
R = 6371000

# get the forward and backward time 
fw_time = forward_data.data_time
fw_dt = fw_time[1] - fw_time[0]
bw_time = backward_data.data_time
bw_dt = bw_time[1] - bw_time[0]
# Find the master time (minmax/maxmin)
t_max = min(fw_time[-1], bw_time[-1])
t_min = max(fw_time[0], bw_time[0])
dt = max(fw_dt, bw_dt)
master_time = np.arange(t_min, t_max, dt)

# initialize sensitivity 
sensitivity = {'radius': [], 'latitude': [], 'longitude': [], 'sensitivity': []}

for index, row in points.iterrows():
    # get coordinates in geographical frame and spherical coords
    lat = row['latitude']
    lon = row['longitude']
    rad = R - row['depth']
    point = [rad, lat, lon]
    
    # get forwards and backward waveforms at this point
    forward_waveform = np.nan_to_num(forward_data.load_data_at_point(point))
    backward_waveform = np.nan_to_num(backward_data.load_data_at_point(point))

    # Compute time derivatives wrt to time
    dfwdt = np.diff(forward_waveform)/fw_dt
    dbwdt = np.diff(backward_waveform)/bw_dt
    
    # Project both arrays on the master time
    interpolated_dfwdt = []
    interpolated_dbwdt = []

    for i in range(3):
        interpolated_dfwdt.append(np.interp(master_time, fw_time, forward_waveform[i]))
        interpolated_dbwdt.append(np.interp(master_time, bw_time, backward_waveform[i]))

    interpolated_dfwdt = np.array(interpolated_dfwdt)
    interpolated_dbwdt = np.array(interpolated_dbwdt)

    # flip backward waveform in the time axis
    reversed_dbwdt = np.flip(interpolated_dbwdt, axis=1)

    # make dot product 
    fw_bw = (interpolated_dfwdt * reversed_dbwdt).sum(axis=0)
    
    # integrate over time the dot product
    sensitivity['radius'].append(rad)
    sensitivity['latitude'].append(lat)
    sensitivity['longitude'].append(lon)
    sensitivity['sensitivity'].append(integrate.simpson(fw_bw, dx=fw_dt))
    
sensitivity_df = pd.DataFrame(sensitivity)
sensitivity_df.to_csv('sensitivity_rho.txt', sep=' ', index=False)