import sys
sys.path.append('/home/adrian/PhD/AxiSEM3D/Output_Handlers')
from AxiSEM3D_Data_Handler.element_output import element_output
from .helper_functions import window_data

import numpy as np
import pandas as pd
from scipy import integrate 
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import os 

class L2Kernel():

    def __init__(self, forward_data_path, backward_data_path, element_output_geometry,
                 window_left, window_right):
        self.forward_data = element_output(forward_data_path, element_output_geometry)
        self.backward_data = element_output(backward_data_path, element_output_geometry)
        self.window_left = window_left
        self.window_right = window_right

        # get the forward and backward time 
        fw_time = self.forward_data.data_time
        self.fw_dt = fw_time[1] - fw_time[0]
        bw_time = self.backward_data.data_time
        self.bw_dt = bw_time[1] - bw_time[0]
        # Find the master time (minmax/maxmin)
        t_max = min(fw_time[-1], bw_time[-1])
        t_min = max(fw_time[0], bw_time[0])
        dt = max(self.fw_dt, self.bw_dt)
        self.master_time = np.arange(t_min, t_max, dt)

        self.fw_time = fw_time[0:-1] + self.fw_dt
        self.bw_time = bw_time[0:-1] + self.bw_dt

    def evaluate_on_mesh(self, path_to_inversion_mesh, sensitivity_out_path):
        # Earth's radius in m
        R = 6371000

        # import inversion mesh
        points = pd.read_csv(path_to_inversion_mesh, sep=" ")

        # initialize sensitivity 
        sensitivity = {'radius': [], 'latitude': [], 'longitude': [], 'sensitivity': []}

        for _, row in points.iterrows():
            latitude = row['latitude']
            longitude = row['longitude']
            radius = R - row['depth']

            # integrate over time the dot product
            sensitivity['radius'].append(radius)
            sensitivity['latitude'].append(latitude)
            sensitivity['longitude'].append(longitude)
            sensitivity['sensitivity'].append(self.evaluate(radius, latitude, longitude))

    
        sensitivity_df = pd.DataFrame(sensitivity)
        sensitivity_df.to_csv(sensitivity_out_path + '/' + 'sensitivity_rho.txt', sep=' ', index=False)

    def evaluate(self, radius, latitude, longitude):
        point = [radius, latitude, longitude]

        # get forwards and backward waveforms at this point
        forward_waveform = np.nan_to_num(self.forward_data.load_data_at_point(point))
        backward_waveform = np.nan_to_num(self.backward_data.load_data_at_point(point))

        # Compute time derivatives wrt to time
        dfwdt = np.diff(forward_waveform) / self.fw_dt
        dbwdt = np.diff(backward_waveform) / self.bw_dt

        # Project both arrays on the master time
        interpolated_dfwdt = []
        interpolated_dbwdt = []

        for i in range(3):
            interpolated_dfwdt.append(np.interp(self.master_time, self.fw_time, dfwdt[i]))
            interpolated_dbwdt.append(np.interp(self.master_time, self.bw_time, dbwdt[i]))
        interpolated_dfwdt = np.array(interpolated_dfwdt)
        interpolated_dbwdt = np.array(interpolated_dbwdt)

        # flip backward waveform in the time axis
        reversed_dbwdt = np.flip(interpolated_dbwdt, axis=1)

        # Window the data
        windowed_time, windowed_interp_dfwdt = window_data(self.master_time, interpolated_dfwdt, self.window_left, self.window_right)
        _, windowed_reversed_dbwdt = window_data(self.master_time, reversed_dbwdt, self.window_left, self.window_right)

        # make dot product 
        fw_bw = (windowed_interp_dfwdt * windowed_reversed_dbwdt).sum(axis=0)
        
        sensitivity = integrate.simpson(fw_bw, dx=(windowed_time[1] - windowed_time[0]))
        return sensitivity