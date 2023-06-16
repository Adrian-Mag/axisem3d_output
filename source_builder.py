from AxiSEM3D_Data_Handler.element_output import element_output
from AxiSEM3D_Data_Handler.obspy_output import ObspyfiedOutput
from .helper_functions import find_phase_window, window_data

import matplotlib.pyplot as plt
import numpy as np
import os




def save_STF(directory, master_time, STF, channel_type):
    for index, channel in enumerate(channel_type):
        # Save results to a text file
        filename = directory + channel + '.txt'
        # Combine time and data arrays column-wise
        combined_data = np.column_stack((master_time, STF[channel]))
        print(STF[channel])
        # Save the combined data to a text file
        np.savetxt(filename, combined_data, fmt='%.16f', delimiter='\t')


def L2_STF_builder(real_data_path, forward_data_path, station, network, location, channel_type, window_left, window_right):
    ################
    # LOAD REAL DATA
    ################
    real_data_obspyobj = ObspyfiedOutput(mseed_file_path = real_data_path)

    stream_real_data  = real_data_obspyobj.stream
    real_data_name = real_data_obspyobj.mseed_file_name
    # get real data time and time step
    real_data_time = stream_real_data[0].times(type="relative")
    dt_real_data = real_data_time[1] - real_data_time[0]
    # Select the specific station data
    stream_real_data = stream_real_data.select(station=station)

    inventory = real_data_obspyobj.inv
    # Extract the station coordinates from that inventory
    inventory = inventory.select(network=network, 
                                station=station, 
                                location=location)
    station_depth = -inventory[0][0].elevation
    station_latitude = inventory[0][0].latitude
    station_longitude = inventory[0][0].longitude    

    #####################
    # LOAD SYNTHETIC DATA
    #####################
    # Get the synthetic data
    element_data_name = forward_data_path.split('/')[-1].split('.')[0]
    element_obj = element_output(forward_data_path, [0, 2, 4])
    stream_forward_data = element_obj.stream(station_depth, 
                                            station_latitude, 
                                            station_longitude)
    forward_time = element_obj.data_time
    dt_forward = forward_time[1] - forward_time[0]

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

    residue = {}
    for _, channel in enumerate(channel_type):
        interpolated_real_data = np.interp(master_time, 
                                        real_data_time, 
                                        stream_real_data.select(channel='LX' + channel)[0].data)
        interpolated_forward_data = np.interp(master_time, 
                                            forward_time, 
                                            stream_forward_data.select(channel='LX' + channel)[0].data)
        _, windowed_real_data = window_data(master_time, interpolated_real_data, window_left, window_right)
        windowed_master_time, windowed_forward_data = window_data(master_time, interpolated_forward_data, window_left, window_right)

        residue[channel] = np.array(windowed_forward_data) - np.array(windowed_real_data)
    STF = residue

    ##########
    # PLOT STF
    ##########
    fig, axs = plt.subplots(3, 1)

    for index, channel in enumerate(channel_type):
        axs[index].plot(windowed_master_time, STF[channel])
        axs[index].text(1.05, 0.5, channel, transform=axs[index].transAxes)
    plt.show()

    ###############
    # SAVE STF FILE
    ###############
    # Save residue as STF.txt file ready to be given to AxiSEM3D
    directory = 'CMB/STFS/' + element_data_name + '_' + real_data_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory created:", directory)
        save_STF(directory, windowed_master_time, STF, channel_type)
        
    else:
        print("Directory already exists:", directory)
        ans = input('Overwrite the existing data [y/n]: ')
        if ans == 'y':
            save_STF(directory, windowed_master_time, STF, channel_type)