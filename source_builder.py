from AxiSEM3D_Data_Handler.element_output import element_output
from obspy import read, read_inventory
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def eliminate_data_outside_range(time_array, data_array, t_min, t_max):
    """
    Eliminate time values and corresponding data outside a specified range.

    Args:
        time_array (numpy.ndarray): Array of time values.
        data_array (numpy.ndarray): Array of corresponding data values.
        t_min (float): Minimum time value to keep.
        t_max (float): Maximum time value to keep.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the filtered time array and corresponding data array.
    """
    mask = (time_array >= t_min) & (time_array <= t_max)
    filtered_time_array = time_array[mask]
    filtered_data_array = data_array[mask]
    return filtered_time_array, filtered_data_array

def save_STF(directory, master_time, STF, channel_type):
    for index, channel in enumerate(channel_type):
        # Save results to a text file
        filename = directory + channel + '.txt'
        # Combine time and data arrays column-wise
        combined_data = np.column_stack((master_time, STF[channel]))
        print(STF[channel])
        # Save the combined data to a text file
        np.savetxt(filename, combined_data, fmt='%.16f', delimiter='\t')

def L2_source_builder(real_data_path, forward_data_path, station, network, location, channel_type, window_left, window_right):
    ################
    # LOAD REAL DATA
    ################
    # Load the mseed file
    try: 
        stream_real_data = read(real_data_path)
    except FileNotFoundError:
        raise FileNotFoundError('File {} not found'.format(real_data_path))
        sys.exit(1)

    # get name to be used for naming files
    real_data_name = real_data_path.split('/')[-1]
    # get real data time and time step
    real_data_time = stream_real_data[0].times(type="relative")
    dt_real_data = real_data_time[1] - real_data_time[0]
    # Select the specific station data
    stream_real_data = stream_real_data.select(station=station)

    # Find inventories in the output directory
    real_data_dir = os.path.dirname(real_data_path)
    real_data_dir = os.path.join(real_data_dir, "")
    inventories = search_files(real_data_dir, 'inv.xml')
    if len(inventories) > 1:
        print('The following inventories have been found: ')
        for index, element in enumerate(inventories):
            print(f'{index}: {element}')
        inventory_index = int(input('Select which one to be used: '))
        inventory = inventories[inventory_index]
    elif len(inventories) == 0:
        raise FileNotFoundError('No inventories have been found!')
        sys.exit(1)
    else:
        inventory = inventories[0]

    # Extract the station coordinates from that inventory
    inventory = read_inventory(inventory)
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
        _, windowed_real_data = eliminate_data_outside_range(master_time, interpolated_real_data, window_left, window_right)
        windowed_master_time, windowed_forward_data = eliminate_data_outside_range(master_time, interpolated_forward_data, window_left, window_right)

        residue[channel] = np.array(windowed_forward_data) - np.array(windowed_real_data)

    # Subtract
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