from obspy.taup import TauPyModel
import sys
import numpy as np

def find_phase_window(event_depth, event_latitude, event_longitude, 
                      station_latitude, station_longitude, T, phase):
    #####################
    # FIND PHASE LOCATION
    #####################
    # Specify the model for travel time calculations
    model = TauPyModel(model='iasp91')

    # Compute the travel times
    arrivals = model.get_travel_times_geo(source_depth_in_km=event_depth, 
                                        source_latitude_in_deg=event_latitude, 
                                        source_longitude_in_deg=event_longitude, 
                                        receiver_latitude_in_deg=station_latitude, 
                                        receiver_longitude_in_deg=station_longitude, 
                                        phase_list=(phase,))

    # Get the arrival time of the desired phase
    arrival_times = []
    if len(arrivals) == 0:
        raise ValueError('No arrivals found')
    else:
        for arrival in arrivals:
            arrival_times.append(arrival.time)

    # Find if all arrivals can fit in the window
    if max(arrival_times) - min(arrival_times) > T:
        print('Not all arrival_times can fit in the time window')
        option = input('Choose an option: \n 0-Modify window to {} so that they all fit \n '
                    '1-Choose only the first arrival_times that fit in window \n 2-Abort'.format(max(arrival_times) - min(arrival_times)))

        if option == 0:
            window_left = min(arrival_times) - T/2
            window_right = max(arrival_times) + T/2
        elif option == 1:
            window_left = min(arrival_times) - T/2
            window_right = window_left + T/2
        else:
            sys.exit(1)
    else:
        window_left = min(arrival_times) - T/2
        window_right = window_left + T

    return [window_left, window_right]

def window_data(time_array, data_array, t_min, t_max):
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
    try:
        filtered_data_array = data_array[:,mask]
    except:
        filtered_data_array = data_array[mask]
    return filtered_time_array, filtered_data_array