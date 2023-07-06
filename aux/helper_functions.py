from obspy.taup import TauPyModel
import sys
import numpy as np


def find_phase_window(event_depth, event_latitude, event_longitude,
                      station_latitude, station_longitude, T, phase):
    """
    Find the time window for a specific seismic phase arrival.

    Args:
        event_depth (float): Event depth in kilometers.
        event_latitude (float): Event latitude in degrees.
        event_longitude (float): Event longitude in degrees.
        station_latitude (float): Station latitude in degrees.
        station_longitude (float): Station longitude in degrees.
        T (float): Desired window duration in seconds.
        phase (str): Seismic phase name.

    Returns:
        list: [window_start, window_end] representing the time window boundaries.

    Raises:
        ValueError: If no arrivals are found for the specified phase.
    """
    # Specify the model for travel time calculations
    model = TauPyModel(model='iasp91')

    # Compute the travel times
    arrivals = model.get_travel_times_geo(
        source_depth_in_km=event_depth,
        source_latitude_in_deg=event_latitude,
        source_longitude_in_deg=event_longitude,
        receiver_latitude_in_deg=station_latitude,
        receiver_longitude_in_deg=station_longitude,
        phase_list=(phase,))

    # Get the arrival times of the desired phase
    arrival_times = [arrival.time for arrival in arrivals]

    if len(arrival_times) == 0:
        raise ValueError('No arrivals found for the specified phase')

    # Sort the arrival times in ascending order
    arrival_times.sort()

    # Calculate the minimum and maximum arrival times
    min_arrival_time = arrival_times[0]
    max_arrival_time = arrival_times[-1]

    # Calculate the window boundaries
    window_start = min_arrival_time - T / 2
    window_end = max_arrival_time + T / 2

    return [window_start, window_end]


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