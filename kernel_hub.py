from obspy.taup import TauPyModel
import sys
import fnmatch
from obspy import read_events
import os 

from .source_builder import L2_source_builder

#@@@@@@@@@@@@@
# L2 Kernels @
#@@@@@@@@@@@@@

#################
# SOURCE BUILDING 
#################

# Path to the mseed file of real data
real_data_path = '/disks/data/PhD/CMB/simu3D_CMB/REAL_DATA/output/obspyfied/REAL_DATA.mseed'
# Path to element output of synthetic data
element_path = '/disks/data/PhD/CMB/simu1D_element/FORWARD_DATA'

# location info
network = 'A'
station = '22'
location = '*'
channel_type = 'RTZ'

# Window size [seconds]
T = 40
# Phase
phase = 'PcP'

L2_source_builder(real_data_path, element_path, station, network, 
                  location, channel_type, phase, T)

####################
# KERNEL COMPUTATION
####################

path_to_backward = '/disks/data/PhD/CMB/simu1D_element/BACKWARD_DATA'
path_to_inversion_mesh = '/disks/data/PhD/CMB/stations/STA_3D_UNIFORM_STA.txt'

element_output_geometry = [0, 2, 4]

################
# IMPLEMENTATION
################

real_data_dir = os.path.dirname(real_data_path)
real_data_dir = os.path.join(real_data_dir, "")

def search_files(directory, keyword, include_subdirectories=True):
    """
    Search for files containing a specific keyword in a directory.

    Args:
        directory (str): The directory to search in.
        keyword (str): The specific keyword to search for in file names.
        include_subdirectories (bool, optional): Determines whether to include subdirectories in the search.
            Defaults to True.

    Returns:
        list: A list of file paths that contain the specified keyword.
    """
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        if not include_subdirectories and root != directory:
            break
        for filename in filenames:
            if fnmatch.fnmatch(filename, '*' + keyword + '*'):
                matches.append(os.path.join(root, filename))
    return matches


def get_event_data(real_data_dir):
    # Find catalogues of real data
    catalogues = search_files(real_data_dir, 'cat.xml')
    if len(catalogues) > 1:
        print('The following catalogues have been found: ')
        for index, element in enumerate(catalogues):
            print(f'{index}: {element}')
        catalogue_index = int(input('Select which one to be used: '))
        catalogue = catalogues[catalogue_index]
    elif len(catalogues) == 0:
        raise FileNotFoundError('No catalogues have been found!')
        sys.exit(1)
    else:
        catalogue = catalogues[0]

    # Extract event information from that catalogue
    try:
        catalogue = read_events(catalogue)
    except FileNotFoundError:
        raise FileNotFoundError('The file {} was not found.'.format(catalogue))

    if len(catalogue) > 1:
        print('The simulation is not based on a single point source. '
            'This code can not cope with that yet.')
        sys.exit(1)
    elif len(catalogue) == 1:
        event_depth = catalogue[0].origins[0].depth * 1e-3  # must be in km for get_travel_times_geo
        event_latitude = catalogue[0].origins[0].latitude
        event_longitude = catalogue[0].origins[0].longitude
    else:
        raise ValueError('No events were found in the catalogue!')
        sys.exit(1)
    
    return [event_depth, event_latitude, event_longitude]


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
        window_right = window_left + T/2

    return [window_left, window_right]