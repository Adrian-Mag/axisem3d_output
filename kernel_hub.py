from .source_builder import L2_STF_builder
from AxiSEM3D_Data_Handler.obspy_output import ObspyfiedOutput
from .helper_functions import find_phase_window
from .kernel import L2Kernel

#@@@@@@@@@@@@@
# L2 Kernels @
#@@@@@@@@@@@@@

###################
# CHOICE PARAMETERS
###################

SOURCE_BUILDING = False
KERNEL_COMPUTATION = True

############################
# SOURCE BUILDING PARAMETERS
############################
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

###############################
# KERNEL COMPUTATION PARAMETERS
###############################

path_to_backward = '/disks/data/PhD/CMB/simu1D_element/BACKWARD_DATA'
path_to_inversion_mesh = '/disks/data/PhD/CMB/stations/STA_3D_UNIFORM_STA.txt'
element_output_geometry = [0, 2, 4]

################
# IMPLEMENTATION
################
real_data_obspyobj = ObspyfiedOutput(mseed_file_path=real_data_path)
inventory = real_data_obspyobj.inv
inventory = inventory.select(network=network, 
                                station=station, 
                                location=location)
station_depth = -inventory[0][0].elevation
station_latitude = inventory[0][0].latitude
station_longitude = inventory[0][0].longitude 
catalogue = real_data_obspyobj.cat
event_depth = catalogue[0].origins[0].depth * 1e-3  # must be in km for get_travel_times_geo
event_latitude = catalogue[0].origins[0].latitude
event_longitude = catalogue[0].origins[0].longitude

window_left, window_right = find_phase_window(event_depth, event_latitude, event_longitude, 
                                              station_latitude, station_longitude, T, phase)

if SOURCE_BUILDING is True:
    L2_STF_builder(real_data_path, element_path, station, network, 
                  location, channel_type, window_left, window_right)

if KERNEL_COMPUTATION is True:
    kernel = L2Kernel(element_path, path_to_backward, element_output_geometry,
                        window_left, window_right)
    kernel.evaluate_on_mesh(path_to_inversion_mesh, '/disks/data/PhD/AxiSEM3D-Kernels/KERNELS')