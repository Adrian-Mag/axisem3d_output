import sys
sys.path.append("/disks/data/PhD/AxiSEM3D_Data_Handler")
from element_output import element_output
from obspy import read
import matplotlib.pyplot as plt


################
# LOAD REAL DATA
################

# Path to the mseed file
mseed_file = '/disks/data/PhD/CMB/simu1D_element/FORWARD/output/obspyfied/entire_earth.mseed'

# Load the mseed file
stream = read(mseed_file)

#####################
# LOAD SYNTHETIC DATA
#####################

element_obj = element_output('/disks/data/PhD/CMB/simu1D_element/FORWARD', [0, 2, 4])


#################
# COMPUTE RESIDUE 
#################

# Save residue as STF.txt file ready to be given to AxiSEM3D
