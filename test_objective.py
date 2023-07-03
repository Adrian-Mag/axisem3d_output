from .objective_function import L2Objective_Function
from AxiSEM3D_Data_Handler.element_output import ElementOutput
from AxiSEM3D_Data_Handler.station_output import StationOutput
from AxiSEM3D_Data_Handler.obspy_output import ObspyfiedOutput


fwd = ElementOutput('/disks/data/PhD/CMB/simu1D_element/FORWARD_DATA/output/elements/entire_earth')
real = ObspyfiedOutput('/disks/data/PhD/CMB/simu3D_CMB/REAL_DATA/output/stations/Station_grid/obspyfied')
obj = L2Objective_Function(forward_data=fwd, real_data=real,
                           window_left=None, window_right=None)
fwd.stream([6371000, 0, 30], ['U']).plot()
real.stream.select(station='22').plot()
obj.evaluate_objective_function(network='A', station='22', location='*')
