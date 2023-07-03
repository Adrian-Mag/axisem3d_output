import numpy as np 
from AxiSEM3D_Data_Handler.element_output import ElementOutput
from AxiSEM3D_Data_Handler.station_output import StationOutput
from AxiSEM3D_Data_Handler.obspy_output import ObspyfiedOutput
from .helper_functions import window_data

import matplotlib.pyplot as plt

class L2Objective_Function:
    def __init__(self, forward_data:ElementOutput, real_data:ObspyfiedOutput,
                 window_left: float, window_right: float):
        self.forward_data_obj = forward_data
        self.real_data_obj = real_data

        # Source data
        self.source_depth = forward_data.source_depth
        self.source_latitude = forward_data.source_lat
        self.source_longitude = forward_data.source_lon

        # Windowing data
        self.window_left = window_left
        self.window_right = window_right


    def evaluate_objective_function(self, network: str, station: str, location: str,
                                plot_residue: bool=True) -> float:
        """
        Evaluates the objective function by computing the integral of the L2 norm of the residue over the windowed time.

        Args:
            network (str): Network identifier.
            station (str): Station identifier.
            location (str): Location identifier.
            plot_residue (bool, optional): Whether to plot the residue. Defaults to True.

        Returns:
            float: The integral of the L2 norm of the residue.

        Raises:
            ValueError: If invalid network, station, or location is provided.
        """

        # Load real data
        stream_real_data = self.real_data_obj.stream.select(station=station)
        real_data_time = stream_real_data[0].times(type="relative")
        dt_real_data = real_data_time[1] - real_data_time[0]

        # Extract station coordinates from inventory
        inventory = self.real_data_obj.inv.select(network=network, station=station, location=location)
        station_depth = -inventory[0][0].elevation
        station_latitude = inventory[0][0].latitude
        station_longitude = inventory[0][0].longitude

        # Load synthetic data
        sta_rad = self.forward_data_obj.Earth_Radius - station_depth
        point = [sta_rad, station_latitude, station_longitude]
        stream_forward_data = self.forward_data_obj.stream(point)
        forward_time = self.forward_data_obj.data_time
        dt_forward = forward_time[1] - forward_time[0]
        channel_type = self.forward_data_obj.coordinate_frame

        # Compute residue
        t_max = min(real_data_time[-1], forward_time[-1])
        t_min = max(real_data_time[0], forward_time[0])
        dt = max(dt_real_data, dt_forward)
        master_time = np.arange(t_min, t_max, dt)

        interpolated_real_data = np.vstack([
            np.interp(master_time, real_data_time, stream_real_data.select(channel='U' + channel)[0].data)
            for channel in channel_type
        ])
        interpolated_forward_data = np.vstack([
            np.interp(master_time, forward_time, stream_forward_data.select(channel='U' + channel)[0].data)
            for channel in channel_type
        ])

        windowed_master_time = master_time
        windowed_real_data = interpolated_real_data
        windowed_forward_data = interpolated_forward_data

        if self.window_left is not None and self.window_right is not None:
            windowed_master_time, windowed_real_data = window_data(master_time, interpolated_real_data, self.window_left, self.window_right)
            _, windowed_forward_data = window_data(master_time, interpolated_forward_data, self.window_left, self.window_right)

        residue = windowed_forward_data - windowed_real_data
        residue_norm = np.linalg.norm(residue, axis=0)
        integral = np.trapz(residue_norm, x=windowed_master_time)

        if plot_residue:
            # Plot windowed_real_data, windowed_forward_data, residue, and L2 norm of residue
            fig, ax = plt.subplots(len(channel_type), 1, figsize=(10, 6))
            fig.suptitle('Windowed Real Data, Windowed Forward Data, and Residue')

            for i, channel in enumerate(channel_type):
                ax[i].plot(windowed_master_time, windowed_real_data[i], label='Windowed Real Data')
                ax[i].plot(windowed_master_time, windowed_forward_data[i], label='Windowed Forward Data')
                ax[i].plot(windowed_master_time, residue[i], label='Residue')
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel('Amplitude')
                ax[i].legend()

            plt.show()

        return 0.5 * integral
