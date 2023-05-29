#include "/home/adrian/PhD/AxiSEM3D/AxiSEM3D-master/SOLVER/src/shared/io/netcdf/NetCDF_Reader.hpp"
#include <iostream>
#include <string.h>
#include <vector>

int main()
{
    const std::string fname = "/home/adrian/PhD/Testing_ground/netcdf_reader/entire_earth/axisem3d_synthetics.nc.rank0";

   try {
        // Create an instance of NetCDF_Reader
        NetCDF_Reader reader;

        // Open the NetCDF file
        reader.open(fname);

        // Check if the file is open
        if (!reader.isOpen()) {
            std::cout << "Failed to open the NetCDF file." << std::endl;
            return 1;
        }

        // Get the id of variable
        const std::string vname = "data_wave__NaG\=5";
        Eigen::Tensor<float, 5, Eigen::RowMajor> data;
        int varid = reader.getVariableID(vname, data, true);
        std::cout << varid << std::endl;

        // Get dimension of variable
        std::vector<numerical::Int> dims;
        reader.getVariableDimensions(varid, dims);
        for (const auto& value : dims) std::cout << value << " " << std::endl;;


        // Read variable
        reader.readTensor(vname, data);

        // Print the data
        int element = 0;
        int coeff = 0;
        int gll_point = 0;
        int channel = 0;
        int time = 0;
        int a = static_cast<int>(data.dimension(4));
        Eigen::array<Eigen::Index, 5> startIndices = {0,0,0,0,1};
        Eigen::array<Eigen::Index, 5> extent = {1,1,1,1,1000};

        Eigen::Tensor<float, 5, Eigen::RowMajor> subset = data.slice(startIndices, extent);

        // Set the format to scientific notation
        std::cout << std::scientific;

        // Print the data
        std::cout << "Subset data:" << std::endl;
        for (int element = 0; element < subset.dimension(0); ++element) {
            for (int coeff = 0; coeff < subset.dimension(1); ++coeff) {
                for (int gll_point = 0; gll_point < subset.dimension(2); ++gll_point) {
                    for (int channel = 0; channel < subset.dimension(3); ++channel) {
                        for (int time = 0; time < subset.dimension(4); time += 10) {
                            std::cout << "Element " << element << ", Coeff " << coeff
                                    << ", GLL Point " << gll_point << ", Channel " << channel
                                    << ", Time " << time << ": " << subset(element, coeff, gll_point, channel, time) << std::endl;
                        }
                    }
                }
            }
        }

        // Reset the format back to default
        std::cout << std::defaultfloat;

        // Close the NetCDF file
        reader.close();
    } catch (const std::exception& e) {
        std::cout << "Exception occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}