#include "/home/adrian/PhD/AxiSEM3D/AxiSEM3D-master/SOLVER/src/shared/io/netcdf/NetCDF_Reader.hpp"
#include <iostream>
#include <string.h>
#include <vector>
#include <filesystem>
#include <Eigen/Core>

void link_location_list(const std::string& dirPath)
{
    // Location lists should be 3D tensors
    Eigen::Tensor<double, 3, Eigen::RowMajor> concatLocationList;

    // Flag for telling 
    bool FIRST_FILE_IS_PROCESSED = true;

    // Iterate over files in the directory
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            const std::string filePath = entry.path().string();
            
            try {
                // Process the file
                std::cout << "Processing file: " << filePath << std::endl;

                // Create an instance of NetCDF_Reader
                NetCDF_Reader reader;

                // Open the NetCDF file
                reader.open(filePath);

                // Check if the file is open
                if (!reader.isOpen()) {
                    std::cout << "Failed to open the NetCDF file." << std::endl;
                    continue;
                }

                // Get the id of variable list_element_coords
                const std::string vname = "list_element_coords";
                Eigen::Tensor<double, 3, Eigen::RowMajor> locationList;
                reader.readTensor(vname, locationList);


                if (FIRST_FILE_IS_PROCESSED) {
                    // Initialize the concatenated list 
                    concatLocationList = locationList;
                    std::cout << locationList.dimensions() << std::endl;
                    FIRST_FILE_IS_PROCESSED = false;
                } else {
                    // Concatenate locationList to concatLocationList
                    std::cout << concatLocationList.dimensions() << " " << locationList.dimensions() << std::endl;
                    Eigen::TensorConcatenationOp<int, Eigen::Tensor<double, 3, Eigen::RowMajor>, Eigen::Tensor<double, 3, Eigen::RowMajor>> concatOp(concatLocationList, locationList, 0);
                    concatLocationList = concatOp.eval();
                    std::cout << locationList.dimensions() << std::endl;
                }
                // Close the NetCDF file
                reader.close();

            } catch (const std::exception& e) {
                std::cout << "Exception occurred: " << e.what() << std::endl;
            }
        }
    }
    std::cout << concatLocationList.dimensions() << std::endl;
}


int main()
{
    const std::string dirPath = "/home/adrian/PhD/Testing_ground/netcdf_reader/entire_earth";

   try {
        link_location_list(dirPath);
    } catch (const std::exception& e) {
        std::cout << "Exception occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
