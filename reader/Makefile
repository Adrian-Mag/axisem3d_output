# Compiler options
CXX = g++
CXXFLAGS = -std=c++17 -lboost_filesystem -Wall -g

# Directories
SRC_DIR = /home/adrian/PhD/AxiSEM3D/AxiSEM3D_Kernels/reader
INC_DIRS = /home/adrian/PhD/AxiSEM3D/AxiSEM3D-master/SOLVER/src/shared \
			/home/adrian/PhD/AxiSEM3D/AxiSEM3D-master/SOLVER/src/shared/eigen \
			 /home/adrian/anaconda3/envs/axisem3d/include/eigen3 \
			 /home/adrian/PhD/AxiSEM3D/AxiSEM3D-master/SOLVER/src/shared/utilities
BUILD_DIR = build
BIN_DIR = bin

# Source file and header file lists
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
INCS = $(foreach dir, $(INC_DIRS), -I$(dir))

# Additional source file
ADDITIONAL_SRC = /home/adrian/PhD/AxiSEM3D/AxiSEM3D-master/SOLVER/src/shared/io/netcdf/NetCDF_Reader.cpp
ADDITIONAL_OBJ = $(BUILD_DIR)/NetCDF_Reader.o

# Target executable
TARGET = $(BIN_DIR)/program

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS) $(ADDITIONAL_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OBJS) $(ADDITIONAL_OBJ) -o $(TARGET) -lnetcdf

# Rule to compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@

# Rule to compile the additional source file
$(ADDITIONAL_OBJ): $(ADDITIONAL_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@

# Clean up
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Phony target to avoid conflicts with file names
.PHONY: all clean
