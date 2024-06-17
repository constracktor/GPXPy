cmake_minimum_required(VERSION 3.16)

project(test_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find HPX
find_package(HPX REQUIRED)
find_package(MKL CONFIG REQUIRED)

# Include directories
include_directories(${HPX_INCLUDE_DIR})
include_directories(${MKL_INCLUDE_DIRS})
include_directories(/home/maksim/simtech/thesis/GPPPy_hpx/hpx_py11/cmake_cpp/include) # Add any additional include directories if needed

# Set the output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)

# Find automobile library
find_library(AUTOMOBILE_LIB automobile HINTS /home/maksim/simtech/thesis/GPPPy_hpx/hpx_py11/cmake_cpp/lib/)

# Add the executable
add_executable(test_cpp src/test.cpp)

# Link the libraries
target_link_libraries(test_cpp PUBLIC 
                      ${AUTOMOBILE_LIB}
                      HPX::hpx 
                      HPX::wrap_main 
                      HPX::iostreams_component
                      MKL::mkl_intel_lp64
                      MKL::mkl_core
                      MKL::MKL
                      MKL::mkl_sequential
                      )
