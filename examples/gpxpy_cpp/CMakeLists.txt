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

# Determine the path to the include directory
get_filename_component(INCLUDE_DIR "install_cpp/include" ABSOLUTE)
# Add the include directory
include_directories(${INCLUDE_DIR})

# Set the output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Determine the path to the library directory relative to the current
# CMakeLists.txt file
get_filename_component(LIB_DIR "install_cpp/lib" ABSOLUTE)
# Find the Gaussian Process library
find_library(GPXPY_LIB gpxpy HINTS ${LIB_DIR} REQUIRED)
message(STATUS "GPXPy library found: ${GPXPY_LIB}")

# Add the executable
add_executable(gpxpy_cpp src/execute.cpp)
# Link the libraries
target_link_libraries(
  gpxpy_cpp
  PUBLIC "${GPXPY_LIB}"
         HPX::hpx
         HPX::wrap_main
         HPX::iostreams_component
         MKL::mkl_intel_lp64
         MKL::mkl_core
         MKL::MKL
         MKL::mkl_sequential)
