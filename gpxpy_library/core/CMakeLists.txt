cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(gpxpy VERSION 0.1.0)

# Option for GPU support with cuBLAS
option(GPXPY_WITH_CUBLAS "Enable GPU support with cuBLAS" OFF)
# Pass variable to C++ code
add_compile_definitions(GPXPY_WITH_CUBLAS=$<BOOL:${GPXPY_WITH_CUBLAS}>)

if(NOT GPXPY_WITH_CUBLAS)
    # TODO: configuration without GPU support
else()
    # TODO: configuration with GPU support
    # find_package(CUDA REQUIRED)
    # target_link_libraries(gpxpy PRIVATE CUDA::cublas)
endif()

find_package(HPX REQUIRED)
find_package(MKL CONFIG REQUIRED)

# Include dir
include_directories(${HPX_INCLUDE_DIRS})
include_directories(${MKL_INCLUDE_DIRS})

# Source files
set(SOURCE_FILES
    adapter_cublas.cpp
    adapter_mkl.cpp
    gp_algorithms_cuda.cpp
    gp_algorithms_mkl.cpp
    gp_functions.cpp
    gp_gradient_descent.cpp
    gp_uncertainty.cpp
    gpxpy_c.cpp
    hpx_cholesky_gpu.cpp
    tiled_algorithms_cpu.cpp
    tiled_algorithms_gpu.cpp
    utils_c.cpp
)

# Headers (to be installed)
set(HEADER_FILES
    adapter_cublas.hpp
    adapter_mkl.hpp
    gp_algorithms_cpu.hpp
    gp_algorithms_cuda.hpp
    gp_functions.hpp
    gp_gradient_descent.hpp
    gp_uncertainty.hpp
    gpxpy_c.hpp
    tiled_algorithms_cpu.hpp
    tiled_algorithms_gpu.hpp
    utils_c.hpp
)

# Add library
add_library(gpxpy SHARED ${SOURCE_FILES})

# Link HPX libraries
target_link_libraries(gpxpy PRIVATE ${HPX_LIBRARIES} MKL::MKL)

# Include directories
target_include_directories(gpxpy PRIVATE include/ ${HPX_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)

# Set the RPATH to include the MKL library paths
set_target_properties(gpxpy PROPERTIES
    INSTALL_RPATH "${MKL_LIBRARIES}"
    BUILD_WITH_INSTALL_RPATH ON
)

# Set default installation prefix if not specified by user
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    get_filename_component(PARENT_DIR "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)
    set(CMAKE_INSTALL_PREFIX "${PARENT_DIR}/test_cpp/install_cpp" CACHE PATH "Installation Directory" FORCE)
endif()

# Install
install(TARGETS gpxpy DESTINATION lib)

# Create base directory
install(DIRECTORY include/ DESTINATION include)
