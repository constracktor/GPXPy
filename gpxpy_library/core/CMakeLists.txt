cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(gpxpy VERSION 0.1.0)

find_package(HPX REQUIRED)
find_package(MKL CONFIG REQUIRED)

# Include dir
include_directories(${HPX_INCLUDE_DIRS})
include_directories(${MKL_INCLUDE_DIRS})

# Source files
set(SOURCE_FILES
    src/gp_optimizer.cpp
    src/gp_functions.cpp
    src/gp_algorithms_mkl.cpp
    src/gpxpy_c.cpp
    src/adapter_mkl.cpp
    src/tiled_algorithms_cpu.cpp
    src/gp_uncertainty.cpp
    src/utils_c.cpp
)

# Headers (to be installed)
set(HEADER_FILES
    include/gp_optimizer.hpp
    include/gp_functions.hpp
    include/gp_algorithms_cpu.hpp
    include/gpxpy_c.hpp
    include/adapter_mkl.hpp
    include/tiled_algorithms_cpu.hpp
    include/gp_uncertainty.hpp
    include/utils_c.hpp
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
