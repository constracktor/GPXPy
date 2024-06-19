cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(gaussian_process VERSION 0.1.0)

find_package(HPX REQUIRED)
find_package(MKL CONFIG REQUIRED)

# Include dir
include_directories(${HPX_INCLUDE_DIRS})
include_directories(${MKL_INCLUDE_DIRS})

# Source files
set(SOURCE_FILES
    src/gpppy_c.cpp
    src/utils_c.cpp
    src/gp_functions.cpp
)

# Headers (to be installed)
set(HEADER_FILES
    include/gp_headers/gpppy_c.hpp
    include/gp_headers/utils_c.hpp
    include/gp_headers/gp_functions.hpp
    include/gp_headers/tiled_algorithms_cpu.hpp
    include/gp_headers/gp_helper_functions.hpp
    include/gp_headers/gp_functions_grad.hpp
    include/gp_headers/mkl_adapter.hpp
    include/gp_headers/uncertainty.hpp
)

# Add library
add_library(gaussian_process SHARED ${SOURCE_FILES})

# Link HPX libraries
target_link_libraries(gaussian_process PRIVATE ${HPX_LIBRARIES} MKL::MKL)

# Include directories
target_include_directories(gaussian_process PRIVATE include/gp_headers ${HPX_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)

# Set the RPATH to include the MKL library paths
set_target_properties(gaussian_process PROPERTIES
    INSTALL_RPATH "${MKL_LIBRARIES}"
    BUILD_WITH_INSTALL_RPATH ON
)

# Set default installation prefix if not specified by user
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  get_filename_component(PARENT_DIR "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)
  set(CMAKE_INSTALL_PREFIX "${PARENT_DIR}/cmake_cpp" CACHE PATH "Installation Directory" FORCE)
endif()

# Install
install(TARGETS gaussian_process DESTINATION lib)

# Install the headers
install(FILES include/gaussian_process DESTINATION include)

# Create base directory
install(DIRECTORY include/gp_headers DESTINATION include)