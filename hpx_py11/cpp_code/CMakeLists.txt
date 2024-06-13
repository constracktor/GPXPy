cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(automobile VERSION 0.1.0)

find_package(HPX REQUIRED)
# cmake .. -DCMAKE_PREFIX_PATH="/home/maksim/spack/opt/spack/linux-ubuntu22.04-skylake/gcc-11.4.0/hpx-1.9.1-geexjwq4h5szdenwju6rug26fad627bb/lib/cmake/HPX"

# Include dir
# include_directories(/usr/local/include)
include_directories(${HPX_INCLUDE_DIRS})

# Src
AUX_SOURCE_DIRECTORY(src SRC_FILES)

# Headers
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include/automobile_bits")

# Source files
set(SOURCE_FILES
    ${PROJECT_INCLUDE_DIR}/motorcycle.hpp
    ${PROJECT_SOURCE_DIR}/motorcycle.cpp
)

# Set up such that XCode organizes the files correctly
# source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

# Add library
add_library(automobile SHARED ${SOURCE_FILES})

# Link HPX libraries
target_link_libraries(automobile PRIVATE ${HPX_LIBRARIES})

# Include directories
target_include_directories(automobile PRIVATE include/automobile_bits ${HPX_INCLUDE_DIRS})

# Set default installation prefix if not specified by user
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/home/maksim/simtech/thesis/GPPPy_hpx/hpx_py11/cmake_cpp" CACHE PATH "Installation Directory" FORCE)
endif()

# Install
install(TARGETS automobile DESTINATION lib)

# Install the headers
install(FILES include/automobile DESTINATION include)

# Create base directory
install(DIRECTORY include/automobile_bits DESTINATION include)
