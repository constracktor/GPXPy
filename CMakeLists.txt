message(STATUS "Building Python language bindings for GPXPy.")

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## try finding pybind11
find_package(pybind11 2.10.3 QUIET)
if (pybind11_FOUND)
    message(STATUS "Found package pybind11.")
else ()
    message(STATUS "Couldn't find package pybind11. Building from source ...")
    include(FetchContent)
    set(FETCHCONTENT_QUIET FALSE)
    set(GPXPy_pybind11_VERSION v2.10.3)
    # fetch pybind11 library for creating Python bindings
    FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG ${GPXPy_pybind11_VERSION}
            GIT_SHALLOW TRUE
            QUIET
            )
    FetchContent_MakeAvailable(pybind11)
    message(STATUS "Installed pybind11 version ${GPXPy_pybind11_VERSION}.")
endif ()

project(gpxpy)
find_package(HPX REQUIRED)
find_package(MKL CONFIG REQUIRED)

include_directories("${CMAKE_SOURCE_DIR}/core/include")
include_directories("${CMAKE_SOURCE_DIR}/bindings")
include_directories(${HPX_INCLUDE_DIRS})
include_directories(${MKL_INCLUDE_DIRS})

file (GLOB SOURCE_FILES "core/src/*.cpp")
file (GLOB HEADER_FILES "core/include/*.hpp")
file (GLOB BINDING_FILES "bindings/*.cpp" "python_code/*.hpp")

source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${HEADER_FILES})
source_group("Binding Files" FILES ${BINDING_FILES})

pybind11_add_module(gpxpy
	${SOURCE_FILES}
	${HEADER_FILES}
        ${BINDING_FILES}
)

# Calculate the parent directory of CMAKE_BINARY_DIR
get_filename_component(PARENT_BINARY_DIR "${CMAKE_BINARY_DIR}" DIRECTORY)

# Set the output directory for the automobile target
set_target_properties(gpxpy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PARENT_BINARY_DIR}/examples/gpxpy_python/install_python"
    ARCHIVE_OUTPUT_DIRECTORY "${PARENT_BINARY_DIR}/examples/gpxpy_python/install_python"
    RUNTIME_OUTPUT_DIRECTORY "${PARENT_BINARY_DIR}/examples/gpxpy_python/install_python"
)


target_link_libraries(gpxpy PUBLIC
                      HPX::hpx 
                      HPX::wrap_main 
                      HPX::iostreams_component
                      MKL::mkl_intel_lp64
                      MKL::mkl_core
                      MKL::MKL
                      MKL::mkl_sequential
                      )

install(TARGETS gpxpy
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
 )
