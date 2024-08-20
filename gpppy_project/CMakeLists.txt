message(STATUS "Building Python language bindings for GPPPy.")

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
    set(GPPPy_pybind11_VERSION v2.10.3)
    # fetch pybind11 library for creating Python bindings
    FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG ${GPPPy_pybind11_VERSION}
            GIT_SHALLOW TRUE
            QUIET
            )
    FetchContent_MakeAvailable(pybind11)
    message(STATUS "Installed pybind11 version ${GPPPy_pybind11_VERSION}.")
endif ()

project(gaussian_process)
find_package(HPX REQUIRED)
find_package(MKL CONFIG REQUIRED)

include_directories("${CMAKE_SOURCE_DIR}/cpp_code/include/gp_headers")
include_directories("${CMAKE_SOURCE_DIR}/python_code")
include_directories(${HPX_INCLUDE_DIRS})
include_directories(${MKL_INCLUDE_DIRS})

file (GLOB SOURCE_FILES "cpp_code/src/*.cpp")
file (GLOB HEADER_FILES "cpp_code/include/gp_headers/*.hpp")
file (GLOB PYTHON_FILES "python_code/*.cpp" "python_code/*.hpp")

source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${HEADER_FILES})
source_group("Python Files" FILES ${PYTHON_FILES})

# find_package(pybind11 REQUIRED)
# include(pybind11.cmake)
pybind11_add_module(gaussian_process 
	${SOURCE_FILES}
	${HEADER_FILES}
	${PYTHON_FILES}
)

# Calculate the parent directory of CMAKE_BINARY_DIR
get_filename_component(PARENT_BINARY_DIR "${CMAKE_BINARY_DIR}" DIRECTORY)

# Set the output directory for the automobile target
set_target_properties(gaussian_process PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PARENT_BINARY_DIR}/test"
    ARCHIVE_OUTPUT_DIRECTORY "${PARENT_BINARY_DIR}/test"
    RUNTIME_OUTPUT_DIRECTORY "${PARENT_BINARY_DIR}/test"
)


target_link_libraries(gaussian_process PUBLIC
                      HPX::hpx 
                      HPX::wrap_main 
                      HPX::iostreams_component
                      MKL::mkl_intel_lp64
                      MKL::mkl_core
                      MKL::MKL
                      MKL::mkl_sequential
                      )

install(TARGETS gaussian_process
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
  )
