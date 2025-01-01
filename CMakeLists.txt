message(STATUS "Building Python language bindings for GPXPy.")

cmake_minimum_required(VERSION 3.16)

project(gpxpy)

include(CMakeDependentOption)

# What to build?
option(GPXPY_BUILD_CORE "Build the core library" ON)
cmake_dependent_option(GPXPY_BUILD_BINDINGS "Build the Python bindings" ON
                       "GPXPY_BUILD_CORE" OFF)

option(GPXPY_ENABLE_FORMAT_TARGETS "Enable clang-format / cmake-format targets"
       ${PROJECT_IS_TOP_LEVEL})

if(GPXPY_ENABLE_FORMAT_TARGETS)
  find_package(format QUIET)
  if(NOT format_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      format
      GIT_REPOSITORY https://github.com/TheLartians/Format.cmake.git
      GIT_TAG v1.8.1
      QUIET)
    FetchContent_MakeAvailable(format)
  endif()
endif()

if(GPXPY_BUILD_CORE)
  set(MKL_INTERFACE_FULL "intel_lp64")
  set(MKL_THREADING "sequential")

  find_package(HPX REQUIRED)
  find_package(MKL CONFIG REQUIRED)

  add_subdirectory(core)
  if(GPXPY_BUILD_BINDINGS)
    add_subdirectory(bindings)
  endif()
endif()
