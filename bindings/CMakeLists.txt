# try finding pybind11
set(GPXPy_pybind11_VERSION 2.10.3)
find_package(pybind11 ${GPXPy_pybind11_VERSION} QUIET)
if(pybind11_FOUND)
  message(STATUS "Found package pybind11.")
else()
  message(STATUS "Couldn't find package pybind11. Building from source ...")
  include(FetchContent)
  set(FETCHCONTENT_QUIET FALSE)
  # fetch pybind11 library for creating Python bindings
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG "v${GPXPy_pybind11_VERSION}"
    GIT_SHALLOW TRUE
    QUIET)
  FetchContent_MakeAvailable(pybind11)
  message(STATUS "Installed pybind11 version ${GPXPy_pybind11_VERSION}.")
endif()

file(GLOB SOURCE_FILES CONFIGURE_DEPENDS "*.cpp")
file(GLOB HEADER_FILES CONFIGURE_DEPENDS "*.hpp")

source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${HEADER_FILES})

pybind11_add_module(gpxpy ${SOURCE_FILES} ${HEADER_FILES})

# Set the output directory for the automobile target
set_target_properties(
  gpxpy
  PROPERTIES
               LIBRARY_OUTPUT_DIRECTORY
               ${CMAKE_INSTALL_PREFIX}
               ARCHIVE_OUTPUT_DIRECTORY
               ${CMAKE_INSTALL_PREFIX}
               RUNTIME_OUTPUT_DIRECTORY
               ${CMAKE_INSTALL_PREFIX})
 
target_link_libraries(gpxpy PUBLIC GPXPy::core)
