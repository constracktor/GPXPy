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

pybind11_add_module(gpxpy_bindings ${SOURCE_FILES} ${HEADER_FILES})
# must match the Python module name!
set_target_properties(gpxpy_bindings PROPERTIES OUTPUT_NAME "gpxpy")

install(TARGETS gpxpy_bindings DESTINATION "${CMAKE_INSTALL_PREFIX}")
target_link_libraries(gpxpy_bindings PUBLIC GPXPy::core)
