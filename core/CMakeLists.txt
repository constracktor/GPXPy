add_library(
  gpxpy_core STATIC
  src/gp_optimizer.cpp
  src/gp_functions.cpp
  src/gp_algorithms_mkl.cpp
  src/gpxpy_c.cpp
  src/adapter_mkl.cpp
  src/tiled_algorithms_cpu.cpp
  src/gp_uncertainty.cpp
  src/utils_c.cpp)

add_library(GPXPy::core ALIAS gpxpy_core)

# Add them as PRIVATE sources here so they show up in project files Can't use
# PUBLIC etc., see: https://stackoverflow.com/a/62465051
file(GLOB_RECURSE header_files CONFIGURE_DEPENDS include/*.hpp)
target_sources(gpxpy_core PRIVATE ${header_files})

# Link HPX libraries
target_link_libraries(
  gpxpy_core PUBLIC HPX::hpx MKL::mkl_intel_lp64
                    MKL::mkl_core MKL::MKL MKL::mkl_sequential)

# Include directories
target_include_directories(
  gpxpy_core PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>")

target_compile_features(gpxpy_core PUBLIC cxx_std_17)

set_property(TARGET gpxpy_core PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS gpxpy_core DESTINATION "${CMAKE_INSTALL_PREFIX}/install/lib")

install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_PREFIX}/install/include")
