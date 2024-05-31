cmake_minimum_required(VERSION 3.18)
project(hpx_project CXX)
find_package(HPX REQUIRED)
find_package(MKL CONFIG REQUIRED)

set(HEADER_FILES src/headers/mkl_adapter.hpp 
                 src/headers/tiled_algorithms_cpu.hpp
                 src/headers/gp_functions.hpp
                 src/headers/gp_functions_grad.hpp)


add_executable(hpx_cholesky src/hpx_cholesky_cpu.cpp ${HEADER_FILES})
target_link_libraries(hpx_cholesky 
                      HPX::hpx 
                      HPX::wrap_main 
                      HPX::iostreams_component
                      MKL::mkl_intel_lp64
                      MKL::mkl_core
                      MKL::MKL
                      MKL::mkl_sequential
                      )
