cmake_minimum_required(VERSION 3.18)
project(hpx_project CXX)
find_package(HPX REQUIRED)

# set header file
set(HEADER_FILES src/headers/ublas_adapter.hpp src/headers/cublas_adapter.hpp src/headers/tiled_algorithms_cpu.hpp src/headers/tiled_algorithms_gpu.hpp src/headers/gp_functions.hpp)

if(BLAS)
   message(STATUS "Run BLAS Benchmark")

   add_executable(hpx_blas src/hpx_blas.cpp ${HEADER_FILES})
   target_link_libraries(hpx_blas HPX::hpx HPX::wrap_main HPX::iostreams_component)
else()
  if(GPU)
    message(STATUS "Run on GPU")

    add_executable(hpx_cholesky src/hpx_cholesky_gpu.cpp ${HEADER_FILES})
  else()
    message(STATUS "Run on CPU")

    add_executable(hpx_cholesky src/hpx_cholesky_cpu.cpp ${HEADER_FILES})
  endif()

  target_link_libraries(hpx_cholesky HPX::hpx HPX::wrap_main HPX::iostreams_component)
endif()
