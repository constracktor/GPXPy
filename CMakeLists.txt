cmake_minimum_required(VERSION 3.18)
project(hpx_project CXX)
find_package(HPX REQUIRED)

# set header file
set(HEADER_FILES src/ublas/ublas_adapter.cpp)

# add ublas adapter library
#add_library(ublas src/ublas/ublas_adapter.cpp ${HEADER_FILES})
#target_link_libraries(ublas HPX::hpx HPX::wrap_main HPX::iostreams_component)

# build various executables
add_executable(hpx_cholesky src/hpx_cholesky.cpp ${HEADER_FILES})
target_link_libraries(hpx_cholesky HPX::hpx HPX::wrap_main HPX::iostreams_component)

add_executable(hpx_blas src/hpx_blas.cpp ${HEADER_FILES})
target_link_libraries(hpx_blas HPX::hpx HPX::wrap_main HPX::iostreams_component)

add_executable(cublas_matmul src/cublas_matmul.cpp)
target_link_libraries(cublas_matmul HPX::hpx HPX::wrap_main HPX::iostreams_component)
