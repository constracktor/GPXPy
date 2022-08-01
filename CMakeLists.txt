cmake_minimum_required(VERSION 3.18)
project(hpx_project CXX)
list(APPEND CMAKE_PREFIX_PATH "/home/user/hpx/build/lib/lib/cmake/HPX/")
find_package(HPX REQUIRED)

add_executable(hpx_cholesky src/hpx_cholesky.cpp)
target_link_libraries(hpx_cholesky HPX::hpx HPX::wrap_main HPX::iostreams_component)

#add_executable(hpx_program src/main.cpp)
#target_link_libraries(hpx_program HPX::hpx HPX::wrap_main HPX::iostreams_component)
#add_executable(lu_tiled src/lu_tiled.cpp)
#target_link_libraries(lu_tiled HPX::hpx HPX::wrap_main HPX::iostreams_component)
#add_executable(lu_tiled_dist src/lu_tiled_dist.cpp)
#target_link_libraries(lu_tiled_dist HPX::hpx HPX::wrap_main HPX::iostreams_component)
