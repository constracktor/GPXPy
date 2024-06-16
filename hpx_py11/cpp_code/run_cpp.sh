#!/bin/bash
################################################################################
# Diagnostics
################################################################################
set +x

################################################################################
# Variables
################################################################################
# export APEX_SCREEN_OUTPUT=1 
#export APEX_CSV_OUTPUT=1
export CMAKE_COMMAND=cmake
###
export HPX_DIR=/home/maksim/spack/opt/spack/linux-ubuntu22.04-skylake/gcc-11.4.0/hpx-1.9.1-geexjwq4h5szdenwju6rug26fad627bb/lib

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build && $CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${HPX_DIR}/cmake/HPX" -DHPX_WITH_DYNAMIC_HPX_MAIN=ON
make all
make install
################################################################################
# Run benchmark script
################################################################################
cd ..

