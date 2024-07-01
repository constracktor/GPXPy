#!/bin/bash
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print each command before executing it.

################################################################################
# Ensure CMAKE_COMMAND is set
################################################################################
export CMAKE_COMMAND=$(which cmake)

if [ -z "$CMAKE_COMMAND" ]; then
  echo "Error: CMAKE_COMMAND environment variable is not set."
  exit 1
fi

export APEX_SCREEN_OUTPUT=1
export HPX_DIR=/home/maksim/spack/opt/spack/linux-ubuntu22.04-skylake/gcc-11.4.0/hpx-1.9.1-geexjwq4h5szdenwju6rug26fad627bb/lib
export MKL_DIR=/home/maksim/mkl/install/mkl/2024.1/lib

export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build
# $CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_LIBRARY_DIR="/usr/local/lib/python3.10/dist-packages" -DPYTHON_EXECUTABLE="/usr/bin/python3" -Dpybind11_DIR="/home/maksim/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11" # Configure the project

$CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_LIBRARY_DIR="/usr/local/lib/python3.10/dist-packages" -DPYTHON_EXECUTABLE="/usr/bin/python3" -DCMAKE_PREFIX_PATH="${HPX_DIR}/cmake/HPX" -DHPX_WITH_DYNAMIC_HPX_MAIN=ON -DMKL_DIR="${MKL_DIR}/cmake/mkl" ${MKL_CONFIG} # Configure the project

make -j4 all           # Build the project

#cd ../test
#python3 test_py.py
