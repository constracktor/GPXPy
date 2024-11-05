#!/bin/bash

################################################################################
# Run C++ test code (see test_cpp/)
################################################################################

# exit on error (non-zero status); print each command before executing it
set -ex

# Configurations ---------------------------------------------------------------

# load spack environment
source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# set cmake command
export CMAKE_COMMAND=$(which cmake)

# activate APEX output to stdout
export APEX_SCREEN_OUTPUT=1

# configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

# Compile code -----------------------------------------------------------------

# goto project directory
cd test_cpp

# reset build directory
rm -rf build && mkdir build && cd build

# configure project
$CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release \
    -DHPX_WITH_DYNAMIC_HPX_MAIN=ON \
    -DCMAKE_C_COMPILER=$(which gcc) \
    -DCMAKE_CXX_COMPILER=$(which g++) \
    ${MKL_CONFIG}

# build project
make -j VERBOSE=1 all

# ------------------------------------------------------------------------------

# Run test code ----------------------------------------------------------------

../test_cpp

# ------------------------------------------------------------------------------
