#!/bin/bash

################################################################################
# This script compiles the GPXPy project (Python Library)
################################################################################

# Exit on error (non-zero status); Print each command before executing it
set -ex

# Configurations ---------------------------------------------------------------

# Load spack environment
source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Set cmake command
export CMAKE_COMMAND=$(which cmake)

# Configure APEX
export APEX_SCREEN_OUTPUT=1

# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

# ------------------------------------------------------------------------------

# Compile code -----------------------------------------------------------------

# Reset build directory
rm -rf build_gpxpy && mkdir build_gpxpy && cd build_gpxpy

# Configure project
$CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_LIBRARY_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])") \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DHPX_WITH_DYNAMIC_HPX_MAIN=ON \
    -DCMAKE_C_COMPILER=$(which gcc) \
    -DCMAKE_CXX_COMPILER=$(which g++) \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    ${MKL_CONFIG}

# Build project
make -j all

# ------------------------------------------------------------------------------
