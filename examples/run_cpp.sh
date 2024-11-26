#!/bin/bash

################################################################################
# Run C++ test code (see test_cpp/)
################################################################################

# Exit on error (non-zero status); Print each command before executing it
set -ex

# Configuration ------------------------------------------------------------ {{{

# Load spack environment
source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Set cmake command
export CMAKE_COMMAND=$(which cmake)

# Activate APEX output to stdout
export APEX_SCREEN_OUTPUT=1

# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic \
                   -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

# Get CUDA architecture
export CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | \
                   awk -F '.' '{print $1$2}')

# }}} ----------------------------------------------------- end of Configuration

# Compilation ------------------------------------------------------------------

# Goto project directory
cd test_cpp

# Reset build directory
rm -rf build && mkdir build && cd build

# Configure project
$CMAKE_COMMAND .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DHPX_WITH_DYNAMIC_HPX_MAIN=ON \
    -DCMAKE_C_COMPILER=$(which clang) \
    -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_FLAGS="--cuda-gpu-arch=sm_${CUDA_ARCH} \
                        --cuda-path=${CUDA_HOME}" \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    ${MKL_CONFIG}

# Build project
make -j VERBOSE=1 all

# ----------------------------------------------------------- end of Compilation

# Running test code ------------------------------------------------------------

../test_cpp

# ----------------------------------------------------- end of Running test code
