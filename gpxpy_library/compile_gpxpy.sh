#!/bin/bash

################################################################################
# This script compiles the GPXPy project (Python Library)
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

# Compiling code & Making Python Library ----------------------------------- {{{

# Reset build directory
rm -rf build_gpxpy && mkdir build_gpxpy && cd build_gpxpy

# Configure project
$CMAKE_COMMAND .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DHPX_WITH_DYNAMIC_HPX_MAIN=ON \
    -DPYTHON_LIBRARY_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])") \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_C_COMPILER=$(which clang) \
    -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_FLAGS="--cuda-gpu-arch=sm_${CUDA_ARCH} \
        --cuda-path=${CUDA_HOME}" \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    ${MKL_CONFIG}

# Build project
make -j all

# }}} ---------------------------- end of Compiling code & Making Python Library
