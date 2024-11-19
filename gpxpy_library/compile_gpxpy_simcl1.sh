#!/bin/bash

################################################################################
# Build GPXPy Python library on simcl1n1 or simcl1n2
#
# Some system specific notes:
# - uses module cuda/12.2.2 and clang/17.0.1
# - requires setup of spack environment gpxpy
# - uses Clang as compiler for C, C++, and CUDA
################################################################################

# Exit on error (non-zero status); Print each command before execution
set -ex

# Configuration ------------------------------------------------------------ {{{

# Load modules
module load clang/17.0.1
module load cuda/12.2.2

# Load spack environment
source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Set cmake command
export CMAKE_COMMAND=$(which cmake)

# Configure APEX
export APEX_SCREEN_OUTPUT=1

# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic \
                   -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

# get CUDA architecture (80 for NVIDIA A30 GPU on simcl1n1 or simcl1n2)
export CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | \
                   awk -F '.' '{print $1$2}')

# }}} ----------------------------------------------------- end of Configuration

# Compile to ./build_gpxpy ------------------------------------------------- {{{

# Reset build directory
rm -rf build_gpxpy && mkdir build_gpxpy && cd build_gpxpy

# Configure the project
$CMAKE_COMMAND .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_LIBRARY_DIR=$(python3 -c "import site; \
                                       print(site.getsitepackages()[0])") \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DHPX_WITH_DYNAMIC_HPX_MAIN=ON \
    -DCMAKE_C_COMPILER=$(which clang) \
    -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_FLAGS="--cuda-gpu-arch=sm_${CUDA_ARCH} \
                        --cuda-path=/usr/local.nfs/sw/cuda/cuda-12.2.2" \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    ${MKL_CONFIG}

# Build project
make -j all

# }}} ------------------------------------------------- Compile to ./build_gpxpy
