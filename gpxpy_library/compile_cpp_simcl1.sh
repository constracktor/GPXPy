#!/bin/bash

################################################################################
# Build GPXPy C++ library on simcl1n1 or simcl1n2
#
# Some system specific notes:
# - uses module cuda/12.2.2 and clang/17.0.1
# - requires setup of spack environment gpxpy
# - assumes NVIDIA A30 GPU with compute capability 8.0
# - uses Clang as compiler for C, C++, and CUDA
################################################################################

# exit on error (non-zero status); print each command before execution
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

# Activate APEX output to stdout
export APEX_SCREEN_OUTPUT=1

# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic ' \
                  '-DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

# }}} ----------------------------------------------------- end of Configuration

# Compiling to ./build_cpp ------------------------------------------------- {{{

# Reset build directory
rm -rf build_cpp && mkdir build_cpp && cd build_cpp

# Configure the project
$CMAKE_COMMAND ../core \
    -DCMAKE_BUILD_TYPE=Release \
    -DHPX_WITH_DYNAMIC_HPX_MAIN=ON \
    -DCMAKE_C_COMPILER=$(which clang) \
    -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_FLAGS="--cuda-gpu-arch=sm_80 --cuda-path=${CUDA_HOME}" \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    ${MKL_CONFIG}

 # Build project
make -j all
make install

# ---------------------------------------------- end of Compiling to ./build_cpp
