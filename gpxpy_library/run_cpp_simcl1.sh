#!/bin/bash
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################
# Load GCC compiler
# module load gcc/13.2.0
module load cmake
module load cuda/12.2.2
# Activate spack environment
spack env activate gpxpy
# Set cmake command
export CMAKE_COMMAND=$(which cmake)
# Configure APEX
export APEX_SCREEN_OUTPUT=1
# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

################################################################################
# Compile code
################################################################################
cd test_cpp
rm -rf build && mkdir build && cd build
# Configure the project
$CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release \
                  -DHPX_WITH_DYNAMIC_HPX_MAIN=ON \
                  -DCMAKE_C_COMPILER=$(which clang) \
                  -DCMAKE_CXX_COMPILER=$(which clang++) \
                  -DCMAKE_CUDA_COMPILER=$(which clang++) \
                  -DCMAKE_CUDA_FLAGS="--cuda-gpu-arch=sm_80 --cuda-path=/usr/local.nfs/sw/cuda/cuda-12.2.2" \
                  -DCMAKE_CUDA_ARCHITECTURES=80 \
                  ${MKL_CONFIG}
 # Build the project
make -j VERBOSE=1 all

################################################################################
# Run code
################################################################################
../test_cpp
