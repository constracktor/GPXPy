#!/bin/bash
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################
# Load GCC compiler
module load gcc/13.2.0
CC_COMPILER=gcc
CXX_COMPILER=g++
# Activate spack environment
spack env activate gpxpy_cpu_gcc
# # Load Clang compiler
# module load clang/17.0.1
# CC_COMPILER=clang
# CXX_COMPILER=clang++
# # Activate spack environment
# spack env activate gpxpy_gpu_clang
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
                  -DCMAKE_C_COMPILER=$(which $CC_COMPILER) \
		  -DCMAKE_CXX_COMPILER=$(which $CXX_COMPILER) \
                  ${MKL_CONFIG}
 # Build the project
make -j VERBOSE=1 all

################################################################################
# Run code
################################################################################
../test_cpp
