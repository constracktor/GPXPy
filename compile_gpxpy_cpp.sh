#!/bin/bash
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################
# Load GCC compiler
module load gcc/13.2.0
module load cmake
export CC=gcc
export CXX=g++
# Activate spack environment
spack env activate gpxpy_cpu_gcc
# # Load Clang compiler
# module load clang/17.0.1
# export CC=clang
# export CXX=clang++
# # Activate spack environment
# spack env activate gpxpy_gpu_clang
# Configure APEX
export APEX_SCREEN_OUTPUT=1
# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

################################################################################
# Compile code
################################################################################
# note: dev defaults to debug builds!
cmake --preset dev-linux -DGPXPY_BUILD_BINDINGS=OFF
cmake --build --preset dev-linux
# ctest --preset dev-linux
