#!/bin/bash
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################
# Load GCC compiler
module load gcc/13.2.0
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
#export APEX_SCREEN_OUTPUT=1

# Bindings
export BINDINGS=ON
export INSTALL_DIR=$(pwd)/examples/gpxpy_python
# export BINDINGS=OFF
# export INSTALL_DIR=$(pwd)/examples/gpxpy_cpp

# Release:	ci-ubuntu
# Debug:	dev-linux
export PRESET=ci-ubuntu

################################################################################
# Compile code
################################################################################
# note: dev defaults to debug builds!
cmake --preset $PRESET -DGPXPY_BUILD_BINDINGS=$BINDINGS -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
cmake --build --preset $PRESET
cmake --install build/$PRESET
# ctest --preset $PRESET
