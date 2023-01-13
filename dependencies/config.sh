#${HPXSC_ROOT:?} ${BUILD_TYPE:?}

export INSTALL_ROOT=${HPXSC_ROOT}/build
export SOURCE_ROOT=${HPXSC_ROOT}/src

################################################################################
# Package Configuration
################################################################################
# CMake
export CMAKE_VERSION=3.19.5

# GCC - not activated
export GCC_VERSION=10.3.0

# clang - not implemented
export CLANG_VERSION=12.0.0

# Boost
export BOOST_VERSION=1.75.0
export BOOST_ROOT=${INSTALL_ROOT}/boost
export BOOST_BUILD_TYPE=$(echo ${BUILD_TYPE/%WithDebInfo/ease} | tr '[:upper:]' '[:lower:]')

# jemalloc
export JEMALLOC_VERSION=5.2.1

# hwloc
export HWLOC_VERSION=1.11.12

# CUDA - not implemented
export CUDA_VERSION=11.0.3

# HPX
export HPX_VERSION=1.8.0

# Max number of parallel jobs
export PARALLEL_BUILD=$(grep -c ^processor /proc/cpuinfo)

export LIB_DIR_NAME=lib
