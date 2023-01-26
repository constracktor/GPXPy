#${HPXSC_ROOT:?} ${BUILD_TYPE:?}

export INSTALL_ROOT=${HPXSC_ROOT}/build
export SOURCE_ROOT=${HPXSC_ROOT}/src

################################################################################
# Package Configuration
################################################################################
# CMake
export CMAKE_VERSION=3.19.5

# GCC
export GCC_VERSION=10.3.0
# specific version of system GCC
export CC_VERSION=-9

# clang
#export CLANG_VERSION=12.0.0
export CLANG_VERSION=release/12.x
# specific version of system clang
#export CC_CLANG_VERSION=-12

# Boost
echo ${BUILD_TYPE}
export BOOST_VERSION=1.75.0
export BOOST_ROOT=${INSTALL_ROOT}/boost
export BOOST_BUILD_TYPE=$(echo ${BUILD_TYPE/%WithDebInfo/ease} | tr '[:upper:]' '[:lower:]')

# jemalloc
export JEMALLOC_VERSION=5.2.1

# hwloc
export HWLOC_VERSION=1.11.12

# CUDA
export CUDA_VERSION=11.0.3

# HPX
export HPX_VERSION=1.8.0

# Max number of parallel jobs
export PARALLEL_BUILD=$(grep -c ^processor /proc/cpuinfo)

export LIB_DIR_NAME=lib
