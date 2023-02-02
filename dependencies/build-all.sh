#!/usr/bin/env bash

################################################################################
# Command-line help
################################################################################
print_usage_abort ()
{
    cat <<EOF >&2
SYNOPSIS
    ${0} {Release|RelWithDebInfo|Debug}
    {with-gcc|with-clang|with-CC|with-CC-clang}
    {with-cuda|without-cuda}
    [gcc|clang]
DESCRIPTION
    Download, configure, build, and install HPXSc and its dependencies.
EOF
    exit 1
}

################################################################################
# Diagnostics
################################################################################
set -e
set -x

################################################################################
# Command-line options
################################################################################
# Determine build type
if [[ "$1" == "Release" || "$1" == "RelWithDebInfo" || "$1" == "Debug" ]]; then
    export BUILD_TYPE=$1
    echo "Build Type: ${BUILD_TYPE}"
else
    echo 'Build type must be provided and has to be "Release", "RelWithDebInfo", or "Debug"' >&2
    print_usage_abort
fi
# Determine GPu support
if [[ "$3" == "without-cuda" ]]; then
    export HPX_WITH_CUDA=OFF
    echo "CUDA Support: Disabled"
elif [[ "$3" == "with-cuda" ]]; then
    export HPX_WITH_CUDA=ON
    echo "CUDA Support: Enabled"
else
    echo 'CUDA support must be specified and has to be "with-cuda" or "without-cuda"' >&2
    print_usage_abort
fi
# Determine compiler
if [[ "$2" == "with-gcc" ]]; then
    export HPX_WITH_CLANG=OFF
    export HPX_WITH_CUDA=OFF
    echo "Using self-built gcc - CUDA Support diasbled"
elif [[ "$2" == "with-clang" ]]; then
    export HPX_WITH_CLANG=ON
    echo "Using self-built clang "
elif [[ "$2" == "with-CC" ]]; then
    export HPX_WITH_CLANG=OFF
    export HPX_USE_CC_COMPILER=ON
    export HPX_WITH_CUDA=OFF
    echo "Using CC / CXX compiler (whatever it may be) - CUDA Support diasbled"
elif [[ "$2" == "with-CC-clang" ]]; then
    export HPX_WITH_CLANG=ON
    export HPX_USE_CC_COMPILER=ON
    echo "Using CC / CXX compiler (but expecting it to be some kind of clang)"
else
    echo 'Compiler must be specified with "with-gcc" or "with-clang" or "with-CC" or "with-CC-clang"' >&2
    print_usage_abort
fi
export HPX_COMPILER_OPTION="$2"

################################################################################
# Configuration
################################################################################
# Script directory
export HPXSC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )"
# Set Build Configuration Parameters
source config.sh

################################################################################
# Create source and installation directories
################################################################################
mkdir -p ${SOURCE_ROOT} ${INSTALL_ROOT}

################################################################################
# Build tools
################################################################################
# Build Compiler and set Compiler Environment Variables
if [[ "${HPX_COMPILER_OPTION}" == "with-gcc" ]]; then
    echo "Building GCC"
    ./build-gcc.sh
    echo "Configuring self-built GCC"
    source gcc-config.sh
elif [[ "${HPX_COMPILER_OPTION}" == "with-clang" ]]; then
    echo "Building clang"
    ./build-clang.sh
    echo "Configuring self-built clang"
    source clang-config.sh
elif [[ "${HPX_COMPILER_OPTION}" == "with-CC" ]]; then
    echo "Configuring GCC"
    source gcc-config.sh
elif [[ "${HPX_COMPILER_OPTION}" == "with-CC-clang" ]]; then
    echo "Configuring clang"
    source clang-config.sh
fi

echo "Building CMake"
./build-cmake.sh
export CMAKE_COMMAND=${INSTALL_ROOT}/cmake/bin/cmake

################################################################################
# Dependencies
################################################################################
if [[ "${HPX_WITH_CUDA}" == "ON" ]]; then
    # load cuda module (on pcsgs05 use e.g. 11.0.3)
    module load cuda/${CUDA_VERSION}

    echo "Building CPPuddle"
    ./build-cppuddle.sh
fi

echo "Building MKL"
#./build-mkl.sh

echo "Building Boost"
./build-boost.sh

echo "Building hwloc"
./build-hwloc.sh

echo "Building jemalloc"
./build-jemalloc.sh

echo "Building HPX"
./build-hpx.sh
