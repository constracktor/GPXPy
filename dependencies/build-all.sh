#!/usr/bin/env bash

################################################################################
# Command-line help
################################################################################
print_usage_abort ()
{
    cat <<EOF >&2
SYNOPSIS
    ${0} {Release|RelWithDebInfo|Debug}
DESCRIPTION
    Download, configure, build, and install HPXSc and its dependencies.
EOF
    exit 1
}

################################################################################
# Command-line options
################################################################################
if [[ "$1" == "Release" || "$1" == "RelWithDebInfo" || "$1" == "Debug" ]]; then
    export BUILD_TYPE=$1
    echo "Build Type: ${BUILD_TYPE}"
else
    echo 'Build type must be provided and has to be "Release", "RelWithDebInfo", or "Debug"' >&2
    print_usage_abort
fi

################################################################################
# Diagnostics
################################################################################
set -e
set -x

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
#echo "Building GCC"
#./build-gcc.sh

echo "Building CMake"
./build-cmake.sh
export CMAKE_COMMAND=${INSTALL_ROOT}/cmake/bin/cmake
################################################################################
# Dependencies
################################################################################
echo "Building Boost"
./build-boost.sh

echo "Building hwloc"
./build-hwloc.sh

echo "Building jemalloc"
./build-jemalloc.sh

echo "Building HPX"
./build-hpx.sh
