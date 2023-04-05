#!/usr/bin/env bash
set -ex

: ${SOURCE_ROOT:?} ${INSTALL_ROOT:?} ${CMAKE_COMMAND:?}

DIR_SRC=${SOURCE_ROOT}/kokkos-hpx
DIR_BUILD=${INSTALL_ROOT}/kokkos-hpx/build
DIR_INSTALL=${INSTALL_ROOT}/kokkos-hpx

cd "${SOURCE_ROOT}"
if [ ! -d kokkos-hpx ] ; then
    git clone https://github.com/STEllAR-GROUP/hpx-kokkos.git
fi
mkdir -p "${DIR_BUILD}"
cd "${DIR_BUILD}"
${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -H${DIR_SRC} \
    -DCMAKE_INSTALL_PREFIX=$DIR_INSTALL/kokkos-hpx \
    -DHPX_DIR="${HPXSC_ROOT}/build/hpx/build/lib/cmake/HPX" \
    -DKokkos_DIR="${HPXSC_ROOT}/build/kokkos/kokkos/lib/cmake/Kokkos" \
    -DHPX_KOKKOS_ENABLE_TESTS=ON \
    -DHPX_HPX_KOKKOS_ENABLE_BENCHMARKS=ON
#make -j${PARALLEL_BUILD} install

#cd $BUILD_ROOT