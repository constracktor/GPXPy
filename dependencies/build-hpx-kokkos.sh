#!/usr/bin/env bash
set -ex

: ${SOURCE_ROOT:?} ${INSTALL_ROOT:?} ${CMAKE_COMMAND:?}

DIR_SRC=${SOURCE_ROOT}/hpx-kokkos
DIR_BUILD=${INSTALL_ROOT}/hpx-kokkos/build
DIR_INSTALL=${INSTALL_ROOT}/hpx-kokkos

cd "${SOURCE_ROOT}"
if [ ! -d hpx-kokkos ] ; then
    git clone https://github.com/STEllAR-GROUP/hpx-kokkos.git
fi
mkdir -p "${DIR_BUILD}"
cd "${DIR_BUILD}"
${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -H${DIR_SRC} \
    -DCMAKE_INSTALL_PREFIX=${DIR_INSTALL}/hpx-kokkos \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DHPX_DIR="${HPXSC_ROOT}/build/hpx/build/lib/cmake/HPX" \
    -DKokkos_DIR="${HPXSC_ROOT}/build/kokkos/kokkos/lib/cmake/Kokkos" \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DHPX_KOKKOS_ENABLE_TESTS=ON \
    -DHPX_HPX_KOKKOS_ENABLE_BENCHMARKS=ON
#make -j${PARALLEL_BUILD} install

#cd $BUILD_ROOT