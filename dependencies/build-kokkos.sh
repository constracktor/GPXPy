#!/usr/bin/env bash
set -ex

: ${SOURCE_ROOT:?} ${INSTALL_ROOT:?} ${CMAKE_COMMAND:?}

DIR_SRC=${SOURCE_ROOT}/kokkos
DIR_BUILD=${INSTALL_ROOT}/kokkos/build
DIR_INSTALL=${INSTALL_ROOT}/kokkos

cd "${SOURCE_ROOT}"
if [ ! -d kokkos ] ; then
    git clone https://github.com/kokkos/kokkos.git
    cd kokkos && git checkout develop 
    cd ..
fi
mkdir -p "${DIR_BUILD}"
cd "${DIR_BUILD}"
${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -H${DIR_SRC} -DCMAKE_INSTALL_PREFIX=$DIR_INSTALL/kokkos -DKokkos_ENABLE_SERIAL=ON -DKokkos_CUDA=ON -DKokkos_ENABLE_HPX=ON -DHPX_DIR="${HPXSC_ROOT}/build/hpx/build/lib/cmake/HPX"
make -j${PARALLEL_BUILD} install

cd $BUILD_ROOT