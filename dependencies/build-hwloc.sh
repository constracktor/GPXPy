#!/usr/bin/env bash

set -ex

: ${SOURCE_ROOT:?} ${INSTALL_ROOT:?} ${GCC_VERSION:?} ${HWLOC_VERSION:?}

DIR_SRC=${SOURCE_ROOT}/hwloc
DIR_BUILD=${INSTALL_ROOT}/hwloc/build
DIR_INSTALL=${INSTALL_ROOT}/hwloc
FILE_MODULE=${INSTALL_ROOT}/modules/hwloc/${HWLOC_VERSION}

DOWNLOAD_URL="https://download.open-mpi.org/release/hwloc/v${HWLOC_VERSION%.*}/hwloc-${HWLOC_VERSION}.tar.gz"

if [[ ! -d ${DIR_SRC} ]]; then
    (
        mkdir -p ${DIR_SRC}
        cd ${DIR_SRC}
        wget -O- ${DOWNLOAD_URL} | tar xz --strip-components=1
    )
fi

(
    mkdir -p ${DIR_BUILD}
    cd ${DIR_BUILD}
    ${DIR_SRC}/configure --prefix=${DIR_INSTALL} --disable-opencl
    make -j ${PARALLEL_BUILD}
    make install
)
