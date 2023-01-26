#!/usr/bin/env bash

set -ex

: ${SOURCE_ROOT:?} ${INSTALL_ROOT:?} ${GCC_VERSION:?}

DIR_SRC=${SOURCE_ROOT}/gcc
DIR_BUILD=${INSTALL_ROOT}/gcc/build
DIR_INSTALL=${INSTALL_ROOT}/gcc
FILE_MODULE=${INSTALL_ROOT}/modules/gcc/${GCC_VERSION}

DOWNLOAD_URL="https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.xz"

if [[ ! -d ${DIR_SRC} ]]; then
    (
        mkdir -p ${DIR_SRC}
        cd ${DIR_SRC}
        wget -O- ${DOWNLOAD_URL} | tar xJ --strip-components=1
        ./contrib/download_prerequisites
    )
fi

(
    unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE

    mkdir -p ${DIR_BUILD}
    cd ${DIR_BUILD}

    ${DIR_SRC}/configure --prefix=${DIR_INSTALL} --enable-languages=c,c++ --disable-multilib --disable-nls
    make -j${PARALLEL_BUILD}
    make install
)
