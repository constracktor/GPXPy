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

    ${DIR_SRC}/configure --prefix=${DIR_INSTALL} --enable-languages=c,c++,fortran --disable-multilib --disable-nls
    make -j${PARALLEL_BUILD}
    make install
)

mkdir -p $(dirname ${FILE_MODULE})
cat >${FILE_MODULE} <<EOF
#%Module
proc ModulesHelp { } {
  puts stderr {GCC}
}
module-whatis {GCC}
set root    ${DIR_INSTALL}
conflict    GCC
prepend-path    CPATH              \$root/include
prepend-path    LD_LIBRARY_PATH    \$root/lib
prepend-path    LD_LIBRARY_PATH    \$root/lib64
prepend-path    LD_LIBRARY_PATH    \$root/lib/gcc/$(ls ${DIR_INSTALL}/lib/gcc/)/${GCC_VERSION}
prepend-path    LIBRARY_PATH       \$root/lib
prepend-path    LIBRARY_PATH       \$root/lib64
prepend-path    MANPATH            \$root/share/man
prepend-path    PATH               \$root/bin
setenv  GCC_VERSION     ${GCC_VERSION}
EOF
