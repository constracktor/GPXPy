#!/usr/bin/env bash

set -ex

: ${SOURCE_ROOT:?} ${INSTALL_ROOT:?} ${GCC_VERSION:?} ${JEMALLOC_VERSION:?}

DIR_SRC=${SOURCE_ROOT}/jemalloc
DIR_INSTALL=${INSTALL_ROOT}/jemalloc
FILE_MODULE=${INSTALL_ROOT}/modules/jemalloc/${JEMALLOC_VERSION}

DOWNLOAD_URL="https://github.com/jemalloc/jemalloc/releases/download/${JEMALLOC_VERSION}/jemalloc-${JEMALLOC_VERSION}.tar.bz2"

if [[ ! -d ${DIR_INSTALL} ]]; then
    (
        mkdir -p ${DIR_SRC}
        cd ${DIR_SRC}
        wget -O- ${DOWNLOAD_URL} | tar xj --strip-components=1
        ./autogen.sh
        ./configure --prefix=${DIR_INSTALL}
        make -j${PARALLEL_BUILD}
        make install
    )
fi

mkdir -p $(dirname ${FILE_MODULE})
cat >${FILE_MODULE} <<EOF
#%Module
proc ModulesHelp { } {
  puts stderr {jemalloc}
}
module-whatis {jemalloc}
set root    ${DIR_INSTALL}
conflict    jemalloc
module load gcc/${GCC_VERSION}
prereq      gcc/${GCC_VERSION}
prepend-path    CPATH              \$root/include
prepend-path    PATH               \$root/bin
prepend-path    PATH               \$root/sbin
prepend-path    MANPATH            \$root/share/man
prepend-path    LD_LIBRARY_PATH    \$root/lib
prepend-path    LIBRARY_PATH       \$root/lib
prepend-path    PKG_CONFIG_PATH    \$root/lib/pkgconfig
setenv          JEMALLOC_ROOT      \$root
setenv          JEMALLOC_VERSION   ${JEMALLOC_VERSION}
EOF
