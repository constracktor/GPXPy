#!/usr/bin/env bash

set -ex

: ${SOURCE_ROOT:?} ${INSTALL_ROOT:?} ${CMAKE_VERSION:?}

DIR_SRC=${SOURCE_ROOT}/cmake
DIR_BUILD=${INSTALL_ROOT}/cmake/build
DIR_INSTALL=${INSTALL_ROOT}/cmake
FILE_MODULE=${INSTALL_ROOT}/modules/cmake/${CMAKE_VERSION}

DOWNLOAD_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz"

if [[ ! -d ${DIR_SRC} ]]; then
    (
        mkdir -p ${DIR_SRC}
        cd ${DIR_SRC}
        wget -O- ${DOWNLOAD_URL} | tar xz --strip-components=1
    )
fi

mkdir -p ${DIR_BUILD}
(
    cd ${DIR_BUILD}
    ${DIR_SRC}/bootstrap --parallel=${PARALLEL_BUILD} --prefix=${DIR_INSTALL} -- -DCMAKE_BUILD_TYPE=Release -DCMAKE_USE_OPENSSL=OFF
    make -j${PARALLEL_BUILD} install
)

mkdir -p $(dirname ${FILE_MODULE})
cat >${FILE_MODULE} <<EOF
#%Module
proc ModulesHelp { } {
  puts stderr {CMake}
}
module-whatis {CMake}
set root    ${DIR_INSTALL}
conflict    cmake
prepend-path    PATH            \$root/bin
prepend-path    ACLOCAL_PATH    \$root/share/aclocal
setenv          CMAKE_COMMAND   \$root/bin/cmake
EOF
