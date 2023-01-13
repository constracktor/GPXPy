#!/usr/bin/env bash

set -ex

: ${SOURCE_ROOT:?} ${INSTALL_ROOT:?} ${GCC_VERSION:?} \
    ${BOOST_VERSION:?} ${BOOST_BUILD_TYPE:?} ${HPXSC_ROOT:?}

DIR_SRC=${SOURCE_ROOT}/boost

DIR_INSTALL=${INSTALL_ROOT}/boost
FILE_MODULE=${INSTALL_ROOT}/modules/boost/${BOOST_VERSION}-${BOOST_BUILD_TYPE}

DOWNLOAD_URL="http://downloads.sourceforge.net/project/boost/boost/${BOOST_VERSION}/boost_${BOOST_VERSION//./_}.tar.bz2"

if [[ ! -d ${DIR_SRC} ]]; then
    (
      # Get from sourceforge
      mkdir -p ${DIR_SRC}
      cd ${DIR_SRC}
      # When using the sourceforge link
      wget -O- ${DOWNLOAD_URL} | tar xj --strip-components=1
    )
fi

(
    cd ${DIR_SRC}

    ./bootstrap.sh --prefix=${DIR_INSTALL} --with-toolset=gcc

    ./b2 -j${PARALLEL_BUILD} "${flag1}" ${flag2} --with-atomic --with-filesystem --with-program_options --with-regex --with-system --with-chrono --with-date_time --with-thread --with-iostreams ${BOOST_BUILD_TYPE} install
)

mkdir -p $(dirname ${FILE_MODULE})
cat >${FILE_MODULE} <<EOF
#%Module
proc ModulesHelp { } {
  puts stderr {boost}
}
module-whatis {boost}
set root    ${DIR_INSTALL}
conflict    boost
module load gcc/${GCC_VERSION}
prereq      gcc/${GCC_VERSION}
prepend-path    CPATH           \$root
prepend-path    LD_LIBRARY_PATH \$root/lib
prepend-path    LIBRARY_PATH    \$root/lib
setenv          BOOST_ROOT      \$root
setenv          BOOST_VERSION   ${BOOST_VERSION}
EOF
