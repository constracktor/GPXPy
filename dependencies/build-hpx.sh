#!/usr/bin/env bash

set -ex

: ${SOURCE_ROOT:?} ${INSTALL_ROOT:?} ${LIB_DIR_NAME:?} ${GCC_VERSION:?} ${BUILD_TYPE:?} \
    ${CMAKE_VERSION:?} ${CMAKE_COMMAND:?} \
    ${BOOST_VERSION:?} ${BOOST_BUILD_TYPE:?} \
    ${JEMALLOC_VERSION:?} ${HWLOC_VERSION:?}${HPX_VERSION:?}

DIR_SRC=${SOURCE_ROOT}/hpx
DIR_BUILD=${INSTALL_ROOT}/hpx/build
DIR_INSTALL=${INSTALL_ROOT}/hpx
FILE_MODULE=${INSTALL_ROOT}/modules/hpx/${HPX_VERSION}-${BUILD_TYPE}

DOWNLOAD_URL="https://github.com/stellar-group/hpx/archive/${HPX_VERSION}.tar.gz"

if [[ ! -d ${DIR_SRC} ]]; then
    (
      mkdir -p ${DIR_SRC}
      cd ${DIR_SRC}
      wget -O- ${DOWNLOAD_URL} | tar xz --strip-components=1
    )
fi

${CMAKE_COMMAND} \
    -H${DIR_SRC} \
    -B${DIR_BUILD} \
    -DCMAKE_INSTALL_PREFIX=${DIR_INSTALL} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${LDCXXFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${LDCXXFLAGS}" \
    -DHPX_WITH_CXX17=ON \
    -DHPX_WITH_FETCH_ASIO=ON\
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DHPX_WITH_THREAD_IDLE_RATES=ON \
    -DHPX_WITH_DISABLED_SIGNAL_EXCEPTION_HANDLERS=ON \
    -DHWLOC_ROOT=${INSTALL_ROOT}/hwloc/ \
    -DHPX_WITH_MALLOC=JEMALLOC \
    -DJEMALLOC_ROOT=${INSTALL_ROOT}/jemalloc \
    -DBOOST_ROOT=${INSTALL_ROOT}/boost \
    -DHPX_WITH_NETWORKING=OFF \
    -DHPX_WITH_MORE_THAN_64_THREADS=ON \
    -DHPX_WITH_MAX_CPU_COUNT=256 \
    -DHPX_WITH_EXAMPLES=OFF \
    -DHPX_WITH_TESTS=OFF \
    -DHPX_WITH_APEX=ON \
    -DHPX_WITH_CUDA=OFF \
    -DHPX_WITH_GPUBLAS=OFF

${CMAKE_COMMAND} --build ${DIR_BUILD} -- -j${PARALLEL_BUILD} VERBOSE=1
${CMAKE_COMMAND} --build ${DIR_BUILD} --target install
cp ${DIR_BUILD}/compile_commands.json ${DIR_SRC}/compile_commands.json

mkdir -p $(dirname ${FILE_MODULE})
cat >${FILE_MODULE} <<EOF
#%Module
proc ModulesHelp { } {
  puts stderr {HPX}
}
module-whatis {HPX}
set root    ${DIR_INSTALL}
conflict    hpx
module load gcc/${GCC_VERSION}
module load boost/${BOOST_VERSION}-${BOOST_BUILD_TYPE}
module load cmake/${CMAKE_VERSION}
module load jemalloc/${JEMALLOC_VERSION}
module load hwloc/${HWLOC_VERSION}
prereq      gcc/${GCC_VERSION}
prereq      boost/${BOOST_VERSION}-${BOOST_BUILD_TYPE}
prereq      cmake/${CMAKE_VERSION}
prereq      jemalloc/${JEMALLOC_VERSION}
prereq      hwloc/${HWLOC_VERSION}
prepend-path    CPATH              \$root/include
prepend-path    PATH               \$root/bin
prepend-path    LD_LIBRARY_PATH    \$root/lib
prepend-path    LIBRARY_PATH       \$root/lib
setenv          HPX_DIR            \$root/${LIB_DIR_NAME}/cmake/HPX
setenv          HPX_VERSION        ${HPX_VERSION}
EOF
