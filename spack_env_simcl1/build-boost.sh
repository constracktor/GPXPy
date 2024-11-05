#!/usr/bin/env bash

################################################################################
# Build Boost from source
################################################################################

# exit on error (non-zero status); print each command before execution
set -ex

INSTALL_ROOT=$HOME/local
BOOST_VERSION=1.86.0

DOWNLOAD_URL="http://downloads.sourceforge.net/project/boost/boost/${BOOST_VERSION}/boost_${BOOST_VERSION//./_}.tar.bz2"

SOURCE_DIR=$(mktemp -d -t boost-${BOOST_VERSION}-XXXX)
echo ${SOURCE_DIR}
cd ${SOURCE_DIR}
wget -O- ${DOWNLOAD_URL} | tar xj --strip-components=1

./bootstrap.sh --prefix=${INSTALL_ROOT} --with-toolset=clang

export PARALLEL_BUILD=$(grep -c ^processor /proc/cpuinfo)

./b2 "${flag1}" ${flag2} -j${PARALLEL_BUILD} --with-atomic \
    --with-filesystem --with-program_options --with-regex --with-system \
    --with-chrono --with-date_time --with-thread --with-iostreams \
    variant=release install

rm -rf ${SOURCE_DIR}
