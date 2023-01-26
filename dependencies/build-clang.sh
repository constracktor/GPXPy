#!/usr/bin/env bash
# clang requires requires cmake and git - gcc does not!
set -ex

: ${SOURCE_ROOT:?} ${INSTALL_ROOT:?} ${CLANG_VERSION:?}

DIR_SRC=${SOURCE_ROOT}/llvm-project
DIR_BUILD=${INSTALL_ROOT}/clang/build
DIR_INSTALL=${INSTALL_ROOT}/clang

cd "${SOURCE_ROOT}"
if [ ! -d llvm-project ] ; then
    git clone https://github.com/llvm/llvm-project
    cd llvm-project
    git checkout ${CLANG_VERSION}
    cd ..
fi
mkdir -p "${DIR_BUILD}"
cd "${DIR_BUILD}"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$DIR_INSTALL/clang -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi" -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" "${DIR_SRC}/llvm"
#installed before cmake -> clang requires cmake on system
#${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$DIR_INSTALL/clang -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi" -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" "${DIR_SRC}/llvm"
make -j${PARALLEL_BUILD} install

cd $BUILD_ROOT
