export CC=clang${CC_CLANG_VERSION}
export CXX=clang++${CC_CLANG_VERSION}
export NVCC_WRAPPER_DEFAULT_COMPILER=clang${CC_CLANG_VERSION}
if [ -z "${HPX_USE_CC_COMPILER}" ]
then
    export CC=${INSTALL_ROOT}/clang/clang/bin/clang
    export CXX=${INSTALL_ROOT}/clang/clang/bin/clang++
    export NVCC_WRAPPER_DEFAULT_COMPILER=${CXX}
    export LD_LIBRARY_PATH=${INSTALL_ROOT}/clang/lib64:${LD_LIBRARY_PATH}
fi

export CFLAGS=-fPIC
export LDCXXFLAGS="${LDFLAGS} -std=c++17 "

case $(uname -i) in
    x86_64)
        export CXXFLAGS="-fPIC -march=native -ffast-math -std=c++17 "
        ;;
    *)
        echo 'Unknown architecture encountered.' 2>&1
        exit 1
        ;;
esac
