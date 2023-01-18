# export CC=gcc
# export CXX=g++
export CC=${INSTALL_ROOT}/gcc/bin/gcc
export CXX=${INSTALL_ROOT}/gcc/bin/g++
export NVCC_WRAPPER_DEFAULT_COMPILER=${CXX}
export LD_LIBRARY_PATH=${INSTALL_ROOT}/gcc/lib64:${LD_LIBRARY_PATH}

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
