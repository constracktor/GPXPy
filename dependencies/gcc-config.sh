export CC=gcc${CC_VERSION}
export CXX=g++${CC_VERSION}
if [ -z "$HPX_USE_CC_COMPILER" ]
then
    export CC=${INSTALL_ROOT}/gcc/bin/gcc
    export CXX=${INSTALL_ROOT}/gcc/bin/g++
    export LD_LIBRARY_PATH=${INSTALL_ROOT}/gcc/lib64:${LD_LIBRARY_PATH}
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