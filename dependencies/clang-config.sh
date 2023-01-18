export CC=clang
export CXX=clang++
export NVCC_WRAPPER_DEFAULT_COMPILER=clang

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
