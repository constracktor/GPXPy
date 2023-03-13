#!/bin/bash
################################################################################
# Diagnostics
################################################################################
set +x

################################################################################
# Variables
################################################################################
export APEX_SCREEN_OUTPUT=1 APEX_CSV_OUTPUT=1
export HPXSC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )/dependencies"
export CMAKE_COMMAND=${HPXSC_ROOT}/build/cmake/bin/cmake
export HPX_DIR=${HPXSC_ROOT}/build/hpx/build/lib
export MKL_DIR=${HPXSC_ROOT}/build/mkl/mkl/2023.0.0/lib
export CPPUDDLE_DIR=${HPXSC_ROOT}/build/cppuddle/build/cppuddle/lib

# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'
################################################################################
# Command-line options
################################################################################
# Determine Code
if [[ "$1" == "gpu" ]]
then
    GPU=1
    BLAS=0
    # load cuda module (on pcsgs05 use e.g. 11.0.3)
    module load cuda/11.0.3
    # set CPPuddle 
    #export CPPUDDLE_DIR=${HPXSC_ROOT}/build/cppuddle/build/cppuddle/lib
elif [[ "$1" == "cpu" ]]
then
    GPU=0
    BLAS=0
elif [[ "$1" == "blas" ]]
then
    GPU=0
    BLAS=1
else
  echo 'Please specify what to run: "cpu", "gpu" or "blas"'
  exit 1
fi
# Set Compiler Environment Variables
export INSTALL_ROOT=${HPXSC_ROOT}/build
export HPX_COMPILER_OPTION="$2"
if [[ "${HPX_COMPILER_OPTION}" == "with-gcc" ]]; then
    echo "Configuring self-built GCC"
    source dependencies/gcc-config.sh
elif [[ "${HPX_COMPILER_OPTION}" == "with-clang" ]]; then
    echo "Configuring self-built clang"
    source dependencies/clang-config.sh
elif [[ "${HPX_COMPILER_OPTION}" == "with-CC" ]]; then
    HPX_USE_CC_COMPILER=ON
    echo "Configuring GCC"
    source dependencies/gcc-config.sh
elif [[ "${HPX_COMPILER_OPTION}" == "with-CC-clang" ]]; then
    HPX_USE_CC_COMPILER=ON
    echo "Configuring clang"
    source dependencies/clang-config.sh
else
  echo 'Please specify compiler: "with-gcc" or "with-clang" or "with-CC" or "with-CC-clang"'
  exit 1
fi

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build && $CMAKE_COMMAND .. -DGPU=$GPU -DBLAS=$BLAS -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${HPX_DIR}/cmake/HPX" -DMKL_DIR="${MKL_DIR}/cmake/mkl" ${MKL_CONFIG} -DCPPuddle_DIR="${CPPUDDLE_DIR}/cmake/CPPuddle" && make all
################################################################################
# Run benchmark script
################################################################################
cd ../benchmark_scripts
if [ $BLAS == 1 ]
then
  # Run BLAS benchmark
  OUTPUT_FILE_BLAS="blas_hpx.txt"
  rm $OUTPUT_FILE_BLAS
  touch $OUTPUT_FILE_BLAS
  ../build/hpx_blas | tee $OUTPUT_FILE_BLAS
else
  # Run scripts for different tiled-decomposition
  #CHOLESKY_VARIANTS="left right top"
  CHOLESKY_VARIANTS="right"
  #LOOP=5
  LOOP=1
  for CHOLESKY in $CHOLESKY_VARIANTS; do
    OUTPUT_FILE_CORES="cores_hpx_${CHOLESKY}.txt"
    OUTPUT_FILE_TILES="tiles_hpx_${CHOLESKY}.txt"
    OUTPUT_FILE_DATA="data_hpx_${CHOLESKY}.txt"
    rm $OUTPUT_FILE_CORES
    rm $OUTPUT_FILE_TILES
    rm $OUTPUT_FILE_DATA
    ##############################################################################
    # Run cores_script
    START=1
    END=128
    STEP=2
    N_TRAIN=20000
    N_TEST=5000
    N_REG=100
    N_TILES=200
    #./cores_script.sh $START $END $STEP $N_TILES $N_TRAIN $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_CORES
    ##############################################################################
    # Run tiles_script for cores 16,128 cores on 2x EPYC 7742 and on 18 cores on Intel i9
    #N_CORES=18
    N_CORES=128
    N_TRAIN=20000
    N_TEST=5000
    N_REG=100
    # from 1 to 8 tiles per dimension
    START=1
    END=8
    STEP=2
    #./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_TILES
    # from 25 to 200 tiles per dimension
    START=25
    END=200
    STEP=2
    ./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_TILES
    # for 500 tiles per dimension
    #./tiles_script.sh 500 500 2 $N_TRAIN $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_TILES
    ##############################################################################
    # Run data_script on 128 cores on 2x EPYC 7742 and on 18 cores on Intel i9
    #N_CORES=18
    N_CORES=128
    TILE_SIZE=100
    N_TEST=5000
    N_REG=100
    # from 10^3 to 10^4
    START=1000
    END=9000
    STEP=1000
    #./data_script.sh $START $END $STEP $TILE_SIZE $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_DATA
    # from 10^4 to 10^5
    START=10000
    END=100000
    STEP=10000
    #./data_script.sh $START $END $STEP $TILE_SIZE $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_DATA
    ##############################################################################
    # Run data_script for testing
    START=10000
    END=10000
    STEP=1000
    N_CORES=18
    TILE_SIZE=200
    N_TEST=5000
    N_REG=100
    #./data_script.sh $START $END $STEP $TILE_SIZE $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_DATA
  done
fi
