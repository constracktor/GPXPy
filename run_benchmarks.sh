#!/bin/bash
# Set variables
export APEX_SCREEN_OUTPUT=1 APEX_CSV_OUTPUT=1
export HPXSC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )/dependencies"
export CMAKE_COMMAND=${HPXSC_ROOT}/build/cmake/bin/cmake
# Compile Code
rm -rf build && mkdir build && cd build && $CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${HPXSC_ROOT}/build/hpx/build/lib/cmake/HPX" && make all
cd ../benchmark_scripts
# Run BLAS benchmark
OUTPUT_FILE_BLAS="blas_hpx.txt"
rm $OUTPUT_FILE_BLAS
touch $OUTPUT_FILE_BLAS
#../build/hpx_blas | tee $OUTPUT_FILE_BLAS
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
  END=256
  STEP=2
  N_TRAIN=20000
  N_TEST=5000
  N_REG=100
  N_TILES=200
  #./cores_script.sh $START $END $STEP $N_TILES $N_TRAIN $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_CORES
  ##############################################################################
  # Run tiles_script for cores 16,23,64,128
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
  #./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_TILES
  # for 500 tiles per dimension
  #./tiles_script.sh 500 500 2 $N_TRAIN $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_TILES
  ##############################################################################
  # Run data_script on 128 cores
  N_CORES=128
  N_TILES=200
  N_TEST=5000
  N_REG=100
  # from 10^3 to 10^4
  START=1000
  END=9000
  STEP=1000
  #./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_DATA
  # from 10^4 to 10^5
  START=10000
  END=100000
  STEP=10000
  #./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_DATA
  ##############################################################################
  # Run data_script for testing
  START=10000
  END=10000
  STEP=1000
  N_CORES=16
  N_TILES=10
  N_TEST=5000
  N_REG=100
  ./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_DATA
done
