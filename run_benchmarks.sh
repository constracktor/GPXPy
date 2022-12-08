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
CHOLESKY_VARIANTS="left"
LOOP=1
for CHOLESKY in $CHOLESKY_VARIANTS; do
  OUTPUT_FILE_CORES="cores_hpx_${CHOLESKY}.txt"
  OUTPUT_FILE_TILES="tiles_hpx_${CHOLESKY}.txt"
  OUTPUT_FILE_DATA="data_hpx_${CHOLESKY}.txt"
  rm $OUTPUT_FILE_CORES
  rm $OUTPUT_FILE_TILES
  rm $OUTPUT_FILE_DATA
  # Run cores_script
  START=1
  END=8
  STEP=2
  N_TRAIN=20000
  N_TEST=5000
  N_REG=100
  N_TILES=200
  #./cores_script.sh $START $END $STEP $N_TILES $N_TRAIN $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_CORES
  START=16
  END=128
  STEP=2
  N_TRAIN=20000
  N_TEST=5000
  N_REG=100
  N_TILES=200
  #./cores_script.sh $START $END $STEP $N_TILES $N_TRAIN $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_CORES
  # Run tiles_script
  START=10
  END=10
  STEP=10
  N_TRAIN=10000
  N_TEST=5000
  N_REG=100
  #./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_TILES
  START=100
  END=100
  #./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_TILES
  # Run tiles_script
  START=25
  END=200
  STEP=2
  N_TRAIN=20000
  N_TEST=5000
  N_REG=100
  #./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_TILES
  # Run data_script
  START=100
  END=900
  STEP=100
  N_TILES=1
  N_TEST=1000
  N_REG=100
  #./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_DATA
  # Run data_script
  START=1000
  END=10000
  STEP=1000
  N_TILES=1
  N_TEST=1000
  N_REG=100
  #./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_DATA
  # Run data_script
  START=10000
  END=100000
  STEP=10000
  N_TILES=200
  N_TEST=5000
  N_REG=100
  #./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_DATA
  # Run data_script
  START=10000
  END=10000
  STEP=10000
  N_TILES=200
  N_TEST=5000
  N_REG=100
  ./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_DATA
done
