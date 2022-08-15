#!/bin/bash
# Set variables
export APEX_SCREEN_OUTPUT=1 APEX_CSV_OUTPUT=1
export HPXSC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )/dependencies"
export CMAKE_COMMAND=${HPXSC_ROOT}/build/cmake/bin/cmake
# Compile Code
rm -rf build && mkdir build && cd build && $CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${HPXSC_ROOT}/build/hpx/build/lib/cmake/HPX" && make all
# Run both scripts for for each tiled-decomposition
cd ../benchmark_scripts
CHOLESKY_VARIANTS="left right top"
LOOP=5
for CHOLESKY in $CHOLESKY_VARIANTS; do
  OUTPUT_FILE_TILES="tiles_hpx_${CHOLESKY}.txt"
  OUTPUT_FILE_DATA="data_hpx_${CHOLESKY}.txt"
  rm $OUTPUT_FILE_TILES
  rm $OUTPUT_FILE_DATA
  # Run cores_script
  START=10
  END=20
  STEP=10
  N_TRAIN=2000
  N_TEST=1000
  N_REG=100
  ./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_TILES
  # Run data_script
  START=1000
  END=4000
  STEP=1000
  N_TILES=10
  N_TEST=1000
  N_REG=100
  ./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $CHOLESKY $LOOP $OUTPUT_FILE_DATA
done
