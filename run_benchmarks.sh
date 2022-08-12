#!/bin/bash
# Set variables
export APEX_SCREEN_OUTPUT=1 APEX_CSV_OUTPUT=1
export HPXSC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )/dependencies"
export CMAKE_COMMAND=${HPXSC_ROOT}/build/cmake/bin/cmake
# Compile Code
rm -rf build && mkdir build && cd build && $CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release && make all
# Run both scripts for for each tiled-decomposition
cd ../benchmark_scripts
rm tiles_result.txt
rm data_result.txt

LOOP=2
# Run cores_script
START=1
END=2
STEP=1
N_TRAIN=2000
N_TEST=1000
N_REG=100
CHOLESKY=left
./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $CHOLESKY $LOOP
CHOLESKY=right
#./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $CHOLESKY $LOOP
CHOLESKY=top
#./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $CHOLESKY $LOOP
# Run data_script
LOOP=2
START=1000
END=2000
STEP=1000
N_TILES=1
N_TEST=1000
N_REG=100
CHOLESKY=left
./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $CHOLESKY $LOOP
CHOLESKY=right
#./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $CHOLESKY $LOOP
CHOLESKY=top
#./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $CHOLESKY $LOOP
