#!/bin/bash
# Set variables
export APEX_SCREEN_OUTPUT=1 APEX_CSV_OUTPUT=1
export HPXSC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )/dependencies"
export CMAKE_COMMAND=${HPXSC_ROOT}/build/cmake/bin/cmake
# Compile Code
cd && cd hpx_project && rm -rf build && mkdir build && cd build && $CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release && make all
# Run both scripts for for each tiled-decomposition
cd && cd hpx_project/benchmark_scripts && rm tiles_result.txt && rm data_result.txt
LOOP=5
# Run cores_script
START=10
END=20
STEP=10
N_TRAIN=1000
N_TEST=1000
N_REG=100
N_CHOLESKY=left
cd && cd hpx_project/benchmark_scripts && ./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $N_CHOLESKY $LOOP
N_CHOLESKY=right
cd && cd hpx_project/benchmark_scripts && ./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $N_CHOLESKY $LOOP
N_CHOLESKY=top
cd && cd hpx_project/benchmark_scripts && ./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $N_CHOLESKY $LOOP
# Run data_script
START=1000
END=1000
STEP=1000
N_TILES=10
N_TEST=1000
N_REG=100
N_CHOLESKY=left
cd && cd hpx_project/benchmark_scripts && ./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $N_CHOLESKY $LOOP
N_CHOLESKY=right
cd && cd hpx_project/benchmark_scripts && ./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $N_CHOLESKY $LOOP
N_CHOLESKY=top
cd && cd hpx_project/benchmark_scripts && ./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $N_CHOLESKY $LOOP
