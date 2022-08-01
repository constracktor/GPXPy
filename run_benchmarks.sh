#!/bin/bash
# Compile Code
cd && cd hpx_project && rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make all
# Run both scripts for for each tiled-decomposition

# Run cores_script
START=10
END=20
STEP=10
N_TRAIN=1000
N_TEST=1000
N_REG=100
N_CHOLESKY=left
cd && cd hpx_project/scripts && chmod +x tiles_script.sh && ./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $N_CHOLESKY | tee -a tiles_result.txt
N_CHOLESKY=right
cd && cd hpx_project/scripts && chmod +x tiles_script.sh && ./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $N_CHOLESKY | tee -a tiles_result.txt
N_CHOLESKY=top
cd && cd hpx_project/scripts && chmod +x tiles_script.sh && ./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $N_CHOLESKY | tee -a tiles_result.txt

# Run data_script
START=1000
END=2000
STEP=1000
N_TILES=10
N_TEST=1000
N_REG=100
N_CHOLESKY=left
cd && cd hpx_project/scripts && chmod +x data_script.sh && ./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $N_CHOLESKY | tee -a data_result.txt
N_CHOLESKY=right
cd && cd hpx_project/scripts && chmod +x data_script.sh && ./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $N_CHOLESKY | tee -a data_result.txt
N_CHOLESKY=top
cd && cd hpx_project/scripts && chmod +x data_script.sh && ./data_script.sh $START $END $STEP $N_TILES $N_TEST $N_REG $N_CHOLESKY | tee -a data_result.txt
