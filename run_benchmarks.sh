#!/bin/bash
################################################################################
# Diagnostics
################################################################################
set +x

################################################################################
# Variables
################################################################################
export APEX_SCREEN_OUTPUT=1 APEX_CSV_OUTPUT=1
export CMAKE_COMMAND=cmake
###
# TODO
export HPX_DIR=/home/maksim/spack/opt/spack/linux-ubuntu22.04-skylake/gcc-11.4.0/hpx-1.9.1-geexjwq4h5szdenwju6rug26fad627bb/lib
export MKL_DIR=/home/maksim/mkl/install/mkl/2024.1/lib
###

# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build && $CMAKE_COMMAND .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${HPX_DIR}/cmake/HPX" -DMKL_DIR="${MKL_DIR}/cmake/mkl" ${MKL_CONFIG} && make all
################################################################################
# Run benchmark script
################################################################################
cd ../benchmark_scripts
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
  #./tiles_script.sh $START $END $STEP $N_TRAIN $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_TILES
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
  START=1000
  END=1000
  STEP=1000
  N_CORES=2
  TILE_SIZE=100
  N_TEST=500
  N_REG=10
  ./data_script.sh $START $END $STEP $TILE_SIZE $N_TEST $N_REG $N_CORES $CHOLESKY $LOOP $OUTPUT_FILE_DATA
done
