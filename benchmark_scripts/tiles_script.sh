#!/bin/bash
START=$1
END=$2
STEP=$3
N_TRAIN=$4
N_TEST=$5
N_REG=$6
CHOLESKY=$7
N_LOOP=$8
APEX_FILE="../build/apex.0.csv"
ERROR_FILE="../build/error.csv"
OUTPUT_FILE="tiles_result.txt"

touch $ERROR_FILE
touch $OUTPUT_FILE && echo 'Cores;Tiles;N_train;N_test;N_regressor;Algorithm;Total_time;Assemble_time;Cholesky_time;Triangular_time;Predict_time;Error' >> $OUTPUT_FILE

for (( i=$START; i<=$END; i=i+$STEP ))
do
  for (( l=0; l<$N_LOOP; l=l+1 ))
  do
    cd ../build && touch $ERROR_FILE
    ./hpx_cholesky --n_train $N_TRAIN --n_test $N_TEST --n_regressors $N_REG --n_tiles $i --cholesky $CHOLESKY
    cd ../benchmark_scripts && ./output_formater.sh $i $N_TRAIN $N_TEST $N_REG $CHOLESKY $APEX_FILE $OUTPUT_FILE $ERROR_FILE
    cd ../build && rm $ERROR_FILE
  done
done
