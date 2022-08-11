#!/bin/bash
START=$1
END=$2
STEP=$3
N_TILES=$4
N_TEST=$5
N_REG=$6
N_CHOLESKY=$7
N_LOOP=$8
APEX_FILE="../build/apex.0.csv"
OUTPUT_FILE="data_result.txt"

touch $OUTPUT_FILE && echo 'Cores;Tiles;N_train;N_test;N_regressor;Algorithm;Total_time;Assemble_time;Cholesky_time;Triangular_time;Predict_time;Error' >> $OUTPUT_FILE

for (( i=$START; i<=$END; i=i+$STEP ))
do
  for (( l=0; l<=$N_LOOP; l=l+1 ))
  do
    cd ../build && ./hpx_cholesky --n_train $i --n_test $N_TEST --n_regressors $N_REG --n_tiles $N_TILES --cholesky $N_CHOLESKY | tee -a $APEX_FILE
    cd ../benchmark_scripts && ./output_formater.sh $N_TILES $i $N_TEST $N_REG $N_CHOLESKY $APEX_FILE $OUTPUT_FILE
  done
done
