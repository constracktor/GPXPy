#!/bin/bash
START=$1
END=$2
STEP=$3
N_TILES=$4
N_TRAIN=$5
N_TEST=$6
N_REG=$7
CHOLESKY=$8
N_LOOP=$9
OUTPUT_FILE=${10}
APEX_FILE="../build/apex_profiles.csv"
ERROR_FILE="../build/error.csv"

touch $ERROR_FILE
touch $OUTPUT_FILE && echo "Algorithm;Cores;Tiles;N_train;N_test;N_regressor;Total_time;Assemble_time;Cholesky_time;Triangular_time;Predict_time;Error;${N_LOOP}" >> $OUTPUT_FILE

for (( i=$START; i<=$END; i=i*$STEP ))
do
  for (( l=0; l<$N_LOOP; l=l+1 ))
  do
    cd ../build && touch $ERROR_FILE
    ./hpx_cholesky -t$i --n_train $N_TRAIN --n_test $N_TEST --n_regressors $N_REG --n_tiles $N_TILES --cholesky $CHOLESKY
    cd ../benchmark_scripts && ./output_formater.sh $N_TILES $N_TRAIN $N_TEST $N_REG $i $CHOLESKY $APEX_FILE $OUTPUT_FILE $ERROR_FILE
    cd ../build && rm $ERROR_FILE
  done
done
