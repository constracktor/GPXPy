#!/bin/bash
TILE_SIZE=$1
N_TRAIN=$2
N_TEST=$3
N_REG=$4
N_CORES=$5
CHOLESKY=$6
APEX_FILE=$7
OUTPUT_FILE=$8
ERROR_FILE=$9

TOTAL_TIME=$(sed -n "$(sed -n '/"APEX MAIN"/=' ${APEX_FILE}) p" ${APEX_FILE})
TOTAL_TIME=$(echo ${TOTAL_TIME%?} | sed 's/.*,//')

ASSEMBLY_TIME=$(sed -n "$(sed -n '/"assemble_tiled"/=' ${APEX_FILE}) p" ${APEX_FILE})
ASSEMBLY_TIME=$(echo ${ASSEMBLY_TIME%?} | sed 's/.*,//')

CHOLESKY_TIME=$(sed -n "$(sed -n '/"cholesky_tiled"/=' ${APEX_FILE}) p" ${APEX_FILE})
CHOLESKY_TIME=$(echo ${CHOLESKY_TIME%?} | sed 's/.*,//')

SOLVE_TIME=$(sed -n "$(sed -n '/"triangular_solve_tiled"/=' ${APEX_FILE}) p" ${APEX_FILE})
SOLVE_TIME=$(echo ${SOLVE_TIME%?} | sed 's/.*,//')

PREDICTION_TIME=$(sed -n "$(sed -n '/"prediction_tiled"/=' ${APEX_FILE}) p" ${APEX_FILE})
PREDICTION_TIME=$(echo ${PREDICTION_TIME%?} | sed 's/.*,//')

ERROR=$(sed -n "$(sed -n '/"error"/=' ${ERROR_FILE}) p" ${ERROR_FILE})
ERROR=$(echo ${ERROR} | sed 's/.*,//')

echo "$CHOLESKY;$N_CORES;$TILE_SIZE;$N_TRAIN;$N_TEST;$N_REG;$TOTAL_TIME;$ASSEMBLY_TIME;$CHOLESKY_TIME;$SOLVE_TIME;$PREDICTION_TIME;$ERROR" >> $OUTPUT_FILE
