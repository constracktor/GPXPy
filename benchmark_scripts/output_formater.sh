#!/bin/bash

#txt_file="$1"

APEX_FILE="../build/apex.0.csv"

N_CORES=$(grep -c ^processor /proc/cpuinfo)

TOTAL_TIME=$(sed -n "$(sed -n '/"APEX MAIN"/=' ${APEX_FILE}) p" ${APEX_FILE})
TOTAL_TIME=$(echo ${TOTAL_TIME%?} | sed 's/.*,//')
echo ${TOTAL_TIME}
ASSEMBLY_TIME=$(sed -n "$(sed -n '/"assemble_tiled"/=' ${APEX_FILE}) p" ${APEX_FILE})
ASSEMBLY_TIME=$(echo ${ASSEMBLY_TIME%?} | sed 's/.*,//')
echo ${ASSEMBLY_TIME}
CHOLESKY_TIME=$(sed -n "$(sed -n '/"cholesky_tiled"/=' ${APEX_FILE}) p" ${APEX_FILE})
CHOLESKY_TIME=$(echo ${CHOLESKY_TIME%?} | sed 's/.*,//')
echo ${CHOLESKY_TIME}
SOLVE_TIME=$(sed -n "$(sed -n '/"triangular_solve_tiled"/=' ${APEX_FILE}) p" ${APEX_FILE})
SOLVE_TIME=$(echo ${SOLVE_TIME%?} | sed 's/.*,//')
echo ${SOLVE_TIME}
PREDICTION_TIME=$(sed -n "$(sed -n '/"prediction_tiled"/=' ${APEX_FILE}) p" ${APEX_FILE})
PREDICTION_TIME=$(echo ${PREDICTION_TIME%?} | sed 's/.*,//')
echo ${PREDICTION_TIME}


cat ${APEX_FILE}
