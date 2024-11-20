#!/bin/bash
# Activate spack environment
spack env activate gpxpy_cpu_gcc
#spack env activate gpxpy_gpu_clang
cd test_gpxpy
python3 execute.py
