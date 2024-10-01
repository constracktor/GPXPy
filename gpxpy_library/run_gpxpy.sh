#!/bin/bash
# Activate spack environment
spack env activate gpxpy
cd test_gpxpy
# 
python3 execute.py
