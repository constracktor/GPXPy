#!/bin/bash

# Activate spack environment
source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Run the python script
cd test_gpxpy
python3 execute.py
