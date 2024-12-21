#!/bin/bash

################################################################################
# Run the GPXPy Python library on a test project
################################################################################

# Activate spack environment
source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Run the python script
python3 execute.py
