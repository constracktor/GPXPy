#!/bin/bash

################################################################################
# Run the GPXPy Python library on a test project
#
# - for simcl1n1 or simcl1n2
################################################################################

# Load necessary modules
module load cuda/12.2.2

# Activate spack environment
source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Run the python script
cd test_gpxpy
python3 execute.py
