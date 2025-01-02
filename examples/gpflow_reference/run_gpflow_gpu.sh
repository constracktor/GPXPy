#!/bin/bash

# Activate spack environment
source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Create & Activate python environment
if [ ! -d "gpflow_env" ]; then
    python3 -m venv gpflow_env
fi
source gpflow_env/bin/activate

# install gpflow if not already installed
if ! python3 -c "import gpflow"; then
    pip3 install gpflow
fi

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME/

# Execute the python script
python3 execute.py --use-gpu
