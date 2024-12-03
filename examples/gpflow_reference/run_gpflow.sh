#!/bin/bash
# Create Python environment
python3 -m venv gpflow_env
# Activate environment
source gpflow_env/bin/activate
# Install requirements
pip3 install gpflow
#
python3 execute.py
