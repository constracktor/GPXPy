#!/bin/bash
# Create Python environment
python3 -m venv gpytorch_env
# Activate environment
source gpytorch_env/bin/activate
# Install requirements
pip3 install gpytorch
#
python3 execute.py
