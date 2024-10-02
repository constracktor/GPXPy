#!/bin/bash
# Activate spack environment
spack env activate gpxpy
# create python enviroment
python3 -m venv gpytorch_env
# activate enviroment
source gpytorch_env/bin/activate
# install requirements
pip3 install gpytorch
# 
python3 execute.py
