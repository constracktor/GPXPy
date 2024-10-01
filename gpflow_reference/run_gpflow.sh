#!/bin/bash
# Activate spack environment
spack env activate gpxpy
# create python enviroment
python3 -m venv gpflow_env
# activate enviroment
source gpflow_env/bin/activate
# install requirements
pip3 install gpflow
# 
python3 execute.py
