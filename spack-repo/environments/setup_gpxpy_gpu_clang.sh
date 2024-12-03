#!/usr/bin/env bash
# Script to setup GPU spack environment for GPXPy on simcl1n1-2
set -e
# create environment and copy config file
spack env create gpxpy_gpu_clang
cp spack_gpu_clang.yaml $HOME/spack/var/spack/environments/gpxpy_gpu_clang/spack.yaml
spack env activate gpxpy_gpu_clang
# find external compiler
module load clang/17.0.1
spack compiler find
# find external packages
#spack external find
spack external find python
spack external find ninja
module load cuda/12.2.2
spack external find cuda
# setup environment
spack concretize -f
spack install
