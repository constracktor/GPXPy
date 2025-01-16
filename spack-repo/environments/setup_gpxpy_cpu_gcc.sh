#!/usr/bin/env bash
# Script to setup CPU spack environment for GPXPy on simcl1n1-4
set -e
# create environment and copy config file
#spack env create gpxpy_cpu_gcc
cp spack_cpu_gcc.yaml $HOME/spack/var/spack/environments/gpxpy_cpu_gcc/spack.yaml
spack env activate gpxpy_cpu_gcc
# find external compiler
module load gcc/13.2.0
spack compiler find
# find external packages
#spack external find
spack external find python
spack external find ninja
# setup environment
spack concretize -f
spack install
