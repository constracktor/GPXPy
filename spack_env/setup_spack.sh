#!/usr/bin/env bash

################################################################################
# script for installing spack with HPX+CUDA with Intel-OneAPI-MKL
# on generic system
#
# - add `source $HOME/spack/share/spack/setup-env.sh` to your .bashrc file to
#   automatically load `spack` when opening a new terminal
################################################################################

# exit on error; print each command
set -ex

# clone Spack repo in $HOME/spack
DIR=$(pwd)
cd $HOME
git clone -c feature.manyFiles=true https://github.com/spack/spack.git

# setup `spack` command
source $HOME/spack/share/spack/setup-env.sh

# intentionally not locating system-local packages

# install dependencies into `gpxpy` spack environment
# (see spack.yaml in same directory for details)
spack env create gpxpy
cp $DIR/spack_env/spack.yaml $HOME/spack/var/spack/environments/gpxpy/spack.yaml
spack env activate gpxpy
spack concretize -f
spack install
