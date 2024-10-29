#!/usr/bin/env bash

################################################################################
# Script to install Spack and HPX on pcsgs02-05 (Version: 14.10.24)
#
# - add `source $HOME/spack/share/spack/setup-env.sh` to your .bashrc file to
#   automatically load `spack` when opening a new terminal
# - currently, does not work (with CUDA)
################################################################################

# exit on error, print each command
set -ex

# clone spack git repository into $HOME/spack
DIR=$(pwd)
cd $HOME
git clone -c feature.manyFiles=true https://github.com/spack/spack.git

# setup `spack` command
source $HOME/spack/share/spack/setup-env.sh

# load necessary modules on pcsgs02-05
module load cuda/12.2.2
module load cmake

# find compilers, external packages, and CUDA
spack compiler find
spack external find
spack external find cuda

# install dependencies into `gpxpy` spack environment
# (see spack.yaml in same directory for details)
spack env create gpxpy
cp $DIR/spack_env/spack.yaml $HOME/spack/var/spack/environments/gpxpy/spack.yaml
spack env activate gpxpy
spack concretize -f
spack install
