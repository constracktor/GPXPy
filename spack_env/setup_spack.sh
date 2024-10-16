#!/usr/bin/env bash

# script to install spack and HPX on pcsgs02-05 (Version: 14.10.24)

set -ex

# clone spack git repository into $HOME/spack
DIR=$(pwd)
cd $HOME
git clone -c feature.manyFiles=true https://github.com/spack/spack.git

# configure spack (add this to your .bashrc file!)
source $HOME/spack/share/spack/setup-env.sh

# load necessary modules on pcsgs02-05
module load cuda/12.2.2
module load cmake

# find compilers and 
spack compiler find
spack external find
spack external find cuda

# install HPX, MKL, and its dependencies (specified in spack.yaml) into 
# spack environment gpxpy (can be loaded with `spack env activate gpxpy`)
spack env create gpxpy
cp $DIR/spack_env/spack.yaml $HOME/spack/var/spack/environments/gpxpy/spack.yaml
spack env activate gpxpy
spack concretize -f
spack install
