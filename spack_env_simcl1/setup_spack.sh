#!/usr/bin/env bash

################################################################################
# Script to install Spack and HPX+CUDA on simcl1n1 or simcl1n2
#
# - afterwards add `source $HOME/spack/share/spack/setup-env.sh` to your
#   `.bashrc` file to automatically load `spack`
# - uses local Clang 17.0.1 and CUDA 12.2.2 modules
# - installs Boost into `~/local/`
# - uses custom HPX package.py to add some CMake options
################################################################################

# exit on error (non-zero status); print each command before execution
set -ex

# Install Spack ----------------------------------------------------------------

# Clone spack git repository into $HOME/spack
DIR=$(pwd)
cd $HOME
git clone -c feature.manyFiles=true https://github.com/spack/spack.git

# use custom HPX package.py: added some CMake options at the end
cp ./hpx_package.py $HOME/spack/var/spack/repos/builtin/packages/hpx/package.py

# setup `spack` command
source $HOME/spack/share/spack/setup-env.sh

# ------------------------------------------------------------------------------

# Install HPX ------------------------------------------------------------------

# Load modules
module load clang/17.0.1
module load cuda/12.2.2

# Install Boost into ~/local/ (spack install of Boost does not work)
./build-boost.sh

# Find compilers, external packages, and CUDA
spack compiler find
spack external find
spack external find cuda

# Install dependencies into `gpxpy` spack environment
# (see ./spack.yaml for details)
spack env create gpxpy
cp $DIR/spack_env/spack.yaml $HOME/spack/var/spack/environments/gpxpy/spack.yaml
spack env activate gpxpy
spack concretize -f
spack install

# ------------------------------------------------------------------------------
