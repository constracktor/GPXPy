#!/usr/bin/env bash

################################################################################
# Script for installing Spack with HPX+CUDA and MKL on a generic system
#
# - add `source $HOME/spack/share/spack/setup-env.sh` to your .bashrc file to
#   automatically load `spack` when opening a new terminal
# - may require manual changes depending on your system
################################################################################

# exit on error (non-zero status); print each command before execution
set -ex

# Install Spack ------------------------------------------------------------ {{{

# Clone spack git repository into $HOME/spack

# check if current directory is spack_env
if [[ $(basename "$(pwd)") != "spack_env" ]]; then
    echo "Please run this script from the spack_envdirectory"
    exit 1
fi
git clone -c feature.manyFiles=true https://github.com/spack/spack.git ~/spack

# use custom HPX package.py: added some CMake options at the end
cp ./hpx_package.py $HOME/spack/var/spack/repos/builtin/packages/hpx/package.py

# setup `spack` command
source $HOME/spack/share/spack/setup-env.sh

# }}} --------------------------------------------------------------------------

# Install HPX -------------------------------------------------------------- {{{

# intentionally not locating system-local packages, against problems with
# locally installed packages

# install dependencies into `gpxpy` spack environment
# (see spack.yaml in same directory for details)
spack env create gpxpy
cp ./spack.yaml $HOME/spack/var/spack/environments/gpxpy/spack.yaml
spack env activate gpxpy
spack concretize -f
mkdir -p $HOME/spack/var/spack/environments/gpxpy/.spack-env/view/include
spack install

# }}} --------------------------------------------------------------------------
