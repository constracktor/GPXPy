#!/usr/bin/env bash
# Script to install and setup spack
set -e
# clone git repository
cd
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
# configure spack (add this to your .bashrc file)
source $HOME/spack/share/spack/setup-env.sh
# find external compilers
spack compiler find
# find external software
spack external find
# add GPXPy spack-repo so spack
spack repo add $(pwd)
