#!/usr/bin/env bash
# script to install spack and hpx (Version: 06.11.23)
set -ex
# clone git repository
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
# configure spack (add this to your .bashrc file)
source spack/share/spack/setup-env.sh
# find external software
spack external find
# install hpx and all its dependencies
spack install hpx@1.9.1
# on pcsgs02-05 use
# spack install hpx@1.9.1%gcc@9.4.0
