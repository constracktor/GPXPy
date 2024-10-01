# script to install spack and HPX on pcsgs02-05 (Version: 05.05.24)
#!/usr/bin/env bash
set -ex
# clone git repository
DIR=$(pwd)
cd
#git clone -c feature.manyFiles=true https://github.com/spack/spack.git
# configure spack (add this to your .bashrc file)
#source $HOME/spack/share/spack/setup-env.sh
# find external gcc
module load gcc/13.2.0
spack compiler find
# find external software
module load cmake
spack external find
# install HPX, MKL and all its dependencies
spack env create gpxpy
cp $DIR/spack.yaml $HOME/spack/var/spack/environments/gpxpy/spack.yaml
spack env activate gpxpy
spack concretize -f
spack install
