# TODO build cuda 11.0.3 for GPU implementation

# Modifications for CUDA support
# set the following options in the HPX build script lines 49,50
-DHPX_WITH_CUDA=ON
-DHPX_WITH_GPUBLAS=ON
# set different toolset in boost build script in line 28
./bootstrap.sh --prefix=${DIR_INSTALL} --with-toolset=clang

# Currently: instructions for pcsgs05
# load cuda
module load cuda/11.0.3
# set CXX compiler to clang-12
export CXX=clang++-12
