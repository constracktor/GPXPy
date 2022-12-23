# TODO Currently only instructions for pcsgs05
# load cuda
module load cuda/11.0.3
# set Compiler to clang-12
#export CC=clang-12
export CXX=clang++-12

# more options to add vor full installation
# add the following options to HPX build script
#-DHPX_WITH_CUDA=ON \
#-DHPX_WITH_GPUBLAS=ON
#-> look in octotiger build chain for if condition between clang and GCC
# set different toolset in boost build script
#./bootstrap.sh --prefix=${DIR_INSTALL} --with-toolset=clang
