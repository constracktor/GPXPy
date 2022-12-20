# TODO Currently only instructions for pcsgs05
# add the following options to HPX build script
-DHPX_WITH_CUDA=ON \
-DHPX_WITH_GPUBLAS=ON
# set different toolset in boost build script
./bootstrap.sh --prefix=${DIR_INSTALL} --with-toolset=clang
#-> look in octotiger build chain for if condition between clang and GCC
# load cuda
module load cuda/11.0.3
# set Compiler to clang-12
#export CC=clang-12
export CXX=clang++-12
# run code with
./cublas_matmul --sizemult=10 --iterations=25 --hpx:threads=8
# add to CMakeLists.txt
add_executable(cublas_matmul src/cublas_matmul.cpp)
target_link_libraries(cublas_matmul HPX::hpx HPX::wrap_main HPX::iostreams_component)
