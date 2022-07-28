# HOW TO
# Build image:
# sudo docker build . -f docker/Dockerfile -t hpx_image
#
# Run container inside hpx_project folder:
# sudo docker run -it --rm --user user --name hpx_container hpx_image
#
# Compile Code: (to speed up ublas '-DNDEBUG -DBOOST_UBLAS_NDEBUG')
# cd && cd hpx_project && rm -rf build && git pull --rebase && mkdir build && cd build && cmake .. && make all
# ./cholesky_hpx --n_train 1000 --n_test 1000 --n_regressors 100 --n_tiles 1 --cholesky left
# ijkalgorithm   prod   axpy_prod  block_prod
#   1.335       7.061    1.330       1.278
# Run Code:
# ./hpx_program
#
# Update Repository:
# git add . && git commit -m "comment" && git push
# git add . && git commit --amend --no-edit && git push -f

FROM ubuntu:22.04

SHELL ["/bin/bash","-c"]

# install essentials
RUN apt-get update -y && apt-get install -y vim make cmake git gcc g++ openmpi-bin libboost-all-dev hwloc libasio-dev

# add user
RUN useradd -ms /bin/bash user
USER user
# create working directory
WORKDIR /home/user/

# clone repository
RUN git clone https://github.com/constracktor/hpx_project.git

# clone hpx
RUN git clone https://github.com/STEllAR-GROUP/hpx.git
# compile hpx
RUN cd hpx && git checkout 1.8.0 && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/home/user/hpx/build/lib -DHPX_WITH_EXAMPLES=0FF -DHPX_WITH_TEST=OFF -DHPX_WITH_MAX_CPU_COUNT=128 -DHPX_WITH_MALLOC=system .. && cmake --build . --target install

# remove docker error message (somehow does not work and needs to be done manually)
# RUN export OMPI_MCA_btl_vader_single_copy_mechanism=none
