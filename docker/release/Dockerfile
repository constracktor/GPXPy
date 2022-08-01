FROM ubuntu:22.04

SHELL ["/bin/bash","-c"]

# install essentials
RUN apt-get update -y && apt-get install -y vim make cmake git gcc g++ libboost-all-dev hwloc libasio-dev

# add user
RUN useradd -ms /bin/bash user
USER user
# create working directory
WORKDIR /home/user/

# clone repository
RUN git clone https://github.com/constracktor/hpx_project.git

# clone hpx
RUN git clone https://github.com/STEllAR-GROUP/hpx.git
# compile hpx in release mode
RUN cd hpx && git checkout 1.8.0 && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/user/hpx/build/lib -DHPX_WITH_EXAMPLES=0FF -DHPX_WITH_TEST=OFF -DHPX_WITH_NETWORKING=OFF -DHPX_WITH_MAX_CPU_COUNT=128 -DHPX_WITH_MALLOC=system .. && cmake --build . --target install
