FROM ubuntu:22.04

SHELL ["/bin/bash","-c"]

# install essentials
RUN apt-get update -y && apt-get install -y git wget xz-utils bzip2 libz-dev build-essential autoconf

# install dev tools (not necessary to run benchmark script)
RUN apt-get install -y vim cmake

# add user
RUN useradd -ms /bin/bash user
USER user
# create working directory
WORKDIR /home/user/

# clone repository
RUN git clone https://github.com/constracktor/hpx_project.git

# install dependencies
RUN cd hpx_project/dependencies && ./build-all.sh Release
